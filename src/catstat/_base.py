"""``_BaseStatEncoder`` -- the shared fit/transform/fit_transform skeleton.

All statistics/leakage logic lives here and in the small helper modules; only ``backends/``
knows pandas vs cuDF. Subclasses define the sklearn ``__init__`` params and two hooks:
``_is_supervised`` and ``_resolve_stat_specs``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from ._aggregations import fit_custom_encoding, fit_stat_encoding
from ._cross_fit import (
    build_joint_keyplan,
    complement_tables,
    decode_joint,
    factorize_active,
    finalize_dispersion_oof,
    finalize_mean_oof,
    finalize_shape_oof,
    finalize_woe_oof,
    gather,
    joint_codes,
    loo_encode,
    make_folds,
    ordered_encode,
    resolve_cv,
)
from ._feature_names import build_columns
from ._numeric import apply_numeric_col, fit_numeric_plan, validate_binning
from ._smoothing import fit_mean_encoding, woe_from_prob
from ._validation import (
    _is_numeric_like,
    check_handle,
    infer_target_type,
    normalize_keys,
    prepare_X,
    select_cols,
)
from .backends import _cpu
from .backends._dispatch import backend_module, select_backend

_VALID_OUTPUT = ("auto", "numpy", "pandas", "polars")
_DEFERRED_OUTPUT = ("cudf", "cupy")
# stats the single-pass kernel can finalize from (fold, key) power sums; skew/kurt need order-4
_ADDITIVE_STATS = frozenset({"mean", "var", "std", "skew", "kurt", "woe"})
_SHAPE_STATS = frozenset({"skew", "kurt"})


@dataclass(frozen=True)
class _UnitEncoding:
    """A fitted encoding for one ``(feature, stat, class)`` column, stored for code-gather.

    ``index`` is the encoding unit's canonical category index, shared *by reference* across all of
    the unit's columns -- so transform factorizes the unit's keys once (``index.get_indexer``) and
    gathers every column by integer code. ``values`` is the float64 encoding aligned to ``index``;
    ``fallback`` is the global statistic used for unknown / tiny-n categories (§11).
    """

    index: pd.Index
    values: np.ndarray
    fallback: float


class _BaseStatEncoder(TransformerMixin, BaseEstimator):
    # ---- hooks the subclasses must implement -------------------------------------------------
    def _is_supervised(self) -> bool:
        raise NotImplementedError

    def _resolve_stat_specs(self):
        raise NotImplementedError

    # ---- scikit-learn estimator tags ---------------------------------------------------------
    # catstat encoders are categorical encoders: they accept string/categorical columns and learn
    # NaN as its own level when handle_missing="value"; supervised encoders additionally require y.
    # Both tag APIs are provided -- __sklearn_tags__ for scikit-learn >= 1.6, and _more_tags for
    # < 1.6 (newer versions ignore it; older ones ignore __sklearn_tags__).
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = self._is_supervised()
        tags.input_tags.categorical = True
        tags.input_tags.string = True
        tags.input_tags.allow_nan = True
        return tags

    def _more_tags(self):
        return {
            "requires_y": self._is_supervised(),
            "X_types": ["categorical", "string", "2darray"],
            "allow_nan": True,
        }

    # ---- key helper --------------------------------------------------------------------------
    @staticmethod
    def _key(meta):
        return (meta.feature, meta.stat, meta.class_label)

    # ---- fit ---------------------------------------------------------------------------------
    def fit(self, X, y=None):
        check_handle("handle_unknown", self.handle_unknown)
        check_handle("handle_missing", self.handle_missing)
        mode = getattr(self, "multi_feature_mode", "independent")
        if mode not in ("independent", "combination"):
            raise ValueError(
                f"multi_feature_mode={mode!r} must be 'independent' or 'combination'."
            )
        if self.output in _DEFERRED_OUTPUT:
            raise NotImplementedError(f"output={self.output!r} is not supported in M0 (CPU).")
        if self.output not in _VALID_OUTPUT:
            raise ValueError(f"output={self.output!r} must be one of {_VALID_OUTPUT}.")
        numeric_mode = self._validate_numeric_params()

        Xdf, was_df, all_cols = prepare_X(X)
        self.n_features_in_ = Xdf.shape[1]
        if was_df:
            self.feature_names_in_ = np.asarray(all_cols, dtype=object)
        self._cat_cols = select_cols(Xdf, self.cols, numeric_mode)
        # Encoding units: independent -> one per column; combination -> one joint unit; then any
        # explicit `interactions` groups are appended as additional joint units (deduped by name).
        if mode == "combination" and len(self._cat_cols) > 1:
            joint = "+".join(str(c) for c in self._cat_cols)
            self._units = [(joint, list(self._cat_cols))]
        else:
            self._units = [(c, [c]) for c in self._cat_cols]
        self._add_interaction_units()
        self._unit_cols = dict(self._units)
        self._fit_numeric(Xdf, numeric_mode)
        # Per-combination-unit int64 joint-code plan, learned once from full X (X-only, so
        # leakage-safe) and reused at fit / per-fold / transform via _unit_keys.
        self._unit_keyplans = self._build_joint_keyplans(Xdf)
        self._specs = self._resolve_stat_specs()
        self.stats_ = [s.name for s in self._specs]

        supervised = self._is_supervised()
        if supervised:
            if y is None:
                raise ValueError(f"{type(self).__name__} requires y to be supplied to fit().")
            y_arr = np.asarray(y)
            if len(y_arr) != Xdf.shape[0]:
                raise ValueError("X and y have inconsistent lengths.")
            self.target_type_ = infer_target_type(y_arr, self.target_type)
            self.classes_ = (
                np.unique(y_arr) if self.target_type_ in ("binary", "multiclass") else None
            )
        else:
            y_arr = None
            self.target_type_ = None
            self.classes_ = None

        for spec in self._specs:
            if spec.continuous_only and self.target_type_ != "continuous":
                raise ValueError(
                    f"stat={spec.name!r} requires a continuous target; got "
                    f"target_type={self.target_type_!r}. Dispersion/order statistics on "
                    "classification targets are not supported."
                )
            # getattr: fitted estimators pickled before the field existed lack it on their specs
            if getattr(spec, "binary_only", False) and self.target_type_ != "binary":
                raise ValueError(
                    f"stat={spec.name!r} requires a binary target; got "
                    f"target_type={self.target_type_!r}. WOE is undefined for "
                    "regression/multiclass targets."
                )

        scheme = getattr(self, "scheme", "kfold")
        if scheme not in ("kfold", "loo", "ordered"):
            raise ValueError(f"scheme={scheme!r} must be 'kfold', 'loo', or 'ordered'.")
        if scheme != "kfold":
            bad = [s.name for s in self._specs if s.target_dependent and s.name != "mean"]
            if bad:
                raise ValueError(
                    f"scheme={scheme!r} cross-fits the mean only (count/frequency are allowed "
                    f"too); got target-dependent stats {bad}. Use scheme='kfold' for those."
                )

        self._columns_meta = build_columns(
            [name for name, _ in self._units], self._specs, self.target_type_, self.classes_
        )
        self.feature_names_out_ = np.asarray([m.name for m in self._columns_meta], dtype=object)

        all_gpu = all(s.gpu_supported for s in self._specs)
        self._backend_mod, self.backend_ = select_backend(
            self.backend, Xdf.shape[0], len(self._cat_cols), all_gpu
        )
        # Combination/interaction units now key on int64 mixed-radix joint codes (host-built and
        # GPU-groupable), so the device path handles them; only CPU-only stats (skew/custom, which
        # have no GPU primitive) force host -> `not all_gpu`.
        host_only = not all_gpu
        if self.backend_ == "gpu" and host_only:
            self._backend_mod, self.backend_ = _cpu, _cpu.NAME

        self._fit_tables = self._fit_all(Xdf, y_arr)

        # public fitted attributes derived from the full-data tables
        self.categories_ = {}
        self.global_stats_ = {}
        for meta in self._columns_meta:
            enc = self._fit_tables[self._key(meta)]
            self.global_stats_[meta.name] = enc.fallback
            if meta.feature in self.categories_:
                continue
            plan = self._unit_keyplans.get(tuple(self._unit_cols[meta.feature]))
            # Combination units now hold an int64 canonical index -> decode back to category tuples
            # so categories_ keeps its tuple representation (public API unchanged).
            cats = decode_joint(plan, enc.index.to_numpy()) if plan is not None else list(enc.index)
            self.categories_[meta.feature] = np.asarray(cats, dtype=object)
        self._set_target_mean()
        return self

    def _add_interaction_units(self):
        """Append one joint encoding unit per ``interactions`` group (additive to the base units).

        Each group is a list of categorical column names encoded together as a single tuple
        category named ``"+".join(group)`` (e.g. ``"a+b"``), on top of the independent/combination
        base. Groups must reference selected categorical columns; units are deduped by name.
        ``interactions=None`` (the default, and any encoder without the param) is a no-op. Joint
        keys are GPU-host-only (KI-018)."""
        interactions = getattr(self, "interactions", None)
        if not interactions:
            return
        if not isinstance(interactions, (list, tuple)) or not all(
            isinstance(g, (list, tuple)) and len(g) > 0 for g in interactions
        ):
            raise ValueError(
                "interactions must be a list of non-empty lists of column names, "
                "e.g. [['a', 'b'], ['a', 'c']]."
            )
        seen = {name for name, _ in self._units}
        for group in interactions:
            cols_g = list(group)
            missing = [c for c in cols_g if c not in self._cat_cols]
            if missing:
                raise ValueError(
                    f"interactions reference columns {missing} not among the encoded categorical "
                    f"columns {list(self._cat_cols)}."
                )
            name = "+".join(str(c) for c in cols_g)
            if name not in seen:
                self._units.append((name, cols_g))
                seen.add(name)

    def _set_target_mean(self):
        if not self._is_supervised() or "mean" not in self.stats_:
            return
        f0 = self._units[0][0]
        if self.target_type_ == "multiclass":
            self.target_mean_ = np.asarray(
                [self._fit_tables[(f0, "mean", c)].fallback for c in self.classes_], dtype=float
            )
        else:
            self.target_mean_ = float(self._fit_tables[(f0, "mean", None)].fallback)

    # ---- numeric-column encoding (opt-in, cardinality-aware) ---------------------------------
    def _validate_numeric_params(self) -> str:
        """Validate the opt-in numeric-encoding params; return the resolved numeric mode.

        These params live on ``TargetEncoder`` only; unsupervised encoders never define ``numeric``,
        so ``getattr`` yields ``"ignore"`` and the whole block is a no-op for them.
        """
        mode = getattr(self, "numeric", "ignore")
        if mode not in ("ignore", "auto", "direct", "bin"):
            raise ValueError(f"numeric={mode!r} must be one of 'ignore', 'auto', 'direct', 'bin'.")
        if mode == "ignore":
            return mode
        validate_binning(getattr(self, "binning", "quantile"))
        n_bins = getattr(self, "n_bins", 20)
        if isinstance(n_bins, bool) or not isinstance(n_bins, (int, np.integer)) or n_bins < 2:
            raise ValueError(f"n_bins={n_bins!r} must be an integer >= 2.")
        ct = getattr(self, "cardinality_threshold", 20)
        ok_int = isinstance(ct, (int, np.integer)) and not isinstance(ct, bool) and ct >= 1
        ok_float = isinstance(ct, float) and 0.0 < ct <= 1.0
        if not (ok_int or ok_float):
            raise ValueError(
                f"cardinality_threshold={ct!r} must be an int >= 1 (absolute unique count) "
                "or a float in (0, 1] (unique/n ratio)."
            )
        mbs = getattr(self, "min_bin_size", None)
        if mbs is not None:
            ok_i = isinstance(mbs, (int, np.integer)) and not isinstance(mbs, bool) and mbs >= 1
            ok_f = isinstance(mbs, float) and 0.0 < mbs <= 1.0
            if not (ok_i or ok_f):
                raise ValueError(
                    f"min_bin_size={mbs!r} must be None, an int >= 1 (absolute), or a float in "
                    "(0, 1] (fraction of n)."
                )
        return mode

    def _fit_numeric(self, Xdf, numeric_mode: str) -> None:
        """Build the per-column numeric encoding plan + introspection attrs (from full training X).

        Bin edges are a function of feature values only (never ``y``), so the plan is leakage-safe;
        the per-bin target statistic is still cross-fitted out-of-fold by ``fit_transform``. The
        plan maps each numeric-encoded column to its strategy and (for ``"bin"``) its edges, and is
        consulted by :meth:`_col_values` at every cross-fit scheme's key-building step.
        """
        plan: dict = {}
        if numeric_mode != "ignore":
            num_cols = [c for c in self._cat_cols if _is_numeric_like(Xdf[c].dtype)]
            plan = fit_numeric_plan(
                Xdf,
                num_cols,
                numeric_mode,
                self.cardinality_threshold,
                self.n_bins,
                self.binning,
                getattr(self, "min_bin_size", None),
            )
        self._numeric_plan_ = plan
        self.numeric_cols_ = list(plan)
        self.numeric_strategy_ = {c: e["strategy"] for c, e in plan.items()}
        self.bin_edges_ = {c: e["edges"] for c, e in plan.items() if e["strategy"] == "bin"}

    def _col_values(self, Xdf, c):
        """Raw values for column ``c``, numeric-encoded (binned / direct) when the plan covers it.

        Columns not in the numeric plan -- categoricals, or any column when numeric encoding is off
        -- pass through unchanged, so existing behavior is preserved exactly.
        """
        vals = Xdf[c].to_numpy()
        plan = getattr(self, "_numeric_plan_", None)
        if plan and c in plan:
            return apply_numeric_col(vals, plan[c])
        return vals

    # ---- per-statistic fitting ---------------------------------------------------------------
    def _fit_all(self, Xdf, y_arr, specs=None):
        """Return the encoding tables ``{(feature, stat, class): _UnitEncoding}``.

        ``specs`` defaults to all stats; the hybrid OOF slow path passes only the non-additive
        specs so the additive columns (already done by the single-pass kernel) cost nothing here.
        Each unit's columns are aligned to one canonical category index (the first column's), so
        transform factorizes the unit's keys once and gathers every column by integer code.
        """
        tables = {}
        hm = self.handle_missing
        specs = self._specs if specs is None else specs
        for feat, cols in self._units:
            keys_full, missing_mask = self._unit_keys(Xdf, cols)
            if hm == "error" and missing_mask.any():
                raise ValueError(
                    f"Missing values in unit {feat!r} with handle_missing='error'."
                )
            sel = np.ones(len(keys_full), dtype=bool) if hm == "value" else ~missing_mask
            keys = keys_full[sel]
            n_total = int(sel.sum())
            entries = []  # (table_key, Series, fallback) for this unit; aligned to canonical below
            for spec in specs:
                if spec.name == "count":
                    s, fb = self._fit_count(keys, False, n_total)
                    entries.append(((feat, "count", None), s, fb))
                elif spec.name == "frequency":
                    s, fb = self._fit_count(keys, True, n_total)
                    entries.append(((feat, "frequency", None), s, fb))
                elif spec.name == "mean":
                    y_sel = y_arr[sel]
                    bk = self._backend_mod
                    if self.target_type_ == "continuous":
                        s, fb = fit_mean_encoding(keys, y_sel.astype(float), self.smooth, bk)
                        entries.append(((feat, "mean", None), s, fb))
                    elif self.target_type_ == "binary":
                        yb = (y_sel == self.classes_[1]).astype(float)
                        s, fb = fit_mean_encoding(keys, yb, self.smooth, bk)
                        entries.append(((feat, "mean", None), s, fb))
                    else:  # multiclass: one-vs-rest per global class
                        for c in self.classes_:
                            yc = (y_sel == c).astype(float)
                            s, fb = fit_mean_encoding(keys, yc, self.smooth, bk)
                            entries.append(((feat, "mean", c), s, fb))
                elif spec.name == "woe":  # binary-only (gated at fit); prior = fold/global mean
                    yb = (y_arr[sel] == self.classes_[1]).astype(float)
                    s, prior = fit_mean_encoding(keys, yb, self.smooth, self._backend_mod)
                    woe = pd.Series(
                        woe_from_prob(s.to_numpy(dtype=float), prior), index=s.index
                    )
                    entries.append(((feat, "woe", None), woe, 0.0))
                else:  # var/std/median/min/max/skew/kurt or custom (continuous-only)
                    min_samples = getattr(self, "min_samples_category", 1)
                    y_sel_f = y_arr[sel].astype(float)
                    if spec.func is not None:
                        s, fb = fit_custom_encoding(keys, y_sel_f, spec.func, min_samples)
                    else:
                        s, fb = fit_stat_encoding(
                            keys, y_sel_f, spec.name, min_samples, self._backend_mod
                        )
                    entries.append(((feat, spec.name, None), s, fb))
            # Align the unit's encodings to one canonical category index (the first column's). Every
            # stat groups the same keys, so they share the category *set*; reindex only reorders --
            # no NaN is introduced -- and a unit's keys can then be factorized once at transform.
            if entries:
                canonical = entries[0][1].index
                for tkey, s, fb in entries:
                    vals = s.reindex(canonical).to_numpy(dtype=float)
                    tables[tkey] = _UnitEncoding(canonical, vals, fb)
        return tables

    @staticmethod
    def _fit_count(keys, normalize, n_total):
        vc = pd.Series(keys).value_counts().astype(float)
        if normalize:
            vc = vc / float(max(n_total, 1))
        return vc, 0.0

    def _build_joint_keyplans(self, Xdf):
        """Per-combination-unit :class:`~._cross_fit._JointKeyPlan` from full X (X-only labeling).

        Each multi-column unit's components are integer-coded once into value-stable maps so a
        combination yields the same int64 joint code at fit, per fold, and transform. Units whose
        radix product overflows int64 are omitted -> :meth:`_unit_keys` falls back to tuple keys.
        Keyed by ``tuple(cols)`` so ``_unit_keys`` (which holds ``cols``) can look the plan up.
        """
        plans = {}
        for _feat, cols in self._units:
            if len(cols) <= 1:
                continue
            comp_keys = [normalize_keys(self._col_values(Xdf, c))[0] for c in cols]
            plan = build_joint_keyplan(comp_keys)
            if plan.use_int:
                plans[tuple(cols)] = plan
        return plans

    def _unit_keys(self, Xdf, cols):
        """Return ``(keys, missing_mask)`` for an encoding unit.

        A single-column unit uses the column's normalized keys directly. A combination unit folds
        its components into one joint key: a mixed-radix int64 code via the unit's fitted
        :class:`~._cross_fit._JointKeyPlan` when present (fast, GPU-ready), else an object tuple per
        row (overflow fallback). The row counts as missing if any component is missing.
        """
        if len(cols) == 1:
            return normalize_keys(self._col_values(Xdf, cols[0]))
        comp_keys = []
        missing = np.zeros(Xdf.shape[0], dtype=bool)
        for c in cols:
            k, m = normalize_keys(self._col_values(Xdf, c))
            comp_keys.append(k)
            missing = missing | m
        plan = getattr(self, "_unit_keyplans", {}).get(tuple(cols))
        if plan is not None:
            return joint_codes(plan, comp_keys), missing
        # Overflow fallback: object tuple per row via a C-level zip (same MISSING-sentinel tuples).
        joint = np.empty(Xdf.shape[0], dtype=object)
        for i, key in enumerate(zip(*comp_keys)):
            joint[i] = key
        return joint, missing

    # ---- transform ---------------------------------------------------------------------------
    def _transform_array(self, Xdf, tables) -> np.ndarray:
        n = Xdf.shape[0]
        out = np.full((n, len(self._columns_meta)), np.nan, dtype=float)
        hm, hu = self.handle_missing, self.handle_unknown
        cache: dict = {}  # feat -> (keys, missing_mask, codes) for the unit's canonical index
        for j, meta in enumerate(self._columns_meta):
            if self._key(meta) not in tables:  # restricted table (hybrid slow path): skip absent
                continue
            feat = meta.feature
            enc = tables[self._key(meta)]
            if feat not in cache:
                keys, missing_mask = self._unit_keys(Xdf, self._unit_cols[feat])
                if hm == "error" and missing_mask.any():
                    raise ValueError(
                        f"Missing values in unit {feat!r} with handle_missing='error'."
                    )
                # factorize the unit's keys once; every column of the unit shares this canonical idx
                codes = enc.index.get_indexer(keys)
                has_unknown = bool((codes < 0).any())  # once per unit, reused across its columns
                cache[feat] = (keys, missing_mask, codes, has_unknown)
            keys, missing_mask, codes, has_unknown = cache[feat]

            col = gather(enc.values, codes, has_unknown)  # drop-in for pd.Series(keys).map(series)
            notfound = np.isnan(col)

            if hm == "return_nan":
                col[missing_mask] = np.nan
                self._apply_unknown(col, notfound & ~missing_mask, enc.fallback, hu, feat)
            elif hm == "value":
                # rows not found are either unseen real categories OR an unseen missing level
                self._apply_unknown(col, notfound, enc.fallback, hu, feat)
            else:  # "error" already raised above if any missing present
                self._apply_unknown(col, notfound & ~missing_mask, enc.fallback, hu, feat)
            out[:, j] = col
        return out

    @staticmethod
    def _apply_unknown(col, mask, fallback, hu, feat):
        if not mask.any():
            return col
        if hu == "error":
            raise ValueError(
                f"Found unknown categories in column {feat!r} with handle_unknown='error'."
            )
        if hu == "value":
            col[mask] = fallback
        # "return_nan": leave the NaN in place
        return col

    # ---- pickle support ----------------------------------------------------------------------
    # A fitted estimator caches its backend *module* in `_backend_mod`; modules aren't picklable,
    # so drop it on pickle and re-resolve it from the recorded backend name (`backend_`) on load.
    def __getstate__(self):
        state = dict(super().__getstate__())
        state.pop("_backend_mod", None)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        if "backend_" in state:
            self._backend_mod = backend_module(state["backend_"])

    def transform(self, X):
        check_is_fitted(self, "_fit_tables")
        Xdf, was_df, _ = prepare_X(X)
        arr = self._transform_array(Xdf, self._fit_tables)
        return self._wrap_output(arr, was_df, Xdf)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        Xdf, was_df, _ = prepare_X(X)
        full = self._transform_array(Xdf, self._fit_tables)

        if self._is_supervised() and any(m.target_dependent for m in self._columns_meta):
            y_arr = np.asarray(y)
            scheme = getattr(self, "scheme", "kfold")
            if scheme == "kfold":
                oof = self._kfold_oof(Xdf, y_arr, full.shape)
            else:
                oof = self._loo_ordered_oof(Xdf, y_arr, scheme, full.shape)
            for j, meta in enumerate(self._columns_meta):
                if meta.target_dependent:
                    full[:, j] = oof[:, j]
        return self._wrap_output(full, was_df, Xdf)

    def _kfold_oof(self, Xdf, y_arr, shape):
        splitter = resolve_cv(self.cv, self.target_type_, self.shuffle, self.random_state)
        folds = make_folds(Xdf.shape[0], y_arr, splitter)
        # Fast path: additive stats (mean/var/std/skew/kurt) are reconstructed from one composite
        # (fold, key) power-sum aggregation via complement subtraction instead of a per-fold
        # group-by loop -- exactly equivalent (allclose; leakage-audited). Hybrid: when both
        # additive and non-additive (median/min/max/custom) stats are requested, the additive
        # columns take the fast kernel and only the non-additive ones fall to the per-fold loop.
        # Needs a partitioning CV.
        add_cols = [
            j
            for j, m in enumerate(self._columns_meta)
            if m.target_dependent and m.stat in _ADDITIVE_STATS
        ]
        fold_id = self._partition_fold_id(folds, Xdf.shape[0]) if add_cols else None
        if fold_id is not None:
            oof = np.full(shape, np.nan)
            self._kfold_oof_additive_fast(Xdf, y_arr, fold_id, len(folds), oof, add_cols)
            non_cols = [
                j
                for j, m in enumerate(self._columns_meta)
                if m.target_dependent and m.stat not in _ADDITIVE_STATS
            ]
            if non_cols:
                self._slow_oof_into(Xdf, y_arr, folds, oof, non_cols)
            return oof
        oof = np.full(shape, np.nan)
        for tr, te in folds:
            tbl = self._fit_all(Xdf.iloc[tr], y_arr[tr])
            oof[te, :] = self._transform_array(Xdf.iloc[te], tbl)
        return oof

    @staticmethod
    def _partition_fold_id(folds, n):
        """Integer fold-id per row if the folds partition ``[0, n)`` (each row in exactly one test
        fold), else ``None``. ``KFold``/``StratifiedKFold`` partition; arbitrary user CV may not, so
        the fast path is gated on this and otherwise falls back to the per-fold loop."""
        fold_id = np.full(n, -1, dtype=np.int64)
        for f, (_tr, te) in enumerate(folds):
            te = np.asarray(te)
            if (fold_id[te] >= 0).any():  # overlapping test folds -> not a partition
                return None
            fold_id[te] = f
        if (fold_id < 0).any():  # some row in no test fold -> not a partition
            return None
        return fold_id

    def _kfold_oof_additive_fast(self, Xdf, y_arr, fold_id, n_folds, oof, col_indices):
        """Fill the additive (mean/var/std/skew/kurt) OOF columns in ``col_indices`` via the
        single-pass kernel. Per unit the keys are factorized once; for a continuous target the
        complement moments are computed once and shared across that unit's columns. A unit with a
        shape stat upgrades the shared pass to order-4 sums shifted by the global mean (exact:
        shift-invariant stats; the mean finalizer un-shifts) -- units without shape stats keep the
        order-2 zero-shift pass bit-identical. Dispersion/shape are continuous-only, so
        classification only ever finalizes the mean, one pass per class."""
        by_unit: dict = {}
        for j in col_indices:
            meta = self._columns_meta[j]
            by_unit.setdefault(meta.feature, []).append((j, meta))
        ms = getattr(self, "min_samples_category", 1)
        for feat, items in by_unit.items():
            keys, missing_mask = self._unit_keys(Xdf, self._unit_cols[feat])
            n, a, codes, n_cat = factorize_active(keys, missing_mask, self.handle_missing)
            fid_a = fold_id[a]
            if self.target_type_ == "continuous":
                y_act = y_arr.astype(float)[a]
                shape_req = any(meta.stat in _SHAPE_STATS for _j, meta in items)
                order = 4 if shape_req else 2
                shift = float(y_act.mean()) if shape_req and y_act.size else 0.0
                tab = complement_tables(
                    n, a, codes, n_cat, fid_a, y_act, n_folds, order=order, shift=shift
                )
                for j, meta in items:
                    if meta.stat == "mean":
                        oof[:, j] = finalize_mean_oof(tab, self.smooth, self.handle_unknown)
                    elif meta.stat in _SHAPE_STATS:
                        oof[:, j] = finalize_shape_oof(tab, meta.stat, ms, self.handle_unknown)
                    else:
                        oof[:, j] = finalize_dispersion_oof(tab, meta.stat, ms, self.handle_unknown)
            elif self.target_type_ == "binary":  # mean/woe share one binarized table pass
                yv = self._mean_y_vector(y_arr, items[0][1])[a]
                tab = complement_tables(n, a, codes, n_cat, fid_a, yv, n_folds)
                for j, meta in items:
                    if meta.stat == "woe":
                        oof[:, j] = finalize_woe_oof(tab, self.smooth, self.handle_unknown)
                    else:
                        oof[:, j] = finalize_mean_oof(tab, self.smooth, self.handle_unknown)
            else:  # multiclass: mean only, one table pass per class (factorize shared)
                for j, meta in items:
                    yv = self._mean_y_vector(y_arr, meta)[a]
                    tab = complement_tables(n, a, codes, n_cat, fid_a, yv, n_folds)
                    oof[:, j] = finalize_mean_oof(tab, self.smooth, self.handle_unknown)
        return oof

    def _slow_oof_into(self, Xdf, y_arr, folds, oof, cols):
        """Per-fold (slow path) OOF for the non-additive columns ``cols`` only, written into ``oof``
        in place. Fits only the specs those columns need, so the additive columns (already encoded
        by the single-pass kernel) cost nothing in this loop."""
        need_stats = {self._columns_meta[j].stat for j in cols}
        needed_specs = [s for s in self._specs if s.name in need_stats]
        for tr, te in folds:
            tbl = self._fit_all(Xdf.iloc[tr], y_arr[tr], specs=needed_specs)
            sub = self._transform_array(Xdf.iloc[te], tbl)
            for j in cols:
                oof[te, j] = sub[:, j]
        return oof

    def _mean_y_vector(self, y_arr, meta):
        if self.target_type_ == "continuous":
            return y_arr.astype(float)
        if self.target_type_ == "binary":
            return (y_arr == self.classes_[1]).astype(float)
        return (y_arr == meta.class_label).astype(float)  # multiclass one-vs-rest

    def _loo_ordered_oof(self, Xdf, y_arr, scheme, shape):
        """Leave-one-out / ordered encodings for the mean columns (validated: mean-only)."""
        oof = np.full(shape, np.nan)
        m = 0.0 if isinstance(self.smooth, str) else float(self.smooth)  # loo pseudo-count
        # ordered prior weight a (CatBoost) must be > 0; default 1 for "auto"/non-positive smooth.
        smooth_pos = (not isinstance(self.smooth, str)) and float(self.smooth) > 0
        a = float(self.smooth) if smooth_pos else 1.0
        perm = (
            check_random_state(self.random_state).permutation(len(y_arr))
            if scheme == "ordered"
            else None
        )
        for j, meta in enumerate(self._columns_meta):
            if not (meta.target_dependent and meta.stat == "mean"):
                continue
            keys, missing_mask = self._unit_keys(Xdf, self._unit_cols[meta.feature])
            yv = self._mean_y_vector(y_arr, meta)
            prior = float(yv.mean())
            if scheme == "loo":
                vals = loo_encode(keys, yv, m, prior)
            else:
                vals = ordered_encode(keys, yv, a, prior, perm)
            if self.handle_missing == "return_nan":
                vals = vals.copy()
                vals[missing_mask] = np.nan
            oof[:, j] = vals
        return oof

    # ---- output container --------------------------------------------------------------------
    def _wrap_output(self, arr, was_df, Xdf):
        if self.output == "numpy":
            return arr
        if self.output == "pandas":
            idx = Xdf.index if was_df else None
            return pd.DataFrame(arr, columns=self.feature_names_out_, index=idx)
        if self.output == "polars":
            try:
                import polars as pl
            except ImportError as e:  # pragma: no cover - exercised only without polars
                raise ImportError("output='polars' requires polars (pip install polars).") from e
            return pl.from_numpy(arr, schema=list(self.feature_names_out_))
        # "auto": mirror the input container
        if was_df:
            return pd.DataFrame(arr, columns=self.feature_names_out_, index=Xdf.index)
        return arr

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "_fit_tables")
        return np.asarray(self.feature_names_out_, dtype=object)
