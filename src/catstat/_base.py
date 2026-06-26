"""``_BaseStatEncoder`` -- the shared fit/transform/fit_transform skeleton.

All statistics/leakage logic lives here and in the small helper modules; only ``backends/``
knows pandas vs cuDF. Subclasses define the sklearn ``__init__`` params and two hooks:
``_is_supervised`` and ``_resolve_stat_specs``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._aggregations import fit_custom_encoding, fit_stat_encoding
from ._cross_fit import make_folds, resolve_cv
from ._feature_names import build_columns
from ._smoothing import fit_mean_encoding
from ._validation import (
    check_handle,
    infer_target_type,
    normalize_keys,
    prepare_X,
    select_cols,
)
from .backends import _cpu
from .backends._dispatch import select_backend

_VALID_OUTPUT = ("auto", "numpy", "pandas")
_DEFERRED_OUTPUT = ("cudf", "cupy", "polars")


class _BaseStatEncoder(TransformerMixin, BaseEstimator):
    # ---- hooks the subclasses must implement -------------------------------------------------
    def _is_supervised(self) -> bool:
        raise NotImplementedError

    def _resolve_stat_specs(self):
        raise NotImplementedError

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

        Xdf, was_df, all_cols = prepare_X(X)
        self.n_features_in_ = Xdf.shape[1]
        if was_df:
            self.feature_names_in_ = np.asarray(all_cols, dtype=object)
        self._cat_cols = select_cols(Xdf, self.cols)
        # Encoding units: independent -> one per column; combination -> one joint unit.
        if mode == "combination" and len(self._cat_cols) > 1:
            joint = "+".join(str(c) for c in self._cat_cols)
            self._units = [(joint, list(self._cat_cols))]
        else:
            self._units = [(c, [c]) for c in self._cat_cols]
        self._unit_cols = dict(self._units)
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

        self._columns_meta = build_columns(
            [name for name, _ in self._units], self._specs, self.target_type_, self.classes_
        )
        self.feature_names_out_ = np.asarray([m.name for m in self._columns_meta], dtype=object)

        all_gpu = all(s.gpu_supported for s in self._specs)
        self._backend_mod, self.backend_ = select_backend(
            self.backend, Xdf.shape[0], len(self._cat_cols), all_gpu
        )
        # GPU can't run tuple keys (combination) or CPU-only stats (skew/custom) -> host only.
        host_only = (not all_gpu) or any(len(cols) > 1 for _, cols in self._units)
        if self.backend_ == "gpu" and host_only:
            self._backend_mod, self.backend_ = _cpu, _cpu.NAME

        self._fit_tables = self._fit_all(Xdf, y_arr)

        # public fitted attributes derived from the full-data tables
        self.categories_ = {}
        self.global_stats_ = {}
        for meta in self._columns_meta:
            enc, fb = self._fit_tables[self._key(meta)]
            self.global_stats_[meta.name] = fb
            self.categories_.setdefault(meta.feature, np.asarray(list(enc.index), dtype=object))
        self._set_target_mean()
        return self

    def _set_target_mean(self):
        if not self._is_supervised() or "mean" not in self.stats_:
            return
        f0 = self._units[0][0]
        if self.target_type_ == "multiclass":
            self.target_mean_ = np.asarray(
                [self._fit_tables[(f0, "mean", c)][1] for c in self.classes_], dtype=float
            )
        else:
            self.target_mean_ = float(self._fit_tables[(f0, "mean", None)][1])

    # ---- per-statistic fitting ---------------------------------------------------------------
    def _fit_all(self, Xdf, y_arr):
        """Return the full encoding tables: ``{(feature, stat, class): (Series, fallback)}``."""
        tables = {}
        hm = self.handle_missing
        for feat, cols in self._units:
            keys_full, missing_mask = self._unit_keys(Xdf, cols)
            if hm == "error" and missing_mask.any():
                raise ValueError(
                    f"Missing values in unit {feat!r} with handle_missing='error'."
                )
            sel = np.ones(len(keys_full), dtype=bool) if hm == "value" else ~missing_mask
            keys = keys_full[sel]
            n_total = int(sel.sum())
            for spec in self._specs:
                if spec.name == "count":
                    tables[(feat, "count", None)] = self._fit_count(keys, False, n_total)
                elif spec.name == "frequency":
                    tables[(feat, "frequency", None)] = self._fit_count(keys, True, n_total)
                elif spec.name == "mean":
                    y_sel = y_arr[sel]
                    bk = self._backend_mod
                    if self.target_type_ == "continuous":
                        tables[(feat, "mean", None)] = fit_mean_encoding(
                            keys, y_sel.astype(float), self.smooth, bk
                        )
                    elif self.target_type_ == "binary":
                        yb = (y_sel == self.classes_[1]).astype(float)
                        tables[(feat, "mean", None)] = fit_mean_encoding(keys, yb, self.smooth, bk)
                    else:  # multiclass: one-vs-rest per global class
                        for c in self.classes_:
                            yc = (y_sel == c).astype(float)
                            tables[(feat, "mean", c)] = fit_mean_encoding(keys, yc, self.smooth, bk)
                else:  # var/std/median/min/max/skew or custom (continuous-only, target-dependent)
                    min_samples = getattr(self, "min_samples_category", 1)
                    y_sel_f = y_arr[sel].astype(float)
                    if spec.func is not None:
                        tables[(feat, spec.name, None)] = fit_custom_encoding(
                            keys, y_sel_f, spec.func, min_samples
                        )
                    else:
                        tables[(feat, spec.name, None)] = fit_stat_encoding(
                            keys, y_sel_f, spec.name, min_samples, self._backend_mod
                        )
        return tables

    @staticmethod
    def _fit_count(keys, normalize, n_total):
        vc = pd.Series(keys).value_counts().astype(float)
        if normalize:
            vc = vc / float(max(n_total, 1))
        return vc, 0.0

    def _unit_keys(self, Xdf, cols):
        """Return ``(keys, missing_mask)`` for an encoding unit.

        A single-column unit uses the column's normalized keys directly. A combination unit uses
        the tuple of its components' keys as one joint category; the row counts as missing if any
        component is missing.
        """
        if len(cols) == 1:
            return normalize_keys(Xdf[cols[0]].to_numpy())
        comp_keys = []
        missing = np.zeros(Xdf.shape[0], dtype=bool)
        for c in cols:
            k, m = normalize_keys(Xdf[c].to_numpy())
            comp_keys.append(k)
            missing = missing | m
        joint = np.empty(Xdf.shape[0], dtype=object)
        for i in range(Xdf.shape[0]):
            joint[i] = tuple(ck[i] for ck in comp_keys)
        return joint, missing

    # ---- transform ---------------------------------------------------------------------------
    def _transform_array(self, Xdf, tables) -> np.ndarray:
        n = Xdf.shape[0]
        out = np.full((n, len(self._columns_meta)), np.nan, dtype=float)
        hm, hu = self.handle_missing, self.handle_unknown
        cache: dict = {}
        for j, meta in enumerate(self._columns_meta):
            feat = meta.feature
            if feat not in cache:
                keys, missing_mask = self._unit_keys(Xdf, self._unit_cols[feat])
                if hm == "error" and missing_mask.any():
                    raise ValueError(
                        f"Missing values in unit {feat!r} with handle_missing='error'."
                    )
                cache[feat] = (keys, missing_mask)
            keys, missing_mask = cache[feat]

            enc_series, fallback = tables[self._key(meta)]
            mapped = pd.Series(keys).map(enc_series).to_numpy(dtype=float)
            col = mapped.copy()
            notfound = np.isnan(mapped)

            if hm == "return_nan":
                col[missing_mask] = np.nan
                self._apply_unknown(col, notfound & ~missing_mask, fallback, hu, feat)
            elif hm == "value":
                # rows not found are either unseen real categories OR an unseen missing level
                self._apply_unknown(col, notfound, fallback, hu, feat)
            else:  # "error" already raised above if any missing present
                self._apply_unknown(col, notfound & ~missing_mask, fallback, hu, feat)
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
            splitter = resolve_cv(self.cv, self.target_type_, self.shuffle, self.random_state)
            folds = make_folds(Xdf.shape[0], y_arr, splitter)
            oof = np.full_like(full, np.nan)
            for tr, te in folds:
                tbl = self._fit_all(Xdf.iloc[tr], y_arr[tr])
                oof[te, :] = self._transform_array(Xdf.iloc[te], tbl)
            for j, meta in enumerate(self._columns_meta):
                if meta.target_dependent:
                    full[:, j] = oof[:, j]
        return self._wrap_output(full, was_df, Xdf)

    # ---- output container --------------------------------------------------------------------
    def _wrap_output(self, arr, was_df, Xdf):
        if self.output == "numpy":
            return arr
        if self.output == "pandas":
            idx = Xdf.index if was_df else None
            return pd.DataFrame(arr, columns=self.feature_names_out_, index=idx)
        # "auto": mirror the input container
        if was_df:
            return pd.DataFrame(arr, columns=self.feature_names_out_, index=Xdf.index)
        return arr

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "_fit_tables")
        return np.asarray(self.feature_names_out_, dtype=object)
