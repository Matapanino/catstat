"""Device-resident fit / OOF orchestration for cuDF input (``backend='auto'/'gpu'``).

This module never imports cudf/cupy (that stays inside ``backends/_gpu``); it calls the backend's
device primitives and manipulates the returned device arrays through operators only. Everything
above the integer-code layer -- smoothing arithmetic, the (fold, key) cell finalizers, fold
assignment (host RNG: the CPU==GPU fold-parity invariant), feature names, fitted attributes -- is
the same host code the CPU path uses, so a device-fitted encoder transforms pandas input with
zero new logic.

Data flow per unit: cuDF factorize (device) -> dense int64 codes (device) + value-stable host
index -> on-device reductions (``code_moments`` / ``oof_moment_tables``) -> small host tables ->
shared host finalization -> device gather back onto the rows. The only D2H copies are the small
per-category tables and the final output matrix (``output='numpy'``); the only H2D copies are the
per-fold ids and the small finalized value tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._aggregations import g1_g2_from_power_sums
from ._base import _SHAPE_STATS, _UnitEncoding
from ._cross_fit import (
    _INT64_MAX,
    _JointKeyPlan,
    dispersion_cells,
    mean_cells,
    shape_cells,
    tables_from_raw,
    woe_cells,
)
from ._smoothing import mean_from_stats, woe_from_prob
from .backends import _gpu

# order stats: no additive complement trick -- a per-fold device group-by loop instead
_ORDER_STATS = frozenset({"median", "min", "max"})


def check_device_fences(est, specs) -> None:
    """Loud, early errors for the device-input combinations that are not implemented (yet)."""
    if est.backend == "cpu":  # precedence over the other fences: the intent error comes first
        raise ValueError(
            "backend='cpu' with a cuDF input would require a silent device->host transfer. "
            "Convert explicitly with X.to_pandas(), or use backend='auto'/'gpu'."
        )
    if getattr(est, "numeric", "ignore") != "ignore":
        raise NotImplementedError(
            "numeric encoding with cuDF input is not supported yet; pass X.to_pandas() or "
            "numeric='ignore'."
        )
    if getattr(est, "scheme", "kfold") != "kfold":
        raise NotImplementedError(
            "scheme='loo'/'ordered' with cuDF input is not supported yet; use scheme='kfold'."
        )
    for spec in specs:
        if spec.func is not None or not spec.gpu_supported:
            raise ValueError(
                f"stat={spec.name!r} is CPU-only (custom callables have no GPU primitive); "
                "convert with X.to_pandas() to use it."
            )


def factorize_units(est, Xg) -> dict:
    """Per-unit ``(dense codes [device], canonical host index, missing mask [device], n_cat)``.

    Single-column units keep their value index; combination/interaction units get device
    mixed-radix joint codes densified on device, an Int64 canonical index of observed joint
    codes, and a host :class:`_JointKeyPlan` (stored on the estimator) so pandas-input transform
    reuses the ordinary host joint-code builder against the same category maps.
    """
    hm = est.handle_missing
    colfac = {}
    for _feat, cols in est._units:
        for c in cols:
            if c not in colfac:
                colfac[c] = _gpu.factorize_column(Xg[c], hm)

    units = {}
    for feat, cols in est._units:
        if len(cols) == 1:
            codes, uniques, missing = colfac[cols[0]]
            units[feat] = (codes, uniques, missing, len(uniques))
            continue
        fac = [colfac[c] for c in cols]
        radices = [len(u) for _c, u, _m in fac]
        product = 1
        for r in radices:
            product *= max(r, 1)
        if product > _INT64_MAX:
            raise ValueError(
                f"combination unit {feat!r}: joint cardinality overflows int64; the device path "
                "has no tuple-key fallback -- reduce the combined columns or use pandas input."
            )
        joint = _gpu.joint_codes_device(radices, [c for c, _u, _m in fac])
        missing = fac[0][2]
        for _c, _u, m in fac[1:]:
            missing = missing | m
        has_negative = hm != "value" and bool(missing.any())
        codes, uniques = _gpu.dense_codes(joint, has_negative)
        est._unit_keyplans[tuple(cols)] = _JointKeyPlan(
            tuple(u for _c, u, _m in fac), tuple(radices), True
        )
        units[feat] = (codes, uniques, missing, len(uniques))
    return units


def _active_codes(codes, missing, handle_missing):
    """(active mask or None, active codes) -- mirrors ``factorize_active``'s row selection."""
    if handle_missing == "value":
        return None, codes
    neg = codes < 0
    if not bool(neg.any()):
        return None, codes
    active = ~neg
    return active, codes[active]


def _unit_y_plan(est, items):
    """(order, wants_shift) for a unit's stats: shape stats force order-4. Continuous targets
    are always shifted by the global mean (exact for the shift-invariant stats, un-shifted for
    the mean) -- it keeps the moment reconstruction and EB weights stable at |mean| >> sd, on
    both backends identically."""
    stats = {meta.stat for _j, meta in items}
    order = 4 if stats & _SHAPE_STATS else 2
    return order, True


def _global_var(cnt_sum, s1_sum, s2_sum) -> float:
    """Sample (ddof=1) variance from global (shifted) sums; 0.0 when undefined (n <= 1)."""
    if cnt_sum <= 1:
        return 0.0
    return float((s2_sum - s1_sum * s1_sum / cnt_sum) / (cnt_sum - 1.0))


def fit_all_device(est, units, y) -> dict:
    """Device twin of ``_BaseStatEncoder._fit_all``: the same ``{key: _UnitEncoding}`` tables
    (host indexes/values), with every heavy reduction on device."""
    tables = {}
    hm = est.handle_missing
    ms = int(getattr(est, "min_samples_category", 1))
    supervised = est._is_supervised()
    for feat, _cols in est._units:
        codes, uniques, missing, n_cat = units[feat]
        if hm == "error" and bool(missing.any()):
            raise ValueError(f"Missing values in unit {feat!r} with handle_missing='error'.")
        _active, codes_act = _active_codes(codes, missing, hm)
        n_total = int(codes_act.shape[0])

        # one shared moment pass per (unit, target vector); count-only pass when unsupervised
        target_specs = [s for s in est._specs if s.target_dependent]
        entries = []
        for spec in est._specs:
            if spec.name == "count":
                (cnt,) = _gpu.code_moments(codes_act, None, n_cat)
                entries.append(((feat, "count", None), pd.Series(cnt, index=uniques), 0.0))
            elif spec.name == "frequency":
                (cnt,) = _gpu.code_moments(codes_act, None, n_cat)
                alpha = float(getattr(est, "laplace_alpha", 0.0) or 0.0)  # validated at resolve
                if alpha > 0.0:  # Laplace add-alpha, mirroring the host _fit_count
                    denom = float(max(n_total, 1)) + alpha * float(n_cat)
                    freq, fb = (cnt + alpha) / denom, alpha / denom
                else:
                    freq, fb = cnt / float(max(n_total, 1)), 0.0
                entries.append(((feat, "frequency", None), pd.Series(freq, index=uniques), fb))
        if supervised and target_specs:
            entries.extend(
                _fit_target_stats_device(est, feat, codes_act, _active, uniques, n_cat, y, ms)
            )
        for tkey, s, fb in entries:
            # every stat shares the unit's canonical index already (code order) -- no reindex
            tables[tkey] = _UnitEncoding(uniques, s.to_numpy(dtype=float), fb)
    return tables


def _fit_target_stats_device(est, feat, codes_act, active, uniques, n_cat, y, ms):
    """Per-unit supervised encodings from shared on-device moment passes."""
    entries = []
    order, wants_shift = _unit_y_plan(
        est, [(None, m) for m in est._columns_meta if m.feature == feat and m.target_dependent]
    )
    if est.target_type_ == "continuous":
        y_dev = _gpu.to_device_float(y)
        y_act = y_dev[active] if active is not None else y_dev
        shift = float(y_act.mean()) if wants_shift and y_act.shape[0] else 0.0
        mom = _gpu.code_moments(codes_act, y_act - shift if shift != 0.0 else y_act, n_cat, order)
        cnt, s1, s2 = mom[0], mom[1], mom[2]
        n_sum, s1_sum, s2_sum = cnt.sum(), s1.sum(), s2.sum()
        gm_shifted = s1_sum / n_sum if n_sum else 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_by_code = s1 / cnt
        for spec in est._specs:
            if not spec.target_dependent:
                continue
            if spec.name == "mean":
                tau2 = float(s2_sum / n_sum - gm_shifted * gm_shifted) if n_sum else 0.0
                enc = mean_from_stats(
                    pd.Series(cnt), pd.Series(mean_by_code), pd.Series(s2),
                    est.smooth, gm_shifted, max(tau2, 0.0),
                )
                enc = pd.Series(enc.to_numpy() + shift, index=uniques)
                entries.append(((feat, "mean", None), enc, float(gm_shifted + shift)))
            elif spec.name in ("var", "std"):
                with np.errstate(invalid="ignore", divide="ignore"):
                    var_raw = np.where(
                        cnt > 1, (s2 - s1 * mean_by_code) / np.where(cnt > 1, cnt - 1.0, 1.0),
                        np.nan,
                    )
                gv = _global_var(n_sum, s1_sum, s2_sum)
                vals = np.where(np.isnan(var_raw) | (cnt < max(ms, 1)), gv, var_raw)
                if spec.name == "std":
                    vals = np.sqrt(np.clip(vals, 0.0, None))
                    gv = float(np.sqrt(max(gv, 0.0)))
                entries.append(((feat, spec.name, None), pd.Series(vals, index=uniques), gv))
            elif spec.name in _SHAPE_STATS:
                s3, s4 = mom[3], mom[4]
                vals = g1_g2_from_power_sums(cnt, s1, s2, s3, s4, spec.name)
                g_arr = g1_g2_from_power_sums(
                    np.asarray([n_sum]), np.asarray([s1_sum]), np.asarray([s2_sum]),
                    np.asarray([s3.sum()]), np.asarray([s4.sum()]), spec.name,
                )
                gv = float(g_arr[0]) if np.isfinite(g_arr[0]) else 0.0
                vals = np.where(np.isnan(vals) | (cnt < max(ms, 1)), gv, vals)
                entries.append(((feat, spec.name, None), pd.Series(vals, index=uniques), gv))
            elif spec.name in _ORDER_STATS:
                # raw (unshifted) y: order stats need actual values, not moments
                vals = _gpu.category_agg_codes(codes_act, y_act, spec.name, n_cat)
                gv = _gpu.global_agg(y_act, spec.name)
                vals = np.where(np.isnan(vals) | (cnt < max(ms, 1)), gv, vals)
                entries.append(((feat, spec.name, None), pd.Series(vals, index=uniques), gv))
        return entries

    # binary / multiclass: mean (per class) and woe (binary) from binarized device passes
    if est.target_type_ == "binary":
        class_vectors = [(None, est.classes_[1])]
    else:  # multiclass: encoded classes only (max_classes may cap)
        class_vectors = [(c, c) for c in est.encoded_classes_]
    for class_label, pos in class_vectors:
        yb = _gpu.binarize_device(y, pos)
        yb_act = yb[active] if active is not None else yb
        cnt, s1, s2 = _gpu.code_moments(codes_act, yb_act, n_cat, 2)
        n_sum = cnt.sum()
        prior = float(s1.sum() / n_sum) if n_sum else 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_by_code = s1 / cnt
        enc = None
        for spec in est._specs:
            if spec.name not in ("mean", "woe"):
                continue
            if enc is None:
                tau2 = prior * (1.0 - prior)  # population var of a 0/1 vector
                enc = mean_from_stats(
                    pd.Series(cnt), pd.Series(mean_by_code), pd.Series(s2),
                    est.smooth, prior, tau2,
                )
            if spec.name == "mean":
                entries.append(
                    ((feat, "mean", class_label), pd.Series(enc.to_numpy(), index=uniques), prior)
                )
            elif spec.name == "woe" and class_label is None:
                woe = woe_from_prob(enc.to_numpy(), prior)
                entries.append(((feat, "woe", None), pd.Series(woe, index=uniques), 0.0))
    return entries


def transform_train_columns(est, units, tables) -> list:
    """Device gather of the fitted encodings back onto the training rows (the ``full`` matrix).

    Codes are the fit-time codes, so every non-negative code is known; ``-1`` (missing under
    'return_nan') gathers NaN, matching the host transform's missing handling at fit rows.
    """
    cols = []
    for meta in est._columns_meta:
        codes, _uniques, _missing, _n_cat = units[meta.feature]
        enc = tables[est._key(meta)]
        cols.append(_gpu.gather_cells(enc.values, codes))
    return cols


def kfold_oof_columns(est, units, y, fold_id, n_folds, cols) -> None:
    """Overwrite the target-dependent entries of ``cols`` with device OOF columns.

    Same fold ids as the CPU path (host RNG, one small H2D); per unit one on-device composite
    (fold, key) pass; host cell finalization on the small tables; device gather of the cells.
    """
    by_unit: dict = {}
    for j, meta in enumerate(est._columns_meta):
        if meta.target_dependent:
            by_unit.setdefault(meta.feature, []).append((j, meta))
    ms = int(getattr(est, "min_samples_category", 1))
    fid_dev = _gpu.to_device(fold_id)
    n = int(fold_id.shape[0])
    for feat, items in by_unit.items():
        codes, _uniques, missing, n_cat = units[feat]
        active, codes_act = _active_codes(codes, missing, est.handle_missing)
        fid_act = fid_dev[active] if active is not None else fid_dev
        order, wants_shift = _unit_y_plan(est, items)
        size = n_folds * n_cat
        comp = fid_act * n_cat + codes_act  # device arithmetic

        if est.target_type_ == "continuous":
            y_dev = _gpu.to_device_float(y)
            y_act = y_dev[active] if active is not None else y_dev
            add_items = [(j, m) for j, m in items if m.stat not in _ORDER_STATS]
            ord_items = [(j, m) for j, m in items if m.stat in _ORDER_STATS]
            tab = None
            if add_items:
                shift = float(y_act.mean()) if wants_shift and y_act.shape[0] else 0.0
                raw = _gpu.oof_moment_tables(
                    comp, y_act - shift if shift != 0.0 else y_act, size, order
                )
                tab = tables_from_raw(n, None, None, n_folds, n_cat, *raw, shift=shift)
                for j, meta in add_items:
                    E = _cells_for(meta.stat, tab, est, ms)
                    cols[j] = _gpu.scatter_active(_gpu.gather_cells(E, comp), active, n)
            if ord_items:
                if tab is None:  # counts tables only (fallback masks): one order-2 pass
                    raw = _gpu.oof_moment_tables(comp, y_act, size, 2)
                    tab = tables_from_raw(n, None, None, n_folds, n_cat, *raw)
                for j, meta in ord_items:
                    E = _order_oof_cells(
                        est, meta.stat, tab, codes_act, y_act, fid_act, n_folds, n_cat, ms
                    )
                    cols[j] = _gpu.scatter_active(_gpu.gather_cells(E, comp), active, n)
        elif est.target_type_ == "binary":
            yb = _gpu.binarize_device(y, est.classes_[1])
            yb_act = yb[active] if active is not None else yb
            raw = _gpu.oof_moment_tables(comp, yb_act, size, 2)
            tab = tables_from_raw(n, None, None, n_folds, n_cat, *raw)
            for j, meta in items:
                E = _cells_for(meta.stat, tab, est, ms)
                cols[j] = _gpu.scatter_active(_gpu.gather_cells(E, comp), active, n)
        else:  # multiclass: one binarized pass per class
            for j, meta in items:
                yb = _gpu.binarize_device(y, meta.class_label)
                yb_act = yb[active] if active is not None else yb
                raw = _gpu.oof_moment_tables(comp, yb_act, size, 2)
                tab = tables_from_raw(n, None, None, n_folds, n_cat, *raw)
                E = mean_cells(tab, est.smooth, est.handle_unknown)
                cols[j] = _gpu.scatter_active(_gpu.gather_cells(E, comp), active, n)


def _order_oof_cells(est, stat, tab, codes_act, y_act, fid_act, n_folds, n_cat, ms):
    """Per-fold device group-by OOF for an order stat (median/min/max), finalized on the
    (fold, key) cells.

    Mirrors the host slow path: fold ``f``'s cells hold the statistic fitted on its complement
    (train side); NaN / low-count cells fall back to the complement-global statistic; unseen
    cells follow ``handle_unknown`` (``'error'`` only if the cell is actually gathered,
    ``fc > 0``). Only the small per-code tables and one scalar per fold leave the device -- the
    per-fold row data (the KI-020 bottleneck) never does.
    """
    hu = est.handle_unknown
    E = np.empty(n_folds * n_cat, dtype=float)
    for f in range(n_folds):
        train = fid_act != int(f)
        vals = _gpu.category_agg_codes(codes_act[train], y_act[train], stat, n_cat)
        gv = _gpu.global_agg(y_act[train], stat)
        sl = slice(f * n_cat, (f + 1) * n_cat)
        cnt_f = tab.cc[sl]  # complement (train-side) counts per code
        vals = np.where(np.isnan(vals) | (cnt_f < max(ms, 1)), gv, vals)
        unseen = cnt_f <= 0.0
        if unseen.any() and hu != "value":  # 'value' already took gv via the low-count mask
            if hu == "error":
                if (unseen & (tab.fc[sl] > 0.0)).any():
                    raise ValueError(
                        "Found unknown categories during out-of-fold encoding "
                        "(handle_unknown='error')."
                    )
            vals = np.where(unseen, np.nan, vals)
        E[sl] = vals
    return E


def _cells_for(stat, tab, est, ms):
    """Dispatch a stat to its (fold, key) cell finalizer -- the same host math as the CPU path."""
    if stat == "mean":
        return mean_cells(tab, est.smooth, est.handle_unknown)
    if stat == "woe":
        return woe_cells(tab, est.smooth, est.handle_unknown)
    if stat in _SHAPE_STATS:
        return shape_cells(tab, stat, ms, est.handle_unknown)
    return dispersion_cells(tab, stat, ms, est.handle_unknown)


def transform_device_columns(est, Xg) -> list:
    """Transform a cuDF frame on device against the fitted (host) tables.

    Mirrors ``_transform_array``: per unit, device codes against the fit-time category index
    (unknown -> -1), one device gather per column, then the unknown/missing policy applied on
    device masks. Fitted on either backend -- the tables are host either way.
    """
    if getattr(est, "_numeric_plan_", None):
        raise NotImplementedError(
            "transform on cuDF input with numeric-encoded columns is not supported; "
            "convert with X.to_pandas()."
        )
    hm, hu = est.handle_missing, est.handle_unknown
    # per-FIT device LUT cache: the uniques H2D dominates repeated small transforms, so the
    # lookup tables are built once per fitted unit and reused across transform(cuDF) calls
    # (invalidated by fit, dropped by __getstate__ -- device objects are not picklable).
    luts = getattr(est, "_device_transform_luts", None)
    if luts is None:
        luts = est._device_transform_luts = {}
    cols: list = []
    cache: dict = {}  # per-CALL unit codes (shared across a unit's stat columns)
    for meta in est._columns_meta:
        feat = meta.feature
        enc = est._fit_tables[est._key(meta)]
        if feat not in cache:
            unit_cols = est._unit_cols[feat]
            if len(unit_cols) == 1:
                if feat not in luts:
                    luts[feat] = _gpu.build_value_lut(enc.index)
                lut, missing_code = luts[feat]
                codes, missing = _gpu.codes_from_lut(lut, missing_code, Xg[unit_cols[0]])
            else:
                plan = est._unit_keyplans.get(tuple(unit_cols))
                if plan is None:  # int64 overflow at fit -> tuple keys, host-only
                    raise NotImplementedError(
                        f"combination unit {feat!r} was fitted with tuple keys (int64 "
                        "overflow); transform it on pandas input."
                    )
                if feat not in luts:
                    luts[feat] = (
                        [_gpu.build_value_lut(u) for u in plan.uniques],
                        _gpu.build_int_lut(enc.index),
                    )
                comp_luts, int_lut = luts[feat]
                comp_codes, missing = [], None
                for (lut, missing_code), col in zip(comp_luts, unit_cols):
                    c, m = _gpu.codes_from_lut(lut, missing_code, Xg[col])
                    comp_codes.append(c)
                    missing = m if missing is None else (missing | m)
                joint = _gpu.joint_codes_device(plan.radices, comp_codes)
                codes = _gpu.codes_from_int_lut(int_lut, joint)
            if hm == "error" and bool(missing.any()):
                raise ValueError(f"Missing values in unit {feat!r} with handle_missing='error'.")
            cache[feat] = (codes, missing)
        codes, missing = cache[feat]

        col = _gpu.gather_cells(enc.values, codes)  # unknown/missing codes (-1) -> NaN
        notfound = codes < 0
        if hm == "return_nan":
            unknown = notfound & ~missing
        elif hm == "value":
            # not-found rows are unseen real categories OR an unseen missing level
            unknown = notfound
        else:  # "error": missing already raised above
            unknown = notfound
        if bool(unknown.any()):
            if hu == "error":
                raise ValueError(
                    f"Found unknown categories in column {feat!r} with handle_unknown='error'."
                )
            if hu == "value":
                col[unknown] = enc.fallback
            # "return_nan": leave NaN
        cols.append(col)
    return cols
