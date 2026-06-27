"""Cross-fitting: deterministic fold assignment for leakage-safe ``fit_transform``.

``catstat`` owns fold assignment so CPU and (future) GPU produce identical out-of-fold encodings.
``random_state`` flows only through the resolved splitter; the global numpy RNG is never touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

# A mixed-radix joint code lives in ``[0, prod(radices))``; beyond int64 we fall back to tuple keys.
_INT64_MAX = int(np.iinfo(np.int64).max)


class _PrecomputedSplitter:
    """Wrap a user-provided iterable of ``(train_idx, test_idx)`` tuples."""

    def __init__(self, splits):
        self._splits = [(np.asarray(tr), np.asarray(te)) for tr, te in splits]

    def split(self, X, y=None, groups=None):
        return iter(self._splits)


def resolve_cv(cv, target_type: str, shuffle: bool, random_state):
    """Return a splitter object for the given ``cv`` argument.

    int -> KFold (continuous) or StratifiedKFold (binary/multiclass). A splitter object is
    returned as-is. An iterable of index pairs is wrapped.
    """
    if hasattr(cv, "split"):
        return cv
    if isinstance(cv, (int, np.integer)):
        rs = random_state if shuffle else None
        if target_type == "continuous":
            return KFold(n_splits=int(cv), shuffle=shuffle, random_state=rs)
        return StratifiedKFold(n_splits=int(cv), shuffle=shuffle, random_state=rs)
    # assume an iterable of (train, test) index arrays
    return _PrecomputedSplitter(cv)


def make_folds(n_rows: int, y, splitter) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialize the ``(train_idx, test_idx)`` folds.

    A dummy feature matrix is passed for shape; stratified splitters use ``y``.
    """
    dummy_X = np.zeros((n_rows, 1))
    return list(splitter.split(dummy_X, y))


def loo_encode(keys, y, m: float, prior: float) -> np.ndarray:
    """Leave-one-out mean encoding (deterministic, leakage-safe for the training set).

    Each row is encoded by its category mean computed from **every other row**:
    ``(cat_sum - y_i + m*prior) / (cat_count - 1 + m)``. With ``m=0`` this is the classic LOO mean;
    singletons (empty denominator) fall back to the global ``prior``.
    """
    yv = np.asarray(y, dtype=float)
    grp = pd.DataFrame({"k": pd.Series(keys), "y": yv}).groupby("k", sort=False)["y"]
    cat_sum = grp.transform("sum").to_numpy()
    cat_cnt = grp.transform("count").to_numpy()
    num = cat_sum - yv + m * prior
    den = cat_cnt - 1.0 + m
    den_safe = np.where(den > 0, den, 1.0)  # avoid 0/0 warning; result is overwritten by prior
    return np.where(den > 0, num / den_safe, prior)


def ordered_encode(keys, y, a: float, prior: float, perm: np.ndarray) -> np.ndarray:
    """CatBoost-style ordered target statistics.

    Walk the rows in a random permutation; each row is encoded from only the **prior** rows of its
    category in that order: ``(prior_sum + a*prior) / (prior_count + a)`` (first occurrence ->
    prior).
    """
    yv = np.asarray(y, dtype=float)
    ks = np.asarray(keys, dtype=object)[perm]
    ys = yv[perm]
    g = pd.Series(ys).groupby(ks, sort=False)
    prior_sum = g.cumsum().to_numpy() - ys  # cumsum includes current -> subtract it
    prior_cnt = g.cumcount().to_numpy()  # 0-based position == count of earlier rows
    enc_perm = (prior_sum + a * prior) / (prior_cnt + a)
    out = np.empty(len(yv), dtype=float)
    out[perm] = enc_perm
    return out


class _OOFMoments(NamedTuple):
    """Per-(fold, key) complement count/sum/sumsq plus per-fold complement totals.

    Computed once per (encoding unit, target vector) by :func:`complement_moments`; each statistic's
    finalizer reads these shared moments, so a unit's mean/var/std cost one factorize and one
    composite ``(fold, key)`` bincount between them -- the per-fold group-by loop is avoided.
    """

    n: int  # total rows (output length)
    a: np.ndarray  # active row indices (rows that enter the statistics)
    fid: np.ndarray  # fold id per active row
    cc: np.ndarray  # complement count per active row
    cs: np.ndarray  # complement sum per active row
    css: np.ndarray  # complement sum-of-squares per active row
    seen: np.ndarray  # cc > 0 (category present in this row's complement)
    cn: np.ndarray  # complement count per fold
    cs_fold: np.ndarray  # complement sum per fold
    css_fold: np.ndarray  # complement sum-of-squares per fold


def factorize_active(keys, missing_mask, handle_missing):
    """Integer-code the active rows' keys once (shared across a unit's stats and classes).

    Active rows are those that enter the statistics -- mirrors ``_fit_all``'s ``sel``: keep all rows
    for handle_missing='value' (MISSING is its own key) and 'error' (no missing present by then);
    drop missing rows for 'return_nan'. Returns ``(n, a, codes, n_cat)``.
    """
    n = len(keys)
    active = ~missing_mask if handle_missing == "return_nan" else np.ones(n, dtype=bool)
    a = np.nonzero(active)[0]
    if a.size == 0:
        return n, a, np.empty(0, dtype=np.intp), 0
    codes, _uniq = pd.factorize(keys[a])  # integer key codes over active rows
    return n, a, codes, int(codes.max()) + 1


def gather(values: np.ndarray, codes: np.ndarray, has_unknown: bool) -> np.ndarray:
    """Reproduce ``pd.Series.map`` over integer codes, as a numpy fancy-index gather.

    ``values`` is a unit's encoding aligned to its canonical category index; ``codes`` come from
    ``index.get_indexer(keys)`` (``>= 0`` for a known category, ``-1`` for a key absent from the
    index). Known codes gather ``values[code]``; unknown codes (``-1``) map to NaN -- identical to
    ``.map`` returning NaN for a key that is not in the encoding's index, so the downstream
    unknown/missing fallback logic in ``_transform_array`` is unchanged.

    ``has_unknown`` (any ``codes < 0``) is precomputed once per unit: when every key is known the
    gather is a single fancy index (the common transform-on-seen-data case); otherwise unknown rows
    are masked to NaN. Both branches return a fresh array safe to mutate in place.
    """
    if not has_unknown:
        return values[codes]  # fast path: every key known -> no -1 to mask
    out = np.empty(codes.shape[0], dtype=float)
    known = codes >= 0
    out[known] = values[codes[known]]
    out[~known] = np.nan
    return out


@dataclass(frozen=True)
class _JointKeyPlan:
    """Encode a ``combination`` unit's joint category as a single int64 code instead of a tuple.

    A combination unit's category is the tuple of its components' (normalized) values. Rather than
    materialize those tuples per row, each component is integer-coded once from the full training X
    into a value-stable ``pd.Index`` (``uniques``); per row the component codes are folded into one
    mixed-radix int64 ``joint = ((c0*n1 + c1)*n2 + c2)...``. The same combination then maps to the
    same int at fit and at transform, so the existing code-gather (``index.get_indexer`` over an
    ``Int64Index``) works unchanged. ``use_int`` is False when the radix product overflows int64 --
    the caller then falls back to the object-tuple key build for that unit.
    """

    uniques: tuple  # one pd.Index per component (value-stable category map, incl. MISSING sentinel)
    radices: tuple  # n_c (number of distinct categories) per component
    use_int: bool


def build_joint_keyplan(component_keys) -> _JointKeyPlan:
    """Build a :class:`_JointKeyPlan` from each component's normalized key array (full training X).

    ``component_keys`` is a list of object arrays already run through ``normalize_keys`` (missing
    entries are the ``MISSING`` sentinel). Each is factorized to a value-stable ``pd.Index``; the
    radix product is checked against int64 so the caller knows whether the int path is safe.
    """
    uniques = []
    radices = []
    product = 1
    for keys in component_keys:
        _codes, uniq = pd.factorize(keys)  # value-stable category map for this component
        uniques.append(pd.Index(uniq))
        radices.append(len(uniq))
        product *= max(len(uniq), 1)
    return _JointKeyPlan(tuple(uniques), tuple(radices), product <= _INT64_MAX)


def joint_codes(plan: _JointKeyPlan, component_keys) -> np.ndarray:
    """Mixed-radix int64 joint code per row from a unit's per-component normalized key arrays.

    Each component value is mapped through its fit-time ``uniques`` (``get_indexer``: ``>= 0``
    known, ``-1`` unknown). Any row with an unknown component is forced to the ``-1`` sentinel --
    never a valid (non-negative) canonical code, so the downstream ``index.get_indexer`` returns -1
    and the row takes the unknown/fallback path, like a tuple key absent from the encoding's index.
    """
    n = len(component_keys[0])
    joint = np.zeros(n, dtype=np.int64)
    unknown = np.zeros(n, dtype=bool)
    for uniq, radix, keys in zip(plan.uniques, plan.radices, component_keys):
        codes = uniq.get_indexer(keys)  # >= 0 known, -1 unknown (value absent from fit uniques)
        unknown |= codes < 0
        joint = joint * np.int64(radix) + codes.astype(np.int64)  # known rows stay in [0, product)
    if unknown.any():
        joint[unknown] = -1  # contaminated arithmetic above is overwritten here
    return joint


def decode_joint(plan: _JointKeyPlan, codes) -> list:
    """Invert :func:`joint_codes` for canonical (non-negative) codes -> component-value tuples.

    Used only to rebuild ``categories_`` in its tuple representation (O(#categories), not per-row).
    """
    codes = np.asarray(codes, dtype=np.int64)
    k = len(plan.radices)
    comp_codes = [None] * k
    rem = codes.copy()
    for c in range(k - 1, -1, -1):  # least-significant component first
        radix = np.int64(plan.radices[c])
        comp_codes[c] = rem % radix
        rem //= radix
    return [
        tuple(plan.uniques[c][int(comp_codes[c][i])] for c in range(k)) for i in range(len(codes))
    ]


def complement_moments(n, a, codes, n_cat, fid_active, yv_active, n_folds) -> _OOFMoments:
    """Out-of-fold ``(count, sum, sumsq)`` per active row, by subtraction from the grand totals of
    one composite ``(fold, key)`` aggregation.

    Under a partitioning CV (``KFold``/``StratifiedKFold``) the complement of test-fold ``f`` is
    exactly its train set, so each fold's moments are ``global - this_fold``. ``yv_active`` is the
    (possibly binarized) target over the active rows. This is the single pass shared by the mean,
    var and std finalizers; parity vs the per-fold path is asserted at allclose by the audit.
    """
    y_a = np.asarray(yv_active, dtype=float)
    y2_a = y_a * y_a

    # one composite (fold, key) aggregation via flattened bincount; per-key globals by summing folds
    comp = fid_active * n_cat + codes
    size = n_folds * n_cat
    fc = np.bincount(comp, minlength=size).astype(float)
    fs = np.bincount(comp, weights=y_a, minlength=size)
    fss = np.bincount(comp, weights=y2_a, minlength=size)
    gc = fc.reshape(n_folds, n_cat).sum(0)
    gs = fs.reshape(n_folds, n_cat).sum(0)
    gss = fss.reshape(n_folds, n_cat).sum(0)

    # complement (all folds but this row's) per active row, by subtraction
    cc = gc[codes] - fc[comp]
    cs = gs[codes] - fs[comp]
    css = gss[codes] - fss[comp]

    # per-fold complement totals (the fold's training set): prior mean / global var fallback
    tn = np.bincount(fid_active, minlength=n_folds).astype(float)
    ts = np.bincount(fid_active, weights=y_a, minlength=n_folds)
    tss = np.bincount(fid_active, weights=y2_a, minlength=n_folds)
    return _OOFMoments(
        n=n,
        a=a,
        fid=fid_active,
        cc=cc,
        cs=cs,
        css=css,
        seen=cc > 0.0,  # category present in the complement (else -> handle_unknown)
        cn=tn.sum() - tn,  # > 0 for n_folds >= 2 (always: KFold rejects n_splits < 2)
        cs_fold=ts.sum() - ts,
        css_fold=tss.sum() - tss,
    )


def finalize_mean_oof(mom: _OOFMoments, smooth, handle_unknown) -> np.ndarray:
    """OOF mean encoding from the shared moments (fixed m-estimate and ``smooth='auto'``
    empirical-Bayes with the per-fold complement population mean/variance). Identical arithmetic to
    the original single-pass mean kernel; mirrors ``_smoothing.fit_mean_encoding`` per fold."""
    out = np.full(mom.n, np.nan, dtype=float)
    if mom.a.size == 0:
        return out
    cn, fid = mom.cn, mom.fid
    g = mom.cs_fold / cn
    g_row = g[fid]
    cc, cs, css, seen = mom.cc, mom.cs, mom.css, mom.seen
    cc_safe = np.where(seen, cc, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_c = cs / cc_safe
        if isinstance(smooth, str):  # 'auto' empirical-Bayes, per fold
            tau2 = mom.css_fold / cn - g * g
            tau2_row = tau2[fid]
            var_pop = np.clip(css / cc_safe - mean_c * mean_c, 0.0, None)
            m = np.where(tau2_row > 0.0, var_pop / np.where(tau2_row > 0.0, tau2_row, 1.0), 0.0)
            lam = cc / (cc + m)
            enc = lam * mean_c + (1.0 - lam) * g_row
        else:
            mm = float(smooth)
            enc = mean_c if mm == 0.0 else (cc * mean_c + mm * g_row) / (cc + mm)

    if not seen.all():
        if handle_unknown == "error":
            raise ValueError(
                "Found unknown categories during out-of-fold encoding (handle_unknown='error')."
            )
        enc = np.where(seen, enc, g_row if handle_unknown == "value" else np.nan)

    out[mom.a] = enc
    return out


def finalize_dispersion_oof(mom: _OOFMoments, stat, min_samples, handle_unknown) -> np.ndarray:
    """OOF var/std encoding from the SAME shared moments -- no smoothing (honesty rule).

    A seen category whose complement count is ``< max(min_samples, 1)`` or ``< 2`` (sample
    variance undefined for a singleton, ddof=1) falls back to the per-fold complement-global
    statistic; unseen categories (absent from the fold's complement) follow ``handle_unknown``.
    Mirrors ``_aggregations.fit_stat_encoding`` fitted on each fold's complement and mapped to the
    held-out rows. The complement-global sample variance is ``(ss - s**2/cn)/(cn - 1)`` (0.0 when
    cn <= 1).
    """
    out = np.full(mom.n, np.nan, dtype=float)
    if mom.a.size == 0:
        return out
    cn, cs_fold, css_fold, fid = mom.cn, mom.cs_fold, mom.css_fold, mom.fid
    cc, cs, css, seen = mom.cc, mom.cs, mom.css, mom.seen
    with np.errstate(invalid="ignore", divide="ignore"):
        cn_d = np.where(cn > 1, cn, 1.0)
        g_var = np.where(cn > 1, (css_fold - cs_fold * cs_fold / cn_d) / (cn_d - 1.0), 0.0)
        g_var_row = g_var[fid]
        cc_pos = np.where(cc > 0, cc, 1.0)
        mean_c = cs / cc_pos
        var_raw = (css - cs * mean_c) / np.where(cc > 1, cc - 1.0, 1.0)  # (ss - s**2/cc)/(cc-1)

    # seen but undersupported (incl. singleton: var is NaN) -> per-fold complement-global stat
    lowcount = (cc < max(int(min_samples), 1)) | (cc < 2)
    enc = np.where(lowcount, g_var_row, var_raw)
    if not seen.all():
        if handle_unknown == "error":
            raise ValueError(
                "Found unknown categories during out-of-fold encoding (handle_unknown='error')."
            )
        enc = np.where(seen, enc, g_var_row if handle_unknown == "value" else np.nan)
    if stat == "std":
        enc = np.sqrt(np.clip(enc, 0.0, None))  # std = sqrt(var); clip guards fp cancellation
    out[mom.a] = enc
    return out


def kfold_mean_oof_fast(
    keys, missing_mask, yv, fold_id, n_folds, smooth, handle_missing, handle_unknown
) -> np.ndarray:
    """Single-pass OOF mean encoding for a partitioning kfold CV (kept for back-compat; thin
    wrapper over :func:`complement_moments` + :func:`finalize_mean_oof`)."""
    n, a, codes, n_cat = factorize_active(keys, missing_mask, handle_missing)
    if a.size == 0:
        return np.full(n, np.nan, dtype=float)
    mom = complement_moments(n, a, codes, n_cat, fold_id[a], np.asarray(yv, float)[a], n_folds)
    return finalize_mean_oof(mom, smooth, handle_unknown)
