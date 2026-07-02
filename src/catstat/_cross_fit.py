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


class _OOFTables(NamedTuple):
    """Per-(fold, key) complement power-sum tables plus per-fold complement totals.

    Every OOF encoding is a function of ``(fold, key)`` only, so each statistic's finalizer builds
    a small ``(n_folds * n_cat)`` value table ``E`` and one gather ``E[comp]`` scatters it to the
    rows -- the same IEEE operations on the same operand values as per-row arithmetic, but the
    elementwise work drops from ``O(n_rows)`` to ``O(n_folds * n_cat)`` per statistic. The raw
    per-cell sums come from a ``moment_tables`` kernel (numpy ``bincount`` on CPU; the GPU twin
    computes the same sums on device and returns small host arrays), which is the backend seam.

    All sums are of ``y' = y - shift`` (``shift=0.0`` unless a shape stat asked for the stabilizing
    global-mean shift; central moments are shift-invariant, and the mean finalizer adds ``shift``
    back). ``c3``/``c4`` (sums of ``y'**3``/``y'**4``) are populated only for ``order=4``.
    """

    n: int  # total rows (output length)
    a: np.ndarray  # active row indices (rows that enter the statistics)
    comp: np.ndarray  # flattened (fold, key) cell index per active row: fid * n_cat + code
    n_folds: int
    n_cat: int
    fc: np.ndarray  # (F*C) rows of this (fold, key) cell -- fc > 0 marks cells actually gathered
    cc: np.ndarray  # (F*C) complement count (all folds but this one, this key)
    cs: np.ndarray  # (F*C) complement sum
    css: np.ndarray  # (F*C) complement sum-of-squares
    cn: np.ndarray  # (F,) complement count per fold
    cs_fold: np.ndarray  # (F,) complement sum per fold
    css_fold: np.ndarray  # (F,) complement sum-of-squares per fold
    c3: np.ndarray | None = None  # (F*C) complement sum of y'**3 (order=4 only)
    c4: np.ndarray | None = None  # (F*C) complement sum of y'**4 (order=4 only)
    c3_fold: np.ndarray | None = None  # (F,) complement sum of y'**3 (order=4 only)
    c4_fold: np.ndarray | None = None  # (F,) complement sum of y'**4 (order=4 only)
    shift: float = 0.0  # subtracted from y before the sums; mean finalizer adds it back


# smallest complement count for which a dispersion/shape stat is defined (ddof=1 variance needs 2,
# bias-corrected G1 skew needs 3, bias-corrected G2 kurtosis needs 4); below it -> fold-global.
_STAT_MIN_N = {"var": 2, "std": 2, "skew": 3, "kurt": 4}




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


def complement_tables(
    n,
    a,
    codes,
    n_cat,
    fid_active,
    yv_active,
    n_folds,
    order: int = 2,
    shift: float = 0.0,
    moment_tables=None,
) -> _OOFTables:
    """Out-of-fold power-sum tables per ``(fold, key)`` cell, by subtraction from the grand
    totals of one composite aggregation.

    Under a partitioning CV (``KFold``/``StratifiedKFold``) the complement of test-fold ``f`` is
    exactly its train set, so each cell's moments are ``per-key global - this fold's cell``.
    ``yv_active`` is the (possibly binarized) target over the active rows. This is the single pass
    shared by the mean, var/std, woe and (with ``order=4``) skew/kurt finalizers; parity vs the
    per-fold path is asserted at allclose by the audit. ``shift`` is subtracted from ``y`` first
    (exact for shift-invariant stats; the mean finalizer adds it back) so the shape stats'
    subtractive moment reconstruction stays numerically stable. ``moment_tables`` is the backend
    kernel (``backends._cpu.oof_moment_tables`` by default; the cupy twin on GPU).
    """
    if moment_tables is None:
        from .backends import _cpu

        moment_tables = _cpu.oof_moment_tables
    y_a = np.asarray(yv_active, dtype=float)
    if shift != 0.0:
        y_a = y_a - shift

    comp = fid_active * n_cat + codes
    size = n_folds * n_cat
    fc, fs, fss, fs3, fs4 = moment_tables(comp, y_a, size, order)
    return tables_from_raw(n, a, comp, n_folds, n_cat, fc, fs, fss, fs3, fs4, shift)


def tables_from_raw(
    n, a, comp, n_folds, n_cat, fc, fs, fss, fs3=None, fs4=None, shift: float = 0.0
) -> _OOFTables:
    """Build :class:`_OOFTables` from the raw per-(fold, key) sums.

    Split out of :func:`complement_tables` so the device path (which computes the raw sums on
    device and holds ``comp``/row indices as cupy arrays) can reuse the complement arithmetic on
    the small host tables. ``comp``/``a`` may then be device arrays: only the *cell* functions
    (:func:`mean_cells` etc.) may be used on such a table, with the caller doing its own gather --
    the host :func:`_scatter_cells` requires host arrays.
    """

    def complement(t):
        """(per-key global - this fold's cell) over the flattened table; also the fold totals."""
        m = t.reshape(n_folds, n_cat)
        return (m.sum(0)[None, :] - m).ravel(), m.sum(1)

    cc, tn = complement(fc)
    cs, ts = complement(fs)
    css, tss = complement(fss)
    c3 = c4 = c3_fold = c4_fold = None
    if fs3 is not None:
        c3, ts3 = complement(fs3)
        c4, ts4 = complement(fs4)
        c3_fold = ts3.sum() - ts3
        c4_fold = ts4.sum() - ts4

    return _OOFTables(
        n=n,
        a=a,
        comp=comp,
        n_folds=n_folds,
        n_cat=n_cat,
        fc=fc,
        cc=cc,
        cs=cs,
        css=css,
        cn=tn.sum() - tn,  # > 0 for n_folds >= 2 (always: KFold rejects n_splits < 2)
        cs_fold=ts.sum() - ts,
        css_fold=tss.sum() - tss,
        c3=c3,
        c4=c4,
        c3_fold=c3_fold,
        c4_fold=c4_fold,
        shift=shift,
    )


def _fold_cells(per_fold, n_cat):
    """Broadcast a per-fold vector to the flattened (fold, key) cell layout."""
    return np.repeat(np.asarray(per_fold, dtype=float), n_cat)


def _apply_unknown_cells(tab: _OOFTables, E, g_cell, handle_unknown):
    """Unknown handling on the value table: a cell is *unseen* when its key is absent from the
    fold's complement (``cc == 0``). Only occupied cells (``fc > 0``) are ever gathered, so
    ``'error'`` raises iff an occupied unseen cell exists -- exactly the rows the per-row path
    raised for; unoccupied cells are don't-cares (left NaN)."""
    unseen = tab.cc <= 0.0
    if not unseen.any():
        return E
    if handle_unknown == "error":
        if (unseen & (tab.fc > 0.0)).any():
            raise ValueError(
                "Found unknown categories during out-of-fold encoding (handle_unknown='error')."
            )
        return np.where(unseen, np.nan, E)
    return np.where(unseen, g_cell if handle_unknown == "value" else np.nan, E)


def _scatter_cells(tab: _OOFTables, E) -> np.ndarray:
    """Gather the (fold, key) value table onto the rows: ``out[a] = E[comp]``; inactive rows NaN."""
    out = np.full(tab.n, np.nan, dtype=float)
    if tab.a.size:
        out[tab.a] = E[tab.comp]
    return out


def _mean_enc_cells(tab: _OOFTables, smooth, handle_unknown):
    """Shared smoothed-mean arithmetic over the (fold, key) cells: ``(E, g_cell)``.

    ``E`` is the per-cell OOF smoothed mean (fixed m-estimate or ``smooth='auto'``
    empirical-Bayes with the per-fold complement population mean/variance; mirrors
    ``_smoothing.fit_mean_encoding`` per fold), with the unknown handling already applied
    (``'value'`` cells are exactly the per-fold prior ``g_cell``). Consumed by the mean finalizer
    (un-shift + scatter) and the WOE finalizer (logit difference against ``g_cell``)."""
    cn = tab.cn
    g = tab.cs_fold / cn
    g_cell = _fold_cells(g, tab.n_cat)
    cc, cs, css = tab.cc, tab.cs, tab.css
    seen = cc > 0.0
    cc_safe = np.where(seen, cc, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_c = cs / cc_safe
        if isinstance(smooth, str):  # 'auto' empirical-Bayes, per fold
            tau2 = tab.css_fold / cn - g * g
            tau2_cell = _fold_cells(tau2, tab.n_cat)
            var_pop = np.clip(css / cc_safe - mean_c * mean_c, 0.0, None)
            m = np.where(tau2_cell > 0.0, var_pop / np.where(tau2_cell > 0.0, tau2_cell, 1.0), 0.0)
            lam = cc / (cc + m)
            E = lam * mean_c + (1.0 - lam) * g_cell
        else:
            mm = float(smooth)
            E = mean_c if mm == 0.0 else (cc * mean_c + mm * g_cell) / (cc + mm)
    E = _apply_unknown_cells(tab, E, g_cell, handle_unknown)
    return E, g_cell


def mean_cells(tab: _OOFTables, smooth, handle_unknown) -> np.ndarray:
    """The mean finalizer's (fold, key) value table. When the moments were computed on shifted
    values (``tab.shift != 0``) the shift is added back on the cells -- both smoothers are
    affine-equivariant (a convex blend of shifted means plus ``shift`` equals the blend of
    unshifted means; the EB weights use only shift-invariant variances), so the result is the
    unshifted encoding exactly (up to fp rounding, covered by the audit)."""
    E, _g_cell = _mean_enc_cells(tab, smooth, handle_unknown)
    if tab.shift != 0.0:
        E = E + tab.shift  # NaN cells survive the add
    return E


def woe_cells(tab: _OOFTables, smooth, handle_unknown) -> np.ndarray:
    """The WOE finalizer's value table: ``logit(smoothed p) - logit(per-fold prior)`` from the
    SAME tables as the mean.

    ``tab`` is computed on the binarized target (positive class = ``classes_[1]``), never
    shifted (binary targets take no shape stats). Unknown cells under ``'value'`` are encoded at
    their fold's prior by :func:`_mean_enc_cells`, so their WOE is exactly 0.0 -- the documented
    unknown fallback."""
    from ._smoothing import woe_from_prob

    E, g_cell = _mean_enc_cells(tab, smooth, handle_unknown)
    return woe_from_prob(E, g_cell)


def finalize_mean_oof(tab: _OOFTables, smooth, handle_unknown) -> np.ndarray:
    """OOF mean encoding: :func:`mean_cells` + the row gather (see :class:`_OOFTables`)."""
    if tab.a.size == 0:
        return np.full(tab.n, np.nan, dtype=float)
    return _scatter_cells(tab, mean_cells(tab, smooth, handle_unknown))


def finalize_woe_oof(tab: _OOFTables, smooth, handle_unknown) -> np.ndarray:
    """OOF WOE encoding: :func:`woe_cells` + the row gather."""
    if tab.a.size == 0:
        return np.full(tab.n, np.nan, dtype=float)
    return _scatter_cells(tab, woe_cells(tab, smooth, handle_unknown))


def dispersion_cells(tab: _OOFTables, stat, min_samples, handle_unknown) -> np.ndarray:
    """The var/std finalizer's (fold, key) value table -- no smoothing (honesty rule).

    A seen cell whose complement count is ``< max(min_samples, 1)`` or ``< _STAT_MIN_N[stat]``
    (sample variance undefined for a singleton, ddof=1) falls back to the per-fold
    complement-global statistic; unseen cells (key absent from the fold's complement) follow
    ``handle_unknown``. Mirrors ``_aggregations.fit_stat_encoding`` fitted on each fold's
    complement and mapped to the held-out rows. The complement-global sample variance is
    ``(ss - s**2/cn)/(cn - 1)`` (0.0 when cn <= 1). Var/std are shift-invariant, so tables
    computed on shifted values need no correction here.
    """
    cn, cs_fold, css_fold = tab.cn, tab.cs_fold, tab.css_fold
    cc, cs, css = tab.cc, tab.cs, tab.css
    with np.errstate(invalid="ignore", divide="ignore"):
        cn_d = np.where(cn > 1, cn, 1.0)
        g_var = np.where(cn > 1, (css_fold - cs_fold * cs_fold / cn_d) / (cn_d - 1.0), 0.0)
        g_cell = _fold_cells(g_var, tab.n_cat)
        cc_pos = np.where(cc > 0, cc, 1.0)
        mean_c = cs / cc_pos
        var_raw = (css - cs * mean_c) / np.where(cc > 1, cc - 1.0, 1.0)  # (ss - s**2/cc)/(cc-1)

    # seen but undersupported (incl. singleton: var is NaN) -> per-fold complement-global stat
    lowcount = (cc < max(int(min_samples), 1)) | (cc < _STAT_MIN_N[stat])
    E = np.where(lowcount, g_cell, var_raw)
    E = _apply_unknown_cells(tab, E, g_cell, handle_unknown)
    if stat == "std":
        E = np.sqrt(np.clip(E, 0.0, None))  # std = sqrt(var); clip guards fp cancellation
    return E


def shape_cells(tab: _OOFTables, stat, min_samples, handle_unknown) -> np.ndarray:
    """The skew/kurt finalizer's (fold, key) value table -- no smoothing (honesty rule; shape
    stats never blend).

    Structure mirrors :func:`dispersion_cells`: a seen cell whose complement count is
    ``< max(min_samples, 1)`` or where the statistic is undefined (``n < 3`` skew / ``n < 4``
    kurt, NaN from the reconstruction) falls back to the per-fold complement-global G1/G2 (itself
    0.0 when undefined, mirroring ``_aggregations.global_stat``); unseen cells follow
    ``handle_unknown``. G1/G2 are shift-invariant, so the shifted sums need no correction.
    """
    from ._aggregations import g1_g2_from_power_sums

    g_fold = g1_g2_from_power_sums(
        tab.cn, tab.cs_fold, tab.css_fold, tab.c3_fold, tab.c4_fold, stat
    )
    g_fold = np.where(np.isnan(g_fold), 0.0, g_fold)  # fold-global undefined -> 0.0
    g_cell = _fold_cells(g_fold, tab.n_cat)
    val = g1_g2_from_power_sums(tab.cc, tab.cs, tab.css, tab.c3, tab.c4, stat)
    # seen but undersupported (NaN: n below the stat's min-n) -> per-fold complement-global stat
    lowcount = (tab.cc < max(int(min_samples), 1)) | np.isnan(val)
    E = np.where(lowcount, g_cell, val)
    return _apply_unknown_cells(tab, E, g_cell, handle_unknown)


def finalize_dispersion_oof(tab: _OOFTables, stat, min_samples, handle_unknown) -> np.ndarray:
    """OOF var/std encoding: :func:`dispersion_cells` + the row gather."""
    if tab.a.size == 0:
        return np.full(tab.n, np.nan, dtype=float)
    return _scatter_cells(tab, dispersion_cells(tab, stat, min_samples, handle_unknown))


def finalize_shape_oof(tab: _OOFTables, stat, min_samples, handle_unknown) -> np.ndarray:
    """OOF skew/kurt encoding: :func:`shape_cells` + the row gather."""
    if tab.a.size == 0:
        return np.full(tab.n, np.nan, dtype=float)
    return _scatter_cells(tab, shape_cells(tab, stat, min_samples, handle_unknown))
