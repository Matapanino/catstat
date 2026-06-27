"""Cross-fitting: deterministic fold assignment for leakage-safe ``fit_transform``.

``catstat`` owns fold assignment so CPU and (future) GPU produce identical out-of-fold encodings.
``random_state`` flows only through the resolved splitter; the global numpy RNG is never touched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


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


def kfold_mean_oof_fast(
    keys, missing_mask, yv, fold_id, n_folds, smooth, handle_missing, handle_unknown
) -> np.ndarray:
    """Out-of-fold mean encoding for a partitioning kfold CV, in a single vectorized pass.

    Mathematically identical to fitting ``_smoothing.fit_mean_encoding`` on each fold's COMPLEMENT
    and mapping the held-out fold's rows -- but computed without the per-fold group-by loop. Under a
    partitioning CV (``KFold``/``StratifiedKFold``), the complement of test-fold ``f`` is exactly
    its train set, so each fold's ``(count, sum, sumsq)`` is ``global - this_fold`` by
    subtraction of one composite ``(fold, key)`` aggregation. The smoothing arithmetic mirrors
    ``fit_mean_encoding`` exactly (fixed m-estimate and ``smooth='auto'`` empirical-Bayes with the
    fold-complement population mean/variance); parity is asserted at allclose by the leakage audit.

    ``yv`` is the (possibly binarized, for classification) target. Returns a float array of length
    ``n``; rows are NaN where ``handle_missing='return_nan'`` drops them, or where an unseen
    category meets ``handle_unknown='return_nan'``.
    """
    n = len(keys)
    out = np.full(n, np.nan, dtype=float)
    yv = np.asarray(yv, dtype=float)

    # active rows are those that enter the statistics -- mirrors `_fit_all`'s `sel`: keep all rows
    # for handle_missing='value' (MISSING is its own key) and 'error' (no missing present by then);
    # drop missing rows for 'return_nan'.
    active = ~missing_mask if handle_missing == "return_nan" else np.ones(n, dtype=bool)
    a = np.nonzero(active)[0]
    if a.size == 0:
        return out

    codes, _uniq = pd.factorize(keys[a])  # integer key codes over active rows
    n_cat = int(codes.max()) + 1
    fid = fold_id[a]
    y_a = yv[a]
    y2_a = y_a * y_a

    # one composite (fold, key) aggregation via flattened bincount; per-key globals by summing folds
    comp = fid * n_cat + codes
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

    # per-fold complement prior: global mean (and population var for 'auto') over the other folds
    tn = np.bincount(fid, minlength=n_folds).astype(float)
    ts = np.bincount(fid, weights=y_a, minlength=n_folds)
    tss = np.bincount(fid, weights=y2_a, minlength=n_folds)
    cn = tn.sum() - tn  # > 0 for n_folds >= 2 (always: KFold rejects n_splits < 2)
    g = (ts.sum() - ts) / cn
    g_row = g[fid]

    seen = cc > 0.0  # category present in the complement (else -> handle_unknown)
    cc_safe = np.where(seen, cc, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_c = cs / cc_safe
        if isinstance(smooth, str):  # 'auto' empirical-Bayes, per fold
            tau2 = (tss.sum() - tss) / cn - g * g
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

    out[a] = enc
    return out
