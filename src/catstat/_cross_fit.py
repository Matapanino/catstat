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
