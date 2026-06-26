"""Cross-fitting: deterministic fold assignment for leakage-safe ``fit_transform``.

``catstat`` owns fold assignment so CPU and (future) GPU produce identical out-of-fold encodings.
``random_state`` flows only through the resolved splitter; the global numpy RNG is never touched.
"""

from __future__ import annotations

import numpy as np
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
