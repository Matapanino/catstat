"""Non-mean target statistics: var / std / median / min / max.

These have **no principled smoothing** (the smoothing honesty rule): order statistics never blend,
and var/std default to no shrinkage. Each falls back to the **global** statistic for unseen
categories and for categories below ``min_samples_category`` (or where the statistic is undefined,
e.g. the sample variance of a singleton). Continuous targets only -- the encoders reject these for
classification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backends import _cpu


def global_stat(y, stat: str) -> float:
    y = np.asarray(y, dtype=float)
    if stat == "var":
        return float(np.var(y, ddof=1)) if len(y) > 1 else 0.0
    if stat == "std":
        return float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    if stat == "median":
        return float(np.median(y))
    if stat == "min":
        return float(np.min(y))
    if stat == "max":
        return float(np.max(y))
    raise ValueError(f"Unknown non-mean stat {stat!r}.")


def fit_stat_encoding(
    keys, y, stat: str, min_samples: int, backend=None
) -> tuple[pd.Series, float]:
    """Return ``(encoding_by_category, global_fallback)`` for a dispersion/order statistic."""
    if backend is None:
        backend = _cpu
    per_cat = backend.category_agg(keys, y, stat)  # Series; NaN where undefined (e.g. var of n=1)
    counts = pd.Series(keys).value_counts().reindex(per_cat.index)
    gv = global_stat(y, stat)
    # fall back to the global statistic for undefined or under-supported categories
    fallback_mask = per_cat.isna() | (counts < max(int(min_samples), 1))
    enc = per_cat.where(~fallback_mask, gv).astype(float)
    return enc, gv
