"""Non-mean target statistics: var / std / median / min / max / skew / kurt.

These have **no principled smoothing** (the smoothing honesty rule): order/shape statistics never
blend, and var/std default to no shrinkage. Each falls back to the **global** statistic for unseen
categories and for categories below ``min_samples_category`` (or where the statistic is undefined,
e.g. the sample variance of a singleton, skew for n < 3, kurt for n < 4). Continuous targets only
-- the encoders reject these for classification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backends import _cpu

# smallest n for which the (bias-corrected) statistic is defined; below it -> global fallback
_SHAPE_MIN_N = {"skew": 3, "kurt": 4}


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
    if stat == "skew":
        s = pd.Series(y).skew()  # NaN for n < 3
        return float(s) if pd.notna(s) else 0.0
    if stat == "kurt":
        s = pd.Series(y).kurt()  # NaN for n < 4
        return float(s) if pd.notna(s) else 0.0
    raise ValueError(f"Unknown non-mean stat {stat!r}.")


def g1_g2_from_power_sums(cnt, s1, s2, s3, s4, stat: str) -> np.ndarray:
    """Bias-corrected sample skewness (G1) / excess kurtosis (G2) from raw power sums.

    Inputs are per-category ``count`` and ``S_k = sum(y'**k)`` of (optionally shifted) values
    ``y'``. Central moments are shift-invariant, so callers pre-shift ``y`` by a global mean:
    the result is algebraically identical but the subtractions below stop cancelling
    catastrophically when ``|mean| >> sd``. Matches pandas ``nanskew``/``nankurt`` (adjusted
    Fisher-Pearson G1; bias-corrected excess kurtosis G2): NaN where undefined (n < 3 / n < 4,
    the caller's global-fallback trigger), 0.0 for a (numerically) constant category. The
    zero-variance guard is relative (``M2 <= 1e-13 * S2``) rather than pandas' absolute 1e-14,
    because a subtractively-computed ``M2`` carries residue that scales with the data.
    """
    n = np.asarray(cnt, dtype=float)
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    s3 = np.asarray(s3, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = s1 / n
        m2 = np.clip(s2 - s1 * mu, 0.0, None)  # sum (y'-mu)^2; clip guards fp cancellation
        zero_var = m2 <= 1e-13 * np.abs(s2)
        m2_safe = np.where(zero_var, 1.0, m2)
        if stat == "skew":
            m3 = s3 - 3.0 * mu * s2 + 2.0 * mu * mu * s1
            res = n * np.sqrt(n - 1.0) / (n - 2.0) * m3 / m2_safe**1.5
        elif stat == "kurt":
            s4 = np.asarray(s4, dtype=float)
            mu2 = mu * mu
            m4 = s4 - 4.0 * mu * s3 + 6.0 * mu2 * s2 - 3.0 * mu2 * mu * s1
            num = n * (n + 1.0) * (n - 1.0) * m4
            den = (n - 2.0) * (n - 3.0) * m2_safe * m2_safe
            res = num / den - 3.0 * (n - 1.0) ** 2 / ((n - 2.0) * (n - 3.0))
        else:
            raise ValueError(f"stat={stat!r} is not a shape statistic (skew/kurt).")
    res = np.where(zero_var, 0.0, res)
    res = np.asarray(res, dtype=float)
    res[n < _SHAPE_MIN_N[stat]] = np.nan
    return res


def fit_custom_encoding(keys, y, fn, min_samples: int) -> tuple[pd.Series, float]:
    """Return ``(encoding_by_category, global_fallback)`` for a custom aggregation (CPU only)."""
    per_cat = _cpu.category_agg_custom(keys, y, fn)
    counts = pd.Series(keys).value_counts().reindex(per_cat.index)
    gv = float(fn(np.asarray(y, dtype=float)))
    fallback_mask = per_cat.isna() | (counts < max(int(min_samples), 1))
    return per_cat.where(~fallback_mask, gv).astype(float), gv


def fit_stat_encoding(
    keys, y, stat: str, min_samples: int, backend=None
) -> tuple[pd.Series, float]:
    """Return ``(encoding_by_category, global_fallback)`` for a dispersion/order/shape statistic."""
    if backend is None:
        backend = _cpu
    y = np.asarray(y, dtype=float)
    if stat in _SHAPE_MIN_N or stat in ("var", "std"):
        # dispersion/shape stats from shifted power sums; the shift is exact (shift-invariant
        # statistics) and keeps the moment reconstruction numerically stable -- it also removes
        # any reliance on the backend's own var/skew implementations (cuDF's one-pass var
        # cancels catastrophically at |mean| >> sd, and differently from pandas' two-pass).
        shift = float(np.mean(y)) if len(y) else 0.0
        mom = backend.category_moments(keys, y - shift)
        cnt, s1, s2 = mom["count"], mom["s1"], mom["s2"]
        if stat in ("var", "std"):
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = s1 / cnt
                vals = np.where(cnt > 1, (s2 - s1 * mu) / np.where(cnt > 1, cnt - 1.0, 1.0),
                                np.nan)
            if stat == "std":
                vals = np.sqrt(np.clip(vals, 0.0, None))
        else:
            vals = g1_g2_from_power_sums(cnt, s1, s2, mom["s3"], mom["s4"], stat)
        per_cat = pd.Series(np.asarray(vals, dtype=float), index=mom.index)
        counts = cnt
    else:
        per_cat = backend.category_agg(keys, y, stat)  # median/min/max
        counts = pd.Series(keys).value_counts().reindex(per_cat.index)
    gv = global_stat(y, stat)
    # fall back to the global statistic for undefined or under-supported categories
    fallback_mask = per_cat.isna() | (counts < max(int(min_samples), 1))
    enc = per_cat.where(~fallback_mask, gv).astype(float)
    return enc, gv
