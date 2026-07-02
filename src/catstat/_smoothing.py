"""Smoothing for mean/probability statistics.

Only mean/probability admit principled smoothing (see docs: the "smoothing honesty rule").
This module implements the fixed m-estimate and the ``smooth='auto'`` empirical-Bayes estimate.

For ``smooth='auto'`` we use the documented empirical-Bayes form ``m_i = sigma_i^2 / tau^2`` with
population (ddof=0) variances, blending ``lambda_i = n_i / (n_i + m_i)`` toward the global mean.
The exact parity with scikit-learn's auto formula is a known follow-up (docs/known_issues KI-010);
the leakage/determinism guarantees do not depend on the smoothing constant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backends import _cpu


def mean_from_stats(count, mean, sumsq, smooth, global_mean: float, tau2: float) -> pd.Series:
    """Smoothed mean encoding from per-category ``(count, mean, sumsq)`` plus global scalars.

    The single source of the m-estimate / empirical-Bayes arithmetic, shared by the host path
    (:func:`fit_mean_encoding`, category-indexed Series) and the device path (code-indexed
    Series built from on-device reductions). ``tau2`` is the population variance of ``y`` (only
    consulted for ``smooth='auto'``).
    """
    if isinstance(smooth, str):
        if smooth != "auto":
            raise ValueError(f"smooth={smooth!r}: only 'auto' or a float >= 0 is allowed.")
        var_pop = (sumsq / count - mean**2).clip(lower=0.0)
        if tau2 > 0:
            m = var_pop / tau2
        else:  # constant target -> every category mean equals the global mean
            m = count * 0.0
        lam = count / (count + m)
        enc = lam * mean + (1.0 - lam) * global_mean
    else:
        m = float(smooth)
        if m < 0:
            raise ValueError("smooth must be >= 0.")
        if m == 0.0:
            enc = mean.copy()
        else:
            enc = (count * mean + m * global_mean) / (count + m)
    return enc.astype(float)


def fit_mean_encoding(
    keys: np.ndarray, y: np.ndarray, smooth, backend=None
) -> tuple[pd.Series, float]:
    """Return ``(encoding_by_category, global_mean)`` for a mean/probability target statistic.

    ``y`` is the (possibly binarized, for classification) target aligned with ``keys``. The heavy
    group-by runs on ``backend`` (CPU by default); the rest is host arithmetic
    (:func:`mean_from_stats`), so CPU and GPU produce the same table (to ``allclose``).
    """
    if backend is None:
        backend = _cpu
    stats = backend.category_reduce(keys, y)
    yv = np.asarray(y, dtype=float)
    global_mean = float(np.mean(yv))
    tau2 = float(np.var(yv)) if isinstance(smooth, str) else 0.0  # population variance
    enc = mean_from_stats(stats["count"], stats["mean"], stats["sumsq"], smooth, global_mean, tau2)
    return enc, global_mean


def woe_from_prob(p, prior):
    """Weight of evidence from the (smoothed) ``P(y=1 | category)`` and the prior.

    ``woe_c = logit(p_c) - logit(prior)`` -- by Bayes this equals the classic credit-scoring
    ``ln(P(c | y=1) / P(c | y=0))``; positive WOE means the category over-indexes on the positive
    class. Deriving it from the already-smoothed probability keeps it inside the honesty rule
    (probability-family smoothing, nothing new invented) -- and, deliberately, nothing extra is
    clipped: a **pure** category (``p in {0, 1}``) yields ``+-inf`` under ``smooth=0`` *and*
    under ``smooth='auto'`` (the EB weight ``m_i = var_i/tau^2`` is 0 at zero within-category
    variance, so pure categories are not shrunk). A fixed m-estimate ``smooth=m > 0`` keeps ``p``
    strictly interior and WOE finite. ``prior`` may be a scalar (full-data fit) or a per-row
    array (per-fold OOF priors); a category encoded at its prior (e.g. the unknown fallback) gets
    exactly 0.0.
    """
    p = np.asarray(p, dtype=float)
    prior = np.asarray(prior, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.log(p) - np.log1p(-p)) - (np.log(prior) - np.log1p(-prior))
