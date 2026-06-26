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


def fit_mean_encoding(
    keys: np.ndarray, y: np.ndarray, smooth, backend=None
) -> tuple[pd.Series, float]:
    """Return ``(encoding_by_category, global_mean)`` for a mean/probability target statistic.

    ``y`` is the (possibly binarized, for classification) target aligned with ``keys``. The heavy
    group-by runs on ``backend`` (CPU by default); the rest is host arithmetic, so CPU and GPU
    produce the same table (to ``allclose``).
    """
    if backend is None:
        backend = _cpu
    stats = backend.category_reduce(keys, y)
    count = stats["count"]
    mean = stats["mean"]
    global_mean = float(np.mean(np.asarray(y, dtype=float)))

    if isinstance(smooth, str):
        if smooth != "auto":
            raise ValueError(f"smooth={smooth!r}: only 'auto' or a float >= 0 is allowed.")
        var_pop = (stats["sumsq"] / count - mean**2).clip(lower=0.0)
        tau2 = float(np.var(np.asarray(y, dtype=float)))  # population variance
        if tau2 > 0:
            m = var_pop / tau2
        else:  # constant target -> every category mean equals the global mean
            m = pd.Series(0.0, index=count.index)
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

    return enc.astype(float), global_mean
