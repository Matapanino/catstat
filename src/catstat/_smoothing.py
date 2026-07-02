"""Smoothing for mean/probability statistics.

Only mean/probability admit principled smoothing (see docs: the "smoothing honesty rule").
This module implements the fixed m-estimate, the ``smooth='auto'`` empirical-Bayes estimate, and
the ``smooth='sigmoid'`` blend (category_encoders parity).

For ``smooth='auto'`` we use the empirical-Bayes form ``m_i = sigma_i^2 / tau^2`` with population
(ddof=0) variances, blending ``lambda_i = n_i / (n_i + m_i)`` toward the global mean. This is
**exactly scikit-learn's** ``TargetEncoder(smooth="auto")`` formula (their
``lambda = n*tau^2 / (n*tau^2 + SS/n)`` is the same expression): verified to fp rounding across
target types and edge cases by ``tests/test_sklearn_auto_parity.py`` (KI-010, resolved).

``smooth='sigmoid'`` (or ``('sigmoid', k, f)``) reproduces category_encoders' ``TargetEncoder``:
``w = 1/(1 + exp(-(n - k)/f))``, ``enc = w*mean + (1-w)*prior``, with a singleton category
(``n == 1``) forced to the prior -- exactly their formula, including that override. The bare
string uses their defaults ``k=20`` (min_samples_leaf), ``f=10.0`` (smoothing).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backends import _cpu

_SIGMOID_DEFAULTS = (20.0, 10.0)  # category_encoders: min_samples_leaf=20, smoothing=10


def sigmoid_params(smooth):
    """``(k, f)`` when ``smooth`` selects the sigmoid blend, else ``None``.

    Accepts ``'sigmoid'`` (category_encoders defaults) or ``('sigmoid', k, f)`` with ``f > 0``.
    Raises for a malformed sigmoid spec; returns ``None`` for every other ``smooth`` value.
    """
    if smooth == "sigmoid":
        return _SIGMOID_DEFAULTS
    if isinstance(smooth, tuple):
        if len(smooth) != 3 or smooth[0] != "sigmoid":
            raise ValueError(
                f"smooth={smooth!r}: tuple form must be ('sigmoid', k, f), e.g. "
                "('sigmoid', 20, 10.0)."
            )
        k, f = float(smooth[1]), float(smooth[2])
        if not np.isfinite(k) or not np.isfinite(f) or f <= 0:
            raise ValueError(f"smooth={smooth!r}: need finite k and f > 0.")
        return k, f
    return None


def mean_from_stats(count, mean, sumsq, smooth, global_mean: float, tau2: float) -> pd.Series:
    """Smoothed mean encoding from per-category ``(count, mean, sumsq)`` plus global scalars.

    The single source of the m-estimate / empirical-Bayes / sigmoid arithmetic, shared by the
    host path (:func:`fit_mean_encoding`, category-indexed Series) and the device path
    (code-indexed Series built from on-device reductions). ``tau2`` is the population variance
    of ``y`` (only consulted for ``smooth='auto'``).
    """
    sig = sigmoid_params(smooth)
    if sig is not None:
        k, f = sig
        w = 1.0 / (1.0 + np.exp(-(count - k) / f))
        enc = w * mean + (1.0 - w) * global_mean
        # category_encoders parity: a singleton category takes the prior outright
        enc = enc.where(count > 1, global_mean)
        return enc.astype(float)
    if isinstance(smooth, str):
        if smooth != "auto":
            raise ValueError(
                f"smooth={smooth!r}: 'auto', 'sigmoid', ('sigmoid', k, f), or a float >= 0."
            )
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
    keys: np.ndarray, y: np.ndarray, smooth, backend=None, shift: bool = True
) -> tuple[pd.Series, float]:
    """Return ``(encoding_by_category, global_mean)`` for a mean/probability target statistic.

    ``y`` is the (possibly binarized, for classification) target aligned with ``keys``. The heavy
    group-by runs on ``backend`` (CPU by default); the rest is host arithmetic
    (:func:`mean_from_stats`), so CPU and GPU produce the same table (to ``allclose``).

    ``shift=True`` (continuous targets) reduces about the global mean: the smoothed mean is
    affine-equivariant and var_pop is shift-invariant, so the result is identical -- but the
    shifted sums keep the EB weights stable when ``|mean| >> sd`` (unshifted
    ``sumsq/count - mean^2`` cancels catastrophically, and CPU/GPU cancel *differently*,
    breaking parity at large offsets). Binarized (0/1) targets pass ``shift=False``: they have
    no offset problem, and shifting would smear fp residue into the exactly-zero variance of a
    *pure* category, breaking WOE's documented ``+-inf`` contract.
    """
    if backend is None:
        backend = _cpu
    yv = np.asarray(y, dtype=float)
    global_mean = float(np.mean(yv))
    tau2 = float(np.var(yv)) if smooth == "auto" else 0.0  # population variance (EB only)
    if shift:
        stats = backend.category_reduce(keys, yv - global_mean)
        enc = mean_from_stats(stats["count"], stats["mean"], stats["sumsq"], smooth, 0.0, tau2)
        return enc + global_mean, global_mean
    stats = backend.category_reduce(keys, yv)
    enc = mean_from_stats(
        stats["count"], stats["mean"], stats["sumsq"], smooth, global_mean, tau2
    )
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
