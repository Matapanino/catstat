"""CPU backend primitives (pandas/numpy).

Only this module (and its future GPU twin) knows about the concrete array library. It exposes
the small set of reductions the encoders need; all statistics/leakage logic lives above it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "cpu"


def category_reduce(keys: np.ndarray, y: np.ndarray | None = None) -> pd.DataFrame:
    """Group ``y`` by ``keys`` and return per-category reductions.

    Returns a DataFrame indexed by category key with column ``count``; when ``y`` is given it
    also has ``sum``, ``mean`` and ``sumsq`` (the latter enables a population-variance estimate
    for ``smooth='auto'`` without a second groupby pass).
    """
    if y is None:
        vc = pd.Series(keys).value_counts()
        return pd.DataFrame({"count": vc.astype(float)})

    yv = np.asarray(y, dtype=float)
    df = pd.DataFrame({"k": pd.Series(keys), "y": yv, "y2": yv * yv})
    g = df.groupby("k", sort=False)
    out = pd.DataFrame(
        {
            "count": g["y"].count().astype(float),
            "sum": g["y"].sum(),
            "mean": g["y"].mean(),
            "sumsq": g["y2"].sum(),
        }
    )
    return out


def category_agg(keys: np.ndarray, y: np.ndarray, stat: str) -> pd.Series:
    """Per-category dispersion/order statistic of ``y`` (var/std/median/min/max).

    Returns a Series indexed by category key; ``var``/``std`` are sample (ddof=1) and are NaN for
    singleton categories (the caller falls those back to the global statistic). Shape stats
    (skew/kurt) do not route here -- they are reconstructed from :func:`category_moments`.
    """
    df = pd.DataFrame({"k": pd.Series(keys), "y": np.asarray(y, dtype=float)})
    g = df.groupby("k", sort=False)["y"]
    if stat == "var":
        return g.var(ddof=1)
    if stat == "std":
        return g.std(ddof=1)
    if stat == "median":
        return g.median()
    if stat == "min":
        return g.min()
    if stat == "max":
        return g.max()
    raise ValueError(f"Unknown non-mean stat {stat!r}.")


def category_moments(keys: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Per-category count and raw power sums ``S1..S4`` of ``y`` (caller pre-shifts ``y``).

    Returns a DataFrame indexed by category key with columns ``count, s1, s2, s3, s4`` -- the
    additive inputs from which bias-corrected skew/kurt are reconstructed
    (``_aggregations.g1_g2_from_power_sums``). Only plain sums, so both backends support it.
    """
    yv = np.asarray(y, dtype=float)
    y2 = yv * yv
    df = pd.DataFrame({"k": pd.Series(keys), "y": yv, "y2": y2, "y3": y2 * yv, "y4": y2 * y2})
    g = df.groupby("k", sort=False)
    return pd.DataFrame(
        {
            "count": g["y"].count().astype(float),
            "s1": g["y"].sum(),
            "s2": g["y2"].sum(),
            "s3": g["y3"].sum(),
            "s4": g["y4"].sum(),
        }
    )


def category_agg_custom(keys: np.ndarray, y: np.ndarray, fn) -> pd.Series:
    """Per-category custom aggregation: ``fn(values: ndarray) -> scalar`` (CPU only)."""
    df = pd.DataFrame({"k": pd.Series(keys), "y": np.asarray(y, dtype=float)})
    return df.groupby("k", sort=False)["y"].apply(lambda s: float(fn(s.to_numpy())))


def oof_moment_tables(comp, y, size, order):
    """Raw per-(fold, key) count + power sums via ``np.bincount`` -- the CPU OOF kernel.

    Returns ``(fc, fs, fss, fs3, fs4)`` (the last two ``None`` unless ``order >= 4``), each a
    float64 array of length ``size = n_folds * n_cat``. The GPU twin computes the same sums with
    ``cupy.bincount`` on device and returns them as host arrays, so everything above this seam
    (``_cross_fit.complement_tables`` and the finalizers) is backend-blind.
    """
    y = np.asarray(y, dtype=float)
    fc = np.bincount(comp, minlength=size).astype(float)
    fs = np.bincount(comp, weights=y, minlength=size)
    y2 = y * y
    fss = np.bincount(comp, weights=y2, minlength=size)
    if order < 4:
        return fc, fs, fss, None, None
    fs3 = np.bincount(comp, weights=y2 * y, minlength=size)
    fs4 = np.bincount(comp, weights=y2 * y2, minlength=size)
    return fc, fs, fss, fs3, fs4
