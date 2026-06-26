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
    singleton categories (the caller falls those back to the global statistic).
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
