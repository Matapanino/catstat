"""Numeric-column encoding for ``TargetEncoder`` (opt-in, cardinality-aware).

A numeric column is turned into categorical *keys* before the existing group-by / cross-fit
machinery sees it, so all leakage-safety, smoothing, fallback, and feature-name logic is reused
unchanged. Two strategies:

* ``"direct"`` -- each raw value is its own category (low-cardinality numerics that are effectively
  categorical). This is the identity transform: :func:`catstat._validation.normalize_keys` already
  treats raw values as category keys.
* ``"bin"`` -- the value is discretized into a bin id using **bin edges computed from X only**
  (quantile / equal-frequency by default, or equal-width). Edges are a function of feature values,
  never the target, so computing them once from the full training data is leakage-safe; the
  per-bin target statistic is still cross-fitted out-of-fold by the caller.

Everything here is host-side and deterministic (``np.quantile`` + ``np.unique`` + ``np.digitize``),
so CPU and GPU produce identical bin ids and CPU/GPU parity holds at allclose. This module imports
only numpy/pandas (no backend, no statistics/leakage logic).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def route_strategy(mode: str, nunique: int, n_rows: int, threshold) -> str:
    """Return ``"direct"`` or ``"bin"`` for one numeric column.

    ``mode="auto"`` routes by cardinality: an ``int`` ``threshold`` compares the absolute number of
    distinct values (``nunique <= threshold`` -> direct); a ``float`` threshold compares the
    unique-to-rows ratio (``nunique / n_rows <= threshold`` -> direct). ``mode`` may also force
    ``"direct"`` or ``"bin"``.
    """
    if mode in ("direct", "bin"):
        return mode
    if isinstance(threshold, float):
        ratio = nunique / max(int(n_rows), 1)
        return "direct" if ratio <= threshold else "bin"
    return "direct" if nunique <= int(threshold) else "bin"


def _bin_edges(finite: np.ndarray, n_bins: int, binning: str) -> tuple[np.ndarray, int]:
    """Interior bin boundaries (outer min/max dropped) + the resulting number of bins.

    Duplicate / degenerate edges (ties, near-constant columns) are collapsed with ``np.unique``; a
    column with fewer than two distinct edge values yields a single bin (everything falls back to
    the global statistic, which is correct and never raises). ``np.digitize`` uses the *interior*
    boundaries, so a value below the training min maps to bin 0 and one above the max maps to the
    last bin -- out-of-range values are clipped to the outer bins by construction.
    """
    if finite.size == 0:
        return np.empty(0, dtype=float), 1
    if binning == "quantile":
        raw = np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1))
    else:  # "uniform" / equal-width
        raw = np.linspace(float(finite.min()), float(finite.max()), n_bins + 1)
    uniq = np.unique(raw)  # drop tie / degenerate edges deterministically
    interior = uniq[1:-1]  # np.digitize uses the interior boundaries
    return interior.astype(float), int(interior.size + 1)


def fit_numeric_plan(Xdf, numeric_cols, mode: str, threshold, n_bins: int, binning: str) -> dict:
    """Build the per-column numeric encoding plan from the **full** training frame.

    Returns ``{col: {"strategy", "edges", "n_bins"}}``. Bin edges come from feature values only
    (never ``y``), so this is leakage-safe; the caller cross-fits the per-bin target statistic.
    """
    plan: dict = {}
    n_rows = len(Xdf)
    for c in numeric_cols:
        vals = pd.to_numeric(Xdf[c], errors="coerce").to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        nunique = int(np.unique(finite).size)
        strategy = route_strategy(mode, nunique, n_rows, threshold)
        if strategy == "direct":
            plan[c] = {"strategy": "direct", "edges": None, "n_bins": None}
        else:
            edges, nb = _bin_edges(finite, n_bins, binning)
            plan[c] = {"strategy": "bin", "edges": edges, "n_bins": nb}
    return plan


def apply_numeric_col(values, entry: dict) -> np.ndarray:
    """Map one numeric column's raw values to category keys per its plan ``entry``.

    ``"direct"`` is the identity (raw values become category keys downstream). ``"bin"`` returns an
    object array of integer bin ids with non-finite entries (NaN / +/-inf) left as ``np.nan`` so
    the existing ``normalize_keys`` routes them to the MISSING level (``handle_missing``). Values
    outside the training range map to the outer bins (``np.digitize`` on interior edges already
    clamps to the ends; ``np.clip`` is a defensive guard).
    """
    if entry["strategy"] == "direct":
        return values
    edges = entry["edges"]
    n_bins = entry["n_bins"]
    v = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(v)
    out = np.empty(v.shape[0], dtype=object)
    out[:] = np.nan
    if edges.size == 0:
        out[finite] = 0  # single degenerate bin
    else:
        out[finite] = np.clip(np.digitize(v[finite], edges), 0, n_bins - 1)
    return out
