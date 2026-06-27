"""Numeric-column encoding for ``TargetEncoder`` (opt-in, cardinality-aware).

A numeric column is turned into categorical *keys* before the existing group-by / cross-fit
machinery sees it, so all leakage-safety, smoothing, fallback, and feature-name logic is reused
unchanged. Two strategies:

* ``"direct"`` -- each raw value is its own category (low-cardinality numerics that are effectively
  categorical). This is the identity transform: :func:`catstat._validation.normalize_keys` already
  treats raw values as category keys.
* ``"bin"`` -- the value is discretized into a bin id using **bin edges computed from X only**
  (quantile / equal-frequency by default, equal-width, or **user-supplied explicit edges** that
  depend on nothing in the data at all). Computed edges are a function of feature values, never the
  target, so deriving them once from the full training data is leakage-safe; explicit edges are
  leakage-safe a fortiori. The per-bin target statistic is still cross-fitted out-of-fold by the
  caller. ``binning`` selects the source: ``"quantile"``/``"uniform"`` (a strategy applied to every
  binned column), a 1-D **edge array** (explicit boundaries for every binned column), or a
  ``{column: strategy-or-edges}`` **dict** for per-column control. ``binning`` governs only *how* a
  column is binned; *whether* it is binned stays with ``numeric`` + ``cardinality_threshold``.

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


def _as_edge_array(spec, ctx: str) -> np.ndarray:
    """Coerce ``spec`` to a validated 1-D float edge array (>= 2 finite, strictly increasing).

    Strings/scalars are rejected here so callers can treat "string -> strategy, else -> edges"
    unambiguously; ``ctx`` names the offending param in the error message.
    """
    if isinstance(spec, (str, bytes)) or np.ndim(spec) == 0:
        raise ValueError(f"{ctx}={spec!r} must be a strategy string or a sequence of >= 2 edges.")
    try:
        arr = np.asarray(spec, dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{ctx}={spec!r} must be a numeric edge sequence.") from e
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{ctx} edge array must be 1-D with >= 2 values; got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{ctx} edge array must be all-finite (no NaN/inf).")
    if not np.all(np.diff(arr) > 0):
        raise ValueError(f"{ctx} edge array must be strictly increasing; got {list(arr)}.")
    return arr


def _check_binning_value(spec, ctx: str) -> None:
    """One binning spec is a strategy string (``"quantile"``/``"uniform"``) or an edge array."""
    if isinstance(spec, str):
        if spec not in ("quantile", "uniform"):
            raise ValueError(f"{ctx}={spec!r} must be 'quantile', 'uniform', or an edge array.")
        return
    _as_edge_array(spec, ctx)


def validate_binning(binning) -> None:
    """Validate the ``binning`` param's structure.

    Accepts a strategy string, a 1-D explicit edge array (applied to every binned column), or a
    ``{column: strategy-or-edges}`` dict for per-column control. Dict *keys* are checked against the
    actual numeric columns later, in :func:`fit_numeric_plan`.
    """
    if isinstance(binning, dict):
        for col, spec in binning.items():
            _check_binning_value(spec, ctx=f"binning[{col!r}]")
        return
    _check_binning_value(binning, ctx="binning")


def _col_binning_spec(c, binning):
    """The binning spec for column ``c``: a strategy string or an explicit edge array.

    A plain string/array applies to every binned column; a dict is consulted per column (a column
    absent from the dict falls back to the default ``"quantile"`` strategy).
    """
    if isinstance(binning, dict):
        return binning.get(c, "quantile")
    return binning


def _resolve_bin_edges(spec, finite: np.ndarray, n_bins: int) -> tuple[np.ndarray, int]:
    """Interior edges + bin count for one binned column from its resolved ``spec``.

    A string ``spec`` computes edges from the column's finite values (the ``n_bins`` strategy); an
    array ``spec`` is explicit boundaries (``np.unique``-sorted, the outer two dropped so
    ``np.digitize`` clamps out-of-range to the end bins). Explicit edges set the bin count, so
    ``n_bins`` is ignored for that column.
    """
    if isinstance(spec, str):
        return _bin_edges(finite, n_bins, spec)
    uniq = np.unique(_as_edge_array(spec, "binning"))
    interior = uniq[1:-1]
    return interior.astype(float), int(interior.size + 1)


def fit_numeric_plan(Xdf, numeric_cols, mode: str, threshold, n_bins: int, binning) -> dict:
    """Build the per-column numeric encoding plan from the **full** training frame.

    Returns ``{col: {"strategy", "edges", "n_bins"}}``. Bin edges come from feature values only
    (computed strategies) or from the user (explicit edges) -- never ``y`` -- so this is
    leakage-safe; the caller cross-fits the per-bin target statistic. ``binning`` may be a strategy
    string, an edge array, or a ``{column: strategy-or-edges}`` dict (dict keys must name numeric
    columns being encoded).
    """
    plan: dict = {}
    n_rows = len(Xdf)
    if isinstance(binning, dict):
        unknown = [c for c in binning if c not in numeric_cols]
        if unknown:
            raise ValueError(
                f"binning keys {unknown} are not numeric columns being encoded "
                f"(numeric columns: {list(numeric_cols)})."
            )
    for c in numeric_cols:
        vals = pd.to_numeric(Xdf[c], errors="coerce").to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        nunique = int(np.unique(finite).size)
        strategy = route_strategy(mode, nunique, n_rows, threshold)
        if strategy == "direct":
            plan[c] = {"strategy": "direct", "edges": None, "n_bins": None}
        else:
            edges, nb = _resolve_bin_edges(_col_binning_spec(c, binning), finite, n_bins)
            plan[c] = {"strategy": "bin", "edges": edges, "n_bins": nb}
    return plan


def apply_numeric_col(values, entry: dict) -> np.ndarray:
    """Map one numeric column's raw values to category keys per its plan ``entry``.

    Keys are emitted as **strings**, with non-finite entries (NaN / +/-inf) left as ``np.nan`` so
    the existing ``normalize_keys`` routes them to the MISSING level (``handle_missing``). Strings
    are required for CPU/GPU parity: cuDF rejects object-dtype *integer* arrays, whereas the
    string-keyed path is the one already validated CPU/GPU-allclose. ``"direct"`` stringifies the
    raw value (each value a category); ``"bin"`` stringifies the bin id. Values outside the training
    range map to the outer bins (``np.digitize`` on interior edges already clamps to the ends;
    ``np.clip`` is a defensive guard).
    """
    s = pd.Series(values)
    out = np.empty(len(s), dtype=object)
    out[:] = np.nan
    if entry["strategy"] == "direct":
        notna = s.notna().to_numpy()
        out[notna] = s[notna].astype(str).to_numpy()  # "3" for ints, "1.5" for floats
        return out
    edges = entry["edges"]
    n_bins = entry["n_bins"]
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(v)
    if edges.size == 0:
        out[finite] = "0"  # single degenerate bin
    else:
        out[finite] = np.clip(np.digitize(v[finite], edges), 0, n_bins - 1).astype(str)
    return out
