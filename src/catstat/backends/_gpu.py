"""GPU backend (cuDF / CuPy).

Mirrors ``_cpu``'s primitives. The heavy per-category group-by runs on the GPU (the part that
scales); the small per-category result is returned as a **pandas** object so the downstream
smoothing / fallback / mapping logic is byte-for-byte identical to the CPU path (CPU and GPU
therefore agree to ``allclose`` -- float reduction order differs). This is the cuML pattern:
device group-by, host orchestration.

Missing values: ``_validation.normalize_keys`` replaces NaN with the host ``MISSING`` sentinel
(a Python object cuDF can't hold). For single-column (object) keys we map that sentinel to a cuDF
**null**, group with ``dropna=False`` so the missing level is its own category, then map the null
result-index entry back to ``MISSING`` so the host ``.map`` lines up. **Combination/interaction
units** arrive as **int64 joint codes** (host-built; a missing component is already folded into an
ordinary integer code, so there is no sentinel) and group directly on the device.

Imports cleanly on CPU-only boxes (``AVAILABLE`` is then ``False`` and nothing here runs).
Validated on Colab (``scripts/colab_gpu_parity.sh``); there is no local GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "gpu"


def _detect() -> bool:
    try:
        import cudf  # noqa: F401
        import cupy  # noqa: F401

        return True
    except Exception:
        return False


AVAILABLE = _detect()


def ensure_available() -> None:
    """Raise a clear error if RAPIDS/GPU is not importable (never a silent fallback)."""
    if not AVAILABLE:
        raise ImportError(
            "backend='gpu' requires RAPIDS (cudf + cupy), which is not importable here. "
            "Install the 'gpu' extra on an NVIDIA/CUDA-12 box, or use backend='cpu'/'auto'."
        )


def _to_nullable(keys: np.ndarray):
    """Return ``(keys_or_copy, had_missing)`` with the host MISSING sentinel -> None (cuDF null).

    Only object key arrays (single-column / numeric string keys) can carry the sentinel; int64
    joint-code keys (combination/interaction units) never do, so they pass straight through.
    """
    keys = np.asarray(keys)
    if keys.dtype != object:
        return keys, False
    from .._validation import MISSING

    mask = np.asarray(keys == MISSING, dtype=bool)
    if not mask.any():
        return keys, False
    arr = np.asarray(keys, dtype=object).copy()
    arr[mask] = None
    return arr, True


def _remap_missing_index(result, had_missing):
    """Map the cuDF null group (NaN after to_pandas) in the result index back to MISSING."""
    if not had_missing:
        return result
    from .._validation import MISSING

    result.index = pd.Index(
        [MISSING if pd.isna(i) else i for i in result.index], dtype=object
    )
    return result


def category_reduce(keys: np.ndarray, y: np.ndarray | None = None) -> pd.DataFrame:
    """GPU group-by; returns a pandas DataFrame (small per-category result copied to host)."""
    import cudf
    import cupy as cp

    key_arr, had_missing = _to_nullable(keys)

    if y is None:
        vc = cudf.Series(key_arr).value_counts(dropna=False)
        res = cudf.DataFrame({"count": vc.astype("float64")}).to_pandas()
        return _remap_missing_index(res, had_missing)

    gdf = cudf.DataFrame({"k": cudf.Series(key_arr), "y": cp.asarray(y, dtype="float64")})
    gdf["y2"] = gdf["y"] * gdf["y"]
    g = gdf.groupby("k", sort=False, dropna=False)
    out = cudf.DataFrame(
        {
            "count": g["y"].count().astype("float64"),
            "sum": g["y"].sum(),
            "mean": g["y"].mean(),
            "sumsq": g["y2"].sum(),
        }
    ).to_pandas()
    return _remap_missing_index(out, had_missing)


def oof_moment_tables(comp, y, size, order):
    """GPU twin of ``_cpu.oof_moment_tables``: per-(fold, key) sums via ``cupy.bincount``.

    ``comp``/``y`` may be host numpy (one H2D copy each -- the whole per-unit transfer on the
    host-origin path) or already-resident cupy arrays (``cp.asarray`` is then a no-op). The
    returned tables are small (``n_folds * n_cat``) host float64 arrays, so the finalizers above
    the seam stay backend-blind. fp64 atomics in ``bincount`` reorder additions vs numpy -->
    parity with CPU holds at allclose, not bitwise (CLAUDE.md invariant #2).
    """
    import cupy as cp

    comp_d = cp.asarray(comp)
    y_d = cp.asarray(y, dtype="float64")
    fc = cp.bincount(comp_d, minlength=size).astype(cp.float64)
    fs = cp.bincount(comp_d, weights=y_d, minlength=size)
    y2 = y_d * y_d
    fss = cp.bincount(comp_d, weights=y2, minlength=size)
    if order < 4:
        return cp.asnumpy(fc), cp.asnumpy(fs), cp.asnumpy(fss), None, None
    fs3 = cp.bincount(comp_d, weights=y2 * y_d, minlength=size)
    fs4 = cp.bincount(comp_d, weights=y2 * y2, minlength=size)
    return cp.asnumpy(fc), cp.asnumpy(fs), cp.asnumpy(fss), cp.asnumpy(fs3), cp.asnumpy(fs4)


def category_moments(keys: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """GPU per-category count + raw power sums ``S1..S4`` (caller pre-shifts ``y``).

    Twin of ``_cpu.category_moments``: only plain group-by sums, which cuDF supports, so the
    shape stats (skew/kurt) reconstructed from these sums are GPU-supported. Small per-category
    result returned as pandas (host).
    """
    import cudf
    import cupy as cp

    key_arr, had_missing = _to_nullable(keys)
    yd = cp.asarray(y, dtype="float64")
    y2 = yd * yd
    gdf = cudf.DataFrame(
        {"k": cudf.Series(key_arr), "y": yd, "y2": y2, "y3": y2 * yd, "y4": y2 * y2}
    )
    g = gdf.groupby("k", sort=False, dropna=False)
    out = cudf.DataFrame(
        {
            "count": g["y"].count().astype("float64"),
            "s1": g["y"].sum(),
            "s2": g["y2"].sum(),
            "s3": g["y3"].sum(),
            "s4": g["y4"].sum(),
        }
    ).to_pandas()
    return _remap_missing_index(out, had_missing)


def category_agg(keys: np.ndarray, y: np.ndarray, stat: str) -> pd.Series:
    """GPU dispersion/order group-by; returns a pandas Series (host)."""
    import cudf
    import cupy as cp

    key_arr, had_missing = _to_nullable(keys)
    gdf = cudf.DataFrame({"k": cudf.Series(key_arr), "y": cp.asarray(y, dtype="float64")})
    g = gdf.groupby("k", sort=False, dropna=False)["y"]
    if stat == "var":
        res = g.var(ddof=1)
    elif stat == "std":
        res = g.std(ddof=1)
    elif stat == "median":
        res = g.median()
    elif stat == "min":
        res = g.min()
    elif stat == "max":
        res = g.max()
    else:
        raise ValueError(f"Unknown non-mean stat {stat!r}.")
    out = res.to_pandas()
    return _remap_missing_index(out.to_frame("v"), had_missing)["v"]
