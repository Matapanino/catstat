"""GPU backend (cuDF / CuPy).

Mirrors ``_cpu``'s primitives. The heavy per-category group-by runs on the GPU (the part that
scales); the small per-category result is returned as a **pandas** object so the downstream
smoothing / fallback / mapping logic is byte-for-byte identical to the CPU path (CPU and GPU
therefore agree to ``allclose`` -- float reduction order differs). This is the cuML pattern:
device group-by, host orchestration.

This module imports cleanly on CPU-only boxes (``AVAILABLE`` is then ``False`` and nothing here
runs). It is **validated on Colab** via ``scripts/colab_gpu_parity.sh`` -- there is no local GPU,
so CI/macOS never exercise it. Scope for now: single-column numeric/string keys without an
injected missing sentinel; combination mode and missing-as-value fall back to CPU (see
docs/known_issues.md).
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


def category_reduce(keys: np.ndarray, y: np.ndarray | None = None) -> pd.DataFrame:
    """GPU group-by; returns a pandas DataFrame (small per-category result copied to host)."""
    import cudf
    import cupy as cp

    if y is None:
        vc = cudf.Series(keys).value_counts()
        return cudf.DataFrame({"count": vc.astype("float64")}).to_pandas()

    gdf = cudf.DataFrame({"k": cudf.Series(keys), "y": cp.asarray(y, dtype="float64")})
    gdf["y2"] = gdf["y"] * gdf["y"]
    g = gdf.groupby("k", sort=False)
    out = cudf.DataFrame(
        {
            "count": g["y"].count().astype("float64"),
            "sum": g["y"].sum(),
            "mean": g["y"].mean(),
            "sumsq": g["y2"].sum(),
        }
    )
    return out.to_pandas()


def category_agg(keys: np.ndarray, y: np.ndarray, stat: str) -> pd.Series:
    """GPU dispersion/order group-by; returns a pandas Series (host)."""
    import cudf
    import cupy as cp

    gdf = cudf.DataFrame({"k": cudf.Series(keys), "y": cp.asarray(y, dtype="float64")})
    g = gdf.groupby("k", sort=False)["y"]
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
    return res.to_pandas()
