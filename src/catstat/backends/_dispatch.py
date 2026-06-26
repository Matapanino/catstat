"""Backend selection.

``backend='auto'`` chooses GPU only when RAPIDS is importable, the stats are GPU-supported, and
the work is large enough to amortize a host->device copy. ``backend='gpu'`` errors loudly if
RAPIDS/GPU is missing -- never a silent fallback. ``backend_`` on the fitted estimator records the
engine actually used. (No local GPU here, so ``auto`` resolves to CPU; the GPU path is exercised
on Colab.)
"""

from __future__ import annotations

from . import _cpu, _gpu

_GPU_CELL_THRESHOLD = 1_000_000  # n_rows * n_cols above which auto considers GPU


def select_backend(backend: str, n_rows: int, n_cols: int, all_gpu_stats: bool):
    """Return ``(backend_module, name)`` for the requested policy."""
    if backend == "cpu":
        return _cpu, _cpu.NAME
    if backend == "gpu":
        _gpu.ensure_available()  # raises ImportError if RAPIDS/GPU is missing
        return _gpu, _gpu.NAME
    if backend != "auto":
        raise ValueError(f"backend={backend!r} must be one of 'auto', 'cpu', 'gpu'.")

    # auto: prefer GPU only if it is available, applicable, and would pay off.
    if (
        _gpu.AVAILABLE
        and all_gpu_stats
        and (n_rows * max(n_cols, 1)) >= _GPU_CELL_THRESHOLD
    ):
        return _gpu, _gpu.NAME
    return _cpu, _cpu.NAME
