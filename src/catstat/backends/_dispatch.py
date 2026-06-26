"""Backend selection.

``backend='auto'`` chooses GPU only when RAPIDS is importable, the stats are GPU-supported, and
the work is large enough to amortize a host->device copy. ``backend='gpu'`` errors loudly if
RAPIDS/GPU is missing -- never a silent fallback. ``backend_`` on the fitted estimator records the
engine actually used. (No local GPU here, so ``auto`` resolves to CPU; the GPU path is exercised
on Colab.)
"""

from __future__ import annotations

from . import _cpu, _gpu

# Colab T4 crossover (2026-06-26, docs/verdicts/2026-06-26-gpu-crossover-verdict.md): the current
# host-orchestrated GPU path is SLOWER than CPU up to 1M rows (cpu/gpu speedup 0.28-0.86), so
# `auto` must NOT pick it -- picking a slower backend by default would be a regression. Explicit
# backend="gpu" stays available (validated CPU/GPU-allclose, incl. missing) for device-resident
# pipelines / much larger data. Re-enable + calibrate the threshold once the device path keeps
# keys/folds on-device (avoids the per-fold host<->device round-trips that currently dominate).
_AUTO_GPU_ENABLED = False
_GPU_CELL_THRESHOLD = 5_000_000  # retained for when _AUTO_GPU_ENABLED flips True


def select_backend(backend: str, n_rows: int, n_cols: int, all_gpu_stats: bool):
    """Return ``(backend_module, name)`` for the requested policy."""
    if backend == "cpu":
        return _cpu, _cpu.NAME
    if backend == "gpu":
        _gpu.ensure_available()  # raises ImportError if RAPIDS/GPU is missing
        return _gpu, _gpu.NAME
    if backend != "auto":
        raise ValueError(f"backend={backend!r} must be one of 'auto', 'cpu', 'gpu'.")

    # auto: only pick GPU if enabled (currently off, see above), available, applicable, and large.
    if (
        _AUTO_GPU_ENABLED
        and _gpu.AVAILABLE
        and all_gpu_stats
        and (n_rows * max(n_cols, 1)) >= _GPU_CELL_THRESHOLD
    ):
        return _gpu, _gpu.NAME
    return _cpu, _cpu.NAME
