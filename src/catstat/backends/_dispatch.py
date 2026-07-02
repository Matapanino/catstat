"""Backend selection.

``backend='auto'`` chooses GPU only when RAPIDS is importable, the stats are GPU-supported, and
the work is large enough to amortize a host->device copy. ``backend='gpu'`` errors loudly if
RAPIDS/GPU is missing -- never a silent fallback. ``backend_`` on the fitted estimator records the
engine actually used. (No local GPU here, so ``auto`` resolves to CPU; the GPU path is exercised
on Colab.)

**Device-resident input** (a cuDF DataFrame) is a categorical signal, not a size heuristic: it
routes to the GPU backend regardless of ``_AUTO_GPU_ENABLED`` (converting to host behind the
user's back would be the forbidden silent device->host transfer), and combining it with
``backend='cpu'`` is an explicit error -- the user must convert with ``.to_pandas()`` themselves.
"""

from __future__ import annotations

import sys

from . import _cpu, _gpu


def is_device_frame(X) -> bool:
    """True iff ``X`` is a cuDF DataFrame -- detected without importing cudf.

    A user can only hand us a cuDF object if cudf is already imported, so ``sys.modules`` is
    consulted instead of an import (keeping the "no cudf/cupy imports outside backends/_gpu.py"
    rule intact on CPU-only boxes, where this is just a dict lookup returning False).
    """
    cudf = sys.modules.get("cudf")
    return cudf is not None and isinstance(X, cudf.DataFrame)

# Colab T4 crossover (2026-06-26, docs/verdicts/2026-06-26-gpu-crossover-verdict.md): the current
# host-orchestrated GPU path is SLOWER than CPU up to 1M rows (cpu/gpu speedup 0.28-0.86), so
# `auto` must NOT pick it -- picking a slower backend by default would be a regression. Explicit
# backend="gpu" stays available (validated CPU/GPU-allclose, incl. missing) for device-resident
# pipelines / much larger data. Re-enable + calibrate the threshold once the device path keeps
# keys/folds on-device (avoids the per-fold host<->device round-trips that currently dominate).
_AUTO_GPU_ENABLED = False
_GPU_CELL_THRESHOLD = 5_000_000  # retained for when _AUTO_GPU_ENABLED flips True


def select_backend(backend: str, n_rows: int, n_cols: int, all_gpu_stats: bool,
                   device_input: bool = False):
    """Return ``(backend_module, name)`` for the requested policy."""
    if backend not in ("auto", "cpu", "gpu"):
        raise ValueError(f"backend={backend!r} must be one of 'auto', 'cpu', 'gpu'.")
    if device_input:
        if backend == "cpu":
            raise ValueError(
                "backend='cpu' with a cuDF input would require a silent device->host transfer. "
                "Convert explicitly with X.to_pandas(), or use backend='auto'/'gpu'."
            )
        _gpu.ensure_available()  # cudf importable by construction, but keep the loud contract
        return _gpu, _gpu.NAME
    if backend == "cpu":
        return _cpu, _cpu.NAME
    if backend == "gpu":
        _gpu.ensure_available()  # raises ImportError if RAPIDS/GPU is missing
        return _gpu, _gpu.NAME

    # auto: only pick GPU if enabled (currently off, see above), available, applicable, and large.
    if (
        _AUTO_GPU_ENABLED
        and _gpu.AVAILABLE
        and all_gpu_stats
        and (n_rows * max(n_cols, 1)) >= _GPU_CELL_THRESHOLD
    ):
        return _gpu, _gpu.NAME
    return _cpu, _cpu.NAME


def backend_module(name: str):
    """Resolve a backend NAME (as recorded in ``backend_``) back to its module.

    Used to restore the (unpicklable) module cached on a fitted estimator after unpickling.
    """
    if name == _cpu.NAME:
        return _cpu
    if name == _gpu.NAME:
        return _gpu
    raise ValueError(f"Unknown backend name {name!r}.")
