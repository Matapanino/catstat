#!/usr/bin/env python3
"""On-VM entrypoint for the catstat CPU/GPU parity + crossover run (Colab GPU).

Invoked by ``scripts/colab_gpu_parity.sh`` via ``colab exec``. Extracts the uploaded working tree,
installs RAPIDS, then:
  1. PARITY -- for several cases (incl. a missing-as-value case) runs the same data + random_state
     through ``backend="cpu"`` and ``backend="gpu"`` and checks ``transform`` and ``fit_transform``
     agree to ``allclose`` (not bitwise). Records CPU vs GPU fit_transform time.
  2. CROSSOVER -- times CPU vs GPU fit_transform of a mean encoder at n = 10k / 100k / 1M to find
     where the GPU starts to pay off (calibrates the ``backend="auto"`` cell threshold).

Writes ``/content/parity.jsonl`` and ``/content/parity_report.md``. No local GPU -> Colab only.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/content/catstat_repo")


def _sh(cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=False)


def setup():
    WORK.mkdir(parents=True, exist_ok=True)
    _sh(["tar", "xzf", "/content/catstat.tar.gz", "-C", str(WORK)])
    # prefer the image's preinstalled RAPIDS (driver-compatible); pip only when absent
    probe = subprocess.run(
        [sys.executable, "-c", "import cudf, cupy; cupy.zeros(1).sum()"], capture_output=True
    )
    if probe.returncode != 0:
        _sh([sys.executable, "-m", "pip", "install", "-q", "cudf-cu12", "cupy-cuda12x"])
    sys.path.insert(0, str(WORK / "src"))
    try:  # RMM pool: stabilizes the 5M/10M timings (no per-alloc cudaMalloc churn)
        import cupy
        import rmm
        from rmm.allocators.cupy import rmm_cupy_allocator

        rmm.reinitialize(pool_allocator=True, initial_pool_size=6 * 1024**3)
        cupy.cuda.set_allocator(rmm_cupy_allocator)
        print("rmm pool: on", flush=True)
    except Exception as exc:  # pragma: no cover - environment-dependent
        print("rmm pool init skipped:", exc, flush=True)


def _med(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return round(statistics.median(ts), 4)


def parity_cases():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n, k = 200_000, 5_000
    g = rng.integers(0, k, size=n).astype(str)
    X = pd.DataFrame({"g": g})
    y_reg = rng.normal(size=n)
    y_bin = (rng.uniform(size=n) < 0.3).astype(int)
    y_mc = rng.integers(0, 4, size=n)

    gm = g.astype(object).copy()
    gm[rng.uniform(size=n) < 0.1] = np.nan  # 10% missing, handled as a category
    Xm = pd.DataFrame({"g": gm})

    base = dict(cols=["g"], cv=5, random_state=0)
    yield "regression_mean", X, y_reg, {**base, "stats": ["mean"]}
    yield "regression_var", X, y_reg, {**base, "stats": ["var"]}
    yield "binary_mean", X, y_bin, {**base, "stats": ["mean"]}
    yield "multiclass_mean", X, y_mc, {**base, "stats": ["mean"]}
    miss_kw = {**base, "stats": ["mean"], "handle_missing": "value"}
    yield "regression_mean_missing", Xm, y_reg, miss_kw

    # numeric-column TE (0.2.0): bin edges are host-side numpy, so bin ids match on both backends.
    lc = rng.integers(0, 8, size=n)  # low cardinality -> direct under "auto"
    hc = rng.normal(size=n)  # high cardinality continuous -> bin
    Xn = pd.DataFrame({"lc": lc, "hc": hc})
    num_base = dict(cols=["lc", "hc"], stats=["mean"], cv=5, random_state=0, n_bins=20)
    yield "numeric_auto", Xn, y_reg, {**num_base, "numeric": "auto"}  # direct(lc)+bin(hc)
    yield "numeric_bin", Xn, y_reg, {**num_base, "numeric": "bin"}

    # combination / interactions: int64 mixed-radix joint codes (host-built, so identical on both
    # backends -> only the device group-by differs). KI-018 unblock; includes a missing-component
    # case (the missing-combo is folded into an ordinary int code, no MISSING sentinel on device).
    a2 = rng.integers(0, 200, size=n).astype(str)
    b2 = rng.integers(0, 200, size=n).astype(str)
    Xc = pd.DataFrame({"a": a2, "b": b2})
    comb = dict(cols=["a", "b"], multi_feature_mode="combination", cv=5, random_state=0)
    yield "combination_mean", Xc, y_reg, {**comb, "stats": ["mean"]}
    yield "combination_var", Xc, y_reg, {**comb, "stats": ["var"]}
    am = a2.astype(object).copy()
    am[rng.uniform(size=n) < 0.1] = np.nan
    Xcm = pd.DataFrame({"a": am, "b": b2})
    yield "combination_mean_missing", Xcm, y_reg, {
        **comb,
        "stats": ["mean"],
        "handle_missing": "value",
    }
    yield "interactions_mean", Xc, y_reg, dict(
        cols=["a", "b"], interactions=[["a", "b"]], stats=["mean"], cv=5, random_state=0
    )

    # stats arc (2026-07): moments-based shape stats (GPU-supported), incl. the 1e9-offset
    # stability case, and WOE (binary, derived from the smoothed probability).
    yield "regression_skew", X, y_reg, {**base, "stats": ["skew"]}
    yield "regression_kurt", X, y_reg, {**base, "stats": ["kurt"]}
    y_off = 1e9 + y_reg
    yield "shape_offset_1e9", X, y_off, {**base, "stats": ["mean", "var", "skew", "kurt"]}
    yield "binary_woe_auto", X, y_bin, {**base, "stats": ["mean", "woe"]}
    yield "binary_woe_m20", X, y_bin, {**base, "stats": ["woe"], "smooth": 20.0}
    yield "regression_median", X, y_reg, {**base, "stats": ["median"]}


def run_parity():
    import numpy as np

    from catstat import TargetEncoder

    rows = []
    for name, X, y, kw in parity_cases():
        rec = {"kind": "parity", "case": name}
        try:
            cpu = TargetEncoder(**kw, backend="cpu", output="numpy")
            gpu = TargetEncoder(**kw, backend="gpu", output="numpy")
            a_t = np.asarray(cpu.fit(X, y).transform(X))
            b_t = np.asarray(gpu.fit(X, y).transform(X))
            rec["backend_cpu"], rec["backend_gpu"] = cpu.backend_, gpu.backend_
            rec["transform_max_abs_diff"] = float(np.max(np.abs(a_t - b_t)))
            rec["transform_allclose"] = bool(np.allclose(a_t, b_t, rtol=1e-5, atol=1e-8))

            a_ft = np.asarray(
                TargetEncoder(**kw, backend="cpu", output="numpy").fit_transform(X, y)
            )
            b_ft = np.asarray(
                TargetEncoder(**kw, backend="gpu", output="numpy").fit_transform(X, y)
            )
            rec["fit_transform_max_abs_diff"] = float(np.nanmax(np.abs(a_ft - b_ft)))
            rec["fit_transform_allclose"] = bool(
                np.allclose(a_ft, b_ft, rtol=1e-5, atol=1e-8, equal_nan=True)
            )
            TargetEncoder(**kw, backend="gpu", output="numpy").fit_transform(X, y)  # gpu warmup
            rec["cpu_ft_s"] = _med(
                lambda: TargetEncoder(**kw, backend="cpu", output="numpy").fit_transform(X, y), 2
            )
            rec["gpu_ft_s"] = _med(
                lambda: TargetEncoder(**kw, backend="gpu", output="numpy").fit_transform(X, y), 2
            )
            ok = rec["transform_allclose"] and rec["fit_transform_allclose"]
            rec["status"] = "ok" if ok else "MISMATCH"
        except Exception as exc:
            rec["status"], rec["error"] = "ERROR", repr(exc)
        print(rec, flush=True)
        rows.append(rec)
    return rows


def run_crossover():
    import numpy as np
    import pandas as pd

    from catstat import TargetEncoder

    rng = np.random.default_rng(1)
    rows = []
    for n in (10_000, 100_000, 1_000_000, 5_000_000, 10_000_000):
        k = max(2, n // 40)
        X = pd.DataFrame({"g": rng.integers(0, k, size=n).astype(str)})
        y = rng.normal(size=n)
        kw = dict(cols=["g"], stats=["mean"], cv=5, random_state=0, output="numpy")
        rec = {"kind": "crossover", "n": n, "cardinality": k}
        try:
            cpu_s = _med(lambda: TargetEncoder(**kw, backend="cpu").fit_transform(X, y), 3)
            TargetEncoder(**kw, backend="gpu").fit_transform(X, y)  # warmup
            gpu_s = _med(lambda: TargetEncoder(**kw, backend="gpu").fit_transform(X, y), 3)
            rec.update(cpu_ft_s=cpu_s, gpu_ft_s=gpu_s, speedup=round(cpu_s / gpu_s, 2))
        except Exception as exc:
            rec["error"] = repr(exc)
        print(rec, flush=True)
        rows.append(rec)
    return rows


def write(rows):
    import platform

    with open("/content/parity.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    par = [r for r in rows if r["kind"] == "parity"]
    cross = [r for r in rows if r["kind"] == "crossover"]
    lines = [
        "# catstat CPU/GPU parity + crossover (Colab)",
        f"- python: {platform.python_version()}",
        "",
        "## Parity (n=200k, 5k categories)",
        "| case | t_allclose | t_maxabs | ft_allclose | ft_maxabs | cpu_ft_s | gpu_ft_s | status |",
        "|------|-----------|----------|-------------|-----------|----------|----------|--------|",
    ]
    for r in par:
        lines.append(
            f"| {r['case']} | {r.get('transform_allclose')} | {r.get('transform_max_abs_diff')} "
            f"| {r.get('fit_transform_allclose')} | {r.get('fit_transform_max_abs_diff')} "
            f"| {r.get('cpu_ft_s')} | {r.get('gpu_ft_s')} | {r.get('status')} |"
        )
    lines += [
        "",
        "## Crossover (mean encoder; fit_transform median seconds)",
        "| n | cardinality | cpu_ft_s | gpu_ft_s | speedup (cpu/gpu) |",
        "|---|-------------|----------|----------|-------------------|",
    ]
    for r in cross:
        lines.append(
            f"| {r['n']} | {r['cardinality']} | {r.get('cpu_ft_s')} | "
            f"{r.get('gpu_ft_s')} | {r.get('speedup')} |"
        )
    Path("/content/parity_report.md").write_text("\n".join(lines) + "\n")
    print("wrote /content/parity.jsonl and /content/parity_report.md", flush=True)


if __name__ == "__main__":
    setup()
    write(run_parity() + run_crossover())
