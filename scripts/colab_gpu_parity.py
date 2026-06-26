#!/usr/bin/env python3
"""On-VM entrypoint for the catstat CPU/GPU parity run (Colab GPU).

Invoked by ``scripts/colab_gpu_parity.sh`` via ``colab exec``. Extracts the uploaded working tree,
installs RAPIDS, then for several cases runs the *same* data + ``random_state`` through
``backend="cpu"`` and ``backend="gpu"`` and checks that ``transform`` and ``fit_transform`` agree
to ``allclose`` (not bitwise -- GPU reduction order differs). Writes ``/content/parity.jsonl`` and
``/content/parity_report.md`` for the driver to download.

There is no local GPU, so this is only ever run on Colab.
"""

from __future__ import annotations

import json
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
    # RAPIDS + CuPy. Colab images often ship cudf; install defensively and pin to CUDA 12 wheels.
    _sh([sys.executable, "-m", "pip", "install", "-q", "cudf-cu12", "cupy-cuda12x"])
    sys.path.insert(0, str(WORK / "src"))


def cases():
    import numpy as np

    rng = np.random.default_rng(0)
    n, k = 200_000, 5_000
    g = rng.integers(0, k, size=n).astype(str)
    eff = rng.normal(size=k)
    y_reg = eff[g.astype(int)] + rng.normal(0, 0.5, n)
    y_bin = (rng.uniform(size=n) < 0.3).astype(int)
    y_mc = rng.integers(0, 4, size=n)
    import pandas as pd

    X = pd.DataFrame({"g": g})
    yield "regression_mean", X, y_reg, dict(cols=["g"], stats=["mean"], cv=5, random_state=0)
    yield "regression_var", X, y_reg, dict(cols=["g"], stats=["var"], cv=5, random_state=0)
    yield "binary_mean", X, y_bin, dict(cols=["g"], stats=["mean"], cv=5, random_state=0)
    yield "multiclass_mean", X, y_mc, dict(cols=["g"], stats=["mean"], cv=5, random_state=0)


def run():
    import numpy as np

    from catstat import TargetEncoder

    rows = []
    for name, X, y, kw in cases():
        rec = {"case": name}
        try:
            cpu = TargetEncoder(**kw, backend="cpu", output="numpy")
            gpu = TargetEncoder(**kw, backend="gpu", output="numpy")

            a_t = np.asarray(cpu.fit(X, y).transform(X))
            t0 = time.perf_counter()
            b_t = np.asarray(gpu.fit(X, y).transform(X))
            rec["gpu_fit_transform_s"] = round(time.perf_counter() - t0, 4)
            rec["backend_cpu"] = cpu.backend_
            rec["backend_gpu"] = gpu.backend_
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
            ok = rec["transform_allclose"] and rec["fit_transform_allclose"]
            rec["status"] = "ok" if ok else "MISMATCH"
        except Exception as exc:
            rec["status"] = "ERROR"
            rec["error"] = repr(exc)
        print(rec, flush=True)
        rows.append(rec)
    return rows


def write(rows):
    import platform

    with open("/content/parity.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    lines = [
        "# catstat CPU/GPU parity (Colab)",
        f"- python: {platform.python_version()}",
        "",
        "| case | t_allclose | t_maxabs | ft_allclose | ft_maxabs | gpu_ft_s | status |",
        "|------|-----------|----------|-------------|-----------|----------|--------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['case']} | {r.get('transform_allclose')} | "
            f"{r.get('transform_max_abs_diff')} | {r.get('fit_transform_allclose')} | "
            f"{r.get('fit_transform_max_abs_diff')} | {r.get('gpu_fit_transform_s')} | "
            f"{r.get('status')} |"
        )
    Path("/content/parity_report.md").write_text("\n".join(lines) + "\n")
    print("wrote /content/parity.jsonl and /content/parity_report.md", flush=True)


if __name__ == "__main__":
    setup()
    write(run())
