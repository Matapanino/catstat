#!/usr/bin/env python3
"""On-VM entrypoint: run catstat's test suite (incl. the gpu-marked tests) on a Colab GPU.

Companion to ``scripts/colab_gpu_parity.py`` (which measures parity/crossover); this one runs
``pytest`` so every ``@pytest.mark.gpu`` test -- kernel parity, device-input parity, fences --
executes on real RAPIDS. Extracts ``/content/catstat.tar.gz``, installs deps, runs the full
suite (the CPU tests double as a Linux/py3.12 check), and writes ``/content/gpu_tests.txt``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

WORK = Path("/content/catstat_repo")


def _sh(cmd, **kw):
    print(">>", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=False, **kw)


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    _sh(["tar", "xzf", "/content/catstat.tar.gz", "-C", str(WORK)])
    _sh([sys.executable, "-m", "pip", "install", "-q", "cudf-cu12", "cupy-cuda12x",
         "pytest", "scikit-learn"])
    env_path = f"{WORK}/src"
    res = _sh(
        [sys.executable, "-m", "pytest", "tests/", "-q", "-rf", "--tb=short",
         "-p", "no:cacheprovider"],
        cwd=str(WORK),
        env={"PYTHONPATH": env_path, "PATH": "/usr/bin:/bin:/usr/local/bin",
             "OMP_NUM_THREADS": "1", "HOME": "/root"},
        capture_output=True,
        text=True,
    )
    out = res.stdout + "\n" + res.stderr
    print(out[-8000:], flush=True)
    Path("/content/gpu_tests.txt").write_text(out)
    print(f"pytest exit code: {res.returncode}", flush=True)
    return res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
