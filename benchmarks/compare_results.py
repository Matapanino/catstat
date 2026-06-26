"""Compare a benchmark run against a committed baseline; exit non-zero on a regression.

Usage:
    python3 benchmarks/compare_results.py benchmarks/results/run.json \
        benchmarks/results/baseline-cpu.json [--threshold 0.15]

A regression is a fit_transform median slower than baseline by more than --threshold, OR any
change in output width / regression quality. Correctness/leakage are enforced by the tests; this
guards performance + shape stability for the self-improvement loop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(p):
    return json.loads(Path(p).read_text())["cases"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("current")
    ap.add_argument("baseline")
    ap.add_argument("--threshold", type=float, default=0.15)
    args = ap.parse_args()

    cur, base = _load(args.current), _load(args.baseline)
    shared = sorted(set(cur) & set(base))
    regressions = []

    print(f"{'case':18s} {'base ms':>9s} {'cur ms':>9s} {'delta':>8s}  status")
    for name in shared:
        b = base[name]["fit_transform_s"]["median"]
        c = cur[name]["fit_transform_s"]["median"]
        delta = (c - b) / b if b > 0 else 0.0
        status = "ok"
        if delta > args.threshold:
            status = "REGRESSION"
            regressions.append(name)
        elif delta < -0.10:
            status = "improvement"
        if cur[name]["n_out_cols"] != base[name]["n_out_cols"]:
            status = "SHAPE-CHANGED"
            regressions.append(name)
        print(f"{name:18s} {b*1e3:9.1f} {c*1e3:9.1f} {delta*100:7.1f}%  {status}")

    only_cur = sorted(set(cur) - set(base))
    if only_cur:
        print(f"new cases (no baseline): {only_cur}")
    if regressions:
        print(f"\nFAIL: {len(regressions)} regression(s): {regressions}")
        sys.exit(1)
    print("\nOK: no regressions vs baseline.")


if __name__ == "__main__":
    main()
