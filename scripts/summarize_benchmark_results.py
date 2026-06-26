"""Print a compact human-readable table from a benchmark run JSON (for verdict docs).

Usage: python3 scripts/summarize_benchmark_results.py benchmarks/results/run.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    payload = json.loads(Path(sys.argv[1]).read_text())
    meta = payload.get("meta", {})
    print(
        f"# run @ {meta.get('ts', '?')}  sha={meta.get('git_sha', '?')}  "
        f"backend={meta.get('backend', '?')}"
    )
    print(f"versions: {meta.get('versions', {})}")
    print()
    hdr = (
        f"| {'case':16s} | {'fit ms':>8s} | {'transform ms':>12s} | "
        f"{'fit_transform ms':>16s} | cols | quality |"
    )
    print(hdr)
    print("|" + "-" * (len(hdr) - 2) + "|")
    for name, rec in payload.get("cases", {}).items():
        print(
            f"| {name:16s} | {rec['fit_s']['median']*1e3:8.1f} | "
            f"{rec['transform_s']['median']*1e3:12.1f} | "
            f"{rec['fit_transform_s']['median']*1e3:16.1f} | "
            f"{rec['n_out_cols']:4d} | {rec.get('quality', {})} |"
        )


if __name__ == "__main__":
    main()
