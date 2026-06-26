"""Append-only JSONL results ledger + a stable row schema.

Every benchmark run appends one row per case here so any future session can diff "now" vs a
committed baseline. The ledger is never edited in place; only appended to.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
LEDGER_PATH = RESULTS_DIR / "ledger.jsonl"


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "nogit"


def versions() -> dict:
    import numpy
    import pandas
    import sklearn

    import catstat

    return {
        "catstat": catstat.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "python": platform.python_version(),
    }


def run_meta(backend: str) -> dict:
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "backend": backend,
        "versions": versions(),
    }


def append_rows(rows: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return LEDGER_PATH


def write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
