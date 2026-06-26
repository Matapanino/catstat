#!/usr/bin/env bash
# Development green-gate: lint (if ruff is installed), tests, and runnable examples.
set -euo pipefail
cd "$(dirname "$0")/.."

# Make the package importable without an install.
export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}src"
# Keep BLAS/OpenMP single-threaded so the small suite is fast and deterministic.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if python3 -m ruff --version >/dev/null 2>&1; then
    python3 -m ruff check src tests examples benchmarks scripts
fi

python3 -m pytest tests/ -q

python3 examples/regression_basic.py
python3 examples/binary_classification_basic.py
python3 examples/multiclass_classification_basic.py
python3 examples/count_frequency_basic.py

echo "All checks passed."
