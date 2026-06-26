#!/usr/bin/env bash
# Build the catstat API reference with pdoc into the output dir (default: site/).
# Requires the docs extra:  pip install -e ".[docs]"
# (CI installs it in .github/workflows/docs.yml before calling this script.)
set -euo pipefail

cd "$(dirname "$0")/.."
OUT="${1:-site}"
# Make the src-layout package importable whether or not catstat is pip-installed.
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$PWD/src"

if ! python3 -c "import pdoc" >/dev/null 2>&1; then
  echo "error: pdoc is not installed. Run:  pip install -e \".[docs]\"" >&2
  exit 1
fi

rm -rf "$OUT"
python3 -m pdoc catstat -o "$OUT"
echo "Wrote API docs to $OUT/ (entry point: $OUT/index.html)."
