# Verdict: CI green — pytest `pythonpath` so `tests` imports (main)

- Date: 2026-06-26
- Branch: `main`
- Backend: cpu (CI/test-config — no compute path touched)
- Artifacts: `pyproject.toml` (`[tool.pytest.ini_options] pythonpath`)
- Related: `docs/known_issues.md` KI-021

## Question
Why is CI red, and does adding pytest `pythonpath` make CI's bare `pytest tests/` collect and pass?

## Evidence

### Correctness
CI has been red since **before this arc**: the workflow runs bare `pytest tests/`, which — unlike
`scripts/check.sh`'s `python -m pytest` — does not put the repo root on `sys.path`, so
`from tests.conftest import …` raised `ModuleNotFoundError: No module named 'tests'` during
collection. CI never ran a single test. Reproduced locally with `unset PYTHONPATH; pytest tests/`.

Fix: `pythonpath = ["src", "."]` in `[tool.pytest.ini_options]` (pytest ≥ 7; local is 9.0.3). After
the fix, bare `pytest tests/` → **89 passed, 3 skipped** (pandas 1.5); `bash scripts/check.sh` green.

### Performance (≥5 reps, median + spread)
N/A.

## Decision
**KEEP** — central, invocation-agnostic fix (works for `pytest`, `python -m pytest`, with/without an
editable install); no library behavior changed. CI should go green once the pandas 3.0 break
(KI-022) is also fixed (next commit).

## Follow-ups
- Next: fix `select_cols` for the pandas 3.0 default string dtype (KI-022).
