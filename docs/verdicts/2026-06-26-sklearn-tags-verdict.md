# Verdict: scikit-learn estimator tags (`__sklearn_tags__` + `_more_tags`) (main)

- Date: 2026-06-26
- Branch: `main`
- Backend: cpu
- Artifacts:
  - `src/catstat/_base.py` (added `__sklearn_tags__` + generalized `_more_tags` on the base)
  - `src/catstat/target_encoder.py` (removed the now-redundant `_more_tags` override)
  - `tests/test_sklearn_compat.py` (tag unit tests; `__sklearn_tags__` guarded to sklearn>=1.6)
- Roadmap target: `docs/roadmap.md` Phase 3 · Related: `docs/known_issues.md` KI-012

## Question
Do the encoders advertise correct scikit-learn tags across sklearn versions, so `check_estimator`
skips inapplicable checks — without changing any encoding behavior?

## Evidence

### Correctness / leakage / parity
No encoding/cross-fit/smoothing path touched — tags are metadata. Both APIs are provided:
`__sklearn_tags__` (sklearn >= 1.6) and `_more_tags` (< 1.6; sklearn >= 1.7 ignores it with no
warning — verified on 1.9). Keyed off the existing `_is_supervised()` hook.

Verified in a fresh **sklearn 1.9.0 / pandas 3.0.3** venv:
`TargetEncoder().__sklearn_tags__()` → `target_tags.required=True`, `input_tags.categorical/string/
allow_nan=True`; `CountEncoder` → `required=False`, `allow_nan=True`. Local green gate (sklearn 1.2)
passes: the `_more_tags` test runs, the `__sklearn_tags__` test skips (<1.6).

### Performance (≥5 reps, median + spread)
N/A — metadata only.

## Decision
**KEEP** — additive, version-robust tags; no invariant, default, or public-API-param change.

## Follow-ups (PRE-EXISTING issues discovered during verification — NOT caused by this change)
- **CI red (KI-021):** CI runs bare `pytest tests/`, which doesn't put the repo root on `sys.path`,
  so `from tests.conftest import …` raises `ModuleNotFoundError: No module named 'tests'`. CI has
  been red since before this arc. Fix next: add `pythonpath` to `[tool.pytest.ini_options]`.
- **pandas 3.0 (KI-022):** `select_cols` recognizes only `object`/`CategoricalDtype`; pandas 3.0's
  default string dtype (`str`/`StringDtype`) is not auto-selected, so `cols="auto"` raises. 3 tests
  fail under pandas 3.0 (pass under the dev box's pandas 1.5). Fix next in `_validation.py`.
- Then Commit 4b: the documented `check_estimator` subset test.
