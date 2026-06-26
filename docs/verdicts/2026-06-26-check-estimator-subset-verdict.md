# Verdict: scikit-learn `check_estimator` documented subset + picklability (main)

- Date: 2026-06-26
- Branch: `main`
- Backend: cpu
- Artifacts:
  - `tests/test_check_estimator.py` (new — documented subset)
  - `src/catstat/_base.py` (`__getstate__`/`__setstate__`), `src/catstat/backends/_dispatch.py`
    (`backend_module`), `tests/test_sklearn_compat.py` (pickle round-trip test)
- Related: `docs/known_issues.md` KI-012

## Question
Which sklearn estimator checks apply to catstat's categorical encoders, do the applicable ones pass,
and can a fitted estimator be pickled?

## Evidence

### Correctness
On sklearn 1.9 (venv) `check_estimator` runs 46–47 checks per encoder. With the 4a tags plus this
commit, ~36 pass; the rest are **waived** in `expected_failed_checks` with one-line reasons —
TargetEncoder 11, Count/Frequency 9 each. Waived categories: sparse input (categorical encoder),
1D/empty/complex input (reshaped or not rejected with sklearn-style messages), by-name `n_features`
(transform tolerates a differing width **by design** — enforcing it broke existing unseen-category
tests, so the guard was reverted and the check waived), and the supervised y-validation messages.
`test_check_estimator.py` runs the suite over the three encoders and raises on any *non-waived*
failure; it is skipped on sklearn < 1.6 (the dev box is 1.2).

**Picklability bug fixed:** `fit` cached the backend *module* in `_backend_mod` →
`TypeError: cannot pickle 'module' object`. `__getstate__` drops it; `__setstate__` re-resolves it
from the recorded `backend_` name via the new `_dispatch.backend_module` (reuses already-imported
`_cpu`/`_gpu` modules — no new RAPIDS import on CPU). A round-trip unit test (all sklearn versions)
asserts identical transform output; sklearn's `check_estimators_pickle` now passes.

Verified: local green gate passes (subset test skips on 1.2; pickle test runs); sklearn 1.9 / pandas
3.0.3 venv full suite **93 passed, 3 skipped**; the 3 `check_estimator` parametrized cases pass.

### Performance (≥5 reps, median + spread)
N/A.

## Decision
**KEEP** — documents and enforces the applicable sklearn subset (KI-012, downgraded S2 → S3) and
fixes a real pickling bug. No invariant, default, or public-API change; the by-name `n_features`
tolerance is preserved deliberately (waived, not "fixed", to avoid breaking subset-column transform).

## Follow-ups
- Commit 5: project-hygiene files (CONTRIBUTING, SECURITY, issue/PR templates).
