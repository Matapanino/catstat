# Verdict: pandas 3.0 compat — `cols="auto"` selects StringDtype (main)

- Date: 2026-06-26
- Branch: `main`
- Backend: cpu
- Artifacts:
  - `src/catstat/_validation.py` (`_is_categorical_like` + `select_cols`)
  - `tests/test_io_types.py` (`test_string_dtype_column_is_auto_selected`)
- Related: `docs/known_issues.md` KI-022

## Question
Does `cols="auto"` work on pandas >= 3.0, where string columns default to `StringDtype` (not
`object`)?

## Evidence

### Correctness / leakage / parity
Root cause: pandas 3.0 types string columns as `StringDtype` (repr `str`); `select_cols` matched
only `object`/`CategoricalDtype`, so `cols="auto"` raised "found no object/category columns". Fix:
a `_is_categorical_like` helper recognizing `object` (via `pd.api.types.is_object_dtype`),
`Categorical`, and `StringDtype`. Column-selection only — no cross-fit/smoothing/transform path
touched, so `leakage-audit` is not required.

Verified: fresh **sklearn 1.9.0 / pandas 3.0.3** venv full suite → **89 passed, 3 skipped** (the 3
prior pandas-3.0 failures — `test_numpy_object_array_in_ndarray_out_auto`,
`test_column_transformer_passthrough`, `test_set_output_pandas_numpy_input` — now pass). Local
**pandas 1.5.2** `scripts/check.sh` green; the new regression test uses an explicit `dtype="string"`
column, so it exercises the fix on pandas 1.5 too. ruff clean (used `is_object_dtype` to avoid the
`== object` E721).

### Performance (≥5 reps, median + spread)
N/A.

## Decision
**KEEP** — restores the intended `cols="auto"` behavior under pandas >= 3.0 (additive; still selects
`object` + `Categorical`). Together with the pytest `pythonpath` fix (KI-021), CI should now be
green end-to-end on the latest pandas/sklearn.

## Follow-ups
- Commit 4b: the documented `check_estimator` subset test (KI-012).
- Confirm CI goes green on the next push (CI installs the latest pandas).
