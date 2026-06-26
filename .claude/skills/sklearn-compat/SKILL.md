---
name: sklearn-compat
description: >-
  Verify that catstat's encoders behave as well-mannered scikit-learn transformers. Invoke for any
  change to the public classes, constructor params, fitted attributes, feature names, or output
  handling, and before a release. Runs tests/test_sklearn_compat.py and spot-checks clone,
  get/set_params, Pipeline, ColumnTransformer, set_output, and get_feature_names_out. Reports
  PASS/FAIL per check plus the documented subset of check_estimator that applies.
---

You verify **scikit-learn protocol compliance** for `catstat`'s public encoders. Full
`check_estimator` compliance is unrealistic for supervised, multi-output transformers — target a
**documented subset** and be explicit about what does/doesn't apply and why.

## When to use
- Changes to `TargetEncoder`/`CountEncoder`/`FrequencyEncoder` public surface: constructor params,
  fitted attributes, feature names, `set_output`/`output=` handling.
- Before a release.

## When NOT to use
- Internal backend/perf changes with no public-surface effect.

## Required inputs
- The class(es) under test; installed `scikit-learn` (record the version — meaningful
  `check_estimator` coverage needs ≥1.4, which also has `TargetEncoder` for parity).

## Commands
```bash
PYTHONPATH=src python3 -m pytest tests/test_sklearn_compat.py -q
```
Spot-checks (must all hold): `sklearn.base.clone(enc)`; `enc.get_params()` /
`enc.set_params(**p)` round-trip; use inside `Pipeline` and `ColumnTransformer`;
`enc.set_output(transform="pandas")` returns a DataFrame; `enc.get_feature_names_out()` length
equals the output width.

## Files to inspect
`target_encoder.py`, `count_encoder.py`, `frequency_encoder.py`, `_base.py`, `_feature_names.py`,
`tests/test_sklearn_compat.py`.

## Failure modes to catch
- Constructor mutates or fails to store a param verbatim (breaks `clone`/`get_params`).
- A fitted attribute missing its trailing underscore, or set before `fit`.
- `get_feature_names_out` length ≠ number of output columns (esp. multiclass class-expansion and
  class-agnostic count/frequency not multiplied by `K`).
- `set_output` / `output=` not honored, or names lost.
- Silent failure inside `ColumnTransformer` (cuML had this historically — assert it actually works).

## Final report format
PASS/FAIL per check; the `scikit-learn` version used; and the **documented** list of
`check_estimator` checks that are (in)applicable to a supervised multi-output transformer, with a
one-line reason each. On FAIL, the precise attribute/method and the protocol expectation it misses.
