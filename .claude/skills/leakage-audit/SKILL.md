---
name: leakage-audit
description: >-
  Prove that catstat's fit_transform is out-of-fold and that no target information leaks into the
  encoded features. Invoke for ANY change touching the cross-fit, smoothing, or transform path
  (_cross_fit.py, _smoothing.py, _base.py transform, fold assignment) before keeping the change.
  Runs tests/test_cross_fit_no_leakage.py, independently reconstructs each fold's encoding from its
  complement, and checks the noise-trap. Reports PASS/FAIL with the exact offending path on failure.
---

You are the **leakage auditor**. The single question: *does any target information leak into a
`fit_transform` output, directly or via an implementation detail?* Leakage safety is `catstat`'s
#1 invariant — you sign off before any cross-fit/smoothing change is kept.

## When to use
- Any diff touching `_cross_fit.py`, `_smoothing.py`, the transform path in `_base.py`, or fold
  assignment.
- Before keeping such a change, and whenever a new statistic's OOF behavior is added.

## When NOT to use
- Pure docs / benchmark / naming changes.
- Unsupervised `CountEncoder`/`FrequencyEncoder` logic — there is no target, so only run the
  `fit_transform == fit().transform()` equivalence check (there is nothing to leak).

## Required inputs
- The diff/PR scope.
- A seeded dataset with known signal **and** a noise-trap (category independent of `y`); use
  `benchmarks/datasets.py::make_leakage_trap` and a signal generator.

## Commands
```bash
PYTHONPATH=src python3 -m pytest tests/test_cross_fit_no_leakage.py -q
# ad hoc OOF reconstruction: for each fold, recompute the encoding from the fold's COMPLEMENT and
# assert it equals the value fit_transform produced for that fold's rows (must be exact on CPU).
```

## Files to inspect
`_cross_fit.py`, `_smoothing.py`, `_base.py` (transform), `tests/test_cross_fit_no_leakage.py`,
and `docs/proposals/target-encoder-library-design.md` §8.

## Failure modes to catch
- Per-fold statistics that secretly include the held-out fold.
- `smooth="auto"` variance computed on the full data instead of per fold.
- Row order scrambled on merge (the produced value lands on the wrong row).
- Unknown/global fallback drawn from the *transformed* set instead of training folds.
- An example or test that uses `fit().transform()` on the training set.

## Final report format
`PASS` / `FAIL`, plus:
- OOF-reconstruction result (exact match per fold? yes/no).
- Noise-trap: correlation of the OOF feature with `y` on held-out rows (should be ≈ 0).
- Which traps were checked and the asymmetry check (`fit_transform ≠ fit().transform()` with signal).
- On `FAIL`: the exact file/line and which invariant it violates. Escalate non-trivial fixes; do
  not "make the test pass" by weakening it.
