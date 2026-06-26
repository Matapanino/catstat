# Verdict: M0 bootstrap — CPU mean TargetEncoder + Count/Frequency (leakage-safe)

- Date: 2026-06-26
- Branch: `n/a` (repo not yet under git)
- Backend: cpu
- Artifacts:
  - `benchmarks/results/baseline-cpu.json` (first committed baseline)
  - `benchmarks/results/ledger.jsonl` (append-only run ledger)
- Roadmap target: `docs/roadmap.md` → **MVP (M0)** · Related: `docs/known_issues.md` KI-010..017

## Question
Does M0 deliver a runnable, **CPU-only**, leakage-safe `TargetEncoder` (mean) for
regression/binary/multiclass plus `CountEncoder`/`FrequencyEncoder`, with `smooth` auto+fixed,
unknown/missing handling, feature names, sklearn-compat, and a benchmark baseline — all green on
`scripts/check.sh`?

## Evidence

### Correctness / leakage (the crown jewels)
- `bash scripts/check.sh`: **ruff clean · 46 passed, 1 skipped (GPU) · all 4 examples run**.
- **OOF reconstruction is exact**: independently recomputing each fold's encoding from its
  complement matches `fit_transform` to `max |Δ| = 0.0` (`test_oof_reconstruction_is_exact`).
- **Noise-trap**: for a high-cardinality category independent of `y`, OOF encoding corr with `y`
  ≈ **-0.006** (< 0.1) while the leaky `fit().transform()`-on-train path corr ≈ **0.66** (> 0.4) —
  the cross-fitting removes the target signal as intended.
- **Determinism**: identical output for equal `random_state`; differs for different seeds.
- Per-stat fallbacks verified: count/frequency unseen → 0/0.0; mean unseen → global; multiclass
  probabilities sum to 1 (raw); m-estimate and `smooth=0` match hand-computed values.

### Coverage
- `pytest --cov=catstat`: **85.87%** total — above the **85%** floor (GPU path omitted from the lane).

### Performance (size=small=10k rows, cpu, 5 reps, median)
| case | fit_transform (ms) | out cols |
|------|-------------------:|---------:|
| regression | 16.1 | 1 |
| binary | 15.0 | 1 |
| multiclass (5 classes) | 55.2 | 5 |
| high_cardinality | 15.6 | 1 |
| count (unsupervised) | 2.0 | 1 |

`compare_results.py` against this baseline reports no regressions on re-run (deltas within noise).

## Decision
**KEEP** — M0 is complete and green. This `baseline-cpu.json` is established as the committed CPU
baseline; future changes compare against it and update it only via a new verdict.

## Follow-ups
- KI-010: verify the `smooth="auto"` formula against sklearn's `_target_encoder_fast.pyx` and add a
  sklearn-parity test (needs `scikit-learn>=1.4`; the dev box has 1.2, so it currently skips).
- Coverage headroom is thin (85.87%); add targeted tests for `backends/_dispatch` auto-branch and
  `_gpu.ensure_available` error paths when convenient.
- Begin Phase 2 (GPU backend + var/std/median/min/max + Colab parity loop) per the roadmap.
- Put the repo under git so verdicts can record a real branch/SHA.
