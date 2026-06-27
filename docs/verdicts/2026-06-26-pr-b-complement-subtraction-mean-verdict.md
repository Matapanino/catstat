# Verdict: complement-subtraction OOF for the mean fast path (PR-B)

- Date: 2026-06-26
- Branch: `feat/perf-complement-subtraction-mean`
- Backend: cpu
- Artifacts: in-process interleaved measurement + 13-config equivalence matrix (numbers embedded).
- Roadmap target: `docs/roadmap.md` Phase 2 (CPU perf) · Related: `docs/known_issues.md` KI-020
  (this is the backend-agnostic kernel the GPU port reuses), KI-011 (leakage invariant).

## Question
The default `fit_transform` (`stats=["mean"]`, `scheme="kfold"`) re-fits a group-by **per fold** —
`cv+1` group-bys, the "cv multiplier" profiled at 77% of a mean `fit_transform`. Replace it, for the
**pure-mean + partitioning-CV** case, with a single-pass **complement-subtraction**: one composite
`(fold, key)` aggregation (factorize + bincounts), then each fold's `(count, sum, sumsq)` by
`global − this_fold`. Does it speed up the default path **without changing output or leaking**?

## Evidence

### Correctness / leakage
- Full suite green: **117 passed, 8 skipped**.
- **Equivalence to the trusted per-fold path** — a 13-config matrix
  {reg / bin / mc} × {smooth=auto / float / 0} × {handle_missing=value / return_nan} ×
  {handle_unknown=value / return_nan / error} × {single-col / combination}: **max|Δ| ≤ 3.6e-15**,
  NaN masks identical, fast path confirmed exercised, determinism identical (same `random_state`).
- **Leakage audit** (mean fast path, fast path confirmed running): independent complement
  reconstruction **max|Δ| = 8.9e-16** (allclose, not bitwise — sum reassociation, not leakage);
  noise-category OOF corr = **0.020** (leaky path 0.645); asymmetry mean|fit_transform − leaky| =
  0.065 (> 0). Crown-jewel `test_cross_fit_no_leakage.py` passes with the fast path active.
- **Two `..._is_exact` reconstruction asserts relaxed** `== 0.0` → `< 1e-9`
  (`test_cross_fit_no_leakage.py`, `test_numeric_encoding.py`): the fast path reassociates fold sums
  (`global − fold`), so the match is **allclose, not bitwise** — the *same standard the project already
  applies to CPU/GPU parity* (CLAUDE.md invariant #2). Leak-detection power is preserved: any real
  leak is orders of magnitude above 1e-9, and the structural noise-trap + asymmetry tests are unchanged.

### Performance — mean `fit_transform` (in-process, interleaved fast/slow per rep, reps 11/9/5)
| scenario | n | before (loop) | after (fast) | speedup |
|----------|---|--------------:|-------------:|--------:|
| regression       | 10k  | 18.4 ms | 5.6 ms  | **3.30×** |
| high_cardinality | 10k  | 17.8 ms | 5.6 ms  | **3.20×** |
| multiclass(5)    | 10k  | 70.9 ms | 20.7 ms | **3.43×** |
| regression       | 100k | 98.6 ms | 40.1 ms | **2.46×** |
| high_cardinality | 100k | 114.7 ms | 43.6 ms | **2.63×** |
| multiclass(5)    | 100k | 333.7 ms | 149.5 ms | **2.23×** |
| regression       | 1M   | 1018.6 ms | 419.5 ms | **2.43×** |
| high_cardinality | 1M   | 1722.1 ms | 694.7 ms | **2.48×** |
| multiclass(5)    | 1M   | 2996.9 ms | 1583.4 ms | **1.89×** |

## Decision
**KEEP.** A **2.2–3.4× speedup** on the default mean path (regression / binary / multiclass, single
and high-cardinality), output identical to ~1e-15, leakage proven, **gated** to the pure-mean
partitioning-CV case — custom non-partition CV and any non-mean target stat (var/std/median/skew/
custom) fall back to the existing per-fold loop, unchanged. `fit_mean_encoding` itself is untouched
(zero risk to `transform`, `categories_`, smoothing). This is also the backend-agnostic kernel the
**GPU** on-device path (PR-D) will reuse to remove the per-fold host↔device round-trips (KI-020). The
committed `baseline-cpu.json` is not changed (stale 0.0.1 M0 baseline).

## Follow-ups
- **PR-C**: extend complement-subtraction to var/std (additive moments) + share the factorize across
  stats within `_fit_all` (the multi-stat redundant-re-factorize hot path).
- **PR-D**: port `kfold_mean_oof_fast` to `backends/_gpu.py` (on-device composite group-by), run a
  fresh Colab crossover, re-enable `auto`-GPU only if it wins. KI-020 host-side groundwork is done.
