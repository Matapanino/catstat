# Verdict: B0 — (fold × cat) table OOF kernel (feat/shape-stats-moments)

- Date: 2026-07-02
- Branch: `feat/shape-stats-moments`
- Backend: cpu
- Artifacts:
  - `benchmarks/results/2026-07-02-b0-table-kernel.jsonl` (AFTER, standard harness, 5 reps)
  - `benchmarks/results/baseline-cpu.json` (BEFORE — stale 0.0.1 M0 baseline, context only)
  - Interleaved old(205b0c9)/new in-process comparison (this file, table below) — the
    attribution-clean measurement per the established method (cross-process drift ~30%).
- Roadmap target: GPU device-resident arc (PR-D prerequisite) · Related: KI-020

## Question
Does restructuring the additive OOF kernel from per-row arithmetic to per-(fold, key)-cell
finalization + one gather (a) preserve values exactly, (b) not regress CPU, and (c) create the
backend seam (`moment_tables`) the B1 device kernel needs?

## Evidence

### Correctness / leakage / parity
- Full suite green (`scripts/check.sh`, 327 tests incl. the fast==slow matrix over
  {var,std,skew,kurt} × min_samples × missing × unknown × mode, hybrid, 1e9-offset).
- Leakage audit re-run post-refactor: OOF reconstruction from complement fits — skew ≤ 6.9e-14,
  kurt ≤ 2.0e-12 rel (allclose), woe exact 0.0; noise-trap corr −0.003/+0.016/−0.03; asymmetry
  (fit_transform ≠ leaky) holds. PASS.
- Old-vs-new value parity per scenario (n=200k, k=10k): max|Δ| ≤ 1.24e-14; woe bit-exact 0.0.
- New targeted test: `handle_unknown='error'` raises iff an occupied cell is unseen
  (fc>0 & cc==0) — table semantics equal per-row semantics.

### Performance (interleaved old/new, n=200k, k=10k, cv=5, 7 reps, median [IQR])
| case | old fit_transform_s | new fit_transform_s | speedup |
|------|--------------------:|--------------------:|--------:|
| mean_only | 0.1036 [0.0137] | 0.0955 [0.0198] | ×1.08 |
| mean_var_std | 0.1891 [0.0292] | 0.1853 [0.0501] | ×1.02 |
| all_additive (mean/var/std/skew/kurt) | 0.2689 [0.0784] | 0.2446 [0.0457] | ×1.10 |
| binary mean+woe | 0.2085 [0.0618] | 0.1726 [0.0147] | ×1.21 |

Standard harness vs the stale M0 baseline: all target-dependent cases −28%…−71% (reflects the
whole perf arc since 0.0.1, not B0 alone); no regressions.

## Decision
**KEEP** — values are preserved (allclose; woe exact), the leakage audit re-passes, and CPU is
neutral-to-modestly-positive (×1.02–1.21; spreads overlap for the smaller wins, so no CPU perf
claim beyond "no regression"). The real payoff is architectural: all OOF finalization is now a
function of small (F·C) tables fed by a single injectable `moment_tables` kernel
(`np_moment_tables`), which is exactly the seam the B1 on-device kernel plugs into. The committed
baseline is **not** updated (unchanged policy: it changes only via a dedicated baseline verdict).

## Follow-ups
- B1: `backends/_gpu.oof_moment_tables` (cupy.bincount) injected through the same seam; GPU
  perf claims deferred to the B5 Colab crossover verdict.
- `kfold_mean_oof_fast` removed (no callers in src/tests — the "kept for back-compat" note was
  stale).
