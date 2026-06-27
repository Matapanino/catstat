# Verdict: extend the single-pass OOF kernel to var/std (PR-C)

- Date: 2026-06-27
- Branch: `feat/perf-additive-var-std` (off `feat/perf-complement-subtraction-mean`, PR-B)
- Backend: cpu
- Artifacts: in-process interleaved before/after (numbers embedded) + the
  {var,std}×{min_samples}×{missing,unknown}×{single,combination} equivalence matrix
  (`tests/test_additive_fast_path.py`).
- Roadmap target: `docs/roadmap.md` Perf arc · Related: PR-B (the mean kernel this extends),
  `docs/notes/2026-06-27-cuml-vs-sklearn-te-levers.md` (the next lever).

## Question
PR-B made the pure-mean kfold path single-pass via complement subtraction, but **var/std still fell
to the per-fold group-by loop** (`cv` re-fits, each re-factorizing the keys). The fast kernel already
accumulates per-(fold,key) **sum-of-squares** alongside count/sum, so var/std are a cheap *finalize*
from the same complement moments: sample var `= (ss − s²/cc)/(cc − 1)` (ddof=1), std `= √var`, with a
per-fold **complement-global** fallback when a category's complement count `< max(min_samples,1)` or
`< 2` (singleton variance is undefined). Does extending the kernel to {mean,var,std} — and a **hybrid**
gate that runs additive columns fast while non-additive stats (median/min/max/skew/custom) keep the
slow loop — speed things up **without changing output or leaking**?

## Evidence

### Correctness / leakage
- Full suite green: **167 passed, 8 skipped** (was 117; +50 new equivalence cases).
- **Equivalence to the per-fold path** — `tests/test_additive_fast_path.py`, a 48-config matrix
  {var,std} × {min_samples 1/2/5} × {handle_missing value/return_nan} ×
  {handle_unknown value/return_nan} × {independent/combination}, plus a hybrid `mean,var,std,median`
  case: all **allclose** (rtol 1e-7, atol 1e-9, NaN masks identical). Disabling the fast gate
  (`_ADDITIVE_STATS=∅`) routes the same request through the trusted slow loop as the reference.
- **Leakage audit** (independent pure-pandas reconstruction, not catstat internals): per-fold
  OOF var/std rebuilt from each fold's complement matches `fit_transform` to **≤ 3.4e-13 (var) /
  7.1e-15 (std)** across min_samples (allclose, not bitwise — the one-pass moment formula
  reassociates sums, CLAUDE.md invariant #2). Noise-trap corr(OOF var[noise], |y|) = **−0.004**
  (signal +0.445); asymmetry mean|fit_transform − leaky| = **20.5** (> 0). `test_cross_fit_no_leakage.py`
  passes (mean path intact after the refactor).

### Performance — `fit_transform` (in-process, interleaved before/after per rep, 7 reps, median)
`before` = pre-PR behavior (mean fast, var/std/median via the slow loop), reproduced by gating
`_ADDITIVE_STATS={"mean"}`; `after` = the new {mean,var,std} kernel. 2 cols, cv=5, allclose verified.

| stats | n | before | after | speedup |
|-------|---|-------:|------:|--------:|
| `var` | 200k | 312.9 ms | 117.3 ms | **2.67×** |
| `mean,var,std` | 200k | 585.4 ms | 215.9 ms | **2.71×** |
| `mean,var,std,median` (hybrid) | 200k | 824.5 ms | 553.0 ms | **1.49×** |
| `var` | 1M | 1667.8 ms | 604.2 ms | **2.76×** |
| `mean,var,std` | 1M | 3003.7 ms | 1064.9 ms | **2.82×** |
| `mean,var,std,median` (hybrid) | 1M | 4174.9 ms | 2840.6 ms | **1.47×** |

## Decision
**KEEP.** **2.7–2.8×** on additive requests (var-only and mean+var+std) and **~1.5×** on the mixed
case where the additive columns now ride the single-pass kernel and only median stays on the slow
loop — output identical to ~1e-13, leakage proven. Within the fast path a unit's mean/var/std share
**one** factorize + one composite bincount (the per-stat re-factorize is gone for additive stats).
Gating is unchanged in spirit: a non-partitioning CV, or a request with no additive stat, still takes
the original per-fold loop; `_smoothing.fit_mean_encoding` and `_aggregations` are untouched
(`transform`, `categories_`, smoothing unaffected). `baseline-cpu.json` not changed (stale 0.0.1).

## Follow-ups
- **Integer-code gather (next arc)** — the research note's #1 CPU lever: replace `pd.Series.map` in
  `_transform_array` (52% `get_indexer`) with factorize-once + numpy gather; integer joint codes also
  unblock GPU combination (KI-018). See `docs/notes/2026-06-27-cuml-vs-sklearn-te-levers.md`.
- **Interactions feature** — `interactions: list[list[str]]` (separate PR); the additive kernel
  already handles multi-col tuple-key units, so interaction columns get the fast path for free.
- **PR-D (GPU)** — port the kernel to `backends/_gpu.py`; fresh Colab crossover before re-enabling
  `auto` (KI-020). Still pending; unchanged by this PR.
