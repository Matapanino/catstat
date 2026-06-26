# Verdict: vectorize the combination joint-key build (PR-A)

- Date: 2026-06-26
- Branch: `feat/perf-vectorize-joint-key`
- Backend: cpu
- Artifacts:
  - In-process interleaved measurement (methodology below); per-size numbers embedded in the table.
  - Corroborating separate-process harness runs (reps=7) — numbers in the Performance section.
- Roadmap target: `docs/roadmap.md` Phase 2 (CPU perf) · Related: `docs/known_issues.md` KI-019 (closed)

## Question
The combination-mode joint-key build in `_base._unit_keys` was a pure-Python per-row loop
(`for i in range(n): joint[i] = tuple(ck[i] for ck in comp_keys)`) — profiled at **58%** of a
combination `fit_transform` (n=200k, the single Python-level row loop in the codebase, KI-019).
Replace it with a C-level `zip` build (`for i, key in enumerate(zip(*comp_keys)): joint[i] = key`).
Does it speed up combination encoding **without changing any output**?

## Evidence

### Correctness / leakage / sklearn
- Full suite green: **117 passed, 8 skipped** (identical to pre-change).
- Leakage audit (combination path, the code changed): OOF reconstruction **exact** vs an independent
  complement recomputation (max|Δ| = 0.000e+00); noise-trap corr(OOF, y_indep) = −0.0025;
  asymmetry mean|fit_transform − fit().transform()| = 0.022 (> 0, genuinely out-of-fold).
- sklearn-compat: `test_sklearn_compat` + `test_check_estimator` pass (10 passed / 4 skipped);
  clone / get_params / `get_feature_names_out` (`a+b__te_mean`) / `set_output('pandas')` / Pipeline
  spot-check OK; `categories_` representation unchanged.
- **Output identity at scale**: the old loop and the new zip produce **byte-identical** encodings —
  `max|Δ(old, new)| = 0.00e+00` at n = 10k / 100k / 1M. The change is provably behavior-preserving;
  the `MISSING`-sentinel tuples are reproduced exactly (verified on data with ~5% missing).

### Performance — combination `fit_transform` (≥5 reps, median + spread)
In-process, **interleaving old/new per rep** to cancel cross-process drift (monkeypatch
`_BaseStatEncoder._unit_keys`); `OMP_NUM_THREADS=1`; reps 11 / 9 / 5:

| case | metric | before (loop) | after (zip) | delta |
|------|--------|--------------:|------------:|------:|
| combination · 10k  | fit_transform_s | 79.8 ms ±20.0 | 51.5 ms ±9.4  | **−35.5% (1.55×)** |
| combination · 100k | fit_transform_s | 873.7 ms ±37.7 | 609.6 ms ±37.7 | **−30.2% (1.43×)** |
| combination · 1M   | fit_transform_s | 10162.6 ms ±321.8 | 7418.8 ms ±965.1 | **−27.0% (1.37×)** |

Corroborating separate-process harness (reps=7): combination 10k 89.8→40.5 ms, 100k 883.3→657.1 ms
(noisier across processes — e.g. the untouched single-column `multiclass` case drifted ±30% between
invocations — which is exactly why the interleaved measurement above is the attributable one). The
six single-column scenarios are untouched code paths.

## Decision
**KEEP.** A consistent **1.37–1.55× speedup** on the only affected scenario, with **byte-identical
output** (zero regression risk), leakage proven exact, and no public-surface change. The relative win
shrinks with n because the unchanged object-tuple **groupby** (tuple hashing) grows as a fraction —
that hashing cost is the target of the deferred integer-joint-code work (mixed-radix codes land with
the GPU device path, PR-D / KI-018), not PR-A. The committed `baseline-cpu.json` is **not** changed:
it is the stale 0.0.1 M0 baseline; leaving it stale only yields informational "improvement" deltas,
never a false regression, and a full refresh is separate housekeeping.

## Follow-ups
- **KI-019 closed** (combination joint-key loop vectorized).
- **PR-B**: complement-subtraction OOF for mean/var/std — collapse the `cv`-fold groupby multiplier
  (the next CPU lever; the leakage-critical core).
- Integer joint codes (mixed-radix) for the tuple-groupby + combination-on-GPU (KI-018) land with PR-D.
