# Verdict: integer mixed-radix JOINT codes for `combination` units (lever #2A)

- Date: 2026-06-27
- Branch: `feat/perf-integer-joint-codes` (stacked on `feat/perf-integer-code-gather`, PR #7)
- Backend: cpu
- Artifacts:
  - In-process interleaved measurement (methodology below); per-size numbers in the table.
  - Combination leakage audit (OOF reconstruction + noise-trap + asymmetry) — all PASS.
- Roadmap target: `docs/roadmap.md` Phase 2 (CPU perf) · Related: `docs/known_issues.md` KI-019
  (closed here), KI-018 (GPU `combination`, foundation laid), KI-031 (gather, shipped in #7).
- Research note: `docs/notes/2026-06-27-cuml-vs-sklearn-te-levers.md` lever #2.

## Question
A `multi_feature_mode="combination"` unit built its joint category key as a Python object-array of
**tuples** (`_base._unit_keys`) and then grouped / looked up on **tuple hashing** — the only
Python-level row loop left, and the reason GPU `combination` is host-only (KI-018). Replace the tuple
key with a vectorized **mixed-radix int64 joint code** (`((c0*n1+c1)*n2+c2)…`) learned once from full
X (value-stable per-component maps, reused at fit/fold/transform), feeding PR #7's gather unchanged.
Does it speed up `combination` **without changing any output or public surface**?

## Evidence

### Correctness / leakage / sklearn
- Full green gate: ruff + **180 passed, 8 skipped** + runnable examples (`bash scripts/check.sh`).
- **Output identity at scale**: the original tuple loop, PR #2's `zip` build, and the new int joint
  codes produce **byte-identical** encodings — `max|Δ| = 0.00e+00` at n=200k and n=1M (combination
  `fit_transform`). The int code is a pure relabeling of the same row grouping.
- Leakage audit (combination, the OOF path the change feeds): OOF reconstruction vs an independent
  tuple-group-by-over-complements recompute is **exact** (`max|Δ| = 4.4e-16`); noise-trap OOF corr
  **0.06** vs leaky **0.84** for both `smooth=0` and `smooth="auto"`; asymmetry
  `mean|fit_transform − fit().transform()| = 0.022 > 0` (genuinely out-of-fold).
- sklearn-compat: `clone` / `get`+`set_params` / Pipeline / ColumnTransformer / `set_output("pandas")`
  / `get_feature_names_out` (`a+b__te_mean`) all OK; **`categories_` is decoded back to its tuple
  representation** (2D object array of category *values*, not int codes) — verified equal between the
  int path and the tuple fallback; **pickle round-trip** transforms identically (the new
  `_unit_keyplans` state pickles cleanly).
- Overflow guard: when `prod(n_c) > int64.max` the unit declines the int path and falls back to the
  tuple build (unit-tested directly + an integration test via a lowered threshold; output allclose).

### Performance — combination, 4 cols card-20, cv=5 (≥5 reps, median; in-process interleaved)
Three `_unit_keys` implementations interleaved **per rep** to cancel CPU drift: `genexpr` = this
branch's BEFORE (original per-row tuple loop), `zip` = PR #2's vectorized tuple build, `intcode` =
AFTER. `OMP_NUM_THREADS=1`.

| n | op | genexpr (before) | zip (PR #2) | **intcode (after)** | vs before | vs zip |
|---|----|-----------------:|------------:|--------------------:|----------:|-------:|
| 200k | fit_transform | 1190.9 ms | 827.4 ms | **800.0 ms** ±41.8 | **1.49×** | 1.03× |
| 200k | transform | 366.0 ms | 215.9 ms | **99.3 ms** ±14.7 | **3.69×** | 2.17× |
| 1M | fit_transform | 7007.6 ms | 4813.0 ms | **2890.0 ms** ±135.6 | **2.42×** | 1.67× |
| 1M | transform | 2272.3 ms | 1531.5 ms | **522.9 ms** ±39.4 | **4.35×** | 2.93× |

The win **grows with N** (tuple hashing is a larger fraction at scale): transform ×3.7–4.4 over the
original loop and ×2.2–2.9 over PR #2's `zip`; `fit_transform` ×1.5–2.4 over the loop (the OOF
moment-build is the remaining cost, not the key build). Critically, the int code beats even PR #2's
vectorized tuple build at every size — it removes tuple construction *and* swaps tuple-hashing
`get_indexer` for int64.

## Decision
**KEEP.** Consistent speedups that scale with N, **byte-identical** output, leakage proven exact, and
**no public-surface change** (`categories_` stays tuples; feature names, §11 fallback, defaults
unchanged). The committed `baseline-cpu.json` is **not** changed (it is the stale 0.0.1 baseline;
leaving it stale only yields informational deltas, never a false regression).

## Follow-ups
- **KI-019 closed** (combination joint-key loop replaced by int64 codes — supersedes PR #2's `zip`,
  which only built tuples faster; recommend closing/rebasing PR #2).
- **Lever #2B (next, separate PR):** unblock GPU `combination` (KI-018) — drop the `len(cols) > 1`
  clause from `host_only` (`_base.py`) and build the joint codes in `backends/_gpu.py`. That changes
  the device path, so **Colab CPU/GPU `allclose` is mandatory** there (unlike this CPU-only PR).
