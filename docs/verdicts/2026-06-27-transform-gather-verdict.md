# Verdict: integer-code gather on the transform path (feat/perf-integer-code-gather)

- Date: 2026-06-27
- Branch: `feat/perf-integer-code-gather` (stacked on `feat/perf-additive-var-std`)
- Backend: cpu
- Artifacts:
  - `benchmarks/results/2026-06-27-transform-gather.jsonl` (AFTER, end-to-end harness @ medium)
  - in-process interleaved before/after micro/transform bench (scratchpad `bench_gather.py`)
  - `benchmarks/results/baseline-cpu.json` (committed baseline — SHA `6a75054`, **pre-perf-arc**)
- Roadmap target: `docs/roadmap.md` "Next CPU lever" · Related: `docs/known_issues.md` KI-031 (this),
  KI-018 (GPU combination, unblocked by the deferred joint-code follow-up)

## Question
Replace the per-column object-key `pd.Series.map` in `_transform_array` with a factorize-once +
numpy fancy-index gather (each unit's keys hashed once via `index.get_indexer`, each column gathered
from a contiguous float64 array). Does it speed up transform without changing outputs, leakage
safety, or the public surface — and is it worth keeping?

## Evidence

### Correctness / leakage / parity
- **Outputs identical:** the interleaved bench asserts `np.allclose(old, new, equal_nan=True)` on
  every case (single-col 1- and 4-stat, high-cardinality, and combination tuple keys). The gather
  reproduces `.map` exactly (unknown code `-1` → NaN; known code → stored value).
- **Full suite:** `170 passed, 8 skipped` (3 new gather regression tests in
  `tests/test_transform_gather.py`: mixed-order multi-stat alignment, combination unknown/known
  joint key, tiny-n baked-global vs unseen-under-`handle_unknown`).
- **leakage-audit: PASS.** `tests/test_cross_fit_no_leakage.py` 4/4 (exact OOF reconstruction).
  Independent noise-trap (4316-level category independent of `y`): OOF `corr≈−0.013` (mean) /
  `−0.012` (median) — both stats route through the modified `_transform_array`, incl. the per-fold
  slow path; leaky `fit().transform()` memorizes (`corr≈+0.65`). `fit_transform ≠ fit().transform()`.
- **sklearn-compat: PASS** (sklearn 1.2.0): clone, get/set_params, Pipeline, ColumnTransformer,
  `set_output`, feature-name width (incl. multiclass K-expansion), and a **pickle round-trip** (the
  new `_UnitEncoding` dataclass pickles and reproduces identical transforms; `categories_` preserved).
- CPU/GPU parity: the gather is host numpy at the same locus as the prior host `.map`; structurally
  parity-preserving. A Colab `var`/`std` parity pass remains a low-risk optional check.

### Performance — transform path, in-process INTERLEAVED before/after (n=1,000,000, 7 reps, allclose)
The cleanest before/after for *this* change: old (`.map` per column) and new (gather) run on the
**same** fitted tables, alternating per rep to cancel drift (the committed baseline is from SHA
`6a75054`, before the entire perf arc, so a cross-process compare to it is not attributable here).

| case (n=1M) | metric | before (`.map`) | after (gather) | delta |
|------|--------|-------:|------:|------:|
| single-col, **4 stats** (mean/var/std/median) | transform_s (median) | 209.8 ms | 92.1 ms | **×2.28** |
| single-col, 1 stat (mean) | transform_s (median) | 88.8 ms | 88.9 ms | ×1.00 (neutral) |
| single-col, 4 stats, high-card (50k) | transform_s (median) | 686.6 ms | 204.3 ms | **×3.36** |
| combination 2-col, 4 stats | transform_s (median) | 2968.3 ms | 1197.4 ms | **×2.48** |

The win scales with stats-per-unit (one `get_indexer` per unit instead of per column) and with
cardinality. Single-stat is exactly neutral (the no-unknown fast path is a single fancy index).

### Performance — end-to-end harness (medium, 7 reps, AFTER only; default single-stat cases)
Confirms no regression on the canonical cases: transform_s ≈ regression 6.6 ms, count 7.7 ms,
high_cardinality 7.8 ms, multiclass 8.1 ms (5 cols, one shared factorize), combination 79.3 ms.
(Default cases use `stats=("mean",)`, so they exercise the neutral single-stat path; the multi-stat
gain is the interleaved table above.)

## Decision
**KEEP.** Outputs are bit-for-bit equivalent (allclose), leakage and sklearn-compat both PASS, and
transform is ×2.3–3.4 faster for multi-stat / high-cardinality units with a neutral single-stat
path. **No default is changed. The committed baseline is NOT updated** (it predates the perf arc;
refresh it once the arc lands on `main`).

## Follow-ups
- Next lever (deferred, separate PR): integer **joint code** (`c_a*n_b+c_b`) to replace the tuple-key
  build loop in `_unit_keys` — vectorizes combination key-building (dominant cost in the combination
  case above) and unblocks GPU `combination` (KI-018). Overlaps PR #2 (`feat/perf-vectorize-joint-key`).
- Append KI-031 to `docs/known_issues.md`; log this in `docs/experiment_log.md`.
- After the perf arc merges to `main`, re-run the harness on the final SHA and refresh the committed
  CPU baseline via a follow-up verdict.
