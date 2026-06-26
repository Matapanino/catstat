# Why is cuML's TargetEncoder faster than sklearn's — and which CPU levers is catstat still missing?

- Date: 2026-06-27
- Scope: a research scout (no code) for the perf arc. catstat **already** has the single-pass
  composite-groupby + OOF-by-complement-subtraction algorithm on CPU (PR-B mean, PR-C var/std). The
  question: what *else* makes cuML fast that ports to a pandas/numpy CPU path?
- Sources read: cuML `python/cuml/cuml/preprocessing/TargetEncoder.py` (`_groupby_agg`,
  `_fit_transform`, `_make_fold_column`); the RAPIDS "Target Encoding with cuML" blog; sklearn
  `preprocessing/_target_encoder.py` + the Cython `_target_encoder_fast.pyx`
  (`_fit_encoding_all_targets`), `utils/_encode.py`, `preprocessing/_encoders.py`.

## Headline finding

**cuML offers nothing to port algorithmically.** Its `_groupby_agg` is exactly the
complement-subtraction trick catstat already has: `groupby([fold]+x_cols).agg` then
`groupby(x_cols).agg` on the small per-fold table, then subtract. It represents categories as **raw
object/string keys** (cuDF GPU hash join) — *no* integer codes. The RAPIDS ~100× is GPU parallelism
over cuDF groupby/hash-join + the ~4× from doing all folds in parallel (which complement-subtraction
already buys). So cuML ≠ a source of CPU levers.

**The CPU levers come from scikit-learn**, whose `TargetEncoder` is the opposite design: it
cross-fits **per fold** (n_folds passes — algorithmically *worse* than catstat) but makes each pass
extremely cheap via an **integer-code representation**:
- `OrdinalEncoder` integer-codes every column **once** at fit → a C-contiguous `int` matrix
  `X_ordinal`. Hashing is paid once, upfront.
- Cython `_fit_encoding_fast` does `sums[code] += y` / `counts[code] += 1` — an **O(1) array
  scatter-add**, no hash lookup on the hot path; smoothing is folded into the accumulator init
  (`sums[c] = smooth*y_mean`, `counts[c] = smooth`) so the m-estimate falls out of the final
  division for free; buffers reused across features; `nogil`.
- Transform is a **pure numpy gather**: `encoding[X_ordinal[rows, col]]` — no pandas, no `.map()`,
  no merge — into a pre-allocated `X_out`.

catstat's fast kernel **already** uses `pd.factorize` + `np.bincount` internally (so the *fit*
accumulation is sklearn-equivalent). The unadopted half is the **transform path** and the **fitted
representation**: `_transform_array` still maps via `pd.Series.map` on object keys (profiled at
**52% `get_indexer`** of a multi-stat transform), and the slow per-fold loop (median/min/max/skew/
custom) re-factorizes per stat.

## Ranked CPU-applicable levers (for a path that already has complement-subtraction)

| # | lever | where it fires | expected payoff | risk | sklearn? | cuML? |
|---|-------|----------------|-----------------|------|----------|-------|
| 1 ✅ | **factorize-once + numpy GATHER** (store encodings as `float64[code]`; transform = `enc[codes]`) — **shipped 2026-06-27** | transform (and fit lookups) | **measured ×2.3–3.4** (multi-stat / high-card; one `get_indexer` per *unit*, not per column; single-stat neutral) | Low–Med | yes | no |
| 2 | **bincount over integer (joint) codes** for the remaining slow-path stats; `joint = code_a*n_b+code_b` | fit accumulation; **joint keys** | 5–30× at large N; integer joint codes also unblock **GPU combination (KI-018)** | Med | yes (scatter-add) | no |
| 3 | **pre-allocated output, no merge at apply-back** | transform/apply | 5–15× | Low | yes | no |
| 4 | **dtype discipline** — int32 codes, C-contiguous, float64 throughout | both, large N | 5–20% | Very low | yes | partial |
| 5 | smoothing folded into accumulator init | fit smoothing | <5% (catstat's vectorized smoothing is already ~free) | trivial | yes | no |
| 6 | column-level parallelism (joblib over independent units) | both | up to n_cores | Med | no | inherent |

## Recommendation for catstat

The single highest-leverage intervention is **#1 + #2 together**: a factorize-once, integer-code
**gather** path. Concretely —
- **`_transform_array`**: at fit, store each unit's encoding as a `float64` array indexed by the
  unit's integer codes (keep the object→code mapping built from `categories_`); at transform,
  `pd.factorize`/`searchsorted` the keys once and gather (`enc[codes]`, unknown = code −1 → fallback),
  replacing `pd.Series.map`. This speeds **every** `transform`/inference call, not just `fit_transform`.
- **joint codes**: `code_a * n_b + code_b` (int64) replaces object tuple keys for combination/
  `interactions` — removing the per-row tuple build (KI-019's residue) and giving cuDF an integer
  column to group on, which **unblocks GPU `combination`** (KI-018).

What to measure (in-process before/after, the attributable method): the **transform** step alone at
n ≥ 1M, single- and multi-column, var the cardinality; expect 10–50× on that step. Watch the small-N
break-even (pandas has low overhead; bincount/gather wins grow with N). Tracked as **KI-031**; this
is the next CPU arc after PR-C, ahead of the GPU on-device port (KI-020).

## Status — lever #1 shipped (2026-06-27)

Lever #1 landed on `feat/perf-integer-code-gather` (stacked on `feat/perf-additive-var-std`):
`_transform_array` factorizes each unit's keys once (`index.get_indexer`) and gathers each column from
a `float64` array aligned to a per-unit canonical index (`_UnitEncoding`), replacing per-column
`pd.Series.map`. Measured (in-process interleaved, n=1M, 7 reps): transform **×2.28** (4-stat),
**×3.36** (4-stat high-card 50k), **×2.48** (combination), **×1.00** single-stat (no-unknown fast path
= a single fancy index). The 10–50× estimate above assumed `get_indexer` could be eliminated entirely;
in practice the gather still pays **one** `get_indexer` per unit to locate *arbitrary* transform keys,
so the real win is "a unit's N stats share one hash" (≈2–3× at 4 stats) plus dropping the pandas
`Series.map` overhead. Outputs allclose; leakage + sklearn-compat PASS;
`docs/verdicts/2026-06-27-transform-gather-verdict.md`. **Lever #2 (integer joint codes → vectorized
combination key-build, KI-019 + GPU combination KI-018) is the next, separate PR.**
