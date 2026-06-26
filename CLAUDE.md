# CLAUDE.md — `catstat` development rules

## Project summary

`catstat` is a unified, sklearn-compatible library for **statistical categorical encoding**:
target encoding generalized to an arbitrary set of statistics (mean, count, frequency, var, std,
median, min, max, quantile, skew, custom), with **leakage-safe** internal cross-fitting and **one
API that runs on CPU (pandas/numpy) or GPU (cuDF/CuPy)** via automatic backend selection. It is
*not* a reimplementation of sklearn or cuML — it is the union of what they each do, made
device-agnostic and statistically general.

Design lives in `docs/proposals/`. Status lives in `docs/roadmap.md`. **Status as of this
writing: design phase — no `src/` code yet.**

## Core invariants (do not break)

1. **Leakage safety.** `fit_transform(X, y)` is **out-of-fold** (encode each fold from its
   complement, then refit on full data for later `transform`). `fit(X, y).transform(X)` on the
   training set is the **leaky** path and is documented as such. `smooth="auto"` variance is
   computed **per fold**. `random_state`/`shuffle` flow through `check_random_state`; **never**
   call global numpy RNG.
2. **CPU/GPU parity.** `catstat` **owns its fold assignment** (deterministic integer fold-ids from
   `random_state`) so CPU and GPU produce the **same** OOF encodings. Parity is asserted at
   **allclose, not bitwise** (GPU reduction reordering). GPU tests run **only on Colab**; CI/macOS
   skip them.
3. **sklearn compatibility.** `BaseEstimator`/`TransformerMixin`; params stored verbatim in
   `__init__`; fitted attrs end in `_`; `get_feature_names_out`, `set_output`, and
   Pipeline/ColumnTransformer all work.

## Smoothing honesty rule

Only **mean/probability** statistics get principled smoothing (m-estimate fixed; empirical-Bayes
`auto`). **count/frequency** get none. **var/std** get optional, *clearly-labeled heuristic*
shrinkage (default off). **median/min/max/quantile/skew/custom never blend** — they fall back to
the **global** statistic when `n < min_samples_category`. Do not add smoothing to a statistic that
has no principled rule.

## Public API (stable shape — change only with instruction)

`TargetEncoder` (supervised, cross-fitted, `stats=[...]`), `CountEncoder` /
`FrequencyEncoder` (unsupervised), over a shared private `_BaseStatEncoder`. Key params:
`cols, stats, target_type, smooth, cv, scheme, shuffle, random_state, handle_unknown,
handle_missing, multi_feature_mode, min_samples_category, backend, output, numeric,
cardinality_threshold, n_bins, binning`. `stats` accepts built-ins + `(name, callable)` custom
aggregations; `scheme ∈ {kfold, loo, ordered}` selects how the mean is cross-fitted (loo/ordered are
mean-only); `numeric ∈ {ignore, auto, direct, bin}` (default `ignore`, TargetEncoder only) opts
numeric columns into direct/binned target encoding (edges from X only; bins target-encoded OOF). See
`docs/proposals/target-encoder-library-design.md` §3.

## Code map (target layout — build to this)

```
src/catstat/
  __init__.py, py.typed
  _base.py               # _BaseStatEncoder: fit/transform/fit_transform skeleton + dispatch
  target_encoder.py      # TargetEncoder (supervised)
  count_encoder.py       # CountEncoder (unsupervised)
  frequency_encoder.py   # FrequencyEncoder (= CountEncoder(normalize=True))
  _stats.py              # stat registry (agg, smoothing policy, class-expanded?, fallback)
  _smoothing.py          # m-estimate + empirical-Bayes (mean/prob only)
  _cross_fit.py          # deterministic folds (CPU==GPU) + OOF orchestration
  _numeric.py            # numeric-col encoding: cardinality routing + X-only quantile bin edges
  _validation.py, _feature_names.py, _typing.py
  backends/
    _dispatch.py         # backend="auto" predicate + 4-primitive interface
    _cpu.py              # pandas/numpy primitives
    _gpu.py              # cudf/cupy primitives (Phase 2, import-guarded)
```

**The boundary that matters:** all statistics/leakage logic is backend-agnostic; only the four
primitives in `backends/` (`groupby_agg`, `assign_folds`, `merge_encodings`, `to_output`) know
pandas vs cuDF. RAPIDS is isolated to `backends/_gpu.py`; the CPU path never imports it.

## Backend selection

`backend="auto"` currently resolves to **CPU always** (`_AUTO_GPU_ENABLED=False` in
`backends/_dispatch.py`): the Colab T4 crossover (2026-06-26) showed the host-orchestrated GPU path
is *slower* than CPU up to 1M rows, so auto must not pick it. Explicit `backend="gpu"` is
**validated** (CPU/GPU allclose, incl. missing) and available for device-resident pipelines / much
larger data; with RAPIDS/GPU missing it **raises** (no silent fallback). `backend_` records the
actual engine. Re-enable auto-GPU + calibrate `_GPU_CELL_THRESHOLD` only after the device path is
optimized (keys/folds on-device) and a fresh crossover verdict supports it.

## Testing — one command

- Green gate: `bash scripts/check.sh` (ruff + pytest + runnable examples, `OMP_NUM_THREADS=1`).
- Targeted: `PYTHONPATH=src python3 -m pytest tests/ -q` (or a single file/node).
- GPU/parity (`@pytest.mark.gpu`): **skipped locally**; run via `scripts/colab_gpu_parity.sh`.
- sklearn-parity tests require `scikit-learn>=1.4` (skipped otherwise).
- Keep synthetic datasets small and seeded. Required coverage for new behavior: encode
  correctness, **OOF/no-leakage**, unknown/missing fallback, feature names, determinism.

## Skills (`.claude/skills/`)

| skill | when to invoke |
|---|---|
| `leakage-audit` | any change to `_cross_fit`/`_smoothing`/transform path — prove OOF holds |
| `sklearn-compat` | any public-API/feature-name/output change — verify estimator protocol |
| `benchmark-harness` | perf-relevant change — run, persist, compare to baseline, draft verdict |

Deferred (build when the work exists): `cpu-gpu-parity` (P2), `target-encoder-research`,
`release-prep` (P3). See `docs/proposals/skills-proposal.md`.

## Workflows

- **Implementation loop:** read `docs/roadmap.md` → implement one PR-sized change + tests →
  `scripts/check.sh` green → (`leakage-audit`/`sklearn-compat` as relevant) → update roadmap.
- **Benchmark loop:** baseline → change → `run_benchmarks.py` → `compare_results.py` → verdict in
  `docs/verdicts/` → update baseline only if the verdict says so.
- **Self-improvement loop** (read order): `CLAUDE.md` → `docs/roadmap.md` → `docs/known_issues.md`
  → `docs/experiment_log.md` → newest `docs/verdicts/*` → `benchmarks/results/baseline-*.json`.
  Details: `docs/proposals/self-improvement-loop-design.md`.
- **Release loop (P3):** green gate → version/CHANGELOG → docs → PyPI checklist (`release-prep`).

## When to use subagents
- A research scout *before* implementing a new method/encoder (→ a dated note, not code).
- A runner for long/detached multi-seed benchmark runs (keeps the main context free).
- An analyst to turn a finished run into a verdict. Fan out independent benchmarks in parallel.

## When NOT to use subagents
- Small local edits where you already hold the context (most MVP PRs) — do it inline.
- Anything needing the current in-context state of files you've read (a cold agent re-derives it).
- Final leakage judgment — that stays with the main session + `leakage-audit`.

## Context-saving rules
- Read `docs/proposals/*` and `docs/roadmap.md` first; don't re-browse external libraries for
  facts already captured there.
- Pass file paths to subagents, not pasted dumps. Keep each agent's scope tight.
- End every change with a one-paragraph summary + the single suggested next task.

## Do not change without explicit instruction
- The three **core invariants**, the **smoothing honesty rule**, and the **public API shape**.
- Default smoothing (`"auto"`), default `cv`, the `backend="auto"` thresholds, and the
  unknown/missing fallback table (`docs/proposals/target-encoder-library-design.md` §11).
- The committed benchmark baselines (update only via a verdict).
- This file's structure — **patch** the relevant section, don't rewrite wholesale.

## Documentation
- Update `docs/roadmap.md` when capabilities change (keep it honest: done vs planned).
- ADRs in `docs/adr/` for non-obvious decisions; verdicts in `docs/verdicts/` for default changes.
- Document known limitations in `docs/known_issues.md`; log every experiment (incl. nulls) in
  `docs/experiment_log.md`.

## Avoid
- A separate `CategoryStatEncoder`/`GeneralizedTargetEncoder` class (generality lives in `stats=`).
- Smoothing a statistic that has no principled rule.
- Importing `cudf`/`cupy` anywhere outside `backends/_gpu.py`.
- Silent host↔device conversions; changing a default on a single benchmark run.
- A giant monolithic encoder class; dataset-specific hacks; hidden global state.
