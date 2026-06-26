# `catstat` — Evaluation Harness Design

> Status: **proposal**. Covers design sections 12 (harness), 17 (testing plan), 18 (benchmark
> plan). The harness is a **first-class part of the library**, built to power the
> [self-improvement loop](./self-improvement-loop-design.md), not just one-off testing.

## 0. Principles
- **Correctness before speed.** A faster encoder that leaks is worthless. Leakage and parity
  tests gate everything.
- **Reproducible.** Pinned seeds, recorded package versions + git SHA, deterministic datasets.
- **Comparable over time.** Results persist as JSON/JSONL with a stable schema so any future
  session can diff "now" vs "baseline" mechanically (`compare_results.py`).
- **Honest about noise.** ≥5 reps, report median + spread; never change a default on one run.
- **CPU-local, GPU-on-Colab.** GPU/parity work runs only via the Colab loop (no local GPU).

## 1. File layout
```
benchmarks/
  README.md                 # how to run; schema; acceptance thresholds
  datasets.py               # synthetic generators (the stress cases)  ← FIRST FILE
  run_benchmarks.py         # CLI: --size {small,medium,large} --backend {cpu,gpu} --reps N --out
  ledger.py                 # append-only JSONL results ledger (schema + writer)
  compare_results.py        # diff current run vs a baseline JSON → regressions/improvements
  results/
    .gitkeep
    baseline-cpu.json       # committed CPU baseline (updated only via a verdict)
tests/
  conftest.py               # fixtures; `gpu` marker (auto-skip when cudf/cupy/GPU absent)
  test_target_encoder_regression.py
  test_target_encoder_binary.py
  test_target_encoder_multiclass.py
  test_cross_fit_no_leakage.py
  test_unknown_missing.py
  test_feature_names.py
  test_determinism.py
  test_count_frequency.py
  test_sklearn_compat.py
  test_io_types.py
  test_stats.py             # per-stat correctness + fallback (grows with Phase 2/3 stats)
  test_cpu_gpu_parity.py    # @pytest.mark.gpu — runs on Colab only
scripts/
  check.sh                  # green gate: ruff + pytest + run examples (OMP_NUM_THREADS=1)
  run_quality_gate.py       # optional richer gate (coverage floor, perf smoke)
  summarize_benchmark_results.py
  colab_gpu_parity.sh       # Phase 2: provision T4 → run parity+bench → pull artifacts
  colab_gpu_parity.py       # Phase 2: the on-VM entrypoint
docs/verdicts/
  .gitkeep
  TEMPLATE-verdict.md
```

## 2. Synthetic dataset generators (`benchmarks/datasets.py`)
Each generator is **seeded** and returns `(X, y, meta)` with a known ground truth so tests can
assert exact encodings. Stress dimensions (one generator may combine several):

| generator | stresses |
|---|---|
| `make_high_cardinality(n, n_levels)` | high cardinality (n_levels ~ n/2) |
| `make_rare_categories(n, rare_frac)` | many singleton/`n<min_samples` levels |
| `make_with_missing(n, nan_frac)` | NaN as a category; `handle_missing` paths |
| `make_unseen_split(n)` | returns train/test where test has categories absent from train |
| `make_regression(n, signal)` | continuous target with controllable category→target signal |
| `make_binary(n, pos_rate)` | binary, incl. **imbalanced** (`pos_rate` ∈ {0.5, 0.05}) |
| `make_multiclass(n, K, imbalance)` | multiclass, many classes, imbalanced |
| `make_multi_column(n, n_cols, interaction)` | several categoricals + an **interaction** that only `combination` mode captures |
| `make_leakage_trap(n, n_levels)` | high-cardinality column that is **pure noise** vs `y` (must encode to ≈ global prior out-of-fold) |
| `make_mixed_dtypes(n)` | object / pandas `category` / int categorical columns together |

`meta` carries the analytically-known per-category statistics so correctness tests compare
against hand-computed truth (not just another implementation).

## 3. Correctness checks (sections 17)
- **Per-stat correctness** (`test_stats`, the per-encoder tests): encodings equal hand-computed
  group statistics on a tiny seeded frame (mean, count, frequency now; var/std/median/… in P2).
- **Smoothing:** fixed `smooth=m` matches the m-estimate formula exactly; `smooth=0` = raw means;
  `smooth="auto"` matches the empirical-Bayes formula (and, when `sklearn>=1.4` is installed,
  sklearn's output within tolerance).
- **Missing/unseen** (`test_unknown_missing`): each `handle_*` value × each stat hits the §11
  fallback table (count→0, frequency→0.0, mean→global, dispersion/order→global, error raises,
  return_nan returns NaN).
- **Feature names** (`test_feature_names`): single / multi-column / multiclass / multi-stat /
  custom-combiner naming; `get_feature_names_out` length == output width; round-trips with
  `set_output("pandas")`.
- **I/O types** (`test_io_types`): pandas DataFrame and numpy array inputs; object/category/int
  dtypes; `output ∈ {auto,pandas,numpy}` returns the right container with the right names.

## 4. Leakage checks (the crown jewels — `test_cross_fit_no_leakage`)
1. **OOF reconstruction (exact):** for each fold, independently recompute the encoding from the
   fold's **complement** and assert it equals the value `fit_transform` produced for that fold's
   rows. This *is* the definition of no-leakage; it must hold bit-for-bit on CPU.
2. **Noise-trap:** on `make_leakage_trap` (category independent of `y`), the `fit_transform`
   encoding of a held-out row must be ≈ the global prior (within fold-size sampling noise), i.e.
   it carries no row-specific target signal. A downstream model on the OOF feature shows **no
   train/holdout gap** beyond chance.
3. **Asymmetry:** when there *is* signal, `fit_transform(X,y) ≠ fit(X,y).transform(X)` on the
   training set (the latter is the leaky path); assert they differ and that the leaky path
   over-fits (higher train correlation with `y`).
4. **Unsupervised:** `Count/FrequencyEncoder` satisfy `fit_transform == fit().transform()`.

## 5. Parity checks
- **sklearn parity** (`@pytest.mark.skipif sklearn<1.4`): for `stats=["mean"]`, regression +
  binary + multiclass, `catstat` matches `sklearn.preprocessing.TargetEncoder` within
  `atol=1e-10` for `transform`, and for `fit_transform` **when the same CV splitter is injected**
  (since fold assignment drives OOF values). Document where we intentionally differ.
- **category_encoders parity** (optional, gated on install): match `MEstimateEncoder`
  (clean m-estimate) for fixed smooth on `transform`; note that we do **not** match their
  no-cross-fit `fit_transform`.
- **CPU/GPU parity** (`test_cpu_gpu_parity.py`, `@pytest.mark.gpu`): same `random_state` ⇒ same
  folds ⇒ `transform` exact and `fit_transform` **allclose** (`rtol=1e-5`, not bitwise — GPU
  reduction reordering). Runs only on Colab.

## 6. Performance checks (section 18)
Measured separately (never just end-to-end), ≥5 reps, median + spread:
- **`fit` / `transform` / `fit_transform` time** (each its own number).
- **Peak memory** (`tracemalloc` CPU; device-mem counter GPU).
- **Conversion overhead:** pandas→cuDF (and back) time vs group-by time — the number that sets
  the `backend="auto"` cell threshold.
- **CPU vs GPU crossover:** sweep `n_rows × cardinality`; record where GPU (incl. conversion)
  beats CPU. Feeds the `auto` predicate.
- **Sizes:** small (1e4 rows), medium (1e6), large (1e7+, GPU/Colab).
- **Shapes:** narrow (1 high-card column) and wide (many columns); multiclass (large `K`).

## 7. Result persistence & comparison
- **Ledger (`ledger.py`):** every run appends one JSON object per case to a JSONL file with a
  stable schema:
  ```json
  {"ts": "...", "git_sha": "...", "backend": "cpu", "case": "regression_highcard_medium",
   "stat": "mean", "n_rows": 1000000, "cardinality": 50000,
   "fit_s": {"median": ..., "spread": ...}, "transform_s": {...}, "fit_transform_s": {...},
   "peak_mem_mb": ..., "quality": {"oof_rmse": ...},
   "versions": {"catstat": "...", "numpy": "...", "pandas": "...", "sklearn": "..."}}
  ```
- **Baseline (`results/baseline-cpu.json`):** committed; **updated only via a verdict doc**.
- **`compare_results.py current.jsonl baseline.json`:** prints a table of per-case deltas, flags
  **regressions** (slower beyond threshold, or any correctness/quality change) and **improvements**,
  and exits non-zero on a regression so the loop/CI can gate on it.
- **`summarize_benchmark_results.py`:** human-readable rollup for a verdict doc.

## 8. Acceptance thresholds (initial; tune via verdicts)
- **Correctness/leakage/parity:** must pass; zero tolerance (allclose tolerances as above).
- **Coverage:** core floor (e.g. 85%, mirroring repleafgbm), optional-dep paths omitted.
- **Performance regression gate:** > **15%** median slowdown on any committed case fails the gate
  (noise-aware: require it to persist across reps). Improvements > 10% are verdict-worthy.
- **Memory:** > 20% peak-memory increase on a committed case fails the gate.

## 9. CPU/GPU comparison via Colab (Phase 2)
Mirrors repleafgbm's loop exactly (`scripts/colab_*.sh` + `.py`, the `colab` CLI):
1. `tar` the working tree (exclude `.git`, caches).
2. `colab new -s catstat-gpu --gpu T4`.
3. `colab upload` the tarball to `/content`.
4. `colab exec -f scripts/colab_gpu_parity.py` — on the VM: pip-install
   `cudf-cu12 cuml-cu12 cupy-cuda12x` (RAPIDS Colab install is heavier/slower than torch — keep
   the job minimal and watchdogged), run `test_cpu_gpu_parity.py` + the GPU benchmark cases,
   write `parity.jsonl` + a verdict markdown.
5. `colab download` artifacts → `benchmarks/results/` + `docs/verdicts/`.
6. `colab stop` (trap-on-exit; `--keep` to iterate). External **watchdog** kills a hung
   `colab exec` if the websocket drops.

## 10. Verdict auto-generation
After a benchmark run, `summarize_benchmark_results.py` + `compare_results.py` emit the body of a
`docs/verdicts/YYYY-MM-DD-<topic>-verdict.md` from `TEMPLATE-verdict.md`: question, evidence
(tables of before/after with median+spread), parity status, and a keep/change/revert decision.
This is what closes each iteration of the self-improvement loop.
