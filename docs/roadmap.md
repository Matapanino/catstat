# `catstat` Roadmap

This roadmap is **honest about status**. **M0 (MVP) shipped 2026-06-26**: the CPU library, tests,
and benchmark harness exist and are green on `scripts/check.sh`. Phase 2 (GPU) and Phase 3 remain
plans.

> `MVP / Phase 2 / Phase 3` are **capability tiers**, not package versions.

## Design phase — done ✅
- Library design (`docs/proposals/target-encoder-library-design.md`).
- Evaluation harness design (`docs/proposals/evaluation-harness-design.md`).
- Self-improvement loop design (`docs/proposals/self-improvement-loop-design.md`).
- CLAUDE.md + first 3 skills (`leakage-audit`, `sklearn-compat`, `benchmark-harness`).
- Roadmap / known_issues / experiment_log / verdict template scaffolding.

## MVP (M0) — CPU, mean, leakage-safe — done ✅ (2026-06-26)
- ✅ `pyproject.toml` (src-layout, hatchling, extras `dev`/`bench`/`gpu`/`docs`, ruff len 100,
  coverage gate 85%), `scripts/check.sh`.
- ✅ `TargetEncoder` with `stats=["mean"]`: regression, binary, multiclass (OvR).
- ✅ `smooth` fixed (m-estimate) **and** `smooth="auto"` (empirical-Bayes), computed per fold.
- ✅ Leakage-safe `fit_transform` (out-of-fold) vs `transform`; `KFold`/`StratifiedKFold`;
  determinism. OOF reconstruction exact (`max |Δ|=0.0`).
- ✅ `CountEncoder`, `FrequencyEncoder` (unsupervised, no cross-fit).
- ✅ `handle_unknown`/`handle_missing` ∈ {value, return_nan, error} with the §11 fallback table.
- ✅ `get_feature_names_out`, `set_output`, Pipeline/ColumnTransformer; pandas + numpy I/O.
- ✅ Harness: `benchmarks/datasets.py`, `run_benchmarks.py`, `ledger.py`, `compare_results.py`;
  committed `results/baseline-cpu.json`; verdict `docs/verdicts/2026-06-26-m0-bootstrap-verdict.md`.
- ✅ 46 tests pass / 1 GPU-skipped; coverage 85.87%.
- ⏳ CI workflow (GitHub Actions) — not yet added.

## Phase 2 — GPU + more statistics — in progress (2026-06-26)
- ✅ `var`, `std`, `median`, `min`, `max` (continuous-target only; cross-fitted; no blending,
  small-n → global fallback). `test_stats.py`.
- ✅ `multi_feature_mode="combination"` (joint group-by → one column). `test_multi_feature.py`.
- ✅ GPU backend `backends/_gpu.py` (cuDF/CuPy group-by, host-orchestrated); selectable backend
  threaded through `_smoothing`/`_aggregations`; `backend="auto"` predicate + loud `gpu` errors.
- ✅ CI workflow (`.github/workflows/ci.yml`); coverage 88.17%.
- ✅ `scripts/colab_gpu_parity.{sh,py}`; `test_cpu_gpu_parity.py` (allclose, gpu-marked).
- ✅ **Colab validation (T4, 2026-06-26)**: CPU/GPU allclose for mean/var × reg/bin/mc **+
  missing-as-value** (cuDF nulls), transform + fit_transform. Two verdicts (parity + crossover).
- ✅ **Crossover measured**: GPU is *slower* than CPU up to 1M rows (speedup 0.28–0.86) →
  `backend="auto"` GPU **disabled** (`_AUTO_GPU_ENABLED=False`); explicit `backend="gpu"` stays. KI-020.
- ⏳ `combination` on GPU (tuple keys, host-only) + vectorize joint-key build (KI-018/019).
- ⏳ **GPU perf**: keep keys/folds on-device to remove the per-fold host↔device round-trips that
  dominate; then re-run the crossover and re-enable `auto` if it wins.

## Phase 3 — advanced — in progress (2026-06-26)
- ✅ `skew` (built-in) + **custom-callable aggregations** (`stats=[("q90", fn)]` or dict form;
  CPU-only, cross-fitted, continuous-only, global fallback). Quantiles/IQR via custom callables;
  `stats=["quantile"]` raises with a helpful hint. `test_phase3.py`.
- ⏳ Ordered (CatBoost-style) and leave-one-out encoding modes.
- ⏳ `set_output("polars")`, advanced metadata routing, estimator-check hardening.
- ⏳ PyPI release + API docs; self-improvement-loop hardening.

## Recommended implementation order (PR-sized)
- ✅ **PR1–PR9** (packaging → validation/stats → CPU backend → mean encoder → binary/multiclass →
  unknown/missing + names → count/frequency → sklearn compat → harness) landed together in the
  **M0 bootstrap (2026-06-26)**.
- ✅ **Phase 2 (CPU + GPU validated)** 2026-06-26: var/std/median/min/max, combination mode,
  GPU backend `backends/_gpu.py` **validated CPU/GPU-allclose on a Colab T4**, CI, Colab loop, `git`.
- **Phase 2 — remaining.** GPU *performance* (on-device keys/folds; KI-020) and `combination` on
  GPU (KI-018) — both optional, gated behind a fresh crossover verdict before re-enabling `auto`.
- **Phase 3.** quantile/skew/custom + ordered/LOO + `set_output("polars")` + PyPI release.

## "Next" pointer (update each session)
> **Next task:** Phase 3 continues — **CatBoost-style ordered** target statistics and a
> **leave-one-out** mode (both opt-in, leakage-safe alternatives to k-fold OOF), then
> `set_output("polars")` and a PyPI release. (Phase 2 GPU perf parked under KI-020.)
