# `catstat` Roadmap

This roadmap is **honest about status**. As of **2026-06-26** the library is **release-ready at
0.1.0**: M0 + Phase 2 (GPU validated; auto-GPU disabled pending perf) + Phase 3 (skew/custom stats,
loo/ordered schemes, polars output) are implemented and green on `scripts/check.sh`; the package
builds and `twine check` passes. A release-polish arc → **0.1.1** is underway (release automation
done; README, API docs, sklearn-compat hardening, hygiene next); the PyPI upload + one-time
Trusted-Publisher setup remain the maintainer's.

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
- ✅ **Phase 3a**: `skew` (built-in) + **custom-callable aggregations** (`stats=[("q90", fn)]` or
  dict form; CPU-only, cross-fitted, continuous-only, global fallback). Quantiles/IQR via custom
  callables; `stats=["quantile"]` raises with a helpful hint. `test_phase3.py`.
- ✅ **Phase 3b**: `scheme="loo"` (leave-one-out) + `scheme="ordered"` (CatBoost-style) cross-fitting
  modes for the mean (default `"kfold"`). Leakage-safe, deterministic, mean-only. `test_scheme.py`.
- ✅ **Phase 3c**: `output="polars"` (returns a polars DataFrame; lazy import, optional dep).
  `test_polars.py`.
- ✅ **Release prep (0.1.0)**: `LICENSE`, `CHANGELOG.md`, version bump (pyproject + `__init__` in
  sync), `docs/publishing_checklist.md`, `release-prep` skill. **Build verified** — sdist + wheel
  build, `twine check` passes, clean-venv install imports on sklearn 1.9. Upload/tag = maintainer.
- ✅ **Release automation (0.1.1)**: `.github/workflows/release.yml` — a `v*` tag builds + publishes
  to PyPI via **Trusted Publishing** (OIDC, no token), guarded by a tag↔version check; checklist
  rewritten (automated path + manual fallback). Opened the 0.1.1 cycle (version bump + CHANGELOG).
- ⏳ README polish, API docs (pdoc + Pages), estimator-check hardening (KI-012), project hygiene;
  then the maintainer's one-time PyPI Trusted-Publisher setup + `v0.1.1` tag.

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
> **Next task:** release-polish arc → **0.1.1** is underway. ✅ Commit 1: release automation
> (`.github/workflows/release.yml`, tokenless Trusted Publishing) + opened 0.1.1. **Next:** README
> polish (badges, honest status, install, stat/feature table, API-docs link); then API docs
> (pdoc + GitHub Pages), sklearn estimator-check hardening (KI-012), and project-hygiene files.
> v0.1.0's PyPI upload + the one-time Trusted-Publisher config remain the maintainer's; the GPU
> on-device perf redesign (KI-020) is the larger optional follow-up.
