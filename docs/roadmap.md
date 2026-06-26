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
- ⏳ **Colab validation** of the GPU path (strings/nulls/missing on device) + GPU baselines — run
  `scripts/colab_gpu_parity.sh`. Until then `backend="gpu"` is written-but-unverified.
- ⏳ Conversion-overhead benchmark → calibrate the `auto` cell threshold (needs GPU).
- ⏳ Vectorize combination joint-key construction (currently a Python loop).

## Phase 3 — advanced — planned
- `quantile`, `skew`, custom-callable aggregations (CPU-only; order-independence required).
- Ordered (CatBoost-style) and leave-one-out encoding modes.
- `set_output("polars")`, advanced metadata routing, estimator-check hardening.
- PyPI release + API docs; self-improvement-loop hardening.

## Recommended implementation order (PR-sized)
- ✅ **PR1–PR9** (packaging → validation/stats → CPU backend → mean encoder → binary/multiclass →
  unknown/missing + names → count/frequency → sklearn compat → harness) landed together in the
  **M0 bootstrap (2026-06-26)**.
- ✅ **Phase 2 (CPU + scaffold)** landed 2026-06-26: var/std/median/min/max, combination mode,
  GPU backend (`backends/_gpu.py`), CI workflow, Colab parity scripts, `git init`.
- **Phase 2 — remaining.** Run `scripts/colab_gpu_parity.sh` to validate/harden the GPU path on a
  T4 (strings/nulls/missing-as-value on device), record GPU baselines, and calibrate the
  `backend="auto"` cell threshold from the conversion-overhead benchmark.
- **Phase 3.** quantile/skew/custom + ordered/LOO + `set_output("polars")` + PyPI release.

## "Next" pointer (update each session)
> **Next task:** Run `scripts/colab_gpu_parity.sh` on a Colab T4 to validate the GPU backend
> (CPU/GPU allclose) and commit the GPU baselines + the downloaded `docs/verdicts/` report. Then
> harden any device-side gaps it surfaces (cuDF string/null handling). After that, Phase 3.
