# `catstat` Roadmap

This roadmap is **honest about status**. As of **2026-06-26** **0.1.1 is released on PyPI**: M0 +
Phase 2 (GPU validated; auto-GPU disabled pending perf) + Phase 3 (skew/custom stats, loo/ordered
schemes, polars output), CI green on Python 3.10–3.12 / pandas 1.5–3.0. **0.2.0 adds opt-in,
cardinality-aware numeric-column target encoding** (implemented + green on `scripts/check.sh`,
verdict-backed), pending the maintainer's `v0.2.0` tag. Publishing is tag-driven (Trusted Publishing).

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
- ✅ combination joint-key build **vectorized** to int64 mixed-radix codes (KI-019, 2026-06-27:
  byte-identical, transform ×3.7–4.4 / fit_transform ×1.5–2.4 at 1M; supersedes PR #2). ⏳ GPU
  `combination` still host-only (KI-018) — now keyed by integers, ready to unblock on-device next.
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
- ✅ **README polish (0.1.1)**: honest status (replaced the stale "M0 alpha — CPU-only" marker),
  CI/PyPI/Python/license badges, install + extras, a statistics/feature table, a leakage-safe note,
  and the API-docs link. `twine check` confirms it renders as the PyPI long-description.
- ✅ **API docs (0.1.1)**: `scripts/build_docs.sh` (pdoc) + `.github/workflows/docs.yml` (GitHub
  Pages). Build verified locally (index/catstat HTML, no import errors). Maintainer enables Pages once.
- ✅ **sklearn tags (0.1.1)**: `__sklearn_tags__` (≥1.6) + `_more_tags` (<1.6) on the base encoder
  — categorical/string/`allow_nan` + `requires_y` (supervised). Verified on sklearn 1.9 in a venv.
- ✅ **CI green (0.1.1)**: added pytest `pythonpath=["src", "."]` so bare `pytest tests/` can import
  `tests.conftest` (KI-021). CI had been red since before this arc (a collection error, not pandas).
- ✅ **pandas 3.0 (0.1.1)**: `cols="auto"` now selects pandas `StringDtype` columns (KI-022). Full
  suite green on sklearn 1.9 / pandas 3.0.3 in a venv; CI should now pass end-to-end.
- ✅ **sklearn-compat hardening (0.1.1)**: documented `check_estimator` subset
  (`tests/test_check_estimator.py`, sklearn ≥ 1.6) — applicable checks pass, inapplicable ones waived
  with reasons (KI-012); fixed estimator pickling (cached backend module → `__getstate__`/`__setstate__`).
- ✅ **Project hygiene (0.1.1)**: `CONTRIBUTING.md`, `SECURITY.md`, GitHub issue + PR templates.
- ✅ **0.1.1 PUBLISHED (2026-06-26)**: `v0.1.1` tagged → release workflow built + published to PyPI
  via Trusted Publishing; `pip install catstat==0.1.1` verified in a clean venv; GitHub release created.
- ⏳ **Maintainer-only:** enable GitHub Pages (Settings → Pages → GitHub Actions) so the Docs
  workflow deploys the API site. (KI-020 GPU perf is the optional larger follow-up.)

## 0.2.0 — numeric-column target encoding — done ✅ (2026-06-26)
- ✅ **Opt-in numeric TE** on `TargetEncoder` (`numeric="ignore"|"auto"|"direct"|"bin"` +
  `cardinality_threshold`/`n_bins`/`binning`). Low-cardinality numerics → direct (each value a
  category); high-cardinality → quantile-binned then target-encoded; `"auto"` routes by cardinality.
  Default `"ignore"` keeps existing behavior. `src/catstat/_numeric.py` (host-side numpy) + one seam
  at `_unit_keys`; all smoothing / fallback / feature-name / parity logic reused unchanged.
- ✅ **Leakage-safe**: bin edges from X only (proven ⊥ y), computed once from full train; per-bin
  encoding cross-fitted OOF. Binned OOF reconstruction exact; noise-trap OOF corr ≈ 0.07. 27 tests,
  `_numeric.py` 100% covered.
- ✅ **Empirically validated**: downstream Ridge 5-fold CV R² **0.034 (raw) → 0.91 (auto, n_bins=10)**;
  defaults `n_bins=10` / `cardinality_threshold=10` set by verdict. `benchmarks/eval_numeric.py`,
  `docs/verdicts/2026-06-26-numeric-te-verdict.md`, `docs/notes/2026-06-26-numeric-te-prior-art.md`.
- ✅ GPU parity for binned/direct numeric **validated on T4 (2026-06-26)**: `numeric_auto`/`numeric_bin`
  CPU/GPU allclose (max|Δ| ~1e-17). First run hit `MixedTypeError` (cuDF rejects object-dtype int keys)
  → fixed by emitting **string** keys.
- ⏳ Numeric binning for `Count`/`Frequency` (KI-030).

## Recommended implementation order (PR-sized)
- ✅ **PR1–PR9** (packaging → validation/stats → CPU backend → mean encoder → binary/multiclass →
  unknown/missing + names → count/frequency → sklearn compat → harness) landed together in the
  **M0 bootstrap (2026-06-26)**.
- ✅ **Phase 2 (CPU + GPU validated)** 2026-06-26: var/std/median/min/max, combination mode,
  GPU backend `backends/_gpu.py` **validated CPU/GPU-allclose on a Colab T4**, CI, Colab loop, `git`.
- **Perf arc (2026-06-26→27, profiling-driven).** ✅ CPU OOF is single-pass via complement
  subtraction: `kfold_mean_oof_fast` replaced the per-fold group-by for pure-mean (2.2–3.4×;
  `docs/verdicts/2026-06-26-pr-b-complement-subtraction-mean-verdict.md`). ✅ **var/std** now ride the
  same kernel (shared complement moments; ddof=1 + complement-global fallback) with a **hybrid** gate
  that keeps median/min/max/skew/custom on the slow loop — 2.7–2.8× on var & mean+var+std, ~1.5× mixed
  (`docs/verdicts/2026-06-27-pr-c-additive-var-std-verdict.md`). The kernel also ports on-device to
  remove per-fold host↔device round-trips (KI-020). ✅ **Transform gather** (2026-06-27): factorize-once
  `index.get_indexer` + numpy gather replaced per-column `pd.Series.map` — transform ×2.3–3.4
  (multi-stat / high-card), single-stat neutral (`docs/verdicts/2026-06-27-transform-gather-verdict.md`,
  KI-031). ✅ **Integer joint codes** (2026-06-27): combination key-build replaced by mixed-radix
  int64 codes — byte-identical, transform ×3.7–4.4 / fit_transform ×1.5–2.4 at 1M, closes KI-019
  (supersedes PR #2); GPU `combination` (KI-018) is the remaining lever.
- **Phase 2 — remaining.** GPU *performance* (on-device keys/folds; KI-020) and `combination` on
  GPU (KI-018) — both optional, gated behind a fresh crossover verdict before re-enabling `auto`.
- **Phase 3.** quantile/skew/custom + ordered/LOO + `set_output("polars")` + PyPI release.

## "Next" pointer (update each session)
> **Next task:** **Integer joint codes done (2026-06-27, lever #2A)** — combination key-build replaced
> by vectorized mixed-radix int64 codes (learned once from full X, reused at fit/fold/transform);
> byte-identical output, combination transform ×3.7–4.4 / fit_transform ×1.5–2.4 at 1M, closes KI-019
> and supersedes PR #2; leakage + sklearn-compat PASS; branch `feat/perf-integer-joint-codes` (stacked
> on `feat/perf-integer-code-gather`). **Next, in order:** (1) **lever #2B — GPU `combination`**: drop
> `len(cols)>1` from `host_only` + build joint codes in `_gpu.py`, with **mandatory** Colab CPU/GPU
> parity (KI-018); (2) **PR-D** GPU on-device kernel + a fresh Colab crossover before re-enabling
> `auto` (KI-020). Also pending on a separate branch: **interactions** `interactions: list[list[str]]`.
> Maintainer-only carryover: tag `v0.2.0` to publish; enable GitHub Pages; optional numeric binning for
> `Count`/`Frequency` (KI-030); reconcile PR #2 (superseded).
