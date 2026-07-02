# `catstat` Roadmap

This roadmap is **honest about status**. As of **2026-06-27** **0.4.0 is released on PyPI**: M0 +
Phase 2 (GPU validated; auto-GPU disabled pending perf) + Phase 3 (skew/custom stats, loo/ordered
schemes, polars output) + the **numeric-encoding arc** — cardinality-aware numeric *target* encoding
(0.2.0), then `Count`/`Frequency` binning + explicit/per-column bin edges + `min_bin_size` (0.4.0).
CI green on Python 3.10–3.12 / pandas 1.5–3.0. Publishing is tag-driven (Trusted Publishing).

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
  byte-identical, transform ×3.7–4.4 / fit_transform ×1.5–2.4 at 1M; supersedes PR #2). ✅ GPU
  `combination`/`interactions` now run on GPU too (host-built int64 codes → device group-by;
  `host_only = not all_gpu`); **CPU/GPU allclose validated on Colab T4** (KI-018 resolved).
- **GPU perf** (re-measured 2026-06-26, T4): host complement-subtraction (PR-B) removed the per-fold
  round-trip → crossover ~parity at ≥5M (0.67×@1M, 1.11×@5M, 1.06×@10M; marginal + noisy). `auto`
  **stays off** (data doesn't justify it); a device-resident path is a niche lever.
  `docs/verdicts/2026-06-26-gpu-crossover-postPRB-verdict.md`.

## Phase 3 — advanced — in progress (2026-06-26)
- ✅ **Phase 3a**: `skew` (built-in) + **custom-callable aggregations** (`stats=[("q90", fn)]` or
  dict form; CPU-only, cross-fitted, continuous-only, global fallback). Quantiles/IQR via custom
  callables; `stats=["quantile"]` raises with a helpful hint. `test_phase3.py`.
- ✅ **Stats arc 1 (2026-07-02)**: `kurt` (built-in, excess kurtosis) + `skew`/`kurt` reworked to
  **power-sum moments** (`category_moments` on both backends; global-mean shift for numerical
  stability) — pandas-matching (adjusted G1 / bias-corrected G2, `n<3`/`n<4` → global fallback,
  constant category → 0.0) and now **GPU-supported** (skew no longer forces the CPU backend).
  `test_shape_stats.py`.
- ✅ **Stats arc 2 (2026-07-02)**: skew/kurt joined the **single-pass additive OOF kernel**
  (`_ADDITIVE_STATS`): the shared `(fold, key)` pass upgrades to order-4 shifted power sums when a
  shape stat is requested (`finalize_shape_oof`; per-stat min-n fallback via `_STAT_MIN_N`), so
  skew/kurt no longer force the per-fold slow loop. fast==slow at allclose across the fallback
  matrix incl. a 1e9-offset case; leakage audit re-passed. `test_additive_fast_path.py`.
- ✅ **B0 — (fold × cat) table OOF kernel (2026-07-02)**: OOF finalizers now build small
  `(n_folds × n_cat)` value tables from an injectable `moment_tables` kernel and scatter with one
  gather — value-identical (max|Δ| ≤ 1.2e-14 vs per-row; leakage audit re-PASS), CPU
  neutral-to-modest (×1.02–1.21 interleaved), and the seam the PR-D device kernel plugs into.
  `docs/verdicts/2026-07-02-b0-table-oof-kernel-verdict.md`.
- ✅ **B3 — device transform + cuDF output (2026-07-02, validated on T4)**: `output="cudf"` is a
  first-class container (device input returns cuDF by default; host input + `output='cudf'` does
  one explicit H2D); sklearn `set_output('pandas'/'polars')` wins over the `output` param;
  `transform` accepts cuDF input on any fitted encoder (device hash-join codes against the
  fit-time host index + device gather + on-device unknown/missing policy).
- ✅ **B4 — device order-stat OOF (2026-07-02, validated on T4)**: median/min/max fit + per-fold
  OOF on device-resident codes/y/fold-ids — only per-code tables and one scalar per fold leave
  the GPU, removing the per-fold row transfers that dominated KI-020's non-additive numbers.
  **T4: full suite 358 passed** incl. the whole device matrix.
- ✅ **B2 — cuDF input, device-resident (2026-07-02, validated on Colab T4)**: `fit`/`fit_transform`
  accept a cuDF DataFrame (+ cupy/cudf/numpy y) and keep it on device end-to-end: cudf factorize
  with a MISSING-level remap mirroring `normalize_keys`, device mixed-radix joint codes densified
  on device (host `_JointKeyPlan` -> pandas-input transform reuses the host machinery), on-device
  moment reductions through the shared host smoothing/finalizer math, device OOF + gather, one
  D2H for the final matrix (`output='numpy'` for now). Device input routes to GPU regardless of
  `_AUTO_GPU_ENABLED` (categorical signal); `backend='cpu'` + cuDF raises. Fences: numeric /
  loo-ordered / median-min-max (B4) / custom / non-numpy output. **T4: full suite 351 passed
  incl. 24 gpu tests** (`tests/test_device_input.py`, `scripts/colab_gpu_tests.py`).
- ✅ **B1 — device additive OOF kernel (2026-07-02, validated on Colab T4 via the B2 suite)**:
  `_gpu.oof_moment_tables` (`cupy.bincount`, order 2/4; one H2D of comp+y per unit, small tables
  back) injected through the B0 seam — under `backend="gpu"` the additive OOF path (mean/var/std/
  skew/kurt/woe) now runs its heavy pass on device instead of never touching the GPU. Parity test
  gpu-marked (`test_oof_moment_tables_gpu_matches_numpy`); perf claims deferred to the B5
  crossover verdict (KI-020).
- ✅ **Stats arc 3 (2026-07-02)**: `stats=["woe"]` — weight of evidence, **binary-only**
  (`StatSpec.binary_only`), derived as `logit(smoothed p) − logit(prior)` from the existing
  mean/probability smoothing (honesty-rule compliant; no new smoothing invented). GPU-eligible and
  on the additive fast kernel; unknown/missing-unseen fallback is exactly **0.0**. Documented
  edge: a *pure* category is ±inf under `smooth=0` **and** `smooth='auto'` (EB shrinks by
  within-category variance = 0); fixed `m>0` guarantees finite. Leakage audit exact; sklearn-compat
  spot checks pass. `test_woe.py`.
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
- ✅ **GitHub Pages enabled (2026-06-27)**: source = GitHub Actions; the Docs workflow now deploys
  the API site to https://matapanino.github.io/catstat/ (the deploy step had been failing only
  because Pages was off). (KI-020 GPU perf is the optional larger follow-up.)

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
- ✅ Numeric binning for `Count`/`Frequency` (KI-030) — **done (2026-06-27, targets 0.4.0)**: a
  numeric column takes each row's **bin count** (`CountEncoder`) / **bin frequency = normalized
  histogram** (`FrequencyEncoder`). Added the four `numeric`/`cardinality_threshold`/`n_bins`/
  `binning` params to both encoders; **all** binning logic reused from the shared `_numeric.py` path
  (no edits to `_base.py`/`_validation.py`/`_numeric.py`). Unsupervised → edges from X only and the
  safety property is plain equivalence (`fit_transform == fit().transform()`). `numpy`-array input
  (all-object after `prepare_X`) and `bool` columns stay categorical, exactly as `TargetEncoder`.
  `tests/test_count_frequency.py` (12 numeric cases).
- ✅ Explicit / per-column **bin edges** (KI-030 follow-up) — **done (2026-06-27, targets 0.4.0)**:
  `binning` also accepts an edge array (`[0,18,65,120]` → 3 bins, applied to every binned column) or
  a `{col: strategy-or-edges}` dict; explicit edges set the bin count (`n_bins` ignored), and
  `binning` controls *how* not *whether* (routing stays with `numeric`+`cardinality_threshold`).
  Logic isolated to `_numeric.py` (cross-fit untouched); OOF stays exact, sklearn-compat (list/dict
  `clone`) + leakage-audit PASS. Branch `feat/numeric-explicit-bin-edges`.
- ✅ **`min_bin_size`** (KI-030 follow-up) — **done (2026-06-27, targets 0.4.0)**: merges adjacent
  sparse bins of the computed `quantile`/`uniform` strategies (greedy, from training X counts only)
  so each surviving bin holds enough rows; explicit edges left exact. int / float-fraction / `None`
  (off). All three encoders; OOF exact (edges ⊥ y), sklearn-compat + leakage-audit PASS. Branch
  `feat/numeric-min-bin-size`. **Numeric follow-ups complete.**

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
  (supersedes PR #2). ✅ **GPU `combination`/`interactions`** unblocked (`host_only` drop + int64
  codes to the device group-by); **CPU/GPU allclose validated on Colab T4** (KI-018 resolved).
- ✅ **Interactions (2026-06-27)**: `interactions=[[...]]` → one joint TE column per group (additive
  to `cols`; generalizes `combination`). `_units` plumbing + one param; OOF / naming / parity reuse
  the unit machinery. `test_interactions.py`; sklearn-compat PASS. Branch `feat/interactions`.
- **Phase 2 — remaining.** GPU *performance* (on-device keys/folds; KI-020) and `combination` on
  GPU (KI-018) — both optional, gated behind a fresh crossover verdict before re-enabling `auto`.
- **Phase 3.** quantile/skew/custom + ordered/LOO + `set_output("polars")` + PyPI release.

## "Next" pointer (update each session)
> **Next task:** **Integer joint codes done (2026-06-27, lever #2A)** — combination key-build replaced
> by vectorized mixed-radix int64 codes (learned once from full X, reused at fit/fold/transform);
> byte-identical output, combination transform ×3.7–4.4 / fit_transform ×1.5–2.4 at 1M, closes KI-019
> and supersedes PR #2; leakage + sklearn-compat PASS. The perf stack (#3 mean, #5 var/std, #7 gather,
> joint codes) and **interactions** (`interactions: list[list[str]]`) are all now merged to main.
> **Lever #2B — GPU `combination` DONE (2026-06-27, `feat/perf-gpu-combination`)**: `host_only = not
> all_gpu` (combination/interaction no longer forced to CPU); host-built int64 joint codes flow to the
> device group-by (`_gpu._to_nullable` non-object guard). **CPU/GPU allclose validated on Colab T4** —
> combination mean/var, missing-component, interactions all pass (max\|Δ\| ≤ 3.8e-15, ft 0.0,
> `backend_=gpu`); KI-018 resolved. **Perf arc complete** — CPU levers exhausted (cuML had nothing to
> port; all came from sklearn's integer-code path). **Next:** **PR-D** GPU on-device kernel + a fresh
> crossover before re-enabling `auto` — but the 2026-06-27 crossover re-confirms GPU only reaches
> ~parity at ≥5M (0.93×@1M, 1.22×@5M, 1.07×@10M), so `auto` **stays off** and PR-D is a niche lever
> (KI-020). **Maintainer carryover:** `0.2.0` and **`0.3.0` are released** (`v0.3.0` tagged + pushed,
> `refs/tags/v0.3.0` → `83d7d74`; Trusted Publishing fires on the tag). ✅ GitHub Pages enabled.
> **`0.4.0` is RELEASED on PyPI (2026-06-27)** — the **numeric-encoding arc**: (1) `Count`/`Frequency`
> numeric binning (KI-030, #12) — per-bin count / normalized-histogram frequency; (2) **explicit /
> per-column bin edges** (`binning=` edge array or `{col: strategy-or-edges}` dict, #13); (3)
> **`min_bin_size`** (merge sparse computed bins from X, #14). All reuse the shared `_numeric.py` path
> with the cross-fit untouched; each shipped `/sklearn-compat` + `/leakage-audit` green. `v0.4.0`
> tagged + pushed → Trusted Publishing built + uploaded the wheel/sdist (run succeeded, live on PyPI),
> GitHub Release created. **Next candidates (no work started):** `smoothing="sigmoid"`
> (category_encoders parity), optional Laplace add-α for frequency (default off), multiclass
> `max_classes` (KI-016); or **PR-D** GPU on-device — niche, `auto` stays off (the 2026-06-27
> crossover keeps GPU at ~parity only ≥5M; KI-020). **Ops nit:** Actions warn on Node 20 deprecation
> (bump `actions/checkout@v4` etc.).
