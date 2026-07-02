# `catstat` — Experiment Log (append-only)

One line (or short block) per experiment, **including null and negative results**, so no future
session retries a dead end. Newest at the top. Each entry links its verdict when one exists.

**Format:**
```
## YYYY-MM-DD — <topic>
- Hypothesis: <what we expected>
- Setup: <dataset(s), seeds, reps, backend, git SHA>
- Result: <KEEP | REJECT | NULL> — <one-line evidence>
- Verdict: docs/verdicts/YYYY-MM-DD-<topic>-verdict.md (if any)
```

---

## 2026-06-26 — project bootstrap (design phase)
- Hypothesis: a unified CPU/GPU, statistically-general, leakage-safe encoder fills a real gap left
  by sklearn (CPU/mean-only), cuML (GPU/RAPIDS-only), and category_encoders (no cross-fit).
- Setup: research pass over the three libraries' docs + source; no code run.
- Result: KEEP (design) — gap confirmed; design recorded in `docs/proposals/`.
- Verdict: n/a (design, not a measured change).

## 2026-06-26 — M0 bootstrap (CPU mean encoder + count/frequency)
- Hypothesis: a CPU mean `TargetEncoder` with out-of-fold `fit_transform` plus unsupervised
  `Count`/`Frequency` encoders can be implemented leakage-safe, sklearn-compatible, and green.
- Setup: pandas 1.5.2 / numpy 1.23.5 / sklearn 1.2.0; `scripts/check.sh`; size=small benchmark, 5 reps.
- Result: KEEP — 46 passed / 1 GPU-skipped; OOF reconstruction exact (`max |Δ|=0.0`); noise-trap
  OOF corr ≈ -0.006 vs leaky 0.66; coverage 85.87%; baseline written.
- Verdict: docs/verdicts/2026-06-26-m0-bootstrap-verdict.md

## 2026-06-26 — Phase 2 (dispersion/order stats, combination, GPU scaffold)
- Hypothesis: var/std/median/min/max can be added as cross-fitted, continuous-only stats; a joint
  combination mode and a cuDF/CuPy backend fit behind the existing structure without regressing CPU.
- Setup: pandas 1.5.2 / numpy 1.23.5 / sklearn 1.2.0; `scripts/check.sh`; size=small benchmark.
- Result: KEEP — 67 passed / 2 GPU-skipped; coverage 88.17%; new stats cross-fitted & correct vs
  pandas groupby; combination joint encoding correct; CPU path byte-unchanged after backend
  threading. GPU path written, **Colab-validation pending** (no local GPU).
- Verdict: docs/verdicts/2026-06-26-phase2-stats-gpu-verdict.md

## 2026-06-26 — GPU backend CPU/GPU parity validated on Colab T4
- Hypothesis: `backends/_gpu.py` (cuDF/CuPy, host-orchestrated, catstat-owned folds) produces the
  same encodings as CPU to allclose.
- Setup: Colab T4, Python 3.12.13, RAPIDS (cudf-cu12); n=200k × 5k cats, cv=5, seed=0;
  `scripts/colab_gpu_parity.sh`.
- Result: KEEP — all 4 cases allclose (mean/var × reg/bin/mc), transform + fit_transform,
  max|Δ|~1e-14. backend_gpu="gpu" confirmed.
- Verdict: docs/verdicts/2026-06-26-gpu-parity-verdict.md (+ harness report + JSONL artifact).

## 2026-06-26 — GPU missing-on-device + CPU/GPU crossover (T4)
- Hypothesis: (1) GPU encodes missing-as-value correctly via cuDF nulls; (2) GPU beats CPU above
  some size, so `backend="auto"` should switch over at a calibrated threshold.
- Setup: Colab T4, Python 3.12.13; parity n=200k incl. 10%-missing case; crossover n=10k/100k/1M.
- Result: (1) KEEP — missing-as-value allclose (max|Δ|~3e-16); MISSING→cuDF-null→back works.
  (2) **NEGATIVE/CHANGE** — GPU is *slower* than CPU at all sizes up to 1M (speedup 0.28/0.27/0.86);
  the per-fold host↔device round-trip dominates. → disabled auto-GPU (`_AUTO_GPU_ENABLED=False`);
  explicit `backend="gpu"` retained.
- Verdict: docs/verdicts/2026-06-26-gpu-crossover-verdict.md (KI-020).

## 2026-06-26 — Phase 3a (skew + custom-callable aggregations)
- Hypothesis: skew (built-in) and arbitrary custom aggregations (quantiles, IQR, ...) fit the
  registry as cross-fitted, continuous-only, CPU stats without disturbing the GPU/CPU paths.
- Setup: pandas/numpy/sklearn; `scripts/check.sh`.
- Result: KEEP — 75 passed / 2 GPU-skipped, coverage 89.37%. skew matches pandas groupby; custom
  q90/IQR correct; custom forces CPU + is cross-fitted; `stats=["quantile"]` gives a helpful hint.
- Verdict: docs/verdicts/2026-06-26-phase3a-skew-custom-verdict.md

## 2026-06-26 — Phase 3b (leave-one-out + ordered/CatBoost schemes)
- Hypothesis: LOO and ordered target statistics can be added as a `scheme` param (mean-only,
  leakage-safe alternatives to k-fold) without changing default behavior.
- Setup: pandas/numpy/sklearn; `scripts/check.sh`.
- Result: KEEP — 86 passed / 2 GPU-skipped, coverage 90.64%. LOO exact-value check passes; both
  schemes leakage-safe (noise OOF corr <0.1 vs leaky >0.4); ordered deterministic per seed;
  non-mean+scheme raises. Bug found+fixed: ordered with smooth=0 gave a=0 → 0/0 nan; default a=1.
- Verdict: docs/verdicts/2026-06-26-phase3b-loo-ordered-verdict.md

## 2026-06-26 — Phase 3c (polars output) + release prep (0.1.0)
- Hypothesis: `output="polars"` can be added via lazy import; the package is release-ready.
- Setup: polars 1.35.2 locally; `python -m build` + `twine check`; clean-venv wheel install.
- Result: KEEP — `output="polars"` returns a polars DataFrame (88 passed / 2 GPU-skipped).
  Version bumped 0.0.1→0.1.0 (pyproject + __init__ in sync); LICENSE/CHANGELOG/checklist/skill
  added. Build verified: sdist+wheel build, twine check PASSED, clean-venv install imports and
  runs on **sklearn 1.9.0** (latest) — no newer-sklearn compat issues. Upload/tag deferred to
  the maintainer.
- Verdict: docs/verdicts/2026-06-26-release-0.1.0-verdict.md
- Published: GitHub Matapanino/catstat (public) — `main` + tag `v0.1.0` + release pushed
  2026-06-26. PyPI upload pending the maintainer's token (twine can't prompt in the sandbox).

## 2026-06-26 — release automation (tag-driven PyPI Trusted Publishing)
- Hypothesis: the manual `twine upload` release step can be replaced by a `v*`-tag-triggered GitHub
  Actions workflow that publishes to PyPI via Trusted Publishing (OIDC, no stored token).
- Setup: new `.github/workflows/release.yml`; version 0.1.0→0.1.1 (pyproject + `__init__` in sync);
  CHANGELOG `## [0.1.1]` opened; `docs/publishing_checklist.md` rewritten. CPU-only macOS;
  `scripts/check.sh`.
- Result: KEEP — green gate still green (88 passed / 2 GPU-skipped); both workflow YAMLs parse;
  `python -m build` + `twine check` PASSED for catstat-0.1.1 sdist+wheel. No library behavior or
  invariant changed; the workflow fires only on a `v*` tag and guards tag↔version.
- Verdict: docs/verdicts/2026-06-26-release-automation-verdict.md
- Maintainer follow-up: one-time PyPI Trusted-Publisher config, then `git tag v0.1.1` to publish.

## 2026-06-26 — README polish (status / badges / feature table)
- Hypothesis: the README can be made accurate and credible for a public 0.1.1 without touching behavior.
- Setup: rewrote `README.md`; fixed the `__init__` module docstring; CPU-only macOS; `scripts/check.sh`
  + `python -m build` + `twine check`.
- Result: KEEP — green gate green (88 passed / 2 GPU-skipped); twine check PASSED (README renders as
  the PyPI long-description). Stale "M0 (alpha) — CPU-only" replaced; badges/install/quickstart/
  stat-table/leakage note/API link added; stat table matches `_stats.py`; GPU honesty matches KI-020.
- Verdict: docs/verdicts/2026-06-26-readme-polish-verdict.md

## 2026-06-26 — API docs (pdoc + GitHub Pages)
- Hypothesis: the API reference can be generated reproducibly with pdoc and published to Pages on push.
- Setup: `scripts/build_docs.sh` + `.github/workflows/docs.yml`; ran the script locally with pdoc 14+.
- Result: KEEP — build produced `site/index.html` + `catstat.html` + `search.js`, no import errors
  (pdoc skips the private `_gpu` module → no RAPIDS import on CPU). `docs.yml` parses; green gate
  unaffected.
- Verdict: docs/verdicts/2026-06-26-api-docs-verdict.md
- Maintainer follow-up: enable Pages (Settings → Pages → Source: GitHub Actions).

## 2026-06-26 — scikit-learn estimator tags (__sklearn_tags__ + _more_tags)
- Hypothesis: the encoders can advertise correct sklearn tags (categorical/string/allow_nan +
  requires_y) across sklearn versions without changing encoding behavior.
- Setup: added both tag APIs on `_BaseStatEncoder` keyed off `_is_supervised()`; tag unit tests
  (`__sklearn_tags__` guarded to >=1.6). Local sklearn 1.2 `scripts/check.sh`; fresh sklearn 1.9.0
  / pandas 3.0.3 venv for the >=1.6 path.
- Result: KEEP — local green (88 passed / 2 GPU-skipped; new _more_tags test runs, __sklearn_tags__
  test skips on 1.2). On sklearn 1.9 the tags resolve correctly (TE required=True, CE required=False,
  categorical/string/allow_nan=True).
- DISCOVERED (pre-existing, not from this change): (1) CI red — bare `pytest tests/` can't import
  `tests` (no repo root on sys.path); ModuleNotFoundError on every CI run since before this arc.
  (2) pandas 3.0 — `select_cols` misses the new default `str`/StringDtype, so `cols="auto"` raises;
  3 tests fail under pandas 3.0 (85 pass). Both fixed in the next two commits.
- Verdict: docs/verdicts/2026-06-26-sklearn-tags-verdict.md

## 2026-06-26 — CI green: pytest pythonpath (tests import)
- Hypothesis: CI is red because bare `pytest tests/` can't import `tests.conftest` (repo root not on
  sys.path); adding a pytest `pythonpath` fixes it.
- Setup: reproduced locally (`unset PYTHONPATH; pytest tests/` → ModuleNotFoundError: tests). Added
  `pythonpath=["src", "."]` to `[tool.pytest.ini_options]`.
- Result: KEEP — bare `pytest tests/` now 89 passed / 3 skipped (pandas 1.5); `scripts/check.sh`
  green. CI had been red since before this arc (a collection error, unrelated to pandas/my changes).
  The pandas 3.0 break (KI-022) is fixed in the next commit.
- Verdict: docs/verdicts/2026-06-26-ci-pytest-pythonpath-verdict.md

## 2026-06-26 — pandas 3.0 compat: cols="auto" selects StringDtype (KI-022)
- Hypothesis: `cols="auto"` fails on pandas >=3.0 because string columns are now StringDtype (not
  object) and select_cols only matched object/Categorical; recognizing StringDtype fixes it.
- Setup: added `_is_categorical_like` (object via `is_object_dtype`, Categorical, StringDtype) in
  `_validation.py`; regression test with an explicit `dtype="string"` column (portable to pandas
  1.5). Verified local pandas 1.5.2 (`scripts/check.sh`) + fresh sklearn 1.9.0 / pandas 3.0.3 venv.
- Result: KEEP — venv full suite 89 passed / 3 skipped (was 3 failed); local green; ruff clean. The
  3 prior pandas-3.0 failures (numpy-object-in, column-transformer-passthrough, set_output-numpy)
  now pass. No cross-fit/smoothing change.
- Verdict: docs/verdicts/2026-06-26-pandas3-string-dtype-verdict.md

## 2026-06-26 — sklearn check_estimator documented subset + estimator pickling (KI-012)
- Hypothesis: a documented `check_estimator` subset can be enforced for the categorical encoders,
  and the unpicklable cached backend module can be fixed, without changing established behavior.
- Setup: discovered the failing checks on sklearn 1.9 (venv); wrote `tests/test_check_estimator.py`
  with per-encoder `expected_failed_checks` + reasons (skipped on sklearn<1.6). Fixed pickling via
  `__getstate__`/`__setstate__` + `_dispatch.backend_module`; added a pickle round-trip unit test.
- Result: KEEP — ~36 checks pass per encoder; the rest waived with reasons (sparse, 1d/empty/complex
  input, by-name n_features, y-messages). A transform n_features guard was tried but REVERTED — it
  broke unseen-category tests (transform selects columns by name and tolerates a differing width by
  design), so that check is waived rather than "fixed". Pickling fixed (round-trip transform equal).
  Local green (subset skips on 1.2); venv full suite 93 passed / 3 skipped. KI-012 downgraded S2→S3.
- Verdict: docs/verdicts/2026-06-26-check-estimator-subset-verdict.md

## 2026-06-26 — project hygiene (CONTRIBUTING, SECURITY, issue/PR templates)
- Hypothesis: standard contributor-facing files round out the public release without touching code.
- Setup: added CONTRIBUTING.md, SECURITY.md, .github/ISSUE_TEMPLATE/{bug_report,feature_request}.md
  + config.yml, .github/PULL_REQUEST_TEMPLATE.md. Meta only (outside ruff/pytest paths).
- Result: KEEP — green gate unaffected (`scripts/check.sh` green); config.yml parses. This closes
  the 0.1.1 release-polish arc; remaining steps (PyPI Trusted-Publisher config, Pages enablement,
  `v0.1.1` tag) are maintainer-only.
- Verdict: docs/verdicts/2026-06-26-project-hygiene-verdict.md

## 2026-06-26 — 0.1.1 PUBLISHED to PyPI (first release)
- Hypothesis: the tag-driven release workflow publishes 0.1.1 to PyPI via Trusted Publishing.
- Setup: finalized the CHANGELOG date; tagged `v0.1.1` and pushed; `release.yml` ran (build +
  publish). The maintainer had configured the (pending) Trusted Publisher.
- Result: KEEP — both jobs green; `catstat 0.1.1` is on PyPI (releases: ['0.1.1']; 0.1.0 stayed
  GitHub-only). Clean-venv `pip install catstat==0.1.1` imports and `fit_transform` works on pandas
  3.0.3. GitHub release v0.1.1 created with the CHANGELOG notes. Pages enablement remains
  maintainer-only (a public-hosting config change).
- Verdict: n/a (release execution; see docs/verdicts/2026-06-26-release-automation-verdict.md).

## 2026-06-26 — opt-in cardinality-aware numeric-column target encoding (0.2.0)
- Hypothesis: routing numeric columns by cardinality (low → direct categories, high → quantile bins)
  then target-encoding them — with bin edges from X only — beats passing raw numerics to a
  downstream linear model, inside one leakage-safe encoder (a confirmed gap; see the prior-art note).
- Setup: pandas 1.5.2 / numpy 1.23.5 / sklearn 1.2.0; `scripts/check.sh`; `benchmarks/eval_numeric.py`
  synthetic playground regression (non-linear + categorical numeric signal), Ridge 5-fold CV, 5 seeds.
- Result: KEEP + CHANGE-DEFAULT — CV R² 0.034 (raw) → 0.910 (auto, n_bins=10) → ~0.94 (n_bins=20/40,
  diminishing returns); binned OOF reconstruction exact, noise-trap OOF corr 0.069 (leaky 0.190),
  edges ⊥ y; 116 passed; defaults set to n_bins=10 / cardinality_threshold=10. leakage-audit +
  sklearn-compat PASS. Prior art: docs/notes/2026-06-26-numeric-te-prior-art.md.
- Verdict: docs/verdicts/2026-06-26-numeric-te-verdict.md

<!-- Append new experiments below this line. Never edit or delete prior entries. -->

## 2026-06-27 — extend the single-pass OOF kernel to var/std + hybrid gate (PR-C)
- Hypothesis: the complement-subtraction kernel already accumulates per-(fold,key) sum-of-squares, so
  var/std are a cheap finalize from the same complement moments — sample var `(ss−s²/cc)/(cc−1)`,
  std `√var` (ddof=1) — with a per-fold complement-global fallback when complement count
  `< max(min_samples,1)` or `< 2` (singleton variance undefined). A hybrid gate runs additive stats
  fast and leaves median/min/max/skew/custom on the per-fold loop.
- Setup: pandas 1.5.2 / numpy 1.23.5 / sklearn 1.2.0; in-process **interleaved** before/after
  (before = pre-PR gate `{"mean"}`: mean fast, var/std slow), 7 reps, n=200k & 1M, 2 cols, cv=5;
  `tests/test_additive_fast_path.py` 48-config equivalence matrix {var,std}×{min_samples 1/2/5}×
  {missing,unknown}×{single,combination} + a hybrid mixed-stat case; independent pure-pandas leakage
  reconstruction; `/leakage-audit`.
- Result: KEEP — 2.67–2.82× on var-only and mean+var+std, 1.47–1.49× mixed (median stays slow);
  output allclose to the per-fold path (≤3.4e-13 var / 7.1e-15 std; allclose-not-bitwise, invariant
  #2), noise-trap OOF corr −0.004 (signal +0.445), asymmetry 20.5; 167 passed, 8 skipped; ruff clean.
  No default changed; `_smoothing`/`_aggregations` untouched. Within the fast path a unit's
  mean/var/std share one factorize + one composite bincount.
- Verdict: docs/verdicts/2026-06-27-pr-c-additive-var-std-verdict.md. Research note (next lever):
  docs/notes/2026-06-27-cuml-vs-sklearn-te-levers.md.

## 2026-06-27 — integer-code gather on the transform path (KI-031)
- Hypothesis: `_transform_array` re-hashed each unit's keys once per (stat,class) column via
  `pd.Series.map`; since a unit's stats share one category *set* (only order differs), factorizing
  the keys once (`index.get_indexer`) and gathering each column from a contiguous float64 array
  aligned to a canonical index cuts transform to one hash per unit + a fancy index per column, with
  bit-identical outputs (unknown code −1 → NaN reproduces `.map`; values bake the §11 global so there
  is no other NaN). Speeds up `transform`, the `fit_transform` refit, and the per-fold slow OOF path.
- Setup: pandas 1.5.2 / numpy 1.23.5 / sklearn 1.2.0; in-process **interleaved** old(`.map`)/new(gather)
  on the same fitted tables, 7 reps, n=1M, single- & multi-col, `stats={mean,var,std,median}`;
  `tests/test_transform_gather.py` (mixed-order multi-stat alignment, combination unknown/known joint
  key, tiny-n baked-global vs unseen-under-`handle_unknown`); independent noise-trap leakage audit;
  `/leakage-audit` + `/sklearn-compat`.
- Result: KEEP — transform ×2.28 (4-stat), ×3.36 (4-stat high-card 50k), ×2.48 (combination), ×1.00
  single-stat (no-unknown fast path = a single fancy index); outputs allclose(equal_nan); 170 passed,
  8 skipped; ruff clean. Leakage PASS (OOF corr −0.013 mean / −0.012 median; leaky +0.65; asymmetric).
  sklearn PASS incl. pickle round-trip of the new `_UnitEncoding`. `categories_` / `global_stats_` /
  `target_mean_` unchanged (canonical = first column's index). No default changed; committed baseline
  NOT updated (it predates the perf arc).
- Verdict: docs/verdicts/2026-06-27-transform-gather-verdict.md. Next lever: integer **joint** codes
  (`c_a*n_b+c_b`) → vectorize combination key-build (KI-019) + unblock GPU `combination` (KI-018).

## 2026-06-27 — integer mixed-radix joint codes for `combination` (lever #2A, KI-019)
- Hypothesis: a `combination` unit built its joint key as a Python object-array of **tuples** then
  grouped/looked-up on tuple hashing (the last per-row Python loop, KI-019; also why GPU is host-only,
  KI-018). Replacing the tuple with a vectorized mixed-radix **int64 joint code**
  (`((c0*n1+c1)*n2+c2)…`), learned once from full X (value-stable per-component maps reused at
  fit/fold/transform) and fed to the PR #7 gather, should be faster with no output change — the code
  is a pure relabeling of the same row grouping. Unknown component → −1 sentinel (existing fallback);
  `prod(n_c) > int64.max` → declines int path, falls back to tuple build.
- Setup: pandas 1.5.2 / numpy 1.23.5 / sklearn 1.2.0; in-process **interleaved** per rep across three
  `_unit_keys` impls — genexpr (original loop), zip (PR #2), intcode (new) — `make_multi_column`
  (4 cols card-20, cv=5), n=200k (7 reps) & 1M (5 reps); `tests/test_joint_codes.py` (stable/distinct
  codes, decode roundtrip, −1 sentinel, overflow fallback), new combination tests in
  `test_multi_feature.py` (joint-unseen, missing value/return_nan, categories_ tuples, determinism),
  combination OOF reconstruction in `test_cross_fit_no_leakage.py`; `/leakage-audit` + `/sklearn-compat`.
- Result: KEEP — byte-identical output (max|Δ|=0.00e+00 at 200k & 1M across all three impls); at 1M
  combination transform ×4.35 vs the loop / ×2.93 vs PR #2's zip, fit_transform ×2.42 / ×1.67 (win
  grows with N); 180 passed, 8 skipped; ruff clean. Leakage PASS (OOF reconstruction max|Δ|=4.4e-16;
  noise-trap OOF corr 0.06 vs leaky 0.84 for smooth=0 and "auto"; asymmetry 0.022>0). sklearn PASS
  incl. pickle round-trip of `_unit_keyplans`; `categories_` decoded back to value tuples (unchanged
  representation), feature names / §11 fallback / defaults unchanged. Committed baseline NOT updated.
- Verdict: docs/verdicts/2026-06-27-integer-joint-codes-verdict.md. Closes KI-019, supersedes PR #2.
  Next lever: **#2B GPU `combination`** (KI-018) — drop `host_only` combination clause + joint codes
  in `_gpu.py`, **mandatory** Colab CPU/GPU parity.

## 2026-06-27 — explicit interaction groups (`interactions=[[...]]`)
- Hypothesis: the engine already treats a "unit" as an arbitrary column group (tuple keys), so an
  explicit `interactions: list[list[str]]` param that appends one joint unit per group is mostly a
  `_units`-construction + param change; OOF / naming / unknown-missing / parity reuse the existing
  unit machinery. Generalizes `multi_feature_mode="combination"` (joint-only) by adding joint columns
  on top of the independent `cols`.
- Setup: pandas 1.5.2 / numpy 1.23.5 / sklearn 1.2.0; `tests/test_interactions.py` (naming, equality
  with the combination encoder, multi-stat, dedup, validation errors, clone/get_params); sklearn-compat
  spot-checks (clone/set_params roundtrip, Pipeline, ColumnTransformer, set_output, feature-name
  width); `scripts/check.sh` green.
- Result: KEEP — `a+b__te_*` columns added additively; the interaction column == the combination
  encoder's column (allclose); duplicates deduped; invalid groups raise; clone/get_params preserve the
  param. sklearn-compat PASS. Branch off main (independent of the perf PRs). Joint keys stay
  GPU-host-only (KI-018).
- Verdict: n/a (feature; no default changed).

## 2026-06-27 — GPU `combination` unblocked (lever #2B, KI-018) — CODE; Colab parity PENDING
- Hypothesis: now that combination/interaction units key on **int64 mixed-radix joint codes** (lever
  #2A, host-built in `_unit_keys`), they no longer need tuple keys on the device — cuDF can group an
  int64 column directly. So dropping the `len(cols) > 1` clause from `host_only` (`host_only = not
  all_gpu`) should let combination run on the GPU backend, with parity intact because the joint codes
  are byte-identical on both backends (built on host) and only the device group-by differs — the same
  situation already validated for single-column. A missing component is folded into an ordinary int
  code on the host, so no MISSING sentinel reaches the device (`_gpu._to_nullable` returns early for
  non-object key arrays).
- Setup: pandas 1.5.2 / numpy 1.23.5 / sklearn 1.2.0 (CPU-only box, **no local GPU**). CPU green gate
  (`scripts/check.sh`); verified `backend='gpu'`+combination still **raises** on a no-GPU box (no
  silent fallback) and `auto`/`cpu` combination unchanged. Added combination (mean/var) +
  missing-component + interactions parity cases to `tests/test_cpu_gpu_parity.py` (gpu-marked, skipped
  locally) and `scripts/colab_gpu_parity.py`.
- Result: CODE COMPLETE, **NOT YET VALIDATED** — the device path changed, so CPU/GPU `allclose` on a
  real GPU is the mandatory gate and **I cannot run it** (no local GPU). Maintainer must run
  `bash scripts/colab_gpu_parity.sh` (T4); combination/missing/interactions must show
  `transform_allclose` + `fit_transform_allclose` true and `backend_gpu == "gpu"`.
- Verdict: **VALIDATED on Colab T4 (2026-06-27)** — combination mean/var, missing-component, and
  interactions all `transform`+`fit_transform` allclose (max|Δ| ≤ 3.8e-15, fit_transform 0.0) with
  `backend_=gpu`; pre-existing single-column/numeric cases still pass. **KI-018 RESOLVED.**
  `docs/verdicts/2026-06-27-gpu-parity-report.md`, `benchmarks/results/2026-06-27-T4-gpu-parity.jsonl`.
  Crossover re-confirms `auto` stays off (GPU ~parity only at ≥5M: 0.93×@1M, 1.22×@5M, 1.07×@10M;
  KI-020 unchanged). KEEP → merge `feat/perf-gpu-combination`.

## 2026-06-27 — 0.3.0 release prep + GitHub Pages enabled (ops, not an experiment)
- 0.2.0 is already on PyPI; `main` gained `interactions=` (new public param), the single-pass OOF
  perf arc, integer joint codes, and GPU combination since — so the next release is **0.3.0** (minor:
  backwards-compatible feature add). Bumped pyproject + `__init__` (in sync), wrote CHANGELOG
  `[0.3.0]`; `python -m build` + `twine check` PASSED (sdist+wheel, `py.typed` present); clean-venv
  install imports `0.3.0` and runs the new `interactions` path + CountEncoder. Merged via PR #10.
  **Tag `v0.3.0` + publish is the maintainer's step** (Trusted Publishing fires on the tag).
- **GitHub Pages enabled** (source = GitHub Actions, via `gh api`); the Docs workflow's deploy had
  been failing only because Pages was off — re-ran it green, site live at
  https://matapanino.github.io/catstat/. Action versions still warn on Node 20 deprecation (future
  bump of `actions/checkout@v4` etc.).

## 2026-06-27 — KI-030: numeric binning for `Count`/`Frequency` (first 0.4.0 feature)
- Hypothesis (pre-profile): the numeric machinery built for `TargetEncoder` is encoder-agnostic, so
  enabling it on the two unsupervised encoders is *just* adding the four `numeric`/`cardinality_
  threshold`/`n_bins`/`binning` params verbatim — confirmed by reading the shared path.
- Setup: `feat/numeric-count-frequency` off `main` (`83d7d74`, `v0.3.0` already tagged+pushed →
  KI-030 lands in **0.4.0**). pandas 1.5.2 / numpy 1.23.5 / sklearn 1.2.0 (CPU box, no local GPU).
- Change: **only** `count_encoder.py` + `frequency_encoder.py` `__init__` (params + docstrings).
  **Zero edits** to `_base.py` / `_numeric.py` / `_validation.py` — `select_cols` already gates on
  `numeric_mode`, `_fit_count` already histograms string keys, `apply_numeric_col` already emits
  GPU-safe strings. A binned column → per-row **bin count** (Count) / **normalized-histogram
  frequency** (Freq); `"auto"` routes by cardinality, `"direct"` counts each value.
- Result: green gate PASS (ruff + **201 passed / 11 GPU-skipped** + examples); `tests/test_count_
  frequency.py` +12 numeric cases. `/sklearn-compat` PASS (clone/get_params/set_params, feature
  names `col__count`/`col__freq`, set_output, Pipeline, ColumnTransformer with a binned encoder).
  `/leakage-audit` PASS — TE OOF reconstruction guard still exact (untouched path); unsupervised
  equivalence `fit_transform == fit().transform()` 12/12 across modes × handle_missing; output and
  `bin_edges_` invariant to a (ignored) `y`.
- Scope boundary (intentional, identical to `TargetEncoder`, noted in KI-030): numpy-array input is
  all-`object` after `prepare_X` so numeric auto-detection doesn't fire, and `bool` stays categorical.
- Verdict: KEEP. No new perf/quality verdict needed (a histogram has no CV-quality claim, no default
  change). pyproject/`__init__` version bump deferred to the 0.4.0 `release-prep`. KI-030 **Resolved**.

## 2026-06-27 — Explicit / per-column bin edges (`binning=` array|dict) — 0.4.0 feature #2
- Goal: extend numeric binning so users can pass explicit bin boundaries (domain knowledge, e.g. age
  `[0,18,65,120]`) instead of only quantile/uniform. API: widen the existing `binning` param to
  accept an edge array (every binned col) or a `{col: strategy-or-edges}` dict — **no `__init__`
  signature change** (the param already exists on all three encoders); confirmed with the user before
  coding (chose "binning を拡張" over a separate `bin_edges=` param).
- Design decision: `binning` controls *how* a column is binned; *whether* stays with `numeric` +
  `cardinality_threshold` (predictable, no routing coupling). Explicit edges = full boundaries
  (k+1 → k bins, interior `[1:-1]` for `np.digitize`, out-of-range clamps), and set the bin count so
  `n_bins` is ignored for that column. Strict validation (≥2 finite, strictly-increasing edges; dict
  keys must name numeric columns being encoded).
- Change: `_numeric.py` (new `validate_binning` + per-column spec resolution) + `_base._validate_
  numeric_params` (call the validator). **Cross-fit / transform / `_smoothing` untouched.** Docstrings
  on TargetEncoder + CountEncoder widened. Branch `feat/numeric-explicit-bin-edges`.
- Result: green gate PASS (ruff + **221 passed / 11 GPU-skipped** + examples); +20 tests
  (`test_numeric_encoding.py` explicit/dict/validation; `test_count_frequency.py` histogram + param
  round-trip). `/sklearn-compat` PASS — list **and** dict `binning` round-trip `clone`/`get_params`/
  `set_params` on all 3 encoders, the cloned dict is a deep copy. `/leakage-audit` PASS — OOF
  reconstruction exact incl. dict binning (max|Δ| 2.2e-16); explicit-edges noise-trap OOF corr 0.0004
  vs leaky 0.108; edges ⊥ y.
- Verdict: KEEP. No default changed, no new perf/quality verdict (no CV-quality claim). min-bin-size
  is the remaining numeric follow-up (PR2). → 0.4.0.

## 2026-06-27 — min_bin_size: merge sparse bins (numeric) — 0.4.0 feature #3
- Goal: a guardrail knob so computed quantile/uniform bins aren't left with too few rows (unstable
  per-bin encoding). `min_bin_size` = int (absolute) / float (fraction of n) / None (off, default).
- Design decision (confirmed with the user): `min_bin_size` refines only the *computed* strategies;
  **explicit edge arrays are honored exactly** (each knob keeps a clean contract — explicit = full
  control, min_bin_size = guardrail for auto binning). Within a dict, decided per column. Merge is a
  deterministic greedy left-to-right pass over training-X bin counts (a sparse trailing group folds
  back into the previous bin); a `min_bin_size` larger than n collapses to a single bin.
- Change: `_numeric.py` (`_resolve_min_count` + `_merge_small_bins`; applied in `_resolve_bin_edges`
  for string strategies only) + `_base._validate_numeric_params` (validate) + `_fit_numeric` (pass
  through). New `min_bin_size=None` param on all three encoders. **Cross-fit / transform / _smoothing
  untouched.** Branch `feat/numeric-min-bin-size`.
- Result: green gate PASS (ruff + **238 passed / 11 GPU-skipped** + examples); +17 tests (merge
  correctness, float fraction, explicit-bypass, dict per-column, single-bin, OOF reconstruction,
  determinism, param round-trip, validation). `/sklearn-compat` PASS (verbatim/clone/set_params for
  None/int/float on all 3 encoders; Pipeline/ColumnTransformer). `/leakage-audit` PASS — OOF exact
  with `min_bin_size` (reconstruction), merged edges invariant to y, noise-trap OOF corr 0.026 vs
  leaky 0.134.
- Verdict: KEEP. Off by default. **Numeric arc (KI-030 + 2 follow-ups) complete.** → 0.4.0; next is
  cutting the 0.4.0 release or adding more features.

## 2026-06-27 — 0.4.0 released to PyPI (ops, not an experiment)
- The numeric arc (KI-030 `Count`/`Frequency` binning + explicit/per-column bin edges + min_bin_size,
  PRs #12–#14) shipped as **0.4.0** — backwards-compatible feature additions, no defaults changed.
- release-prep: bumped pyproject + `__init__` to 0.4.0 (in sync), dated CHANGELOG `[0.4.0]`;
  `python -m build` + `twine check` PASSED (sdist+wheel, `py.typed` present); clean-venv install
  imports 0.4.0 and runs the new numeric features. Merged via PR #15.
- Published: `v0.4.0` tagged + pushed; the **Release Actions workflow (Trusted Publishing)** built
  and uploaded the wheel/sdist (Build ✓ + Publish ✓), **live on PyPI** (`pypi.org` latest = 0.4.0).
  Local `twine upload` from the dev box fails (no TTY for the token; Trusted Publishing is
  GH-Actions-only) — expected and irrelevant; the tag-triggered workflow is the publish path. GitHub
  Release `v0.4.0` created from the `[0.4.0]` notes (first GH release since 0.1.1).
- Carryover: Actions still warn on Node 20 deprecation (bump `actions/checkout@v4` etc.). Next
  feature candidates: `smoothing="sigmoid"`, Laplace add-α for frequency, multiclass `max_classes`.

## 2026-07-02 — Stats arc: kurt + moments-based skew (A1a/A1b), WOE (A2)
- Hypothesis: skew/kurt reconstructed from shifted power sums S1..S4 match pandas' bias-corrected
  G1/G2 at allclose, become GPU-supported (plain sums, cuDF-safe) and additive (fast OOF kernel);
  WOE derived as `logit(smoothed p) − logit(prior)` inherits the principled mean smoothing.
- Change: `category_moments` (both backends) + `g1_g2_from_power_sums`; `_OOFMoments` → order-4 +
  global-mean shift; `finalize_shape_oof` + `_STAT_MIN_N`; `StatSpec.binary_only` + `woe` via
  `_mean_enc_cells`/`finalize_woe_oof`. Branch `feat/shape-stats-moments`.
- Result: green gate PASS (327 tests); pandas parity incl. y=1e9±1 offsets; fast==slow allclose
  across the fallback matrix; leakage audits PASS (skew ≤7e-14, kurt ≤2e-12 rel reconstruction;
  woe exact 0.0; noise traps ≈0); sklearn-compat PASS (1.9.0).
- **Null/edge finding:** `smooth="auto"` (EB, m_i = σ²_i/τ²) applies **no shrinkage to pure
  categories** → WOE is ±inf under auto as well as smooth=0; only fixed m>0 guarantees finite.
  Documented (docstring/CHANGELOG/test-locked), not "fixed" — the auto formula is a protected
  default.
- Verdict: KEEP (feature additions; no defaults changed).

## 2026-07-02 — B0: (fold × cat) table OOF kernel (PR-D groundwork)
- Hypothesis: every OOF encoding is a function of (fold, key), so finalizing on small (F·C)
  tables + one gather is value-identical and creates the backend seam (`moment_tables`) for the
  B1 device kernel.
- Change: `complement_tables`/`np_moment_tables`/`_mean_enc_cells`/`_apply_unknown_cells`/
  `_scatter_cells` replace the per-row kernel; `kfold_mean_oof_fast` deleted (no callers —
  the "back-compat" note was stale).
- Result: old(205b0c9)-vs-new interleaved in-process (n=200k, k=10k, 7 reps): ×1.02–1.21
  (mean-only ×1.08, +woe ×1.21; spreads overlap for the small wins → "no regression", not a CPU
  perf claim). Value parity max|Δ| ≤ 1.24e-14 (woe exact). Leakage audit re-PASS. Standard
  harness: no regressions vs baseline. `docs/verdicts/2026-07-02-b0-table-oof-kernel-verdict.md`.
- Verdict: KEEP; baseline unchanged. Next: B1 `oof_moment_tables` on device (cupy.bincount).
