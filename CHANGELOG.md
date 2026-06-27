# Changelog

All notable changes to `catstat` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/).

## [Unreleased]

### Added
- **Numeric binning for `CountEncoder` / `FrequencyEncoder`** (KI-030): the `numeric`,
  `cardinality_threshold`, `n_bins`, and `binning` parameters — previously `TargetEncoder`-only —
  now work on both unsupervised encoders. A numeric column is routed by cardinality (`numeric="auto"`)
  or forced (`"direct"` / `"bin"`); a binned column takes each row's **bin count** (`CountEncoder`)
  or **normalized-histogram frequency** (`FrequencyEncoder`), while `"direct"` counts each distinct
  value. Because the encoders are unsupervised there is no target, so bin edges come from the training
  feature only and `fit_transform` equals `fit().transform()`. Inspect `numeric_cols_` /
  `numeric_strategy_` / `bin_edges_`. The binning reuses `TargetEncoder`'s numeric machinery unchanged,
  so feature names, unknown/missing handling, and CPU/GPU string-key parity all carry over; `numpy`-
  array input and `bool` columns stay categorical, matching `TargetEncoder`.
- **Explicit & per-column bin edges** for numeric encoding: the `binning` parameter (on
  `TargetEncoder`, `CountEncoder`, and `FrequencyEncoder`) now also accepts an explicit **edge
  array** — `binning=[0, 18, 65, 120]` (→ 3 bins, applied to every binned numeric column) — or a
  per-column **dict** mixing strategies and edges — `binning={"age": [0, 18, 65, 120], "income":
  "quantile"}`. Explicit edges set the bin count (`n_bins` is ignored for that column) and
  out-of-range values clamp to the outer bins, as with the `"quantile"`/`"uniform"` strategies.
  `binning` controls only *how* a column is binned; *whether* it is binned stays with `numeric` +
  `cardinality_threshold`. User-supplied edges depend on nothing in the data, so they are
  leakage-safe a fortiori (OOF reconstruction stays exact).

## [0.3.0] — 2026-06-27

### Added
- **Explicit interaction groups** via a new `interactions` parameter on `TargetEncoder`:
  `interactions=[["a", "b"], ...]` adds **one joint target-encoded column per group** (additive to
  the per-column `cols` encodings), generalizing `multi_feature_mode="combination"` (which encodes a
  single joint column only). Out-of-fold cross-fitting, feature naming (`a+b__te_*`), and the
  unknown/missing fallback all reuse the existing encoding-unit machinery.
- **GPU `backend="gpu"` now supports `combination` / `interactions`** (joint units). They key on
  int64 mixed-radix joint codes (built on the host, so identical on both backends) which the cuDF
  group-by consumes directly; CPU/GPU `allclose` validated on a Colab T4 (mean/var, missing
  component, interactions). Previously these were forced to the CPU.

### Performance
All performance work below is **output-identical** (allclose; leakage-audited) — no behavior or API
change. The committed benchmark baseline is unchanged (it predates this arc; see the verdicts).
- **Single-pass out-of-fold encoding** via complement subtraction for `mean` and `var`/`std`,
  replacing the per-fold group-by loop (a shared per-`(fold, key)` moment builder; a hybrid gate
  keeps `median`/`min`/`max`/`skew`/custom on the per-fold path). ~2.2–3.4× (mean), ~2.7–2.8×
  (var/std).
- **Factorize-once integer-code transform gather**: `transform` now hashes each unit's keys once
  (`index.get_indexer`) and gathers each `(stat, class)` column from a contiguous `float64` array,
  replacing the per-column `pd.Series.map`. transform ~2.3–3.4× (multi-stat / high-cardinality).
- **Integer mixed-radix joint codes** for `combination` / `interactions` units, replacing the
  per-row Python tuple build: combination transform ~3.7–4.4× / fit_transform ~1.5–2.4× at 1M rows.
  (Closes KI-019.)

### Changed
- **GPU crossover re-measured** on a Colab T4 after the single-pass kernel: the host-orchestrated GPU
  path reaches only ~parity at ≥5M rows (≈0.9× at 1M, ≈1.2× at 5M), so `backend="auto"` continues to
  resolve to **CPU**; explicit `backend="gpu"` stays available and parity-validated. (KI-020.)

## [0.2.0] — 2026-06-26

### Added
- **Opt-in, cardinality-aware numeric-column target encoding** on `TargetEncoder`, via a new
  `numeric` parameter (default `"ignore"` keeps existing behavior — `cols="auto"` still skips
  numeric columns). `"auto"` routes each numeric column by cardinality: at most
  `cardinality_threshold` distinct values are encoded **directly** (each value its own category),
  higher-cardinality columns are **binned** into `n_bins` (`binning="quantile"` equal-frequency by
  default, or `"uniform"` equal-width) and the bins are target-encoded. `"direct"`/`"bin"` force one
  strategy. Bin **edges are computed from feature values only** (never `y`), once from the full
  training data, so the per-bin encoding stays out-of-fold and CPU/GPU parity is preserved.
  `cardinality_threshold` accepts an int (absolute unique count) or a float in (0, 1] (unique/n
  ratio). New fitted attributes `numeric_cols_`, `numeric_strategy_`, `bin_edges_`; output column
  names are unchanged (`col__te_mean`). Defaults (`cardinality_threshold=10`, `n_bins=10`) are set by
  an empirical CV verdict (`docs/verdicts/2026-06-26-numeric-te-verdict.md`): downstream 5-fold CV
  R² rises from 0.03 (raw numerics) to 0.91 on a synthetic non-linear/categorical benchmark.

## [0.1.1] — 2026-06-26

### Changed
- Releases now build and publish to PyPI automatically on a `v*` tag, via GitHub Actions and PyPI
  **Trusted Publishing** (OIDC — no stored API token). See `docs/publishing_checklist.md`.
- `TargetEncoder` / `CountEncoder` / `FrequencyEncoder` now advertise scikit-learn estimator tags
  via both `__sklearn_tags__` (sklearn ≥ 1.6) and `_more_tags` (< 1.6): categorical/string input,
  `allow_nan` (NaN is learned as a level under `handle_missing="value"`), and `requires_y` for the
  supervised encoder — so `check_estimator` skips inapplicable checks.

### Documentation
- Rewrote the README: status badges, an honest CPU/GPU status, install + extras, a statistics/
  feature table, a "leakage-safe by design" note, and a link to the API reference.
- Published an API reference built with `pdoc` (`scripts/build_docs.sh`), deployed to GitHub Pages
  via `.github/workflows/docs.yml`.

### Fixed
- `cols="auto"` now selects pandas `StringDtype` columns, so auto-detection works on pandas ≥ 3.0
  (where string columns default to `StringDtype` rather than `object`). (KI-022)
- Fitted estimators are now picklable; a cached backend module previously raised
  `TypeError: cannot pickle 'module' object`. (KI-012)

## [0.1.0] — 2026-06-26

First public release. Leakage-safe, sklearn-compatible statistical categorical encoding with one
API across CPU (pandas/numpy) and, optionally, GPU (cuDF/CuPy).

### Added
- **`TargetEncoder`** — supervised, leakage-safe. `fit_transform` is cross-fitted; `transform`
  uses full-data encodings for new data. Regression, binary, and multiclass (one-vs-rest) targets.
- **`CountEncoder` / `FrequencyEncoder`** — unsupervised category-prevalence encoders.
- **Statistics** (`stats=`): `mean`, `count`, `frequency`, `var`, `std`, `median`, `min`, `max`,
  `skew`, and **custom `(name, callable)` aggregations** (quantiles, IQR, …). Only mean/probability
  are smoothed (m-estimate fixed; empirical-Bayes `smooth="auto"`); other statistics fall back to
  the global value for small/unseen categories and never blend.
- **Cross-fitting schemes** (`scheme=`): `kfold` (default, out-of-fold), `loo` (leave-one-out),
  `ordered` (CatBoost-style ordered target statistics). loo/ordered apply to the mean.
- **Multi-column**: `multi_feature_mode="independent"` (default) or `"combination"` (joint).
- **Missing / unseen handling**: `handle_missing` / `handle_unknown` ∈ {`value`, `return_nan`,
  `error`}, with per-statistic fallbacks (count/frequency → 0; mean → global; etc.).
- **Backends**: `backend="cpu"` (default via `auto`), `backend="gpu"` (cuDF/CuPy, validated
  CPU/GPU-allclose on a Colab T4, incl. missing). `auto` resolves to CPU for now — the current GPU
  path is not yet faster than CPU up to 1M rows (see `docs/known_issues.md` KI-020).
- **Output**: `output` ∈ {`auto`, `numpy`, `pandas`, `polars`}; sklearn `set_output`,
  `get_feature_names_out`, `Pipeline` / `ColumnTransformer` compatibility.
- Test suite (88 tests), synthetic benchmark harness, Colab GPU-parity loop, and CI
  (Python 3.10–3.12).

### Known limitations
- GPU `auto` disabled pending an on-device redesign (KI-020); `combination` and `skew`/custom
  aggregations run on CPU. sklearn-parity tests require `scikit-learn>=1.4`. See
  `docs/known_issues.md`.

[0.1.1]: https://github.com/Matapanino/catstat/releases/tag/v0.1.1
[0.1.0]: https://github.com/Matapanino/catstat/releases/tag/v0.1.0
