# Changelog

All notable changes to `catstat` are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versioning is [SemVer](https://semver.org/).

## [0.1.1] — unreleased

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

[0.1.1]: https://github.com/Matapanino/catstat/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Matapanino/catstat/releases/tag/v0.1.0
