# catstat

[![CI](https://github.com/Matapanino/catstat/actions/workflows/ci.yml/badge.svg)](https://github.com/Matapanino/catstat/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/catstat.svg)](https://pypi.org/project/catstat/)
[![Python](https://img.shields.io/pypi/pyversions/catstat.svg)](https://pypi.org/project/catstat/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Unified CPU/GPU **statistical categorical encoding**: leakage-safe target encoding generalized to
arbitrary statistics, behind one scikit-learn-compatible API.

Runs on CPU (pandas/numpy) today. The GPU path (cuDF/CuPy) is **parity-validated** (CPU/GPU
allclose) but not yet faster than CPU up to ~1M rows, so `backend="auto"` resolves to **CPU**;
explicit `backend="gpu"` is available for device-resident pipelines and larger data. See
[`docs/roadmap.md`](docs/roadmap.md) and [`docs/known_issues.md`](docs/known_issues.md) (KI-020).

## Install

```bash
pip install catstat
```

Optional extras: `catstat[gpu]` (RAPIDS cuDF/CuPy, CUDA 12), `catstat[polars]`
(`output="polars"`), `catstat[docs]` (API-reference build), `catstat[dev]` (tests + lint + build).

## Quickstart

```python
from catstat import TargetEncoder, CountEncoder, FrequencyEncoder

enc = TargetEncoder(cols="auto", stats=["mean"], smooth="auto", cv=5, random_state=42)
X_train_enc = enc.fit_transform(X_train, y_train)   # out-of-fold (leakage-safe)
X_test_enc  = enc.transform(X_test)                 # full-data encodings for new data
```

## Why catstat

**sklearn**'s `TargetEncoder` is CPU and mean-only; **cuML** is GPU-only (RAPIDS-locked, few stats);
**category_encoders** has no internal cross-fitting (leakage risk). `catstat` is the union: one API,
CPU today and GPU when it pays off, generalized statistics, always leakage-safe.

## What it encodes

Three encoders over a shared core: **`TargetEncoder`** (supervised, cross-fitted) and the
unsupervised **`CountEncoder`** / **`FrequencyEncoder`**. `TargetEncoder(stats=[...])` selects the
statistics to emit:

| `stats=` entry | smoothing | target | GPU | column infix |
|---|---|---|---|---|
| `"mean"` | m-estimate (fixed) / empirical-Bayes (`smooth="auto"`) | regression / binary / multiclass | ✅ | `te_mean` |
| `"count"` | — | unsupervised | ✅ | `count` |
| `"frequency"` | — | unsupervised | ✅ | `freq` |
| `"var"`, `"std"` | — (global fallback) | regression | ✅ | `te_var`, `te_std` |
| `"median"`, `"min"`, `"max"` | — (global fallback) | regression | ✅ | `te_median` / `te_min` / `te_max` |
| `"skew"`, `"kurt"` | — (global fallback) | regression | ✅ | `te_skew`, `te_kurt` |
| `("name", callable)` — custom (quantiles, IQR, …) | — (global fallback) | regression | CPU only | `name` |

**Smoothing honesty:** only mean/probability statistics are smoothed. Count/frequency get none;
order/shape statistics never blend — below `min_samples_category` (or where undefined) they fall
back to the **global** statistic. (`stats=["quantile"]` raises with a hint to pass a custom
callable such as `("q90", lambda v: np.quantile(v, 0.9))`.)

Other knobs: `scheme ∈ {kfold, loo, ordered}` (cross-fitting for the mean; `loo`/`ordered` are
mean-only), `multi_feature_mode ∈ {independent, combination}` (joint group-by), `handle_unknown` /
`handle_missing ∈ {value, return_nan, error}`, `backend ∈ {auto, cpu, gpu}`, and `output ∈
{auto, numpy, pandas, polars}`.

## Leakage-safe by design

- `fit_transform(X, y)` is **out-of-fold**: each fold is encoded from its complement, then the
  encoder refits on the full data for later `transform` of *new* rows. `fit(X, y).transform(X)` on
  the *training* set is the leaky path and is documented as such.
- `smooth="auto"` variance is computed **per fold**; folds flow only through `random_state`
  (`catstat` owns fold assignment, so CPU and GPU produce the same encodings — asserted allclose).
- Deterministic given `random_state`.

## scikit-learn compatibility

`BaseEstimator` / `TransformerMixin`; works in `Pipeline` and `ColumnTransformer`, supports
`set_output(transform="pandas"|"polars")` and `get_feature_names_out`. The supported subset of
`sklearn.utils.estimator_checks.check_estimator` is documented and tested (see
[`docs/known_issues.md`](docs/known_issues.md), KI-012).

## API reference

Rendered API docs: **https://matapanino.github.io/catstat/** (built with `pdoc`; see
`scripts/build_docs.sh`).

## Develop

```bash
pip install -e ".[dev]"
bash scripts/check.sh        # ruff + pytest + examples (the green gate)
PYTHONPATH=src python3 -m pytest tests/ -q
PYTHONPATH=src python3 -m benchmarks.run_benchmarks --size small --backend cpu --reps 5 \
    --out benchmarks/results/run.json
```

See [`CLAUDE.md`](CLAUDE.md) for the development rules and [`docs/`](docs/) for the design.

## License

MIT
