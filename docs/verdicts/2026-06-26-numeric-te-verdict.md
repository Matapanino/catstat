# Verdict: opt-in, cardinality-aware numeric-column target encoding (0.2.0)

- Date: 2026-06-26
- Branch: `feat/numeric-target-encoding`
- Backend: cpu (GPU parity deferred to the Colab loop; binning is host-side numpy)
- Artifacts:
  - `benchmarks/results/2026-06-26-numeric-te-eval.json` (CV-quality eval, 5 seeds)
  - `benchmarks/eval_numeric.py` (reproducible eval), `docs/notes/2026-06-26-numeric-te-prior-art.md`
  - `tests/test_numeric_encoding.py` (27 tests)
- Roadmap target: `docs/roadmap.md` → numeric target encoding · Related: `docs/known_issues.md` KI-030

## Question
Should `TargetEncoder` offer first-class, opt-in numeric-column target encoding — low-cardinality
numerics encoded **directly** (each value a category), high-cardinality numerics **binned** (quantile)
then target-encoded, auto-routed by cardinality — and does it measurably help a downstream model?
And what should the default `cardinality_threshold` / `n_bins` be?

## Design
New `TargetEncoder(numeric="ignore"|"auto"|"direct"|"bin", cardinality_threshold, n_bins, binning)`.
`"ignore"` (default) reproduces today's behavior exactly (`cols="auto"` skips numerics; an explicit
numeric column still encodes each raw value as a category). Bin **edges are computed once from the
full training X** (never `y`) and stored — leakage-safe because edges are a function of feature
values only — while the per-bin target statistic is cross-fitted out-of-fold by the existing
machinery. The numeric→keys transform lives in a small host-side module (`_numeric.py`,
numpy/pandas only) consulted at the single key-building seam (`_unit_keys`), so all smoothing,
fallback, feature-name, and CPU/GPU-parity logic is reused unchanged. `cardinality_threshold`
accepts an int (absolute unique count) or a float in (0, 1] (unique/n ratio).

## Evidence

### Correctness / leakage / parity
- `bash scripts/check.sh`: **ruff clean · 116 passed, 6 skipped (GPU) · examples run.** Coverage
  90.6% overall; `_numeric.py` 100%.
- **Leakage audit PASS.** Binned OOF reconstruction from each fold's complement is **exact**
  (`max|Δ| = 0.0`). Noise-trap on a continuous numeric independent of `y`: OOF corr **0.069** (≈0)
  vs leaky **0.190** (>2× OOF). Bin **edges are invariant to permuting/replacing `y`** (proves
  edges ⊥ target). `fit_transform ≠ fit().transform()` on signal.
- **sklearn-compat PASS.** New params stored verbatim (`clone`/`get_params`/`set_params` round-trip);
  `numeric_cols_`/`numeric_strategy_`/`bin_edges_` set only after `fit`; `get_feature_names_out`
  width matches output (binned/direct names stay transparent: `col__te_mean`); `set_output`,
  `Pipeline`, `ColumnTransformer` all work with binned-numeric encoders.

### Quality (downstream Ridge, 5-fold CV R², 5 seeds, median) — `benchmarks/eval_numeric.py`
Synthetic playground regression: a high-cardinality continuous feature with a non-monotone effect
(`sin(2.5x)+0.6x²`) plus a low-cardinality integer code with arbitrary per-value offsets — signal a
single linear model cannot use from raw numbers.

| strategy | CV R² (median) | spread (min … max) |
|---|---:|---|
| raw numeric (passthrough) | **+0.034** | -0.002 … +0.114 |
| direct (both cols) | +0.375 | +0.262 … +0.653 |
| auto, n_bins=5 | +0.772 | +0.729 … +0.869 |
| **auto, n_bins=10** | **+0.910** | +0.895 … +0.947 |
| auto, n_bins=20 | +0.939 | +0.928 … +0.963 |
| auto, n_bins=40 | +0.943 | +0.935 … +0.964 |

Routing `@threshold=10`: `{x_hc: bin, c_lc: direct}` (correct). Binning the high-card numeric is the
decisive lever — `direct` alone reaches only 0.375 because continuous values become near-singletons.
`n_bins=10` captures ~96% of the achievable gain; returns past 20 are flat.

## Decision
**KEEP + CHANGE-DEFAULT.** Numeric TE is correct, leakage-safe, opt-in (zero change to existing
defaults), and delivers a large CV-quality win (R² 0.034 → 0.91). Default `n_bins` and
`cardinality_threshold` set to **10** (from the provisional 20): `n_bins=10` captures nearly all of
the gain while keeping ~N/10 samples/bin for stable smoothing on small/typical data (the marginal
20-vs-10 gain is large-`n`-specific and risks variance on small data); `cardinality_threshold=10`
errs low because mis-routing a modest-cardinality *continuous* feature to `direct` (singletons → no
signal, the 0.375 case) is worse than binning a categorical code. Both match the prior-art note's
literature-backed recommendation. No committed perf baseline (`baseline-cpu.json`) is changed.

## Follow-ups
- GPU parity for the binned/direct numeric cases via `scripts/colab_gpu_parity.sh` (maintainer-run);
  binning is host-side numpy so allclose is expected — confirm before claiming GPU support.
- Numeric binning for `CountEncoder`/`FrequencyEncoder` (count/frequency of bins) — see KI-030.
- Custom/explicit bin edges and a min-bin-size knob — `n_bins` + `min_samples_category` cover MVP.
