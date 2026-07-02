# `catstat` — Known Issues & Limitations

Honest list of current limitations, intentional deferrals, and open risks. Severity: **S1** (blocks
correctness/leakage), **S2** (functional gap), **S3** (polish). Update as items are opened/closed.

## Status: M0 shipped (2026-06-26) — CPU only
The deferrals below are **intentional scope boundaries** for M0, not bugs. The leakage risk
(KI-011) is now actively guarded by `tests/test_cross_fit_no_leakage.py` (OOF reconstruction is
exact). KI-010 (auto-smoothing parity) remains open.

## Intentional deferrals (not bugs — do not "fix" without a roadmap change)
| id | sev | item | notes |
|----|-----|------|-------|
| KI-001 | S3 | GPU **validated** (incl. missing-as-value) but not faster yet | CPU/GPU allclose on **T4 2026-06-26** for mean/var × reg/bin/mc + missing. But GPU is *slower* than CPU up to 1M rows → `auto` disabled (KI-020). Explicit `backend="gpu"` works. |
| KI-002 | — | ~~quantile/skew/custom stats absent~~ | **Resolved 2026-06-26**: skew + custom-callable aggregations (quantiles via custom callables). |
| KI-003 | — | ~~`multi_feature_mode="combination"` not implemented~~ | **Resolved 2026-06-26** (joint group-by). |
| KI-004 | S3 | Ordered (CatBoost) / leave-one-out modes absent | P3 options. |
| KI-005 | S3 | `set_output("polars")` not supported | pandas/numpy/`set_output("pandas")` work; polars in P3. |
| KI-018 | — | ~~GPU `combination` forced to CPU~~ | **Resolved 2026-06-27**: combination/interaction now run on GPU — `host_only = not all_gpu` + host-built **int64 joint codes** (KI-019) flow straight to the device group-by (`_gpu._to_nullable` skips the MISSING remap for non-object keys; a missing component is already folded into an integer code). **CPU/GPU `allclose` validated on Colab T4 (2026-06-27)**: combination mean/var, missing-component, and interactions all `transform`+`fit_transform` allclose (max\|Δ\| ≤ 3.8e-15, fit_transform 0.0), `backend_=gpu`. `docs/verdicts/2026-06-27-gpu-parity-report.md`. `backend='gpu'` still raises without RAPIDS (no silent fallback); `auto` stays CPU (KI-020 crossover unchanged — ~parity only at ≥5M). |
| KI-019 | — | ~~combination joint-key build is a Python loop~~ | **Resolved 2026-06-27**: replaced by vectorized mixed-radix **int64 joint codes** (`((c0*n1+c1)*n2+c2)…`), learned once from full X and reused at fit/fold/transform; byte-identical (max\|Δ\|=0 at 200k–1M), combination transform ×3.7–4.4 / fit_transform ×1.5–2.4 vs the loop. **Supersedes PR #2** (which only built tuples faster). `docs/verdicts/2026-06-27-integer-joint-codes-verdict.md`. |
| KI-020 | S2 | GPU reaches ~parity only at ≥5M rows (T4); `auto` stays off | Post-complement-subtraction (host): per-fold round-trip removed → crossover **0.67×@1M → 1.11×@5M, 1.06×@10M** (marginal + noisy: 1M was 0.67 vs 0.98 across runs). GPU scales sublinearly but the win is within noise; `auto` stays disabled, explicit gpu validated (allclose, mean ft now exact). `docs/verdicts/2026-06-26-gpu-crossover-postPRB-verdict.md`. **2026-07-02 (B0/B1)**: OOF finalization moved to (fold×cat) tables fed by an injectable `oof_moment_tables` kernel; the GPU twin (`cupy.bincount`, one H2D of comp+y per unit) is wired in — the additive OOF path now actually uses the device under `backend='gpu'`. Fresh T4 crossover pending (B5) before any `auto` decision. |
| KI-030 | — | ~~Numeric TE (0.2.0): `Count`/`Frequency` don't bin~~ | **Resolved 2026-06-27** (→ 0.4.0): `Count`/`Frequency` now bin numeric columns — `numeric`/`cardinality_threshold`/`n_bins`/`binning` added verbatim to both encoders, reusing `TargetEncoder`'s shared `_numeric.py` plumbing (**no `_base.py`/`_validation.py` edit** — `select_cols` already gates on `numeric_mode`, `_fit_count` already histograms string keys). A binned column takes each row's **bin count** / **normalized-histogram frequency**; `"auto"` routes by cardinality, `"direct"` counts each value. Unsupervised → edges from full-train X only (no `y`; the safety property is plain `fit_transform == fit().transform()`). **Remaining boundary (intentional, identical to `TargetEncoder`):** numpy-array input is all-`object` after `prepare_X` so numeric auto-detection doesn't fire, and `bool` is excluded from numeric (already two-level categorical) — both stay categorical. **GPU:** keys emitted as **strings** (cuDF rejects object-dtype *integer* arrays), the same path already validated CPU/GPU-allclose on T4 (2026-06-26, max\|Δ\| ~1e-17). `tests/test_count_frequency.py`. |
| KI-031 | S3 | Transform `map`→**gather done**; non-additive stats still re-fit per fold | **2026-06-27**: `_transform_array` now factorizes each unit's keys once (`index.get_indexer`) and **gathers** each column from a contiguous float64 array (`_UnitEncoding`), replacing per-column `pd.Series.map` — transform ×2.3–3.4 (multi-stat / high-card), single-stat neutral, outputs allclose, leakage + sklearn-compat PASS (`docs/verdicts/2026-06-27-transform-gather-verdict.md`). **Still open:** median/min/max/skew/custom re-fit per fold in the hybrid OOF slow path (now faster via the gather, but not on the single-pass kernel). **Follow-up:** ✅ integer **joint** codes (`c_a*n_b+c_b`) done — combination key-build vectorized (KI-019, 2026-06-27); GPU `combination` (KI-018) remains. See `docs/notes/2026-06-27-cuml-vs-sklearn-te-levers.md`. |

## Open risks to track (carry into implementation)
| id | sev | risk | mitigation |
|----|-----|------|-----------|
| KI-010 | S1 | `smooth="auto"` exact formula unverified | local sklearn is 1.2 (no `TargetEncoder`); verify against `_target_encoder_fast.pyx` before claiming sklearn parity. |
| KI-011 | S1 | Leakage via implementation detail | OOF reconstruction test + `leakage-audit` skill gate every cross-fit/smoothing change. |
| KI-012 | S3 | sklearn `check_estimator` — documented subset | **2026-06-26**: applicable checks pass (`tests/test_check_estimator.py`, sklearn ≥ 1.6); inapplicable ones waived with reasons (sparse, 1d/empty/complex input, by-name `n_features`, y-messages). Estimator pickling fixed. |
| KI-013 | S2 | cuDF weak on object/high-cardinality strings | `auto` avoids GPU for those; document. |
| KI-014 | S2 | pandas↔cuDF NaN/dtype semantics differ | parity at allclose; normalize dtypes in `_validation`. |
| KI-015 | S3 | Custom aggregations must be order-independent | warn otherwise; CPU-only; no smoothing. |
| KI-016 | S3 | Multiclass column explosion for large `K` | class-agnostic stats not `×K`; width warning; class subset. |
| KI-017 | S3 | RAPIDS install on Colab is slow/fragile | keep parity job minimal + watchdogged. |
| KI-021 | — | ~~CI red: bare `pytest` can't import `tests`~~ | **Resolved 2026-06-26**: CI ran `pytest tests/` (not `python -m pytest`), so the repo root was off `sys.path` and `tests.conftest` failed to import. Added `pythonpath=["src","."]` to the pytest config. |
| KI-022 | — | ~~`cols="auto"` misses pandas ≥3.0 default string dtype~~ | **Resolved 2026-06-26**: `select_cols` now also selects pandas `StringDtype` (pandas 3.0 types strings as `StringDtype`, not `object`). Verified on sklearn 1.9 / pandas 3.0.3 — full suite green. |

## Environment notes
- Dev box (macOS) is CPU-only: pandas 1.5.2, numpy 1.23.5, **sklearn 1.2.0** (no `TargetEncoder`),
  no RAPIDS. sklearn-parity tests require `scikit-learn>=1.4`; GPU/parity tests run only on Colab.
