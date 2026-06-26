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
| KI-018 | S3 | GPU `combination` (tuple keys) forced to CPU | missing-as-value now works on GPU (validated 2026-06-26); only combination remains host-only. |
| KI-019 | — | ~~combination joint-key build is a Python loop~~ | **Resolved 2026-06-26**: C-level `zip` build replaces the per-row genexpr; output byte-identical (max\|Δ\|=0 at 10k–1M), combination `fit_transform` ~1.37–1.55× faster. `docs/verdicts/2026-06-26-pr-a-vectorize-jointkey-verdict.md`. |
| KI-020 | S2 | GPU not faster than CPU up to 1M rows (T4) | host↔device round-trip per OOF fold dominates. `auto` disabled; perf needs on-device keys/folds. `docs/verdicts/2026-06-26-gpu-crossover-verdict.md`. |
| KI-030 | S3 | Numeric TE (0.2.0): `Count`/`Frequency` don't bin; numpy-object & bool route to categorical | `numeric=` is `TargetEncoder`-only. Numeric auto-detection needs real numeric dtypes, so numpy-array input (all-object after `prepare_X`) and bool columns are treated as categorical/direct, not binned. Edges are computed once from full-train X (leakage-safe, ⊥ y). **GPU:** numeric keys are emitted as **strings** — the first Colab T4 run hit `MixedTypeError` (cuDF rejects object-dtype *integer* arrays) with int bin-ids/values; fixed by stringifying keys (matches the validated string-categorical path). CPU/GPU allclose **validated on T4 (2026-06-26)** for `numeric_auto`/`numeric_bin` (max\|Δ\| ~1e-17). |

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
