# Verdict: Phase 3b — leave-one-out + ordered (CatBoost) cross-fitting schemes

- Date: 2026-06-26
- Branch: `main`
- Backend: cpu
- Artifacts: `tests/test_scheme.py`
- Roadmap target: `docs/roadmap.md` → Phase 3 (ordered/LOO modes)

## Question
Can leave-one-out and CatBoost-style ordered target statistics be added as alternative,
leakage-safe ways to cross-fit the **mean** on the training set, without disturbing the default
k-fold path or the other statistics?

## Design
New `scheme="kfold" | "loo" | "ordered"` parameter on `TargetEncoder` (default `"kfold"`, so
existing behavior is unchanged). The scheme **only** affects `fit_transform`'s training encodings
for the mean; `transform` (new data) always uses the full-data mean, so `fit().transform()` is
identical across schemes. `loo`/`ordered` are **mean-only** (count/frequency may accompany them;
any other target-dependent stat raises). `loo`: `(cat_sum - y_i + m·prior)/(cat_count - 1 + m)`
(singletons → prior); `ordered`: random permutation, each row from prior rows of its category,
`(prior_sum + a·prior)/(prior_count + a)`, with `a` from `smooth` (default 1 for "auto"/≤0).

## Evidence
- `bash scripts/check.sh`: **ruff clean · 86 passed, 2 skipped (GPU) · examples run**.
  Coverage **90.64%**.
- LOO exact: `["a","a","a"]` with y `[1,2,3]` → `[2.5, 2.0, 1.5]` (each = mean of the *other* rows).
- `transform` uses the full category mean regardless of scheme.
- **Leakage-safe** (both schemes): on a noise category, OOF corr with y < 0.1 while the full-data
  (leaky) path > 0.4.
- `ordered` is deterministic per `random_state` and seed-sensitive.
- binary + multiclass shapes correct; `count` alongside a scheme stays full-data; non-mean stat +
  scheme raises; invalid scheme raises; default is still `"kfold"`.

## Decision
**KEEP** — both schemes are correct, leakage-safe, and opt-in behind a defaulted parameter. The
honesty rule is preserved (they cross-fit the mean only; no smoothing invented for other stats).

## Follow-ups
- `ordered` uses one shared permutation per fit (Python-free, vectorized via groupby cumsum/cumcount);
  multi-permutation averaging (CatBoost's full scheme) is a possible future refinement.
- Optional `sigma` multiplicative noise (category_encoders-style) deferred.
- Next: `set_output("polars")` + PyPI release prep.
