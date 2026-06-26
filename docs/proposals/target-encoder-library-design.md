# `catstat` — Design Proposal: Unified CPU/GPU Statistical Categorical Encoding

> Status: **proposal** (research/planning). No production code exists yet.
> Audience: implementers + reviewers of the first milestones.
> Companion docs: [`evaluation-harness-design.md`](./evaluation-harness-design.md),
> [`self-improvement-loop-design.md`](./self-improvement-loop-design.md),
> [`claude-md-proposal.md`](./claude-md-proposal.md),
> [`skills-proposal.md`](./skills-proposal.md), [`../roadmap.md`](../roadmap.md).

---

## 1. Problem statement

Target encoding (replacing a categorical level with a statistic of the target conditioned on
that level) is one of the most effective tabular features for high-cardinality categoricals,
but the existing implementations each solve only part of the problem:

| Need | sklearn `TargetEncoder` | cuML `TargetEncoder` | `category_encoders` |
|---|---|---|---|
| sklearn-compatible API | ✅ | partial | ✅ |
| Leakage-safe cross-fitting | ✅ (built in) | ✅ (`n_folds`) | ❌ (none by default) |
| CPU | ✅ | ❌ | ✅ |
| GPU | ❌ | ✅ | ❌ |
| One API for both devices | ❌ | ❌ | ❌ |
| Stats beyond mean/probability | ❌ | `{mean,var,median}` | mean (+ encoder variants) |
| Count / frequency encoding | ❌ | ❌ | ✅ (`CountEncoder`) |
| Custom aggregation | ❌ | ❌ | ❌ |

**The gap.** No single library offers (a) **one** sklearn-compatible API that (b) **auto-selects
CPU or GPU** without the user rewriting code, (c) generalizes target encoding to an **arbitrary
set of statistics** (mean, count, frequency, var, std, median, min, max, quantile, skew, custom),
while (d) staying **leakage-safe** via internal cross-fitting, with (e) robust behavior for
missing values, unseen categories, high cardinality, and tiny category counts.

**Value proposition.** `catstat` is that library: write `TargetEncoder(...).fit_transform(X, y)`
once; it runs on pandas/numpy today and on cuDF/CuPy when a GPU is present and worthwhile — same
results (allclose), same column names, same leakage guarantees.

**Why unified CPU/GPU backend selection matters.** Practitioners prototype on a laptop (pandas)
and scale on a GPU box (cuDF). Today that means two code paths (sklearn vs cuML) with *different
parameter names, different fold strategies, and different output types*. A user who switches gets
silently different encodings. `catstat` makes the device an implementation detail behind
`backend="auto"`, and — crucially — **owns its own fold assignment** so that `fit_transform`
produces the *same* out-of-fold encodings on CPU and GPU (cuML's fold strategies differ from
sklearn's, so neither can be delegated to without breaking parity).

**Why generalized statistical encoding matters.** "Target encoding" is just the **mean** of the
target per category. The same group-by machinery trivially yields count, frequency, variance,
std, median, quantiles, skew, or a user callable — each a different, often complementary, signal
(e.g., per-category *variance* captures heteroscedastic risk; *count* captures support). Exposing
these behind one `stats=[...]` parameter turns a single-purpose encoder into a general
category-statistics feature factory — **but only mean/probability admit principled smoothing**
(§7), and the library must be honest about that rather than pretend otherwise.

---

## 2. Existing implementation survey

### 2.1 scikit-learn `TargetEncoder` (since 1.3; multiclass since 1.4)
- **Source:** `sklearn/preprocessing/_target_encoder.py` + Cython `_target_encoder_fast.pyx`;
  tests `sklearn/preprocessing/tests/test_target_encoder.py`.
- **Docs:** <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html>;
  user guide §"Target Encoder". PRs: #25334 (1.3 intro, T. Fan), #26674 (1.4 multiclass, L. Liu).
- **Constructor:** `TargetEncoder(categories="auto", target_type="auto", smooth="auto", cv=5,
  shuffle=True, random_state=None)`.
- **Smoothing:** m-estimate blend `encoding_i = λ_i·mean_i + (1−λ_i)·global_mean`,
  `λ_i = n_i/(n_i + m)`. For fixed `smooth=m`, `m` is the constant pseudo-count. For
  `smooth="auto"`, an **empirical-Bayes** per-category `m_i = σ²_i / τ²` (within-category variance
  over global variance) — high within-category variance ⇒ more shrinkage to the global mean.
  ⚠️ **Verify the exact `auto` derivation against `_target_encoder_fast.pyx` at implementation
  time** (the local dev box has sklearn 1.2, which predates `TargetEncoder`).
- **Cross-fitting:** `fit_transform` is out-of-fold — `KFold` for continuous, `StratifiedKFold`
  for binary/multiclass; then it refits on the full data and stores `encodings_` for `transform`.
  `fit().transform(X)` on the *training* set leaks; `fit_transform` is the safe path.
- **Multiclass:** one-vs-rest; output `(n_samples, n_features × n_classes)`; names
  `"{feature}_{class}"`.
- **Fitted attrs:** `encodings_` (list per feature; multiclass interleaves classes),
  `categories_`, `target_type_`, `classes_`, `target_mean_`, `n_features_in_`,
  `feature_names_in_`.
- **Missing/unseen:** NaN is treated as its own category; unseen → `target_mean_`.
- **Limits:** mean/probability **only**; no count/std/median/custom; no `sample_weight`.

### 2.2 RAPIDS cuML `TargetEncoder` (GPU)
- **Source:** `python/cuml/cuml/preprocessing/TargetEncoder.py` (rapidsai/cuml).
  **Docs:** <https://docs.rapids.ai/api/cuml/stable/> → `cuml.preprocessing.TargetEncoder`.
  *(Version/date specifics from the research pass are approximate — RAPIDS uses CalVer; confirm
  against the pinned release at implementation time.)*
- **Constructor:** `TargetEncoder(n_folds=4, smooth=0, seed=42, split_method="interleaved",
  output_type=None, stat="mean", multi_feature_mode="combination")`.
- **`stat`:** `{"mean","var","median"}`. **`smooth`:** absolute count, m-estimate
  `(sum + smooth·μ)/(n + smooth)`, `0` disables. **`split_method`:**
  `interleaved` (`idx % n_folds`), `random` (seeded), `continuous` (contiguous blocks),
  `customize` (user `fold_ids`). **`multi_feature_mode`:** `combination` (joint group-by → **one**
  output column) or `independent` (per-column). **`output_type`:** `cupy/numpy/cudf/pandas/auto`
  — host transfer happens for `numpy`/`pandas`.
- **GPU strategy:** cuDF group-by + agg per fold (the bottleneck, ~70–80% of `fit_transform`),
  CuPy fold ids, cuDF left-merge back to rows (carries an internal `__INDEX__` to preserve order).
- **Leakage prevention:** each fold encoded from the other `n_folds−1`; `transform` uses
  global stats. **Lesson for us:** cuML's fold assignment ≠ sklearn's ⇒ different OOF values ⇒
  we must own fold assignment for CPU/GPU parity.
- **Pitfalls:** object/string categories are weak on GPU; pinning `categories` falls back to
  CPU; host↔device conversion dominates small inputs; single-GPU; historical
  `ColumnTransformer`/non-default-index bugs.

### 2.3 `category_encoders` (scikit-learn-contrib, ~2.8)
- **Source:** `category_encoders/{target_encoder,count,leave_one_out,m_estimate,cat_boost,glmm}.py`,
  `utils.py`. **Repo:** <https://github.com/scikit-learn-contrib/category_encoders>.
- **`TargetEncoder`:** `min_samples_leaf=20, smoothing=10`, **sigmoid** blend
  `w = σ((n − min_samples_leaf)/smoothing)`, `enc = prior·(1−w) + mean·w`. **No cross-fitting** —
  `fit().transform()` on train **leaks** (documented community footgun).
- **`CountEncoder`:** `normalize` (count vs frequency), `min_group_size`, `combine_min_nan_groups`.
- **`LeaveOneOutEncoder`:** `(Σy − y_i)/(n−1)` at fit, optional `sigma` noise; finer-grained than
  k-fold OOF. **`CatBoostEncoder`:** ordered TS, cumulative `(cumsum − y_i + a·prior)/(cumcount + a)`.
  **`MEstimateEncoder`:** `(n·mean + m·prior)/(n + m)` — the clean single-`m` m-estimate.
- **Base machinery:** `cols=None`→object/category auto-select; `return_df`; `handle_unknown` /
  `handle_missing ∈ {value, return_nan, error}`; ordinal pre-step; `BaseEstimator` +
  supervised/unsupervised mixins.
- **Worth borrowing:** `handle_unknown`/`handle_missing` vocabulary; `return_df`/auto-`cols`;
  `sigma` noise and ordered/LOO encoders (Phase 3). **Avoid:** no-cross-fit default; inconsistent
  smoothing-parameter naming across encoders.

### 2.4 Other references (for Phase 3 / inspiration, not MVP)
- **CatBoost** ordered target statistics (permutation-based, online) — the strongest leakage
  defense for small data; deferred.
- **Kaggle-style** leave-one-out / k-fold target encoders — the de-facto pattern we standardize.
- **GLMM / hierarchical** encoders (`category_encoders.GLMMEncoder`) — out of scope.

---

## 3. Proposed public API

```python
from catstat import TargetEncoder, CountEncoder, FrequencyEncoder

enc = TargetEncoder(
    cols="auto",                      # "auto"→object/category (int opt-in) | list[str] | list[int]
    stats=["mean"],                   # str | list[str] | {name: callable}
    target_type="auto",               # auto | continuous | binary | multiclass
    smooth="auto",                    # "auto" (empirical-Bayes) | float>=0 (m-estimate)
    cv=5,                             # int | sklearn CV splitter | iterable of (train,test) idx
    shuffle=True,
    random_state=42,
    handle_unknown="value",           # value | return_nan | error
    handle_missing="value",           # value | return_nan | error
    multi_feature_mode="independent", # independent | combination (joint)
    min_samples_category=1,           # below this, non-mean stats fall back to the global stat
    backend="auto",                   # auto | cpu | gpu
    output="auto",                    # auto | pandas | numpy | cudf | cupy | polars
)

X_train_enc = enc.fit_transform(X_train, y_train)   # leakage-safe (out-of-fold)
X_test_enc  = enc.transform(X_test)                 # full-data encodings
names        = enc.get_feature_names_out()
```

### Recommended class structure (chosen)
**One generalized `TargetEncoder` + two thin unsupervised wrappers**, sharing a private
`_BaseStatEncoder`:

- **`TargetEncoder`** — supervised, cross-fitted, `stats=[...]` (the generality lives in a
  *parameter*, not a class).
- **`CountEncoder`** — unsupervised (`stat="count"`, no `y`, no `cv`).
- **`FrequencyEncoder`** — `CountEncoder(normalize=True)`.

**Rejected alternatives** (and why): a separate `CategoryStatEncoder`/`GeneralizedTargetEncoder`
duplicates the supervised path for no API gain — fold generality into `TargetEncoder.stats`. A
single mega-class that also does count/frequency conflates supervised (needs `y`, cross-fitted)
and unsupervised (no `y`) semantics, making `fit(X)` vs `fit(X, y)` ambiguous. Three small,
sklearn-named classes give the smallest discoverable surface with one code path.

**Note on `device=`.** The user's sketch had both `backend=` and `device=`; we **fold `device`
into `backend`** (`auto|cpu|gpu`) to shrink the surface, and keep `output=` for the *return
container*. This is documented as a deliberate simplification.

---

## 4. Backend design

- **`backend="auto"` selects GPU iff all hold:** `cudf` **and** `cupy` import successfully; a GPU
  is visible; **and** (the input is *already* cuDF/CuPy **or** `n_rows · n_cols ≥ ~1e6`); **and**
  no CPU-only stat (`skew`, custom callables in Phase 3) is requested. Otherwise CPU. The `~1e6`
  cell threshold is a starting heuristic to be calibrated by the conversion-overhead benchmark
  (§ harness) — it exists to avoid paying a host→device copy that the group-by won't amortize.
- **Explicit beats silent.** `backend="gpu"` with RAPIDS/GPU missing raises a **clear
  `ImportError`** (never a silent CPU fallback — a typo on a GPU box must be visible).
  `backend="auto"` *may* fall back to CPU silently (that is its purpose); the actual engine is
  always exposed in `backend_`.
- **One math implementation, four backend primitives.** All encoding logic (smoothing, OOF
  orchestration, fallbacks, naming) is backend-agnostic and calls a tiny interface implemented
  once per device:
  1. `groupby_agg(frame, keys, target, aggs) -> stats_frame`
  2. `assign_folds(n_rows, splitter, y) -> int_fold_ids`  *(identical output CPU & GPU)*
  3. `merge_encodings(frame, keys, enc_frame) -> frame`  *(order-preserving)*
  4. `to_output(frame, output_type) -> container`
  This mirrors repleafgbm's `BaseSplitBackend` split. Adding a backend = implementing 4 functions.
- **Avoid conversions.** Stay in the input's native container; never implicitly move pandas→cuDF
  unless GPU was *chosen* and the move amortizes. Convert only to satisfy an explicit `output=`.
- **When NOT to use GPU even if available:** small inputs (copy dominates); object/high-cardinality
  string columns (weak cuDF strings); `categories` pinned (cuML-style CPU fallback); a CPU-only
  stat requested. These are encoded directly in the `auto` predicate above.
- **Graceful degradation:** import guards isolate `cudf`/`cupy` to `backends/_gpu.py`; the core
  and CPU path never import them. The package installs and runs fully on CPU-only machines.

---

## 5. Internal architecture

```
src/catstat/
  __init__.py            # exports TargetEncoder, CountEncoder, FrequencyEncoder, __version__
  py.typed               # PEP 561 marker (ship inline types)
  _base.py               # _BaseStatEncoder(BaseEstimator, TransformerMixin): the fit/transform/
                         #   fit_transform skeleton, validation hookup, dispatch, feature names
  target_encoder.py      # TargetEncoder      (public, supervised, cross-fitted)
  count_encoder.py       # CountEncoder       (public, unsupervised)
  frequency_encoder.py   # FrequencyEncoder   (= CountEncoder(normalize=True))
  _stats.py              # STAT REGISTRY: name → StatSpec(agg, smoothing_policy, class_expanded,
                         #   gpu_supported, min_samples_default, global_fallback)
  _smoothing.py          # m-estimate (fixed) + empirical-Bayes (auto); mean/probability only
  _cross_fit.py          # deterministic fold assignment (CPU==GPU) + OOF orchestration + CV resolve
  _validation.py         # input checks, target_type inference (type_of_target), dtype normalization
  _feature_names.py      # get_feature_names_out + naming scheme / feature_name_combiner
  _typing.py             # type aliases, Protocols (ArrayLike, FrameLike, Backend)
  backends/
    __init__.py
    _dispatch.py         # backend="auto" predicate + the 4-primitive interface (Protocol)
    _cpu.py              # pandas/numpy: groupby_agg, assign_folds, merge_encodings, to_output
    _gpu.py              # cudf/cupy equivalents (Phase 2); import-guarded
```

**Design rule:** the *statistics and leakage logic live in backend-agnostic modules*
(`_stats`, `_smoothing`, `_cross_fit`, `_base`). Only the four primitives in `backends/` know
about pandas vs cuDF. This is what makes "one API, two devices, same results" tractable.

`_stats.StatSpec` (the conceptual core) looks like:
```python
@dataclass(frozen=True)
class StatSpec:
    name: str
    agg: Callable | str                # pandas/cudf groupby agg name or callable
    smoothing: Literal["mean", "none", "dispersion_optin"]
    class_expanded: bool               # multiclass: emit one column per class?
    gpu_supported: bool
    min_samples_default: int
    global_fallback: Callable          # value used for unseen / tiny-n categories
```

---

## 6. Data model and fitted attributes

Set on `fit`/`fit_transform` (sklearn convention — trailing underscore):

| attribute | meaning |
|---|---|
| `categories_` | list per feature: unique levels seen at fit (NaN included when `handle_missing="value"`) |
| `n_features_in_` | number of encoded input features |
| `feature_names_in_` | input names (when X is a DataFrame) |
| `target_type_` | resolved `continuous` / `binary` / `multiclass` (None for unsupervised) |
| `classes_` | class labels (binary/multiclass), else `None` |
| `encodings_` | nested map `feature → stat → [class →] category → value` (the learned table) |
| `global_stats_` | per-stat global fallback; `target_mean_` is an alias for the mean |
| `smooth_` | resolved smoothing (scalar for fixed; per-category vector for `"auto"`) |
| `cv_` | resolved splitter actually used |
| `backend_` | `"cpu"` or `"gpu"` — the engine actually selected |
| `stats_` | resolved list of stat names |
| `multi_feature_mode_` | `"independent"` / `"combination"` |
| `feature_names_out_` | output column names |

---

## 7. Smoothing design — *the honest rule*

> **Only mean/probability statistics admit principled smoothing.** Every other statistic gets a
> **small-sample fallback to the global statistic** (governed by `min_samples_category`), and
> **order/shape/custom statistics never blend** — blending a min, max, median, or quantile toward
> a global value yields a number that was *never observed* and destroys the statistic's meaning.

**Mean / probability (`mean`, binary prob, multiclass per-class prob):**
- Fixed `smooth=m` (m-estimate): `enc_i = (n_i·mean_i + m·μ)/(n_i + m)`.
- `smooth="auto"` (empirical-Bayes): per-category `m_i = σ²_i/τ²`, then the same blend. Computed
  **per fold** inside `fit_transform`. *(Replicate sklearn's exact formula; verify vs source.)*
- A future `smoothing="sigmoid"` option can reproduce `category_encoders`'
  `w = σ((n − min_samples_leaf)/smoothing)`.

**count / frequency:** **no** smoothing — a count is exact; "shrinking a count toward a global
count" is meaningless. (Optional Laplace add-α for frequency is a *separate* future knob, default
off.)

**var / std (dispersion):** there is **no clean Bayesian shrinkage** here. Default = **no
shrinkage** + small-sample fallback (`n<2 → global`). An **opt-in, explicitly-labeled heuristic**
`λ`-shrink (`var_enc = λ_i·var_i + (1−λ_i)·global_var`, same `λ_i = n_i/(n_i+m)`) is available for
users who want it — documented as a heuristic, not principled.

**median / min / max / quantile / skew / custom:** **never blend.** Default = raw statistic with
fallback to the **global** statistic when `n < min_samples_category` (skew needs `n≥3`). Min/max
blending is *actively wrong* (produces unobserved values). `skew` is high-variance; documented as
"use with care." Custom callables must be order-independent (warn otherwise); no smoothing; global
fallback.

This honesty is itself a feature — the library does not pretend all statistics are equally
regularizable, which is exactly the trap a naive generalization of target encoding falls into.

---

## 8. Cross-fitting design (leakage safety = invariant #1)

- **`fit_transform(X, y)` is out-of-fold.** For each fold: compute the per-category encodings from
  the **complement** of the fold, apply them to the held-out fold; concatenate all held-out
  pieces into the output (same shape/order as input). **Then** refit encodings on the **full**
  data and store in `encodings_` for later `transform`. `smooth="auto"` variance is computed
  **per fold**, never from full data.
- **`fit(X, y).transform(X) ≠ fit_transform(X, y)`.** `fit` learns full-data encodings; applying
  them back to the *training rows* leaks the target into the feature. `fit_transform` is the only
  leakage-safe path for the training set; `transform` is for *new* data. This asymmetry is
  intentional (it matches sklearn) and must be loudly documented.
- **Splitters by target type:** `KFold` (continuous), `StratifiedKFold` (binary/multiclass,
  stratified on the **original** `y`, not the OvR-binarized columns). `cv` accepts an int, a
  splitter object, or an iterable of `(train_idx, test_idx)`.
- **Determinism:** `random_state` + `shuffle` flow through a `check_random_state` helper; same
  seed ⇒ identical folds ⇒ identical output. **Never** call global numpy RNG.
- **Unsupervised encoders** (`Count`/`Frequency`): **no cross-fitting** — there is no target, so
  no target leakage; `fit_transform == fit().transform()`. Documented.
- **Future:** CatBoost-style **ordered** TS and **leave-one-out** as opt-in modes (Phase 3).
- **Implementation leakage traps** (the `leakage-audit` skill checks these): per-fold global
  stats must exclude the held fold; auto-smoothing variance must be per-fold; row order must be
  preserved on merge (carry an explicit index on GPU, as cuML does); `target_mean_`/global stats
  for unknowns must come from training folds only, never the transformed set.

---

## 9. Multiclass design

- **One-vs-rest:** binarize `y` into `K` indicator columns; for each `(feature, class)` compute the
  per-class statistic (mean of the indicator = `P(class | category)`).
- **Output shape (independent mode):** **class-expanded** stats (mean→prob, var, std, …) emit
  `n_features × n_classes` columns; **class-agnostic** stats (`count`, `frequency` — properties of
  the *category*, not the target) emit `n_features` columns (not `×K`), avoiding duplicate
  columns. Total = `Σ_stats n_features × (n_classes if class_expanded else 1)`.
- **Feature names:** `"{feat}__te_{stat}__class_{label}"` for class-expanded stats;
  `"{feat}__{stat}"` for class-agnostic.
- **Memory:** columns scale with `K`; emit a warning when output width exceeds a threshold; allow
  selecting a subset of classes. For very large `K`, recommend `stats=["mean"]` only.
- **GPU feasibility:** fine (per-class group-by), but device memory scales with `K`; documented.

---

## 10. Multiple-statistic design

- `stats` is a `str`, `list[str]`, or `{name: callable}`. Output one column per `(feature, stat)`
  (× class where class-expanded).
- **Naming (default):** supervised `"{feat}__te_{stat}"`; unsupervised `"{feat}__count"` /
  `"{feat}__freq"`; class-expanded `"{feat}__te_{stat}__class_{label}"`. Override with a
  `feature_name_combiner` callable (mirrors sklearn 1.5 `OneHotEncoder`). Example:
  `country__te_mean`, `country__te_count`, `country__te_std`, `country__te_skew`.
- **Custom aggregations:** `stats={"p90": lambda v: np.quantile(v, 0.90)}`. Limitations: must be a
  reduction over the group's target values, must be **order-independent** (warn otherwise), gets
  **no smoothing** and a global-value fallback, and is **CPU-only** until/unless a GPU-expressible
  form exists (cuDF UDFs are constrained). Phase 3.

---

## 11. Missing and unseen category behavior

- **`handle_missing` (NaN at fit):** `"value"` (default) → NaN is its own category with its own
  learned encoding; `"return_nan"` → output NaN; `"error"` → raise.
- **`handle_unknown` (unseen at transform):** `"value"` (default) → per-stat global fallback;
  `"return_nan"`; `"error"`.
- **What `"value"` means per stat** (the fallback table):

  | stat | unknown / tiny-n fallback | rationale |
  |---|---|---|
  | mean, binary/mc prob | global mean / probability (`target_mean_`) | the prior when there's no category signal |
  | count | **0** | an unseen category was observed 0 times in training |
  | frequency | **0.0** | 0 occurrences ⇒ 0 frequency |
  | var / std | global var / std | safest dispersion prior (a 1-sample category has undefined variance) |
  | median / min / max / quantile | global statistic over all `y` | order stats have no meaningful blend |
  | skew | global skew (else 0 if undefined) | needs `n≥3`; otherwise the global shape |
  | custom | global value of the callable | consistent with the above |

- A category that was missing-at-fit but appears at transform, when `handle_missing="value"`: if
  NaN was seen at fit it has a learned encoding; if it was never seen, it is treated as unknown.

---

## 12–18 (covered in companion docs)

Sections 12 (evaluation harness), 13 (self-improvement loop), 14 (CLAUDE.md), 15 (skills),
16 (MVP scope), 17 (testing plan), and 18 (benchmark plan) are specified in
[`evaluation-harness-design.md`](./evaluation-harness-design.md),
[`self-improvement-loop-design.md`](./self-improvement-loop-design.md),
[`claude-md-proposal.md`](./claude-md-proposal.md), [`skills-proposal.md`](./skills-proposal.md),
and [`../roadmap.md`](../roadmap.md).

---

## 19. Risks and design decisions

| risk | mitigation / decision |
|---|---|
| **Leakage via implementation detail** (the dominant risk) | OOF in `fit_transform`; per-fold auto-smoothing; order-preserving merge; dedicated `test_cross_fit_no_leakage` (recompute each fold from its complement) + `leakage-audit` skill gating every change to `_cross_fit`/`_smoothing` |
| Over-/under-smoothing | `smooth="auto"` default (empirical-Bayes) + benchmark on rare-category datasets; sigmoid as future option |
| Multiclass column explosion | class-agnostic stats not `×K`; width warning; class subset; recommend mean-only for large `K` |
| GPU dependency complexity | RAPIDS isolated behind `backends/_gpu.py` + extras `[gpu]`; CPU path never imports it; `backend="auto"` degrades silently, explicit `"gpu"` errors loudly |
| pandas↔cuDF dtype/NaN differences | parity tests at **allclose, not bitwise**; categorical/object normalization in `_validation`; document NaN semantics |
| sklearn `check_estimator` burden | supervised transformers + multi-output make *full* compliance unrealistic; target a **documented subset** (clone, get/set_params, Pipeline, ColumnTransformer, set_output) |
| custom-aggregation correctness | require order-independence (warn), no smoothing, global fallback, CPU-only — Phase 3 |
| Benchmark noise | ≥5 reps, median+spread, pinned seeds/versions/SHA; never change a default on a single run |
| False positives in the self-improvement loop | `experiment_log.md` records null/negative results; require repeated runs before default changes (see loop doc) |
| One class vs many | **decided: one `TargetEncoder` + Count/Frequency wrappers** over a shared `_BaseStatEncoder` |

---

## 20. Recommended implementation order (PR-sized)

1. **PR1 — packaging skeleton:** `pyproject.toml` (src-layout, hatchling, extras
   `dev`/`bench`/`gpu`/`docs`, ruff line-length 100, coverage gate), `src/catstat/__init__.py`,
   `py.typed`, `scripts/check.sh`, CI stub.
2. **PR2 — validation + stat registry:** `_validation.py` (target-type inference, dtype norm),
   `_stats.py` (mean/count/frequency specs). Unit tests for both.
3. **PR3 — CPU backend primitives:** `backends/_cpu.py` + `backends/_dispatch.py` (CPU-only auto).
4. **PR4 — mean `TargetEncoder` (regression):** `_base.py`, `_smoothing.py` (fixed + auto),
   `_cross_fit.py`, `target_encoder.py`. Tests: regression correctness, smoothing, determinism,
   **leakage (OOF reconstruction)**.
5. **PR5 — binary + multiclass:** OvR expansion, `classes_`, feature names. Tests: binary,
   multiclass shape/names.
6. **PR6 — unknown/missing + `get_feature_names_out` + `_feature_names.py`.** Tests: the fallback
   table, names for single/multi/multiclass/multi-stat.
7. **PR7 — `CountEncoder` / `FrequencyEncoder`.** Tests: values, unseen→0, normalize.
8. **PR8 — sklearn compat:** `set_output`, Pipeline/ColumnTransformer, partial `check_estimator`.
9. **PR9 — harness:** `benchmarks/datasets.py` + `run_benchmarks.py` + `ledger.py` +
   `compare_results.py`; commit first baseline JSON; first verdict doc.  → **End of M0 (MVP).**
10. **Phase 2:** GPU backend (`backends/_gpu.py`), var/std/median/min/max,
    `multi_feature_mode="combination"`, `scripts/colab_gpu_parity.{sh,py}`, CPU/GPU parity tests,
    benchmark baselines.
11. **Phase 3:** quantile/skew/custom, ordered/LOO encoding, advanced metadata routing /
    `set_output("polars")`, estimator-check hardening, PyPI release + docs.

---

### Appendix — chosen names & first steps (quick reference)
- **Package:** `catstat` (chosen). Alts: `targetstats`, `flex_target_encoder`, `statencoder`.
- **MVP classes:** `TargetEncoder`, `CountEncoder`, `FrequencyEncoder` (+ private `_BaseStatEncoder`).
- **First 5 files:** `pyproject.toml` → `_validation.py` → `_stats.py` → `_cross_fit.py` →
  `target_encoder.py` (with `backends/_cpu.py` close behind).
- **First benchmark file:** `benchmarks/datasets.py`.
- **First milestone:** M0 = CPU mean encoder, leakage-safe, reg/bin/mc, count/frequency, green
  `scripts/check.sh`, one committed baseline JSON.
