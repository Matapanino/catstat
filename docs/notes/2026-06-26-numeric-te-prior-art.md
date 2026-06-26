# Prior-Art Research: Opt-In Cardinality-Aware Numeric Target Encoding

**Date:** 2026-06-26
**Author:** research scout (Claude Sonnet 4.6)
**Purpose:** Pre-implementation survey for adding numeric-column support to `catstat`'s
`TargetEncoder`. Do not modify `src/` based on this note alone.

---

## Gap Statement

No mainstream encoding library offers a single, leakage-safe encoder that (a) accepts raw
numeric input columns and (b) automatically routes low-cardinality numerics to direct
category-level encoding while binning high-cardinality ones before applying target encoding —
all within one `fit_transform` that is out-of-fold throughout. The existing idiom requires
two separate sklearn steps (`KBinsDiscretizer` → `TargetEncoder` in a Pipeline), which
introduces a subtle OOF leakage asymmetry (see below), and every tool surveyed either drops
non-categorical inputs silently or requires explicit pre-discretization by the user.

---

## Tool Survey: Numeric Columns + Binning + Leakage Safety

| Tool | Accepts raw numeric cols? | Auto-bins? | Cardinality routing? | Leakage-safe? |
|---|---|---|---|---|
| **sklearn `KBinsDiscretizer` → `TargetEncoder` (Pipeline)** | Yes, via separate step | Manual (`uniform`/`quantile`/`kmeans`, `n_bins`) | No — user must split cols manually | Mostly: bin edges fit on full train, not per TE-subfold (subtle X-leakage; y-leakage = zero) |
| **sklearn `TargetEncoder` alone** | Only if pre-discretized to category-like integers | No | No | Yes (`fit_transform` OOF) |
| **`category_encoders` `TargetEncoder`** | `cols=None` selects string/object only; numeric ignored | No | No | No (no cross-fitting at all) |
| **H2O `TargetEncoder`** | No — docs say "any non-categorical columns are automatically dropped" | No | No | Yes (fold-based, leaky path documented) |
| **feature-engine `EqualFrequencyDiscretiser`** | Yes (numeric only) | Quantile, `q` bins (no default) | No | User responsibility (separate step) |
| **TALENT Q_bins / T_bins (2024, arxiv)** | Yes | Q_bins = quantile; T_bins = supervised (uses y!) | No — research preprocessing, not encoder class | T_bins is NOT safe (uses y for bin boundaries) |
| **Automunge** | Yes (normalizes by default) | Equal-population and fixed-width available | Partial: hashing for highest-cardinality; no numeric-to-TE routing | User responsibility |
| **OptBinning `ContinuousOptimalBinning`** | Yes | Supervised (uses y) optimal bins | No | NOT safe unless refitted per fold by user |
| **`catstat` (current)** | Only if pre-discretized | No | No | Yes (OOF, owns fold assignment) |

Sources:
- sklearn KBinsDiscretizer: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html
- sklearn TargetEncoder: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
- sklearn cross-fitting example: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html
- sklearn TE + cardinality routing example (255-threshold): https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder.html
- category_encoders TargetEncoder: https://contrib.scikit-learn.org/category_encoders/targetencoder.html
- H2O TargetEncoder: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/target-encoding.html
- feature-engine EqualFrequencyDiscretiser: https://feature-engine.trainindata.com/en/1.7.x/user_guide/discretisation/EqualFrequencyDiscretiser.html
- TALENT (arxiv 2407.04057): https://arxiv.org/pdf/2407.04057
- OptBinning continuous: https://gnpalencia.org/optbinning/binning_continuous.html
- Automunge: https://github.com/Automunge/AutoMunge

---

## Idiomatic Sklearn Approach and Its Leakage Caveat

The idiomatic sklearn pattern is a two-step Pipeline:

```python
Pipeline([
    ("disc", KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")),
    ("te",   TargetEncoder(cv=5, smooth="auto")),
])
```

When this Pipeline is fitted via `fit_transform(X_train, y_train)`, the Pipeline calls
`disc.fit_transform(X_train)` first (on the full training set), then passes the discretized
output to `te.fit_transform(disc_X, y_train)`. `TargetEncoder`'s internal cross-fitting then
splits `disc_X` into 5 folds and fits each fold from the other 4 — but the bin edges were
already computed from all 5 folds' X values combined.

**y-leakage: zero.** Quantile bin edges depend only on the distribution of X values, not y.

**X-distribution leakage: present but mild.** Fold-out samples contribute to the quantile
edge computation, so the binning step has seen a slightly broader X distribution than each
TE-subfold's training complement would provide. For large N this difference is negligible
(bin edges are stable); for small N or very few unique values per bin it is non-trivial. The
standard remedy — fitting `KBinsDiscretizer` inside the cross-validation loop — is not
possible with sklearn's architecture because `TargetEncoder`'s internal CV is not exposed.
`catstat` can fix this by running binning per fold before the OOF aggregation step, achieving
strictly correct OOF fidelity.

sklearn's own documentation confirms that `KBinsDiscretizer` must be fitted to the training
set only: "to avoid information leakage, the preprocessing procedure [should be] repeated for
each train/test split." The Pipeline + TE combination satisfies this at the outer model-selection
level but not at TE's internal subfold level.

Source: https://medium.com/@silva.f.francis/avoiding-data-leakage-in-cross-validation-ba344d4d55c0

---

## Recommended Defaults

### 1. Binning Strategy: `quantile` (equal-frequency)

**Recommendation:** default to quantile (equal-frequency) binning.

**Justification:**
- Uniform (equal-width) bins allocate most of their range budget to the sparse tails of skewed
  distributions, producing near-empty bins that have unstable target-mean estimates and
  disproportionate smoothing pull. A Kaggle grandmaster feature-engineering guide explicitly
  recommends quantile bucketing for skewed data as the starting point
  (https://www.kaggle.com/getting-started/183076).
- Quantile binning by construction gives ~N/k samples per bin, so every bin's mean estimate
  has the same expected variance — the smoothing formula is then applied fairly across bins.
- sklearn `KBinsDiscretizer` defaults to `strategy="quantile"`, confirming this as the
  established safe default (https://sklearn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html).
- The Google ML crash course recommends quantile bucketing for skewed numeric features
  (https://developers.google.com/machine-learning/crash-course/numerical-data/binning).
- Exception: k-means binning beats quantile on some regression benchmarks with highly
  multi-modal distributions (arxiv 2505.12460), but is harder to explain and slower to compute;
  offer as an option, not the default.

### 2. n_bins Default: `10`

**Recommendation:** default `n_bins=10` for continuous features.

**Justification:**
- sklearn's `KBinsDiscretizer` default of `n_bins=5` is widely regarded as too coarse for
  target encoding: it collapses the continuous range into 5 levels, discarding most of the
  ordinal signal. Feature-engineering practitioners use 10–20 bins as a starting point
  (https://medium.com/@adnan.mazraeh1993/advanced-considerations-in-binning-f685bcf1813e).
- Per-bin sample size rule of thumb: for m-estimate / empirical-Bayes smoothing to be stable,
  each bin should have at least 30–50 samples. With `n_bins=10` and a training set of N rows,
  each bin has ~N/10 samples. At N=300 (smallest reasonable training set), that is 30
  samples/bin — right at the boundary. At N=1000, it is 100/bin (comfortable).
- Going to `n_bins=20` gives finer resolution but halves per-bin sample counts; it is a
  reasonable choice for N≥2000 (100 samples/bin). Exposing `n_bins` as a user parameter (with
  default 10) covers both cases.
- OptBinning's supervised approach does not fix `n_bins` but uses a pre-binning step with a
  larger max then optimizes — its default pre-bins are ~20
  (https://gnpalencia.org/optbinning/binning_continuous.html).

### 3. Cardinality Threshold: `≤ 10` unique values → direct encoding

**Recommendation:** default `numeric_cardinality_threshold=10`.

**Justification:**
- There is no single authoritative standard. The closest published guidance comes from two
  sources:
  1. **sklearn's own cardinality example** uses a threshold of 255 for routing to
     `TargetEncoder` vs `OrdinalEncoder`, but this is driven by `HistGradientBoostingRegressor`'s
     native categorical limit, not a general recommendation
     (https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder.html).
  2. **Domain practitioner heuristic** (survey, multiple sources): a feature is treated as
     nominal/categorical when its unique-count/total-row ratio is ≤ 0.05; as ordinal when ≤ 0.2
     (https://medium.com/data-science/dealing-with-features-that-have-high-cardinality-1bc6d8fd7b13).
- For numeric columns specifically, values ≤ 10 unique integers correspond to typical discrete
  codes (0–9, weekday-of-week, star-ratings, 5-point scales). These should be target-encoded
  directly — one row per unique value — not binned, because binning would merge semantically
  distinct ordinal values.
- At 11–15 unique values the distinction blurs; the threshold is a heuristic that should be
  exposed as a tunable parameter.
- A ratio-based threshold (unique/n ≤ some fraction) is more robust for variable-N datasets
  and can be offered as an alternative (`numeric_cardinality_mode="ratio"` with default 0.05),
  but the absolute threshold is simpler as the primary default.

---

## Confirm / Refute: "No Mainstream Library Offers Cardinality-Aware Numeric TE in One Leakage-Safe Encoder"

**CONFIRMED.** Every library surveyed either:
- Requires explicit pre-discretization by the user before calling the encoder (sklearn, H2O,
  category_encoders, feature-engine),
- Silently drops non-categorical inputs (H2O, category_encoders with default `cols`), or
- Uses y-informed binning (TALENT T_bins, OptBinning) which is not leakage-safe without
  per-fold refitting.

The closest approximation is the sklearn two-step Pipeline (`KBinsDiscretizer` + `TargetEncoder`),
but it requires the user to (a) identify numeric columns, (b) choose bin count and strategy,
(c) wire up `ColumnTransformer` manually, and (d) accept the subtle per-fold X-distribution
leakage described above. No single `fit_transform` call handles it end-to-end.

`catstat`'s proposed feature is a genuine gap.

---

## 5-Bullet Summary

1. **sklearn two-step is the only established idiom.** `KBinsDiscretizer(strategy="quantile") →
   TargetEncoder` via Pipeline is the de-facto pattern, but it requires manual column
   identification and `ColumnTransformer` wiring. No cardinality routing is built in.

2. **y-leakage is zero for X-only bin edges; X-distribution leakage is real but mild.** Quantile
   edges are target-independent. However, fitting `KBinsDiscretizer` on the full training set
   (before `TargetEncoder`'s internal CV) means fold-out X values slightly influence bin
   boundaries. `catstat`'s per-fold binning during OOF orchestration would be strictly tighter.

3. **No library auto-routes numeric columns by cardinality.** H2O drops them; category_encoders
   ignores them; feature-engine requires separate discretization; TALENT's T_bins uses y
   (supervised, not safe). The cardinality-aware routing is a genuine gap.

4. **Recommended defaults (all adjustable as parameters):**
   - Binning strategy: `quantile` (equal-frequency) — safer than uniform for skewed features,
     is sklearn's KBinsDiscretizer default, endorsed by Google and Kaggle practitioners.
   - n_bins: `10` — gives ~N/10 samples/bin; coarse enough for smoothing stability on small
     datasets, fine enough not to lose ordinal signal.
   - Cardinality threshold: `10` unique values — direct encoding below, binning above; matches
     typical discrete-code feature semantics.

5. **TALENT (2024) distinguishes Q_bins (safe) from T_bins (supervised, leaky).** This confirms
   the design principle: bin edges must come from X only; supervised (y-informed) bin edges
   require per-fold refitting to be leakage-safe, adding implementation complexity that is not
   justified for a default.
