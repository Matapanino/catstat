"""0.5.x features: shape stats (skew/kurt), WOE, sigmoid smoothing, max_classes, laplace_alpha.

Runnable on CPU (part of scripts/check.sh). The cuDF device-resident path uses the same API --
pass a cudf.DataFrame instead and the encode stays on the GPU (see the README quickstart).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from catstat import FrequencyEncoder, TargetEncoder

rng = np.random.default_rng(42)
n = 1_200
city = rng.choice(["tokyo", "osaka", "nagoya", "sapporo", "fukuoka"], size=n)
X = pd.DataFrame({"city": city})

# --- shape statistics: per-category spread and tail behavior of a continuous target ------------
scale = {"tokyo": 1.0, "osaka": 2.0, "nagoya": 0.5, "sapporo": 3.0, "fukuoka": 1.5}
y_reg = np.array([rng.gamma(2.0, scale[c]) for c in city])
enc = TargetEncoder(
    cols=["city"],
    stats=["mean", "var", "skew", "kurt"],
    smooth="sigmoid",  # category_encoders' blend; or ("sigmoid", k, f), a float m, or "auto"
    cv=5,
    random_state=0,
    output="pandas",
)
Xt = enc.fit_transform(X, y_reg)  # out-of-fold (leakage-safe)
print("shape stats + sigmoid smoothing:", list(enc.get_feature_names_out()))
print(Xt.head(3).round(3).to_string())

# --- WOE: the credit-scoring encoding for binary targets ---------------------------------------
p = {"tokyo": 0.7, "osaka": 0.45, "nagoya": 0.3, "sapporo": 0.55, "fukuoka": 0.2}
y_bin = (rng.uniform(size=n) < np.array([p[c] for c in city])).astype(int)
woe = TargetEncoder(cols=["city"], stats=["mean", "woe"], cv=5, random_state=0, output="pandas")
print("\nWOE (binary):")
print(woe.fit_transform(X, y_bin).head(3).round(3).to_string())

# --- max_classes: cap the multiclass one-vs-rest expansion -------------------------------------
y_mc = rng.choice(np.arange(6), size=n, p=np.array([1, 2, 3, 4, 5, 6]) / 21.0)
mc = TargetEncoder(cols=["city"], cv=5, random_state=0, max_classes=3, output="pandas")
mc.fit(X, y_mc)
print("\nmax_classes=3 keeps the most frequent classes:", list(mc.encoded_classes_))
print("columns:", list(mc.get_feature_names_out()))

# --- laplace_alpha: smoothed frequencies with a nonzero unseen fallback ------------------------
freq = FrequencyEncoder(cols=["city"], laplace_alpha=1.0).fit(X)
probe = pd.DataFrame({"city": ["tokyo", "NEVER_SEEN"]})
print("\nLaplace-smoothed frequency (unseen gets alpha/(n+aK), not 0):")
print(freq.transform(probe).round(5).to_string())
