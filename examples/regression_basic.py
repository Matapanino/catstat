"""Leakage-safe target encoding for a regression target."""

import numpy as np
import pandas as pd

from catstat import TargetEncoder

rng = np.random.default_rng(0)
n = 1000
city = rng.choice(["tokyo", "osaka", "kyoto", "nara", "kobe"], size=n)
effect = {"tokyo": 5.0, "osaka": 3.0, "kyoto": 2.0, "nara": 1.0, "kobe": 0.5}
y = np.array([effect[c] for c in city]) + rng.normal(0, 0.5, n)
X = pd.DataFrame({"city": city})

enc = TargetEncoder(cols=["city"], smooth="auto", cv=5, random_state=42)
X_oof = enc.fit_transform(X, y)  # out-of-fold (leakage-safe) on the training set
print("feature names:", list(enc.get_feature_names_out()))
print("backend:", enc.backend_, "| global target mean:", round(enc.target_mean_, 3))
print(X_oof.head())

new = pd.DataFrame({"city": ["tokyo", "sapporo"]})  # 'sapporo' is unseen -> global mean
print("transform new:\n", enc.transform(new))
