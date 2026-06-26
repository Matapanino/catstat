"""Unsupervised count / frequency encoding (no target, no cross-fitting)."""

import numpy as np
import pandas as pd

from catstat import CountEncoder, FrequencyEncoder

rng = np.random.default_rng(3)
n = 1000
sku = rng.choice([f"sku_{i}" for i in range(10)], size=n)
X = pd.DataFrame({"sku": sku})

count = CountEncoder(cols=["sku"]).fit_transform(X)
freq = FrequencyEncoder(cols=["sku"]).fit_transform(X)
print("count names:", list(CountEncoder(cols=["sku"]).fit(X).get_feature_names_out()))
print("freq names:", list(FrequencyEncoder(cols=["sku"]).fit(X).get_feature_names_out()))
print("count head:\n", count.head())
print("freq head:\n", freq.head())

# unseen category -> 0
unseen = CountEncoder(cols=["sku"]).fit(X).transform(pd.DataFrame({"sku": ["sku_999"]}))
print("unseen count:", unseen.iloc[0, 0])
