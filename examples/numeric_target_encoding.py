"""Opt-in, cardinality-aware numeric-column target encoding.

`numeric="auto"` routes each numeric column by cardinality: a low-cardinality integer code is
encoded directly (each value a category), while a high-cardinality continuous feature is
quantile-binned and the bins are target-encoded. `fit_transform` stays leakage-safe (out-of-fold);
bin edges come from the feature values only, never the target.

Run: python3 examples/numeric_target_encoding.py
"""

import numpy as np
import pandas as pd

from catstat import TargetEncoder

rng = np.random.default_rng(0)
n = 2000
rating = rng.integers(1, 6, size=n)  # 5 distinct integer codes -> encoded directly
amount = rng.gamma(2.0, 1.0, size=n)  # continuous, high cardinality -> quantile-binned
y = np.array([0.0, 0.5, 1.0, 2.5, 4.0])[rating - 1] + np.sin(amount) + rng.normal(0, 0.3, n)
X = pd.DataFrame({"rating": rating, "amount": amount})

enc = TargetEncoder(numeric="auto", n_bins=10, random_state=0, output="pandas")
encoded = enc.fit_transform(X, y)  # leakage-safe, out-of-fold

print("routing:", enc.numeric_strategy_)  # {'rating': 'direct', 'amount': 'bin'}
print("amount bin edges:", np.round(enc.bin_edges_["amount"], 2))
print("feature names:", list(enc.get_feature_names_out()))
print(encoded.head().round(3))
