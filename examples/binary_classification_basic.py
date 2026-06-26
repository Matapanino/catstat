"""Target encoding for a binary classification target (encodes P(positive | category))."""

import numpy as np
import pandas as pd

from catstat import TargetEncoder

rng = np.random.default_rng(1)
n = 1000
plan = rng.choice(["free", "basic", "pro", "enterprise"], size=n)
churn_p = {"free": 0.4, "basic": 0.25, "pro": 0.1, "enterprise": 0.05}
y = (rng.uniform(size=n) < np.array([churn_p[p] for p in plan])).astype(int)
X = pd.DataFrame({"plan": plan})

enc = TargetEncoder(cols=["plan"], smooth="auto", random_state=0)
X_oof = enc.fit_transform(X, y)
print("classes_:", enc.classes_, "| names:", list(enc.get_feature_names_out()))
print("global positive rate:", round(enc.target_mean_, 3))
print(X_oof.head())
