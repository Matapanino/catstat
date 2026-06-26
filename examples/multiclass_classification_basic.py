"""Multiclass target encoding: one column per (feature, class) of P(class | category)."""

import numpy as np
import pandas as pd

from catstat import TargetEncoder

rng = np.random.default_rng(2)
n = 1200
region = rng.choice(["north", "south", "east", "west"], size=n)
y = rng.integers(0, 3, size=n)  # 3 classes
X = pd.DataFrame({"region": region})

enc = TargetEncoder(cols=["region"], smooth=10.0, random_state=0)
X_enc = enc.fit_transform(X, y)
print("classes_:", enc.classes_)
print("feature names:", list(enc.get_feature_names_out()))
print("output shape (n_features * n_classes):", X_enc.shape)
print(X_enc.head())
