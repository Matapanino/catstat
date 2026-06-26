import numpy as np
import pandas as pd
from tests.conftest import make_multiclass

from catstat import TargetEncoder


def test_multiclass_shape_and_names():
    X, y = make_multiclass(classes=3)
    enc = TargetEncoder(cols=["g"], smooth=0.0)
    Xt = enc.fit_transform(X, y)
    assert Xt.shape == (len(X), 3)  # 1 feature * 3 classes
    assert list(enc.get_feature_names_out()) == [
        "g__te_mean__class_0",
        "g__te_mean__class_1",
        "g__te_mean__class_2",
    ]
    assert list(enc.classes_) == [0, 1, 2]


def test_multiclass_probabilities_sum_to_one_full_data():
    X, y = make_multiclass(classes=4, seed=4)
    enc = TargetEncoder(cols=["g"], smooth=0.0, output="numpy").fit(X, y)
    out = enc.transform(X)  # full-data probabilities, raw (smooth=0)
    rowsums = out.sum(axis=1)
    assert np.allclose(rowsums, 1.0)


def test_multiclass_unseen_is_global_class_rates():
    X, y = make_multiclass(classes=3, seed=7)
    enc = TargetEncoder(cols=["g"]).fit(X, y)
    tr = enc.transform(pd.DataFrame({"g": ["NOPE"]})).to_numpy().ravel()
    assert np.allclose(tr, enc.target_mean_)
    assert np.allclose(np.sum(enc.target_mean_), 1.0)
