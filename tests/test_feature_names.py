import numpy as np
import pandas as pd
from tests.conftest import make_multiclass, make_regression

from catstat import CountEncoder, FrequencyEncoder, TargetEncoder


def test_single_feature_single_stat():
    X, y = make_regression()
    enc = TargetEncoder(cols=["g"]).fit(X, y)
    assert list(enc.get_feature_names_out()) == ["g__te_mean"]


def test_multi_feature_multi_stat():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.choice(list("xy"), 200), "b": rng.choice(list("pq"), 200)})
    y = rng.normal(size=200)
    enc = TargetEncoder(cols=["a", "b"], stats=["mean", "count"]).fit(X, y)
    # feature-major, then stat order
    assert list(enc.get_feature_names_out()) == [
        "a__te_mean",
        "a__count",
        "b__te_mean",
        "b__count",
    ]


def test_multiclass_names_have_class_suffix():
    X, y = make_multiclass(classes=3)
    enc = TargetEncoder(cols=["g"]).fit(X, y)
    names = list(enc.get_feature_names_out())
    assert names == ["g__te_mean__class_0", "g__te_mean__class_1", "g__te_mean__class_2"]


def test_count_and_frequency_names():
    X, _ = make_regression()
    assert list(CountEncoder(cols=["g"]).fit(X).get_feature_names_out()) == ["g__count"]
    assert list(FrequencyEncoder(cols=["g"]).fit(X).get_feature_names_out()) == ["g__freq"]


def test_names_length_matches_output_width():
    X, y = make_multiclass(classes=3)
    enc = TargetEncoder(cols=["g"], stats=["mean", "count"])
    out = enc.fit_transform(X, y)
    assert out.shape[1] == len(enc.get_feature_names_out())
