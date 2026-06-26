import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_binary

from catstat import TargetEncoder


def test_binary_single_column_and_classes():
    X, y = make_binary()
    enc = TargetEncoder(cols=["g"], smooth=0.0, output="numpy")
    out = enc.fit(X, y).transform(X)
    assert out.shape == (len(X), 1)
    assert list(enc.classes_) == [0, 1]
    assert enc.target_type_ == "binary"


def test_binary_values_are_probabilities():
    X, y = make_binary(seed=5)
    enc = TargetEncoder(cols=["g"], smooth=0.0, output="numpy").fit(X, y)
    out = enc.transform(X).ravel()
    assert np.all((out >= 0.0) & (out <= 1.0))
    # smooth=0 -> raw P(y=1 | category)
    p = pd.DataFrame({"g": X["g"], "y": y}).groupby("g")["y"].mean()
    assert np.allclose(out, X["g"].map(p).to_numpy())


def test_binary_unseen_is_global_positive_rate():
    X, y = make_binary(seed=6)
    enc = TargetEncoder(cols=["g"]).fit(X, y)
    tr = enc.transform(pd.DataFrame({"g": ["ZZZ"]}))
    assert tr.iloc[0, 0] == pytest.approx(enc.target_mean_)
    assert enc.target_mean_ == pytest.approx(y.mean())
