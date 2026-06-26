import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_regression

from catstat import TargetEncoder


def test_smooth_zero_is_raw_category_mean():
    X, y = make_regression(seed=0)
    enc = TargetEncoder(cols=["g"], smooth=0.0, output="numpy")
    enc.fit(X, y)  # full-data table
    out = enc.transform(X).ravel()
    means = pd.DataFrame({"g": X["g"], "y": y}).groupby("g")["y"].mean()
    expected = X["g"].map(means).to_numpy()
    assert np.allclose(out, expected)


def test_fixed_smooth_matches_m_estimate():
    X, y = make_regression(seed=1)
    m = 7.0
    enc = TargetEncoder(cols=["g"], smooth=m, output="numpy").fit(X, y)
    out = enc.transform(X).ravel()
    df = pd.DataFrame({"g": X["g"], "y": y})
    grp = df.groupby("g")["y"]
    n = grp.count()
    mean = grp.mean()
    gmean = y.mean()
    enc_table = (n * mean + m * gmean) / (n + m)
    expected = X["g"].map(enc_table).to_numpy()
    assert np.allclose(out, expected)


def test_auto_smoothing_is_convex_combination():
    X, y = make_regression(seed=2, k=8)
    enc = TargetEncoder(cols=["g"], smooth="auto", output="numpy").fit(X, y)
    out = enc.transform(X).ravel()
    df = pd.DataFrame({"g": X["g"], "y": y})
    mean = df.groupby("g")["y"].mean()
    gmean = y.mean()
    # each encoding lies between its category mean and the global mean (a convex blend)
    lo = np.minimum(X["g"].map(mean).to_numpy(), gmean)
    hi = np.maximum(X["g"].map(mean).to_numpy(), gmean)
    assert np.all(out >= lo - 1e-9)
    assert np.all(out <= hi + 1e-9)


def test_shapes_and_new_data():
    X, y = make_regression()
    enc = TargetEncoder(cols=["g"])
    Xt = enc.fit_transform(X, y)
    assert Xt.shape == (len(X), 1)
    new = pd.DataFrame({"g": ["c0", "UNSEEN"], "x_num": [0.0, 0.0]})
    tr = enc.transform(new)
    assert tr.shape == (2, 1)
    # unseen category -> global target mean
    assert tr.iloc[1, 0] == pytest.approx(enc.target_mean_)
