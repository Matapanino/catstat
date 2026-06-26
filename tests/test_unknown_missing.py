import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_regression

from catstat import CountEncoder, FrequencyEncoder, TargetEncoder


def _train():
    X, y = make_regression(seed=0)
    return X, y


def test_handle_unknown_value_uses_global_for_mean():
    X, y = _train()
    enc = TargetEncoder(cols=["g"], handle_unknown="value").fit(X, y)
    tr = enc.transform(pd.DataFrame({"g": ["UNSEEN"], "x_num": [0.0]}))
    assert tr.iloc[0, 0] == pytest.approx(enc.target_mean_)


def test_handle_unknown_return_nan():
    X, y = _train()
    enc = TargetEncoder(cols=["g"], handle_unknown="return_nan").fit(X, y)
    tr = enc.transform(pd.DataFrame({"g": ["UNSEEN"], "x_num": [0.0]}))
    assert np.isnan(tr.iloc[0, 0])


def test_handle_unknown_error():
    X, y = _train()
    enc = TargetEncoder(cols=["g"], handle_unknown="error").fit(X, y)
    with pytest.raises(ValueError, match="unknown categories"):
        enc.transform(pd.DataFrame({"g": ["UNSEEN"], "x_num": [0.0]}))


def test_count_unknown_is_zero():
    X, _ = _train()
    ce = CountEncoder(cols=["g"], handle_unknown="value").fit(X)
    assert ce.transform(pd.DataFrame({"g": ["UNSEEN"]})).iloc[0, 0] == 0.0
    fe = FrequencyEncoder(cols=["g"]).fit(X)
    assert fe.transform(pd.DataFrame({"g": ["UNSEEN"]})).iloc[0, 0] == 0.0


def test_handle_missing_value_learns_nan_category():
    X, y = _train()
    X = X.copy()
    X.loc[X.index[:50], "g"] = np.nan  # make NaN a real, learnable level
    enc = TargetEncoder(cols=["g"], handle_missing="value", smooth=0.0).fit(X, y)
    # a NaN at transform maps to the learned NaN encoding (not the global mean)
    out_nan = enc.transform(pd.DataFrame({"g": [np.nan], "x_num": [0.0]})).iloc[0, 0]
    expected = y[:50].mean()
    assert out_nan == pytest.approx(expected)


def test_handle_missing_return_nan():
    X, y = _train()
    X = X.copy()
    X.loc[X.index[:10], "g"] = np.nan
    enc = TargetEncoder(cols=["g"], handle_missing="return_nan").fit(X, y)
    out = enc.transform(pd.DataFrame({"g": [np.nan], "x_num": [0.0]})).iloc[0, 0]
    assert np.isnan(out)


def test_handle_missing_error():
    X, y = _train()
    X = X.copy()
    X.loc[X.index[:5], "g"] = np.nan
    with pytest.raises(ValueError, match="Missing values"):
        TargetEncoder(cols=["g"], handle_missing="error").fit(X, y)
