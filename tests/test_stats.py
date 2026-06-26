"""Phase 2 dispersion/order statistics: var / std / median / min / max."""

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_regression

from catstat import TargetEncoder


@pytest.mark.parametrize("stat", ["var", "std", "median", "min", "max"])
def test_stat_matches_pandas_groupby_full_data(stat):
    # categories all have >= 2 samples so nothing falls back to the global value
    X, y = make_regression(n=600, k=6, seed=0)
    enc = TargetEncoder(cols=["g"], stats=[stat], output="numpy").fit(X, y)
    out = enc.transform(X).ravel()

    g = pd.DataFrame({"g": X["g"], "y": y}).groupby("g")["y"]
    table = {
        "var": g.var(ddof=1),
        "std": g.std(ddof=1),
        "median": g.median(),
        "min": g.min(),
        "max": g.max(),
    }[stat]
    expected = X["g"].map(table).to_numpy()
    assert np.allclose(out, expected)


@pytest.mark.parametrize("stat", ["var", "std", "median", "min", "max"])
def test_unseen_falls_back_to_global(stat):
    X, y = make_regression(seed=1)
    enc = TargetEncoder(cols=["g"], stats=[stat]).fit(X, y)
    val = enc.transform(pd.DataFrame({"g": ["UNSEEN"], "x_num": [0.0]})).iloc[0, 0]
    assert val == pytest.approx(enc.global_stats_[f"g__te_{stat}"])


def test_variance_singleton_falls_back_to_global():
    # 'solo' appears once -> sample variance undefined -> global var fallback.
    # (Non-integer values so type_of_target reads this as continuous, not multiclass.)
    X = pd.DataFrame({"g": ["a", "a", "a", "b", "b", "solo"]})
    y = np.array([1.1, 2.4, 3.3, 10.6, 12.2, 7.9])
    enc = TargetEncoder(cols=["g"], stats=["var"], output="numpy").fit(X, y)
    out = enc.transform(pd.DataFrame({"g": ["solo"]})).ravel()[0]
    assert out == pytest.approx(float(np.var(y, ddof=1)))


def test_dispersion_stats_require_continuous_target():
    X = pd.DataFrame({"g": ["a", "b", "a", "b"] * 10})
    y_bin = np.array([0, 1] * 20)
    with pytest.raises(ValueError, match="continuous target"):
        TargetEncoder(cols=["g"], stats=["std"]).fit(X, y_bin)


def test_multi_stat_columns_and_names():
    X, y = make_regression()
    enc = TargetEncoder(cols=["g"], stats=["mean", "std", "median"])
    out = enc.fit_transform(X, y)
    assert out.shape[1] == 3
    assert list(enc.get_feature_names_out()) == ["g__te_mean", "g__te_std", "g__te_median"]


def test_dispersion_stats_are_cross_fitted():
    # many small categories => out-of-fold std differs from the full-data (leaky) std
    X, y = make_regression(n=500, k=40, seed=2)
    enc = TargetEncoder(cols=["g"], stats=["median"], cv=5, random_state=0, output="numpy")
    oof = np.asarray(enc.fit_transform(X, y)).ravel()
    leaky = np.asarray(enc.fit(X, y).transform(X)).ravel()
    assert not np.allclose(oof, leaky)
