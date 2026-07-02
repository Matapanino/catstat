"""Shape statistics (skew/kurt) via power-sum moments: pandas parity, stability, fallbacks."""

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_regression

from catstat import TargetEncoder


def _pandas_reference(X, y, stat):
    g = pd.DataFrame({"g": X["g"], "y": y}).groupby("g")["y"]
    table = g.skew() if stat == "skew" else g.apply(lambda s: s.kurt())
    return X["g"].map(table).to_numpy()


@pytest.mark.parametrize("stat", ["skew", "kurt"])
def test_matches_pandas_groupby(stat):
    X, y = make_regression(n=2000, k=8, seed=0)
    enc = TargetEncoder(cols=["g"], stats=[stat], output="numpy").fit(X, y)
    out = enc.transform(X).ravel()
    assert np.allclose(out, _pandas_reference(X, y, stat), rtol=1e-9, atol=1e-9)
    assert list(enc.get_feature_names_out()) == [f"g__te_{stat}"]


@pytest.mark.parametrize("stat", ["skew", "kurt"])
def test_offset_stability_matches_pandas(stat):
    # y = 1e9 + N(0,1): raw power sums would cancel catastrophically; the global-mean shift
    # must keep the moment reconstruction allclose to pandas' two-pass computation.
    X, _ = make_regression(n=2000, k=8, seed=1)
    rng = np.random.default_rng(1)
    y = 1e9 + rng.normal(size=len(X))
    enc = TargetEncoder(cols=["g"], stats=[stat], output="numpy").fit(X, y)
    out = enc.transform(X).ravel()
    ref = _pandas_reference(X, y, stat)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_mean_var_offset_matches_pandas():
    """y = 1e9 + N(0,1): the fit path's shifted reductions must stay allclose to pandas'
    two-pass var and the exact group mean (the unshifted sums cancel catastrophically -- and
    differently per backend, which broke CPU/GPU parity at large offsets)."""
    X, _ = make_regression(n=2000, k=8, seed=9)
    rng = np.random.default_rng(9)
    y = 1e9 + rng.normal(size=len(X))
    enc = TargetEncoder(cols=["g"], stats=["mean", "var"], smooth=0.0, output="numpy").fit(X, y)
    out = enc.transform(X)
    g = pd.DataFrame({"g": X["g"], "y": y}).groupby("g")["y"]
    assert np.allclose(out[:, 0], X["g"].map(g.mean()).to_numpy(), rtol=1e-12)
    assert np.allclose(out[:, 1], X["g"].map(g.var(ddof=1)).to_numpy(), rtol=1e-6, atol=1e-6)
    # the EB weights are the fragile part: 'auto' must stay finite and within the data's range
    enc_auto = TargetEncoder(cols=["g"], stats=["mean"], output="numpy").fit(X, y)
    mcol = enc_auto.transform(X).ravel()
    assert np.isfinite(mcol).all() and mcol.min() >= y.min() and mcol.max() <= y.max()


@pytest.mark.parametrize("stat", ["skew", "kurt"])
@pytest.mark.parametrize("scale", [1.0, 1e9])
def test_constant_category_is_zero(stat, scale):
    # A (numerically) constant category has a defined value of 0.0 -- not a fallback.
    X = pd.DataFrame({"g": ["a"] * 6 + ["b"] * 6})
    y = np.array([scale] * 6 + [scale + 1.0, scale + 2.0, scale + 3.0] * 2)
    enc = TargetEncoder(
        cols=["g"], stats=[stat], target_type="continuous", output="numpy"
    ).fit(X, y)
    out = enc.transform(pd.DataFrame({"g": ["a"]})).ravel()
    assert out[0] == 0.0


def test_small_categories_fall_back_to_global():
    # skew undefined for n < 3, kurt for n < 4 -> global statistic (honesty rule, never blend).
    X = pd.DataFrame({"g": ["a"] * 2 + ["b"] * 3 + ["c"] * 20})
    rng = np.random.default_rng(2)
    y = rng.normal(size=len(X))
    enc = TargetEncoder(cols=["g"], stats=["skew", "kurt"], output="numpy").fit(X, y)
    out = enc.transform(pd.DataFrame({"g": ["a", "b", "c"]}))
    g_skew = float(pd.Series(y).skew())
    g_kurt = float(pd.Series(y).kurt())
    assert out[0, 0] == pytest.approx(g_skew)  # n=2: skew undefined
    assert out[0, 1] == pytest.approx(g_kurt)  # n=2: kurt undefined
    assert out[1, 0] != pytest.approx(g_skew)  # n=3: skew defined
    assert out[1, 1] == pytest.approx(g_kurt)  # n=3: kurt undefined
    assert out[2, 0] != pytest.approx(g_skew)
    assert out[2, 1] != pytest.approx(g_kurt)


@pytest.mark.parametrize("stat", ["skew", "kurt"])
def test_unseen_falls_back_to_global(stat):
    X, y = make_regression(n=600, k=6, seed=3)
    enc = TargetEncoder(cols=["g"], stats=[stat]).fit(X, y)
    val = enc.transform(pd.DataFrame({"g": ["UNSEEN"], "x_num": [0.0]})).iloc[0, 0]
    ref = float(pd.Series(y).skew() if stat == "skew" else pd.Series(y).kurt())
    assert val == pytest.approx(ref)
    assert enc.global_stats_[f"g__te_{stat}"] == pytest.approx(ref)


def test_global_kurt_undefined_is_zero():
    # global fallback with n < 4 target: pandas kurt is NaN -> catstat defines it as 0.0.
    X = pd.DataFrame({"g": ["a", "b", "c"]})
    y = np.array([1.0, 2.0, 5.0])
    enc = TargetEncoder(
        cols=["g"], stats=["kurt"], target_type="continuous", min_samples_category=1
    ).fit(X, y)
    assert enc.global_stats_["g__te_kurt"] == 0.0


def test_kurt_requires_continuous_target():
    X = pd.DataFrame({"g": ["a", "b"] * 30})
    y = np.array([0, 1] * 30)
    with pytest.raises(ValueError, match="continuous target"):
        TargetEncoder(cols=["g"], stats=["kurt"]).fit(X, y)


@pytest.mark.parametrize("stat", ["skew", "kurt"])
def test_oof_is_cross_fitted_and_deterministic(stat):
    X, y = make_regression(n=800, k=40, seed=4)
    kw = dict(cols=["g"], stats=[stat], cv=5, random_state=0, output="numpy")
    oof1 = np.asarray(TargetEncoder(**kw).fit_transform(X, y))
    oof2 = np.asarray(TargetEncoder(**kw).fit_transform(X, y))
    leaky = np.asarray(TargetEncoder(**kw).fit(X, y).transform(X))
    assert np.array_equal(oof1, oof2)  # same random_state -> identical folds -> identical OOF
    assert not np.allclose(oof1, leaky)  # fit_transform is out-of-fold, not the leaky path


def test_missing_value_learns_its_own_level():
    X, y = make_regression(n=600, k=6, seed=5)
    Xm = X.copy()
    Xm.loc[Xm.index[:80], "g"] = np.nan
    enc = TargetEncoder(cols=["g"], stats=["kurt"], handle_missing="value", output="numpy")
    out = enc.fit(Xm, y).transform(Xm).ravel()
    miss = Xm["g"].isna().to_numpy()
    ref = float(pd.Series(y[miss]).kurt())
    assert np.allclose(out[miss], ref)
