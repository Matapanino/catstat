"""Phase 3: skew + custom-callable aggregations (which subsume quantiles)."""

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_regression

from catstat import TargetEncoder


def test_skew_matches_pandas_groupby():
    X, y = make_regression(n=600, k=6, seed=0)
    enc = TargetEncoder(cols=["g"], stats=["skew"], output="numpy").fit(X, y)
    out = enc.transform(X).ravel()
    table = pd.DataFrame({"g": X["g"], "y": y}).groupby("g")["y"].skew()
    assert np.allclose(out, X["g"].map(table).to_numpy())
    assert list(enc.get_feature_names_out()) == ["g__te_skew"]


def test_custom_quantile_and_iqr():
    X, y = make_regression(n=600, k=6, seed=1)
    enc = TargetEncoder(
        cols=["g"],
        stats=[
            ("q90", lambda v: np.quantile(v, 0.9)),
            ("iqr", lambda v: np.subtract(*np.percentile(v, [75, 25]))),
        ],
        output="numpy",
    ).fit(X, y)
    out = enc.transform(X)
    assert list(enc.get_feature_names_out()) == ["g__q90", "g__iqr"]
    q90 = pd.DataFrame({"g": X["g"], "y": y}).groupby("g")["y"].quantile(0.9)
    assert np.allclose(out[:, 0], X["g"].map(q90).to_numpy())


def test_custom_dict_form():
    X, y = make_regression(seed=2)
    enc = TargetEncoder(cols=["g"], stats={"p10": lambda v: np.quantile(v, 0.1)})
    out = enc.fit_transform(X, y)
    assert out.shape[1] == 1
    assert list(enc.get_feature_names_out()) == ["g__p10"]


def test_custom_and_skew_force_cpu_backend():
    X, y = make_regression(seed=3)
    assert TargetEncoder(cols=["g"], stats=["skew"], backend="auto").fit(X, y).backend_ == "cpu"
    enc = TargetEncoder(cols=["g"], stats=[("q90", lambda v: np.quantile(v, 0.9))], backend="auto")
    assert enc.fit(X, y).backend_ == "cpu"


def test_custom_unseen_falls_back_to_global():
    X, y = make_regression(seed=4)
    fn = lambda v: np.quantile(v, 0.9)  # noqa: E731
    enc = TargetEncoder(cols=["g"], stats=[("q90", fn)]).fit(X, y)
    val = enc.transform(pd.DataFrame({"g": ["UNSEEN"], "x_num": [0.0]})).iloc[0, 0]
    assert val == pytest.approx(float(fn(y)))


def test_custom_is_cross_fitted():
    X, y = make_regression(n=500, k=40, seed=5)
    kw = dict(cols=["g"], stats=[("q90", lambda v: np.quantile(v, 0.9))], cv=5, random_state=0)
    oof = np.asarray(TargetEncoder(**kw, output="numpy").fit_transform(X, y)).ravel()
    leaky = np.asarray(TargetEncoder(**kw, output="numpy").fit(X, y).transform(X)).ravel()
    assert not np.allclose(oof, leaky)


def test_skew_requires_continuous_target():
    X = pd.DataFrame({"g": ["a", "b"] * 30})
    y = np.array([0, 1] * 30)
    with pytest.raises(ValueError, match="continuous target"):
        TargetEncoder(cols=["g"], stats=["skew"]).fit(X, y)


def test_quantile_string_gives_helpful_error():
    X, y = make_regression(seed=6)
    with pytest.raises(ValueError, match="custom aggregation"):
        TargetEncoder(cols=["g"], stats=["quantile"]).fit(X, y)
