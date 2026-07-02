"""smooth="sigmoid" -- category_encoders-parity blend w = sigmoid((n-k)/f) toward the prior."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_binary, make_regression

import catstat._base as base
from catstat import TargetEncoder


def _hand_sigmoid(X, y, k, f):
    g = pd.DataFrame({"g": X["g"], "y": np.asarray(y, dtype=float)}).groupby("g")["y"]
    n, mean = g.count().astype(float), g.mean()
    prior = float(np.mean(y))
    w = 1.0 / (1.0 + np.exp(-(n - k) / f))
    enc = w * mean + (1.0 - w) * prior
    return enc.where(n > 1, prior), prior


@pytest.mark.parametrize("smooth", [("sigmoid", 5, 2.0), "sigmoid"])
def test_matches_category_encoders_formula(smooth):
    X, y = make_regression(n=800, k=10, seed=0)
    k, f = (5, 2.0) if isinstance(smooth, tuple) else (20, 10.0)
    enc = TargetEncoder(cols=["g"], stats=["mean"], smooth=smooth, output="numpy").fit(X, y)
    out = enc.transform(X).ravel()
    table, _prior = _hand_sigmoid(X, y, k, f)
    assert np.allclose(out, X["g"].map(table).to_numpy(), rtol=1e-12)


def test_singleton_takes_prior():
    # ce parity: a singleton category is forced to the prior, not just heavily blended
    X = pd.DataFrame({"g": ["solo"] + ["a"] * 20 + ["b"] * 20})
    rng = np.random.default_rng(1)
    y = rng.normal(size=len(X))
    enc = TargetEncoder(
        cols=["g"], stats=["mean"], smooth=("sigmoid", 2, 1.0), target_type="continuous"
    ).fit(X, y)
    val = enc.transform(pd.DataFrame({"g": ["solo"], "x_num": [0.0]})).iloc[0, 0]
    assert val == pytest.approx(float(np.mean(y)))


def test_fast_oof_equals_slow_and_is_out_of_fold():
    X, y = make_regression(n=500, k=40, seed=2)
    kw = dict(
        cols=["g"], stats=["mean"], smooth=("sigmoid", 5, 2.0), cv=5, shuffle=True,
        random_state=0, output="numpy",
    )
    fast = np.asarray(TargetEncoder(**kw).fit_transform(X, y))
    saved = base._ADDITIVE_STATS
    base._ADDITIVE_STATS = frozenset()
    try:
        slow = np.asarray(TargetEncoder(**kw).fit_transform(X, y))
    finally:
        base._ADDITIVE_STATS = saved
    np.testing.assert_allclose(fast, slow, rtol=1e-7, atol=1e-9, equal_nan=True)
    leaky = np.asarray(TargetEncoder(**kw).fit(X, y).transform(X))
    assert not np.allclose(fast, leaky)


def test_woe_with_sigmoid_smoothing():
    X, y = make_binary(n=600, k=8, seed=3)
    enc = TargetEncoder(cols=["g"], stats=["woe"], smooth=("sigmoid", 5, 2.0)).fit(X, y)
    out = enc.transform(X).to_numpy()
    assert np.isfinite(out).all()
    assert enc.transform(pd.DataFrame({"g": ["UNSEEN"]})).iloc[0, 0] == 0.0


def test_invalid_sigmoid_specs_raise():
    X, y = make_regression(seed=4)
    for bad in (("sigmoid", 5), ("sigmoid", 5, 0.0), ("sigmoid", 5, -1.0), ("logit", 5, 2.0)):
        with pytest.raises(ValueError, match="sigmoid|smooth"):
            TargetEncoder(cols=["g"], smooth=bad).fit(X, y)
    with pytest.raises(ValueError, match="smooth"):
        TargetEncoder(cols=["g"], smooth="sigmund").fit(X, y)


def test_rejected_for_loo_and_ordered():
    X, y = make_regression(seed=5)
    for scheme in ("loo", "ordered"):
        with pytest.raises(ValueError, match="kfold"):
            TargetEncoder(cols=["g"], smooth="sigmoid", scheme=scheme).fit(X, y)


def test_clone_and_params_roundtrip():
    from sklearn.base import clone

    e = TargetEncoder(cols=["g"], smooth=("sigmoid", 5, 2.0))
    c = clone(e)
    assert c.get_params()["smooth"] == ("sigmoid", 5, 2.0)  # stored verbatim (tuple survives)
    X, y = make_regression(seed=6)
    kw = dict(cols=["g"], cv=5, random_state=0, output="numpy")
    a = np.asarray(TargetEncoder(**kw, smooth="sigmoid").fit_transform(X, y))
    b = np.asarray(TargetEncoder(**kw, smooth=("sigmoid", 20, 10.0)).fit_transform(X, y))
    np.testing.assert_allclose(a, b, rtol=1e-12)  # bare string == explicit ce defaults
