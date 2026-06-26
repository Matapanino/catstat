"""Phase 2: multi_feature_mode = independent vs combination (joint encoding)."""

import numpy as np
import pandas as pd
import pytest

from catstat import TargetEncoder


def _make(n=600, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.choice(list("xy"), n)
    b = rng.choice(list("pq"), n)
    eff = {("x", "p"): 1.0, ("x", "q"): -1.0, ("y", "p"): 2.0, ("y", "q"): 0.0}
    y = np.array([eff[(ai, bi)] for ai, bi in zip(a, b)]) + rng.normal(0, 0.1, n)
    return pd.DataFrame({"a": a, "b": b}), y


def test_combination_makes_one_joint_column():
    X, y = _make()
    enc = TargetEncoder(
        cols=["a", "b"], multi_feature_mode="combination", smooth=0.0, output="numpy"
    ).fit(X, y)
    assert list(enc.get_feature_names_out()) == ["a+b__te_mean"]
    out = enc.transform(X).ravel()
    table = (
        pd.DataFrame({"k": list(zip(X["a"], X["b"])), "y": y}).groupby("k")["y"].mean().to_dict()
    )
    expected = np.array([table[(ai, bi)] for ai, bi in zip(X["a"], X["b"])])
    assert np.allclose(out, expected)


def test_independent_vs_combination_shapes():
    X, y = _make()
    ind = TargetEncoder(cols=["a", "b"], multi_feature_mode="independent").fit_transform(X, y)
    comb = TargetEncoder(cols=["a", "b"], multi_feature_mode="combination").fit_transform(X, y)
    assert ind.shape[1] == 2
    assert comb.shape[1] == 1


def test_combination_unseen_combo_is_global():
    X, y = _make()
    enc = TargetEncoder(cols=["a", "b"], multi_feature_mode="combination").fit(X, y)
    tr = enc.transform(pd.DataFrame({"a": ["z"], "b": ["p"]})).iloc[0, 0]
    assert tr == pytest.approx(enc.target_mean_)


def test_invalid_mode_raises():
    X, y = _make()
    with pytest.raises(ValueError, match="multi_feature_mode"):
        TargetEncoder(cols=["a", "b"], multi_feature_mode="bogus").fit(X, y)
