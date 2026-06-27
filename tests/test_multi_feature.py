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


def test_combination_both_components_known_but_joint_unseen_is_global():
    # x, y, p, q are each seen individually, but the joint (x, q) never co-occurs at fit, so its
    # integer joint code is absent from the canonical index -> unknown -> global.
    a, b = ["x", "x", "y", "y"], ["p", "p", "q", "q"]
    enc = TargetEncoder(
        cols=["a", "b"],
        multi_feature_mode="combination",
        stats=("mean",),
        target_type="continuous",
        smooth=0.0,
        handle_unknown="value",
    ).fit(pd.DataFrame({"a": a, "b": b}), np.array([1.0, 1.0, 2.0, 2.0]))
    out = np.asarray(enc.transform(pd.DataFrame({"a": ["x"], "b": ["q"]})))
    assert out[0, 0] == pytest.approx(enc.target_mean_)


def test_combination_categories_are_value_tuples_not_codes():
    X, y = _make()
    enc = TargetEncoder(
        cols=["a", "b"], multi_feature_mode="combination", smooth=0.0, random_state=0
    ).fit(X, y)
    cats = enc.categories_["a+b"]
    assert cats.dtype == object  # decoded category VALUES, never the int64 joint codes
    assert {tuple(row) for row in cats} == set(zip(X["a"], X["b"]))


def test_combination_missing_value_is_its_own_joint_level():
    # handle_missing="value": a NaN component becomes the MISSING sentinel, a real joint category.
    a, b = ["x", "x", None, "x"], ["p", "q", "p", "p"]
    enc = TargetEncoder(
        cols=["a", "b"],
        multi_feature_mode="combination",
        stats=("mean",),
        target_type="continuous",
        smooth=0.0,
        handle_missing="value",
        handle_unknown="value",
    ).fit(pd.DataFrame({"a": a, "b": b}), np.array([1.0, 2.0, 3.0, 4.0]))
    # (MISSING, p) is a seen joint level with mean 3.0; a NaN-a transform row recovers it.
    out = np.asarray(enc.transform(pd.DataFrame({"a": [None], "b": ["p"]})))
    assert out[0, 0] == pytest.approx(3.0)


def test_combination_missing_return_nan_propagates():
    a, b = ["x", "x", "y"], ["p", "q", "p"]
    enc = TargetEncoder(
        cols=["a", "b"],
        multi_feature_mode="combination",
        stats=("mean",),
        target_type="continuous",
        smooth=0.0,
        handle_missing="return_nan",
    ).fit(pd.DataFrame({"a": a, "b": b}), np.array([1.0, 2.0, 3.0]))
    out = np.asarray(enc.transform(pd.DataFrame({"a": [None], "b": ["p"]})))
    assert np.isnan(out[0, 0])


def test_combination_determinism():
    X, y = _make()
    kw = dict(
        cols=["a", "b"], multi_feature_mode="combination", cv=5, random_state=0, output="numpy"
    )
    first = TargetEncoder(**kw).fit_transform(X, y)
    second = TargetEncoder(**kw).fit_transform(X, y)
    assert np.array_equal(first, second)
