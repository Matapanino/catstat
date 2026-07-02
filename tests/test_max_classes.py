"""max_classes: cap the multiclass one-vs-rest expansion to the most frequent classes (KI-016)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from catstat import TargetEncoder


def _mc_data(n=600, k=8, classes=6, seed=0, skew_freq=True):
    rng = np.random.default_rng(seed)
    g = rng.choice([f"c{i}" for i in range(k)], size=n)
    if skew_freq:  # class j has weight ~ (j+1): frequencies strictly ordered
        w = np.arange(1, classes + 1, dtype=float)
        y = rng.choice(np.arange(classes), size=n, p=w / w.sum())
    else:
        y = rng.integers(0, classes, size=n)
    return pd.DataFrame({"g": g}), y


def test_caps_to_most_frequent_and_matches_full_encoder():
    X, y = _mc_data()
    full = TargetEncoder(cols=["g"], cv=5, random_state=0, output="numpy").fit(X, y)
    capped = TargetEncoder(
        cols=["g"], cv=5, random_state=0, max_classes=3, output="numpy"
    ).fit(X, y)
    # classes_ keeps every observed class; encoded_classes_ = 3 most frequent (3, 4, 5 here)
    assert list(capped.classes_) == list(full.classes_)
    assert list(capped.encoded_classes_) == [3, 4, 5]
    assert list(capped.get_feature_names_out()) == [
        "g__te_mean__class_3", "g__te_mean__class_4", "g__te_mean__class_5",
    ]
    assert capped.target_mean_.shape == (3,)
    # the kept columns are identical to the full encoder's corresponding columns
    fout, cout = full.transform(X), capped.transform(X)
    full_names = list(full.get_feature_names_out())
    idx = [full_names.index(n_) for n_ in capped.get_feature_names_out()]
    assert np.allclose(fout[:, idx], cout, rtol=1e-12)


def test_oof_is_cross_fitted_and_deterministic():
    X, y = _mc_data(seed=1)
    kw = dict(cols=["g"], cv=5, random_state=0, max_classes=2, output="numpy")
    a = np.asarray(TargetEncoder(**kw).fit_transform(X, y))
    b = np.asarray(TargetEncoder(**kw).fit_transform(X, y))
    leaky = np.asarray(TargetEncoder(**kw).fit(X, y).transform(X))
    assert np.array_equal(a, b)
    assert a.shape[1] == 2 and not np.allclose(a, leaky)


def test_none_and_large_cap_are_unchanged():
    X, y = _mc_data(seed=2)
    base = TargetEncoder(cols=["g"], cv=5, random_state=0, output="numpy").fit(X, y)
    same = TargetEncoder(
        cols=["g"], cv=5, random_state=0, max_classes=99, output="numpy"
    ).fit(X, y)
    assert list(base.get_feature_names_out()) == list(same.get_feature_names_out())
    assert np.allclose(base.transform(X), same.transform(X), rtol=0)


def test_binary_and_continuous_ignore_max_classes():
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"g": rng.choice(list("abcd"), 200)})
    yb = rng.integers(0, 2, 200)
    enc = TargetEncoder(cols=["g"], max_classes=1, output="numpy").fit(X, yb)
    assert list(enc.get_feature_names_out()) == ["g__te_mean"]
    yc = rng.normal(size=200)
    enc2 = TargetEncoder(cols=["g"], max_classes=1, output="numpy").fit(X, yc)
    assert list(enc2.get_feature_names_out()) == ["g__te_mean"]


def test_tie_break_and_order_stability():
    # classes 0/1 equally frequent and more frequent than 2: keep {0, 1}, in class order
    y = np.array([0, 1] * 40 + [2] * 10)
    X = pd.DataFrame({"g": list("ab") * 45})
    enc = TargetEncoder(cols=["g"], cv=2, random_state=0, max_classes=2).fit(X, y)
    assert list(enc.encoded_classes_) == [0, 1]


def test_width_warning_without_cap():
    rng = np.random.default_rng(4)
    n_classes = 120
    y = np.arange(n_classes).repeat(3)
    X = pd.DataFrame({"g": rng.choice(list("abcdef"), len(y))})
    with pytest.warns(UserWarning, match="max_classes"):
        TargetEncoder(cols=["g"], cv=2, random_state=0).fit(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # capped -> no warning
        TargetEncoder(cols=["g"], cv=2, random_state=0, max_classes=10).fit(X, y)


def test_invalid_max_classes_raises():
    X, y = _mc_data(seed=5)
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="max_classes"):
            TargetEncoder(cols=["g"], max_classes=bad).fit(X, y)


def test_clone_params_roundtrip():
    from sklearn.base import clone

    e = TargetEncoder(cols=["g"], max_classes=4)
    assert clone(e).get_params()["max_classes"] == 4
