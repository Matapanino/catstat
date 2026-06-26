"""Phase 3: leave-one-out and ordered (CatBoost-style) cross-fitting schemes."""

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_leakage_trap, make_regression

from catstat import TargetEncoder


def test_loo_exact_values():
    X = pd.DataFrame({"g": ["a", "a", "a", "b", "b"]})
    y = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    enc = TargetEncoder(
        cols=["g"], scheme="loo", smooth=0.0, target_type="continuous", output="numpy"
    )
    out = enc.fit_transform(X, y).ravel()
    # each row = mean of the OTHER rows of its category
    assert np.allclose(out, [2.5, 2.0, 1.5, 20.0, 10.0])


def test_loo_transform_uses_full_category_mean():
    X = pd.DataFrame({"g": ["a", "a", "a", "b", "b"]})
    y = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    enc = TargetEncoder(cols=["g"], scheme="loo", smooth=0.0, target_type="continuous").fit(X, y)
    # transform (new data) uses the full-data mean, not LOO
    assert enc.transform(pd.DataFrame({"g": ["a"]})).iloc[0, 0] == pytest.approx(2.0)


@pytest.mark.parametrize("scheme", ["loo", "ordered"])
def test_scheme_is_leakage_safe(scheme):
    X, y = make_leakage_trap(n=2000, n_levels=1000, seed=3)
    kw = dict(cols=["g"], scheme=scheme, smooth=0.0, random_state=0, output="numpy")
    oof = np.asarray(TargetEncoder(**kw).fit_transform(X, y)).ravel()
    leaky = np.asarray(TargetEncoder(**kw).fit(X, y).transform(X)).ravel()
    assert abs(np.corrcoef(oof, y)[0, 1]) < 0.1  # noise category -> no target signal
    assert abs(np.corrcoef(leaky, y)[0, 1]) > 0.4  # full-data path over-fits


def test_ordered_deterministic_and_seed_sensitive():
    X, y = make_regression(n=500, k=12, seed=0)
    a = TargetEncoder(cols=["g"], scheme="ordered", random_state=7).fit_transform(X, y)
    b = TargetEncoder(cols=["g"], scheme="ordered", random_state=7).fit_transform(X, y)
    c = TargetEncoder(cols=["g"], scheme="ordered", random_state=8).fit_transform(X, y)
    assert np.allclose(np.asarray(a), np.asarray(b))
    assert not np.allclose(np.asarray(a), np.asarray(c))


@pytest.mark.parametrize("scheme", ["loo", "ordered"])
def test_binary_and_multiclass(scheme):
    X, y = make_regression(n=400, k=8, seed=1)
    yb = (y > y.mean()).astype(int)
    out_b = TargetEncoder(cols=["g"], scheme=scheme, random_state=0).fit_transform(X, yb)
    assert out_b.shape == (len(X), 1)
    rng = np.random.default_rng(0)
    ym = rng.integers(0, 3, len(X))
    out_m = TargetEncoder(cols=["g"], scheme=scheme, random_state=0).fit_transform(X, ym)
    assert out_m.shape == (len(X), 3)


def test_count_allowed_alongside_scheme_and_is_full_data():
    X, y = make_regression(seed=2)
    enc = TargetEncoder(cols=["g"], stats=["mean", "count"], scheme="loo", output="numpy")
    out = enc.fit_transform(X, y)
    assert out.shape[1] == 2
    # the count column is target-independent -> equals the full-data count (scheme doesn't touch it)
    counts = X["g"].map(X["g"].value_counts()).to_numpy().astype(float)
    assert np.allclose(out[:, 1], counts)


def test_non_mean_stat_with_scheme_raises():
    X, y = make_regression(seed=3)
    with pytest.raises(ValueError, match="cross-fits the mean only"):
        TargetEncoder(cols=["g"], stats=["mean", "std"], scheme="loo").fit_transform(X, y)


def test_invalid_scheme_raises():
    X, y = make_regression(seed=4)
    with pytest.raises(ValueError, match="scheme="):
        TargetEncoder(cols=["g"], scheme="bogus").fit(X, y)


def test_kfold_still_default():
    X, y = make_regression(seed=5)
    enc = TargetEncoder(cols=["g"])
    assert enc.scheme == "kfold"
    enc.fit_transform(X, y)  # smoke
