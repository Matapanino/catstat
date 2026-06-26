import numpy as np
from sklearn.model_selection import KFold
from tests.conftest import make_regression

from catstat import TargetEncoder


def test_same_random_state_is_identical():
    X, y = make_regression(n=500, k=12, seed=0)
    a = TargetEncoder(cols=["g"], smooth="auto", cv=5, random_state=7).fit_transform(X, y)
    b = TargetEncoder(cols=["g"], smooth="auto", cv=5, random_state=7).fit_transform(X, y)
    assert np.allclose(np.asarray(a), np.asarray(b))


def test_different_random_state_differs():
    X, y = make_regression(n=500, k=12, seed=0)
    a = TargetEncoder(cols=["g"], smooth=0.0, cv=5, random_state=7).fit_transform(X, y)
    c = TargetEncoder(cols=["g"], smooth=0.0, cv=5, random_state=8).fit_transform(X, y)
    assert not np.allclose(np.asarray(a), np.asarray(c))


def test_explicit_splitter_is_reproducible():
    X, y = make_regression(n=400, k=10, seed=1)
    kf = KFold(n_splits=4, shuffle=True, random_state=123)
    a = TargetEncoder(cols=["g"], smooth=0.0, cv=kf).fit_transform(X, y)
    b = TargetEncoder(cols=["g"], smooth=0.0, cv=kf).fit_transform(X, y)
    assert np.allclose(np.asarray(a), np.asarray(b))
