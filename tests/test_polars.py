"""Phase 3: output='polars'. Skipped if polars is not installed."""

import pytest
from tests.conftest import make_regression

from catstat import TargetEncoder

pl = pytest.importorskip("polars")


def test_polars_fit_transform():
    X, y = make_regression()
    out = TargetEncoder(cols=["g"], stats=["mean", "count"], output="polars").fit_transform(X, y)
    assert isinstance(out, pl.DataFrame)
    assert out.columns == ["g__te_mean", "g__count"]
    assert out.shape == (len(X), 2)


def test_polars_transform_and_multiclass():
    import numpy as np

    X, _ = make_regression()
    rng = np.random.default_rng(0)
    ym = rng.integers(0, 3, len(X))
    enc = TargetEncoder(cols=["g"], output="polars").fit(X, ym)
    out = enc.transform(X)
    assert isinstance(out, pl.DataFrame)
    assert out.columns == [
        "g__te_mean__class_0",
        "g__te_mean__class_1",
        "g__te_mean__class_2",
    ]
