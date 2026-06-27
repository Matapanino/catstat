"""`interactions=[[...]]` -> one joint target-encoded column per group, additive to `cols`.

The engine already treats a "unit" as an arbitrary column group (tuple keys); interactions just
append more units, so OOF / naming / unknown-missing / parity all work unchanged. This pins the
public behavior: naming, equality with combination, dedup, validation, and sklearn clone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from catstat import TargetEncoder


def _xy(n=300, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        {
            "a": np.array([f"a{v}" for v in rng.randint(0, 6, n)], dtype=object),
            "b": np.array([f"b{v}" for v in rng.randint(0, 5, n)], dtype=object),
            "c": np.array([f"c{v}" for v in rng.randint(0, 4, n)], dtype=object),
        }
    )
    return X, pd.Series(rng.randn(n))


def test_interaction_adds_joint_column_and_names():
    X, y = _xy()
    enc = TargetEncoder(
        cols=["a", "b", "c"], stats=["mean"], interactions=[["a", "b"]], random_state=0
    )
    out = enc.fit_transform(X, y)
    assert list(enc.get_feature_names_out()) == [
        "a__te_mean",
        "b__te_mean",
        "c__te_mean",
        "a+b__te_mean",
    ]
    assert np.asarray(out).shape == (len(X), 4)


def test_interaction_column_matches_combination_encoder():
    """The a+b interaction column equals the single column the combination(a,b) encoder produces
    (same fold assignment + joint-key path)."""
    X, y = _xy()
    inter = np.asarray(
        TargetEncoder(
            cols=["a", "b"], stats=["mean"], interactions=[["a", "b"]], random_state=0
        ).fit_transform(X, y),
        dtype=float,
    )
    comb = np.asarray(
        TargetEncoder(
            cols=["a", "b"], stats=["mean"], multi_feature_mode="combination", random_state=0
        ).fit_transform(X, y),
        dtype=float,
    )
    # inter columns are [a, b, a+b]; comb has the single a+b column
    np.testing.assert_allclose(inter[:, 2], comb[:, 0], rtol=1e-7, atol=1e-9, equal_nan=True)


def test_interaction_multi_stat_and_dedup():
    X, y = _xy()
    enc = TargetEncoder(
        cols=["a", "b"],
        stats=["mean", "var"],
        interactions=[["a", "b"], ["a", "b"]],  # duplicate group -> deduped
        random_state=0,
    )
    enc.fit_transform(X, y)
    names = list(enc.get_feature_names_out())
    assert len(names) == 6  # units {a, b, a+b} x stats {mean, var}; duplicate interaction deduped
    assert [n for n in names if n.startswith("a+b__")] == ["a+b__te_mean", "a+b__te_var"]


@pytest.mark.parametrize("bad", ["ab", [["a"], "b"], [[]], [["a", "zzz"]]])
def test_interaction_validation_errors(bad):
    X, y = _xy()
    with pytest.raises(ValueError):
        TargetEncoder(cols=["a", "b"], interactions=bad).fit(X, y)


def test_interactions_clone_and_get_params():
    enc = TargetEncoder(cols=["a", "b"], interactions=[["a", "b"]])
    assert enc.get_params()["interactions"] == [["a", "b"]]
    cloned = clone(enc)
    assert cloned.get_params()["interactions"] == [["a", "b"]]


def test_default_no_interactions_unchanged():
    X, y = _xy()
    base = np.asarray(
        TargetEncoder(cols=["a", "b"], stats=["mean"], random_state=0).fit_transform(X, y)
    )
    same = np.asarray(
        TargetEncoder(
            cols=["a", "b"], stats=["mean"], interactions=None, random_state=0
        ).fit_transform(X, y)
    )
    np.testing.assert_array_equal(base, same)
