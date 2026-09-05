"""Custom training supports must survive both fast and slow OOF dispatch."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from catstat import TargetEncoder
from catstat._base import _BaseStatEncoder


def purged_folds():
    return [
        (np.array([3, 4, 5]), np.array([0, 1, 2])),
        (np.array([0, 1, 2]), np.array([3, 4, 5])),
        (np.array([0, 1, 2, 3, 4]), np.array([6, 7])),
    ]


@pytest.mark.parametrize("stats", [["mean"], ["mean", "var", "median"], ["median"]])
def test_purged_cv_matches_explicit_training_fits_and_excludes_labels(stats):
    X = pd.DataFrame({"g": ["a", "a", "b", "a", "b", "b", "a", "c"]})
    y = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
    kw = dict(cols=["g"], stats=stats, smooth=2.0, target_type="continuous", output="numpy")
    folds = purged_folds()
    actual = TargetEncoder(cv=folds, **kw).fit_transform(X, y)
    for tr, te in folds:
        expected = TargetEncoder(**kw).fit(X.iloc[tr], y[tr]).transform(X.iloc[te])
        np.testing.assert_allclose(actual[te], expected)
    changed = y.copy()
    changed[6:] += 1000  # excluded from the first fold's train AND validation
    mutated = TargetEncoder(cv=folds, **kw).fit_transform(X, changed)
    np.testing.assert_array_equal(actual[:3], mutated[:3])


def test_complement_gate_accepts_reordered_training_indices_only_if_complete():
    folds = list(KFold(4).split(np.zeros((8, 1))))
    reordered = [(tr[::-1], te) for tr, te in folds]
    np.testing.assert_array_equal(
        _BaseStatEncoder._partition_fold_id(reordered, 8), np.repeat(np.arange(4), 2)
    )
    assert _BaseStatEncoder._partition_fold_id(purged_folds(), 8) is None


@pytest.mark.parametrize("stats", [["mean", "var", "std"], ["mean", "median"]])
def test_standard_kfold_keeps_fast_path_and_matches_fold_reconstruction(stats, monkeypatch):
    X = pd.DataFrame({"g": ["a", "b", "a", "c"] * 4})
    y = np.arange(16, dtype=float) ** 2
    folds = list(KFold(4, shuffle=True, random_state=7).split(X))
    kw = dict(cols=["g"], stats=stats, smooth="auto", target_type="continuous", output="numpy")
    enc = TargetEncoder(cv=folds, **kw)
    calls = []
    original = enc._kfold_oof_additive_fast

    def tracked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(enc, "_kfold_oof_additive_fast", tracked)
    out = enc.fit_transform(X, y)
    assert calls == [True]
    for tr, te in folds:
        expected = TargetEncoder(**kw).fit(X.iloc[tr], y[tr]).transform(X.iloc[te])
        np.testing.assert_allclose(out[te], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("stats", [["mean"], ["median"]])
@pytest.mark.parametrize(
    "tr,te,match",
    [
        ([2, 2, 3], [0, 1], "duplicate"),
        ([2, 3], [0, 0, 1], "duplicate"),
        ([-1, 2], [0, 1], "range"),
        ([2, 4], [0, 1], "range"),
        ([2, 3], [-1, 0], "range"),
        ([2, 3], [0, 4], "range"),
        ([1, 2, 3], [0, 1], "overlap"),
        ([2.0, 3.0], [0, 1], "integer"),
        ([2, 3], [False, True], "integer"),
        ([[2, 3]], [0, 1], "one-dimensional"),
    ],
)
def test_invalid_indices_rejected_before_oof_dispatch(stats, tr, te, match):
    X = pd.DataFrame({"g": ["a"] * 4})
    with pytest.raises(ValueError, match=match):
        TargetEncoder(
            cols=["g"], stats=stats, cv=[(tr, te)], target_type="continuous"
        ).fit_transform(X, np.arange(4, dtype=float))
