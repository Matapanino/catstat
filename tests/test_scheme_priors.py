"""WP1 strict label-exclusion priors, including empty and missing support."""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.utils import check_random_state

from catstat import TargetEncoder


def encode(keys, y, scheme, smooth=3.0, **kwargs):
    return (
        TargetEncoder(
            cols=["g"],
            scheme=scheme,
            smooth=smooth,
            target_type="continuous",
            random_state=7,
            output="numpy",
            **kwargs,
        )
        .fit_transform(pd.DataFrame({"g": keys}), y)
        .ravel()
    )


@pytest.mark.parametrize("smooth", [0.0, 3.0, "auto"])
@pytest.mark.parametrize("singleton", [False, True])
def test_loo_own_label_mutation_cannot_change_own_output(smooth, singleton):
    keys = list("abcdef") if singleton else list("aaabbb")
    y = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    before = encode(keys, y, "loo", smooth)
    for i in range(len(y)):
        mutated = y.copy()
        mutated[i] = 1e16  # no total-minus-self cancellation through either prior or category sum
        after = encode(keys, mutated, "loo", smooth)
        assert after[i] == before[i]


@pytest.mark.parametrize("smooth", [0.0, 3.0, "auto"])
@pytest.mark.parametrize("singleton", [False, True])
def test_ordered_current_and_future_mutation_cannot_change_prefix(smooth, singleton):
    keys = list("abcdef") if singleton else list("aaabbb")
    y = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    perm = check_random_state(7).permutation(len(y))
    before = encode(keys, y, "ordered", smooth)
    for t in range(len(y)):
        mutated = y.copy()
        mutated[perm[t:]] = 1e16
        after = encode(keys, mutated, "ordered", smooth)
        np.testing.assert_array_equal(after[perm[: t + 1]], before[perm[: t + 1]])


@pytest.mark.parametrize("scheme", ["loo", "ordered"])
@pytest.mark.parametrize("smooth", [0.0, 3.0, "auto"])
def test_exact_label_excluded_prior_arithmetic(scheme, smooth):
    keys = np.array(list("aabccc"))
    y = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    actual = encode(keys, y, scheme, smooth)
    perm = check_random_state(7).permutation(len(y)) if scheme == "ordered" else np.arange(len(y))
    m = (
        (1.0 if smooth == "auto" or smooth == 0 else smooth)
        if scheme == "ordered"
        else (0.0 if smooth == "auto" else smooth)
    )
    for t, i in enumerate(perm):
        support = perm[:t] if scheme == "ordered" else np.delete(perm, t)
        prior = y[support].mean() if len(support) else 0.0
        peers = support[keys[support] == keys[i]]
        expected = (y[peers].sum() + m * prior) / (len(peers) + m) if len(peers) + m else prior
        assert actual[i] == pytest.approx(expected)


@pytest.mark.parametrize("scheme", ["loo", "ordered"])
def test_empty_history_uses_fixed_zero_and_full_transform_keeps_full_prior(scheme):
    assert encode(["singleton"], [41.0], scheme)[0] == 0.0
    X = pd.DataFrame({"g": ["a", "a", "b"]})
    y = np.array([1.0, 2.0, 9.0])
    enc = TargetEncoder(cols=["g"], scheme=scheme, smooth=3.0, target_type="continuous")
    enc.fit_transform(X, y)
    new = pd.DataFrame({"g": ["a", "new"]})
    fitted = TargetEncoder(cols=["g"], smooth=3.0, target_type="continuous").fit(X, y)
    pd.testing.assert_frame_equal(enc.transform(new), fitted.transform(new))
    pd.testing.assert_frame_equal(
        pickle.loads(pickle.dumps(enc)).transform(new), enc.transform(new)
    )


@pytest.mark.parametrize("scheme", ["loo", "ordered"])
def test_missing_return_nan_is_excluded_from_prior_support(scheme):
    keys = ["a", None, "b", "a", None, "b"]
    y = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    before = encode(keys, y, scheme, handle_missing="return_nan")
    y[[1, 4]] = 1e16
    after = encode(keys, y, scheme, handle_missing="return_nan")
    np.testing.assert_array_equal(before, after)
    assert np.isnan(after[[1, 4]]).all()


@pytest.mark.parametrize("scheme", ["loo", "ordered"])
@pytest.mark.parametrize("target_type", ["binary", "multiclass"])
def test_classification_mutation_with_fixed_class_schema(scheme, target_type):
    y = np.array([0, 1, 2] * 4) if target_type == "multiclass" else np.array([0, 1] * 6)
    X = pd.DataFrame({"g": [str(i) for i in range(len(y))]})
    kw = dict(
        cols=["g"],
        scheme=scheme,
        smooth=3.0,
        target_type=target_type,
        random_state=7,
        output="numpy",
    )
    before = TargetEncoder(**kw).fit_transform(X, y)
    order = check_random_state(7).permutation(len(y))
    i = order[-2]
    mutated = y.copy()
    mutated[i] = (y[i] + 1) % (3 if target_type == "multiclass" else 2)
    after = TargetEncoder(**kw).fit_transform(X, mutated)
    rows = [i] if scheme == "loo" else order[:-1]
    np.testing.assert_array_equal(before[rows], after[rows])
