"""laplace_alpha: optional add-alpha smoothing for frequencies (default off, counts stay exact)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from catstat import CountEncoder, FrequencyEncoder


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"g": rng.choice(list("abcde"), size=n)})


def test_alpha_matches_hand_formula_and_unknown_fallback():
    X = _data()
    n, alpha = len(X), 2.0
    vc = X["g"].value_counts()
    k = len(vc)
    enc = FrequencyEncoder(cols=["g"], laplace_alpha=alpha).fit(X)
    out = enc.transform(X).to_numpy().ravel()
    ref = X["g"].map((vc + alpha) / (n + alpha * k)).to_numpy()
    assert np.allclose(out, ref, rtol=1e-12)
    unseen = enc.transform(pd.DataFrame({"g": ["ZZZ"]})).iloc[0, 0]
    assert unseen == pytest.approx(alpha / (n + alpha * k))
    assert enc.global_stats_["g__freq"] == pytest.approx(alpha / (n + alpha * k))


def test_default_off_is_unchanged():
    X = _data(seed=1)
    a = FrequencyEncoder(cols=["g"]).fit(X)
    b = FrequencyEncoder(cols=["g"], laplace_alpha=0.0).fit(X)
    assert np.allclose(a.transform(X).to_numpy(), b.transform(X).to_numpy(), rtol=0)
    assert a.transform(pd.DataFrame({"g": ["ZZZ"]})).iloc[0, 0] == 0.0


def test_learned_frequencies_sum_to_one():
    X = _data(seed=2)
    enc = FrequencyEncoder(cols=["g"], laplace_alpha=0.7).fit(X)
    vals = enc._fit_tables[("g", "frequency", None)].values
    assert float(np.sum(vals)) == pytest.approx(1.0)  # sum over the K learned categories


def test_fit_transform_equals_fit_then_transform():
    X = _data(seed=3)
    enc = FrequencyEncoder(cols=["g"], laplace_alpha=1.5)
    a = np.asarray(enc.fit_transform(X))
    b = np.asarray(FrequencyEncoder(cols=["g"], laplace_alpha=1.5).fit(X).transform(X))
    assert np.array_equal(a, b)  # unsupervised: no OOF, identical paths


def test_counts_reject_alpha_and_bad_alpha_rejected():
    X = _data(seed=4)
    with pytest.raises(ValueError, match="honesty"):
        CountEncoder(cols=["g"], laplace_alpha=1.0).fit(X)
    for bad in (-0.5, float("nan")):
        with pytest.raises(ValueError, match="laplace_alpha"):
            FrequencyEncoder(cols=["g"], laplace_alpha=bad).fit(X)


def test_missing_level_counts_toward_k():
    X = _data(seed=5).copy()
    X.loc[X.index[:20], "g"] = np.nan  # MISSING becomes its own learned level (K includes it)
    alpha = 1.0
    enc = FrequencyEncoder(cols=["g"], laplace_alpha=alpha, handle_missing="value").fit(X)
    vals = enc._fit_tables[("g", "frequency", None)].values
    assert float(np.sum(vals)) == pytest.approx(1.0)
    miss_freq = enc.transform(pd.DataFrame({"g": [np.nan]})).iloc[0, 0]
    assert miss_freq == pytest.approx((20 + alpha) / (len(X) + alpha * len(vals)))


def test_clone_params_roundtrip():
    from sklearn.base import clone

    e = FrequencyEncoder(cols=["g"], laplace_alpha=0.3)
    assert clone(e).get_params()["laplace_alpha"] == 0.3
    e2 = CountEncoder(cols=["g"], normalize=True, laplace_alpha=0.3)
    assert clone(e2).get_params()["laplace_alpha"] == 0.3
