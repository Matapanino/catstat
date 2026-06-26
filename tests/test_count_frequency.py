import numpy as np
import pandas as pd
from tests.conftest import make_regression

from catstat import CountEncoder, FrequencyEncoder


def test_count_matches_value_counts():
    X, _ = make_regression(seed=0)
    enc = CountEncoder(cols=["g"], output="numpy")
    out = enc.fit_transform(X).ravel()
    vc = X["g"].value_counts()
    expected = X["g"].map(vc).to_numpy().astype(float)
    assert np.allclose(out, expected)


def test_frequency_is_count_over_n():
    X, _ = make_regression(seed=1)
    out = FrequencyEncoder(cols=["g"], output="numpy").fit_transform(X).ravel()
    freq = X["g"].value_counts(normalize=True)
    expected = X["g"].map(freq).to_numpy()
    assert np.allclose(out, expected)
    # frequencies are in (0, 1]
    assert np.all((out > 0) & (out <= 1.0))


def test_frequency_encoder_equals_count_normalize_true():
    X, _ = make_regression(seed=2)
    a = FrequencyEncoder(cols=["g"], output="numpy").fit_transform(X)
    b = CountEncoder(cols=["g"], normalize=True, output="numpy").fit_transform(X)
    assert np.allclose(np.asarray(a), np.asarray(b))


def test_unseen_counts_zero():
    X, _ = make_regression(seed=3)
    ce = CountEncoder(cols=["g"]).fit(X)
    assert ce.transform(pd.DataFrame({"g": ["NEW"]})).iloc[0, 0] == 0.0
