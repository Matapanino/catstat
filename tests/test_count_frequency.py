import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
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


# ---- numeric binning (KI-030) --------------------------------------------------------------------
# Count/Frequency now bin numeric columns (numeric != "ignore"), reusing TargetEncoder's numeric
# plumbing: a binned column takes each row's bin count (a histogram) / bin frequency (normalized).
# Unsupervised, so bin edges come from X only and the no-leakage property is plain equivalence.
def _numeric_frame(n=600, seed=0):
    """Low-card int (-> "direct") + high-card continuous (-> "bin") numeric columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"lc": rng.integers(0, 5, size=n), "hc": rng.normal(size=n)})


def test_numeric_bin_is_per_bin_count():
    X = _numeric_frame()
    ce = CountEncoder(cols=["hc"], numeric="bin", n_bins=4, output="numpy")
    out = ce.fit_transform(X).ravel()
    # reconstruct the histogram by hand from the fitted interior edges
    edges = ce.bin_edges_["hc"]
    bin_id = np.clip(np.digitize(X["hc"].to_numpy(), edges), 0, edges.size)
    counts = pd.Series(bin_id).value_counts()
    expected = pd.Series(bin_id).map(counts).to_numpy().astype(float)
    assert np.allclose(out, expected)
    assert counts.sum() == len(X)  # every row counted in exactly one bin


def test_numeric_bin_frequency_is_normalized_histogram():
    X = _numeric_frame()
    n = len(X)
    ce = CountEncoder(cols=["hc"], numeric="bin", n_bins=5, output="numpy")
    cnt = ce.fit_transform(X).ravel()
    fe = FrequencyEncoder(cols=["hc"], numeric="bin", n_bins=5, output="numpy")
    freq = fe.fit_transform(X).ravel()
    assert np.allclose(freq, cnt / n)
    assert np.all((freq > 0) & (freq <= 1.0))
    # one representative frequency per distinct bin sums to 1
    edges = fe.bin_edges_["hc"]
    bin_id = np.clip(np.digitize(X["hc"].to_numpy(), edges), 0, edges.size)
    per_bin = {b: freq[bin_id == b][0] for b in np.unique(bin_id)}
    assert np.isclose(sum(per_bin.values()), 1.0)


@pytest.mark.parametrize("mode", ["bin", "auto"])
def test_numeric_fit_transform_equals_fit_then_transform(mode):
    """Unsupervised: the no-leakage property is fit_transform == fit().transform()."""
    X = _numeric_frame()
    for enc_cls in (CountEncoder, FrequencyEncoder):
        enc = enc_cls(numeric=mode, n_bins=6, output="numpy")
        ft = np.asarray(enc.fit_transform(X))
        tt = np.asarray(enc.fit(X).transform(X))
        assert np.allclose(ft, tt)


def test_numeric_auto_routes_by_cardinality():
    X = _numeric_frame()
    ce = CountEncoder(numeric="auto", cardinality_threshold=10, n_bins=8).fit(X)
    assert ce.numeric_strategy_ == {"lc": "direct", "hc": "bin"}
    assert "hc" in ce.bin_edges_ and "lc" not in ce.bin_edges_
    assert ce.bin_edges_["hc"].size == 7  # 8 bins -> 7 interior edges
    assert set(ce.numeric_cols_) == {"lc", "hc"}


def test_numeric_direct_counts_each_value():
    X = _numeric_frame()
    ce = CountEncoder(cols=["lc"], numeric="direct", output="numpy")
    out = ce.fit_transform(X).ravel()
    expected = X["lc"].map(X["lc"].value_counts()).to_numpy().astype(float)
    assert np.allclose(out, expected)


def test_numeric_bin_out_of_range_clamps_to_outer_bin():
    X = pd.DataFrame({"v": np.arange(100, dtype=float)})
    ce = CountEncoder(cols=["v"], numeric="bin", n_bins=4, output="numpy").fit(X)
    train = ce.transform(X).ravel()
    hi = ce.transform(pd.DataFrame({"v": [1e9]})).ravel()[0]
    lo = ce.transform(pd.DataFrame({"v": [-1e9]})).ravel()[0]
    # out-of-range values clamp into the outer bins and take that bin's (non-zero) count,
    # not the unseen-category fallback of 0
    assert hi == train[-1] > 0
    assert lo == train[0] > 0


def test_numeric_bin_missing_handling():
    X = pd.DataFrame({"v": np.r_[np.arange(50, dtype=float), [np.nan] * 10]})
    # "value": NaN is its own MISSING level, counted (the 10 missing rows share it)
    val = (
        CountEncoder(cols=["v"], numeric="bin", n_bins=4, handle_missing="value", output="numpy")
        .fit_transform(X)
        .ravel()
    )
    assert np.allclose(val[-10:], 10.0)
    # "return_nan": NaN rows pass through as NaN
    nan = (
        CountEncoder(
            cols=["v"], numeric="bin", n_bins=4, handle_missing="return_nan", output="numpy"
        )
        .fit_transform(X)
        .ravel()
    )
    assert np.all(np.isnan(nan[-10:]))


def test_numeric_direct_unseen_value_counts_zero():
    X = _numeric_frame()
    ce = CountEncoder(cols=["lc"], numeric="direct").fit(X)
    # a numeric value never seen in training is an unknown category -> 0
    assert ce.transform(pd.DataFrame({"lc": [999]})).iloc[0, 0] == 0.0


def test_numeric_feature_names():
    X = _numeric_frame()
    ce = CountEncoder(numeric="auto").fit(X)
    assert list(ce.get_feature_names_out()) == ["lc__count", "hc__count"]
    fe = FrequencyEncoder(numeric="auto").fit(X)
    assert list(fe.get_feature_names_out()) == ["lc__freq", "hc__freq"]
    assert fe.transform(X).shape[1] == len(fe.get_feature_names_out())


def test_numeric_bin_is_deterministic():
    X = _numeric_frame()
    a = CountEncoder(cols=["hc"], numeric="bin", n_bins=6, output="numpy")
    b = CountEncoder(cols=["hc"], numeric="bin", n_bins=6, output="numpy")
    oa, ob = a.fit_transform(X), b.fit_transform(X)
    assert np.allclose(a.bin_edges_["hc"], b.bin_edges_["hc"])
    assert np.allclose(np.asarray(oa), np.asarray(ob))


def test_numeric_params_roundtrip_and_clone():
    for enc_cls in (CountEncoder, FrequencyEncoder):
        enc = enc_cls(numeric="bin", cardinality_threshold=7, n_bins=12, binning="uniform")
        params = enc.get_params()
        assert params["numeric"] == "bin"
        assert params["cardinality_threshold"] == 7
        assert params["n_bins"] == 12
        assert params["binning"] == "uniform"
        assert clone(enc).get_params() == params
        enc.set_params(numeric="auto", n_bins=3)
        assert enc.numeric == "auto" and enc.n_bins == 3
