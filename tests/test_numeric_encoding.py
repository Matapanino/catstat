"""Opt-in, cardinality-aware numeric-column target encoding.

Covers: cardinality routing (direct vs bin), encode correctness, OOF/no-leakage incl. bin edges
being independent of y, unknown/out-of-range + NaN handling, determinism, feature names, param
validation, and edge cases (near-constant, all-NaN, bool exclusion, threshold boundary, ratio
threshold, uniform binning, combination mode).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from catstat import TargetEncoder


def make_numeric(n=1500, seed=0):
    """A low-cardinality int feature and a high-cardinality continuous feature, both predictive."""
    rng = np.random.default_rng(seed)
    lc = rng.integers(0, 5, size=n)  # 5 distinct -> "direct" under default threshold
    hc = rng.normal(size=n)  # continuous -> "bin"
    y = 1.5 * lc + 2.0 * hc + rng.normal(0, 0.3, size=n)
    return pd.DataFrame({"lc": lc, "hc": hc}), y


# ---- routing -------------------------------------------------------------------------------------
def test_auto_routes_by_cardinality():
    X, y = make_numeric()
    enc = TargetEncoder(numeric="auto", cardinality_threshold=20, n_bins=10, random_state=0)
    enc.fit(X, y)
    assert enc.numeric_strategy_ == {"lc": "direct", "hc": "bin"}
    assert "hc" in enc.bin_edges_ and "lc" not in enc.bin_edges_
    assert enc.bin_edges_["hc"].size == 9  # 10 bins -> 9 interior edges


def test_threshold_boundary_is_inclusive():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.integers(0, 10, size=400)})  # 10 distinct values
    y = rng.normal(size=400)
    inc = TargetEncoder(numeric="auto", cardinality_threshold=10).fit(X, y)
    exc = TargetEncoder(numeric="auto", cardinality_threshold=9).fit(X, y)
    assert inc.numeric_strategy_["a"] == "direct"  # nunique <= threshold -> direct
    assert exc.numeric_strategy_["a"] == "bin"


def test_ratio_threshold_routes_by_fraction():
    rng = np.random.default_rng(2)
    n = 1000
    X = pd.DataFrame({"a": rng.integers(0, 30, size=n)})  # ~30 unique / 1000 = 0.03
    y = rng.normal(size=n)
    lo = TargetEncoder(numeric="auto", cardinality_threshold=0.05).fit(X, y)
    hi = TargetEncoder(numeric="auto", cardinality_threshold=0.01).fit(X, y)
    assert lo.numeric_strategy_["a"] == "direct"
    assert hi.numeric_strategy_["a"] == "bin"


def test_bool_excluded_from_numeric_auto():
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"g": rng.choice(list("abc"), size=300), "flag": rng.random(300) > 0.5})
    y = rng.normal(size=300)
    enc = TargetEncoder(numeric="auto", random_state=0).fit(X, y)
    assert enc.numeric_cols_ == []  # bool 'flag' is neither categorical-auto nor numeric
    assert list(enc.get_feature_names_out()) == ["g__te_mean"]


# ---- encode correctness --------------------------------------------------------------------------
def test_direct_equals_categorical_encoding_of_same_values():
    """numeric='direct' on an int column == the status-quo 'encode each value as a category'."""
    X, y = make_numeric()
    kw = dict(cols=["lc"], smooth=0.0, cv=5, random_state=0, output="numpy")
    direct = TargetEncoder(numeric="direct", **kw).fit_transform(X, y)
    asis = TargetEncoder(**kw).fit_transform(X, y)  # numeric='ignore'
    np.testing.assert_allclose(direct, asis)


def test_bin_assigns_same_value_within_a_bin():
    X, y = make_numeric()
    enc = TargetEncoder(cols=["hc"], numeric="bin", n_bins=8, smooth=0.0, random_state=0,
                        output="numpy").fit(X, y)
    edges = enc.bin_edges_["hc"]
    binid = np.clip(np.digitize(X["hc"].to_numpy(float), edges), 0, edges.size)
    enc_vals = enc.transform(X).ravel()  # full-table encodings are deterministic per bin
    for b in np.unique(binid):  # every row sharing a bin gets an identical encoded value
        vals = enc_vals[binid == b]
        assert np.allclose(vals, vals[0])


def test_numeric_keys_are_gpu_safe_strings():
    # cuDF rejects object-dtype integer arrays, so numeric-encoded keys must be strings for CPU/GPU
    # parity (the string path is the validated one). Guards the MixedTypeError regression.
    X, y = make_numeric()
    enc = TargetEncoder(numeric="auto", n_bins=8, random_state=0).fit(X, y)
    assert set(enc.numeric_cols_) == {"lc", "hc"}  # lc -> direct, hc -> bin
    for col in enc.numeric_cols_:
        assert all(isinstance(c, str) for c in enc.categories_[col])


# ---- leakage safety --------------------------------------------------------------------------
def test_bin_edges_are_independent_of_y():
    """Edges come from X only: permuting or replacing y must not change them."""
    X, y = make_numeric()
    rng = np.random.default_rng(9)
    base = TargetEncoder(numeric="bin", n_bins=12, random_state=0).fit(X, y)
    perm = TargetEncoder(numeric="bin", n_bins=12, random_state=0).fit(X, rng.permutation(y))
    other = TargetEncoder(numeric="bin", n_bins=12, random_state=0).fit(X, rng.normal(size=len(y)))
    np.testing.assert_array_equal(base.bin_edges_["hc"], perm.bin_edges_["hc"])
    np.testing.assert_array_equal(base.bin_edges_["hc"], other.bin_edges_["hc"])


def test_binned_oof_reconstruction_is_exact():
    X, y = make_numeric(n=1600, seed=4)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    enc = TargetEncoder(cols=["hc"], numeric="bin", n_bins=10, smooth=0.0, cv=kf, output="numpy")
    oof = enc.fit_transform(X, y).ravel()

    edges = enc.bin_edges_["hc"]
    binid = np.clip(np.digitize(X["hc"].to_numpy(float), edges), 0, edges.size)
    gmean = y.mean()
    recon = np.empty(len(y))
    for tr, te in kf.split(X, y):
        means = pd.DataFrame({"b": binid[tr], "y": y[tr]}).groupby("b")["y"].mean()
        for i in te:
            recon[i] = means.get(binid[i], gmean)
    # allclose, not bitwise: the fast kfold-mean path reassociates fold sums (global - fold), like
    # CPU/GPU parity (CLAUDE.md invariant #2). A real leak would be orders of magnitude > 1e-15.
    assert np.nanmax(np.abs(oof - recon)) < 1e-9


def test_binned_noise_does_not_leak():
    rng = np.random.default_rng(5)
    n = 2000
    X = pd.DataFrame({"hc": rng.normal(size=n)})  # continuous, independent of y
    y = rng.normal(size=n)
    kw = dict(numeric="bin", n_bins=100, smooth=0.0, random_state=0, output="numpy")
    oof = TargetEncoder(cv=5, **kw).fit_transform(X, y).ravel()
    leaky = TargetEncoder(**kw).fit(X, y).transform(X).ravel()
    oof_corr = abs(np.corrcoef(oof, y)[0, 1])
    leaky_corr = abs(np.corrcoef(leaky, y)[0, 1])
    assert oof_corr < 0.1  # OOF binned encoding of noise carries no target signal
    assert leaky_corr > 2 * oof_corr  # the leaky path over-fits the bins; OOF does not


# ---- unknown / missing / edge cases ----------------------------------------------------------
def test_out_of_range_clips_to_outer_bins():
    X, y = make_numeric()
    enc = TargetEncoder(cols=["hc"], numeric="bin", n_bins=10, smooth=0.0, random_state=0,
                        output="numpy").fit(X, y)
    lo, hi = X["hc"].min(), X["hc"].max()
    out = enc.transform(pd.DataFrame({"hc": [lo - 100, hi + 100]})).ravel()
    assert np.isfinite(out).all()  # extrapolated to the outer bins, not unknown -> finite
    edge_out = enc.transform(pd.DataFrame({"hc": [lo, hi]})).ravel()
    np.testing.assert_allclose(out, edge_out)  # same outer bins as the extreme in-range values


def test_nan_numeric_routes_through_handle_missing():
    X, y = make_numeric()
    Xn = X.copy()
    Xn.loc[:9, "hc"] = np.nan
    kw = dict(cols=["hc"], numeric="bin", n_bins=8, smooth=0.0, random_state=0, output="numpy")
    val = TargetEncoder(handle_missing="value", **kw).fit_transform(Xn, y)
    assert np.isfinite(val).all()  # NaN is its own (missing) level with a learned encoding
    nan_enc = TargetEncoder(handle_missing="return_nan", **kw).fit(Xn, y)
    nanout = nan_enc.transform(Xn).ravel()
    assert np.isnan(nanout[:10]).all()


def test_uniform_binning_uses_equal_width_edges():
    X, y = make_numeric()
    enc = TargetEncoder(cols=["hc"], numeric="bin", binning="uniform", n_bins=10,
                        smooth=0.0, random_state=0, output="numpy").fit(X, y)
    edges = enc.bin_edges_["hc"]
    assert edges.size == 9
    np.testing.assert_allclose(np.diff(edges), edges[1] - edges[0])  # equal-width


def test_all_nan_numeric_column_under_bin():
    rng = np.random.default_rng(11)
    X = pd.DataFrame({"c": np.full(300, np.nan)})
    y = rng.normal(size=300)
    enc = TargetEncoder(numeric="bin", n_bins=8, smooth=0.0, handle_missing="value",
                        random_state=0, output="numpy")
    out = enc.fit_transform(X, y)
    assert enc.bin_edges_["c"].size == 0  # no finite values -> single degenerate bin
    assert out.shape == (300, 1) and np.isfinite(out).all()


def test_near_constant_column_is_single_bin():
    rng = np.random.default_rng(6)
    X = pd.DataFrame({"c": np.ones(300)})  # all identical
    y = rng.normal(size=300)
    enc = TargetEncoder(numeric="bin", n_bins=10, smooth=0.0, random_state=0, output="numpy")
    out = enc.fit_transform(X, y).ravel()
    assert enc.bin_edges_["c"].size == 0  # degenerate -> single bin
    assert np.isfinite(out).all()


# ---- feature names / determinism / params ----------------------------------------------------
def test_feature_names_unchanged_for_numeric():
    X, y = make_numeric()
    enc = TargetEncoder(numeric="auto", stats=["mean", "count"], random_state=0).fit(X, y)
    names = list(enc.get_feature_names_out())
    assert names == ["lc__te_mean", "lc__count", "hc__te_mean", "hc__count"]
    assert len(names) == enc.transform(X).shape[1]


def test_determinism():
    X, y = make_numeric()
    kw = dict(numeric="auto", n_bins=12, cv=5, random_state=7, output="numpy")
    a = TargetEncoder(**kw).fit_transform(X, y)
    b = TargetEncoder(**kw).fit_transform(X, y)
    np.testing.assert_array_equal(a, b)


def test_combination_mode_with_numeric_component():
    rng = np.random.default_rng(8)
    n = 800
    X = pd.DataFrame({"g": rng.choice(list("abc"), size=n), "hc": rng.normal(size=n)})
    y = rng.normal(size=n)
    enc = TargetEncoder(numeric="bin", n_bins=6, multi_feature_mode="combination",
                        smooth=0.0, random_state=0, output="numpy")
    out = enc.fit_transform(X, y)
    assert out.shape == (n, 1)  # one joint unit
    assert enc.numeric_strategy_["hc"] == "bin"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"numeric": "weird"},
        {"numeric": "bin", "binning": "kmeans"},
        {"numeric": "bin", "n_bins": 1},
        {"numeric": "bin", "n_bins": 2.5},
        {"numeric": "auto", "cardinality_threshold": 0},
        {"numeric": "auto", "cardinality_threshold": 1.5},
        {"numeric": "auto", "cardinality_threshold": -3},
    ],
)
def test_param_validation_raises(kwargs):
    X, y = make_numeric(n=200)
    with pytest.raises(ValueError):
        TargetEncoder(random_state=0, **kwargs).fit(X, y)


def test_ignore_default_is_backward_compatible():
    X, y = make_numeric()
    # numeric defaults to 'ignore'; explicitly listing a numeric col still encodes it as categories
    enc = TargetEncoder(cols=["lc"], smooth=0.0, cv=5, random_state=0, output="numpy")
    out = enc.fit_transform(X, y)
    assert out.shape == (len(y), 1)
    assert enc.numeric_cols_ == []  # no special numeric handling under 'ignore'


# ---- explicit / per-column bin edges (binning = edge array | dict) ----------------------------
def test_explicit_edge_array_sets_the_bins():
    X, y = make_numeric()
    enc = TargetEncoder(cols=["hc"], numeric="bin", binning=[-2.0, -0.5, 0.5, 2.0],
                        smooth=0.0, random_state=0, output="numpy").fit(X, y)
    # full boundaries -> the outer two are dropped (np.digitize uses interior); 4 edges -> 3 bins
    np.testing.assert_allclose(enc.bin_edges_["hc"], [-0.5, 0.5])
    binid = np.clip(np.digitize(X["hc"].to_numpy(float), enc.bin_edges_["hc"]), 0, 2)
    vals = enc.transform(X).ravel()
    for b in np.unique(binid):  # each bin -> a single encoded value
        v = vals[binid == b]
        assert np.allclose(v, v[0])


def test_explicit_edges_ignore_n_bins():
    X, y = make_numeric()
    enc = TargetEncoder(cols=["hc"], numeric="bin", binning=[-3.0, 0.0, 3.0], n_bins=99,
                        smooth=0.0, random_state=0, output="numpy").fit(X, y)
    np.testing.assert_allclose(enc.bin_edges_["hc"], [0.0])  # 3 edges -> 2 bins; n_bins ignored


def test_explicit_out_of_range_clips_to_outer_bins():
    X, y = make_numeric()
    enc = TargetEncoder(cols=["hc"], numeric="bin", binning=[-1.0, 0.0, 1.0],
                        smooth=0.0, random_state=0, output="numpy").fit(X, y)
    out = enc.transform(pd.DataFrame({"hc": [-1e9, 1e9]})).ravel()
    edge = enc.transform(pd.DataFrame({"hc": [-0.5, 0.5]})).ravel()  # inside the outer bins
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, edge)  # extrapolation clamps to the same outer bins


def test_dict_binning_per_column_mixes_strategy_and_edges():
    X, y = make_numeric()  # lc (low-card int), hc (continuous) -- both bin under numeric='bin'
    enc = TargetEncoder(numeric="bin", binning={"hc": [-2.0, 0.0, 2.0], "lc": "uniform"},
                        n_bins=4, smooth=0.0, random_state=0, output="numpy").fit(X, y)
    np.testing.assert_allclose(enc.bin_edges_["hc"], [0.0])  # explicit -> 2 bins
    le = enc.bin_edges_["lc"]
    assert le.size == 3  # 'uniform' with n_bins=4 -> 3 interior edges
    np.testing.assert_allclose(np.diff(le), le[1] - le[0])  # equal-width


def test_dict_binning_missing_column_defaults_to_quantile():
    X, y = make_numeric()
    # the dict only mentions lc, so hc falls back to the default "quantile" strategy
    via_dict = TargetEncoder(numeric="bin", binning={"lc": "uniform"}, n_bins=10,
                             random_state=0).fit(X, y)
    ref = TargetEncoder(cols=["hc"], numeric="bin", binning="quantile", n_bins=10,
                        random_state=0).fit(X, y)
    np.testing.assert_array_equal(via_dict.bin_edges_["hc"], ref.bin_edges_["hc"])


def test_binning_controls_how_not_whether_under_auto():
    X, y = make_numeric()  # lc: 5 distinct <= threshold -> 'direct' under numeric='auto'
    enc = TargetEncoder(numeric="auto", cardinality_threshold=10,
                        binning={"lc": [0.0, 2.0, 4.0]}, random_state=0).fit(X, y)
    # binning is "how", not "whether": lc still routes to direct, its dict edges go unused
    assert enc.numeric_strategy_["lc"] == "direct"
    assert "lc" not in enc.bin_edges_


def test_explicit_edges_oof_reconstruction_is_exact():
    X, y = make_numeric(n=1600, seed=4)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    enc = TargetEncoder(cols=["hc"], numeric="bin", binning=[-3.0, -1.0, 0.0, 1.0, 3.0],
                        smooth=0.0, cv=kf, output="numpy")
    oof = enc.fit_transform(X, y).ravel()
    edges = enc.bin_edges_["hc"]
    binid = np.clip(np.digitize(X["hc"].to_numpy(float), edges), 0, edges.size)
    gmean = y.mean()
    recon = np.empty(len(y))
    for tr, te in kf.split(X, y):
        means = pd.DataFrame({"b": binid[tr], "y": y[tr]}).groupby("b")["y"].mean()
        for i in te:
            recon[i] = means.get(binid[i], gmean)
    assert np.nanmax(np.abs(oof - recon)) < 1e-9  # user edges -> OOF still exact (edges _|_ y)


def test_explicit_binning_param_roundtrips_and_clones():
    from sklearn.base import clone
    enc = TargetEncoder(numeric="bin", binning=[0.0, 1.0, 2.0], random_state=0)
    assert clone(enc).get_params()["binning"] == [0.0, 1.0, 2.0]
    encd = TargetEncoder(numeric="auto", binning={"hc": [0, 1, 2], "lc": "uniform"}, random_state=0)
    assert clone(encd).get_params()["binning"] == {"hc": [0, 1, 2], "lc": "uniform"}


@pytest.mark.parametrize(
    "binning",
    [
        [5],                  # < 2 edges
        [3, 2, 1],            # not strictly increasing
        [0, 0, 1],            # duplicate -> not strictly increasing
        [0, np.nan, 1],       # non-finite
        [[0, 1], [2, 3]],     # not 1-D
        "kmeans",             # unknown strategy string
        5,                    # scalar
        {"hc": "kmeans"},     # dict: bad strategy
        {"hc": [1]},          # dict: bad edges
        {"nope": [0, 1, 2]},  # dict: unknown column
    ],
)
def test_binning_validation_raises(binning):
    X, y = make_numeric(n=200)
    with pytest.raises(ValueError):
        TargetEncoder(numeric="bin", binning=binning, random_state=0).fit(X, y)
