"""WOE (weight of evidence): correctness, smoothing derivation, OOF safety, fallbacks.

``woe = logit(smoothed p) - logit(prior)`` is derived from the existing mean/probability
smoothing (honesty rule: probability-family smoothing, nothing new), so it is binary-only,
rides the additive fast OOF kernel, and its unknown/missing fallback is exactly 0.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.conftest import make_binary

import catstat._base as base
from catstat import TargetEncoder


def _logit(p):
    p = np.asarray(p, dtype=float)
    return np.log(p) - np.log1p(-p)


def _hand_woe(X, y, smooth):
    """Independent reference: smoothed P(y=1|c) via the documented m-estimate / EB formulas."""
    yb = np.asarray(y, dtype=float)
    df = pd.DataFrame({"g": X["g"], "y": yb})
    grp = df.groupby("g")["y"]
    n, p_raw = grp.count().astype(float), grp.mean()
    prior = yb.mean()
    if smooth == "auto":
        var_pop = p_raw * (1.0 - p_raw)  # sumsq/count - mean^2 for 0/1 targets
        tau2 = yb.var()  # population (ddof=0)
        m = var_pop / tau2
    else:
        m = float(smooth)
    lam = n / (n + m)
    p = lam * p_raw + (1.0 - lam) * prior
    return pd.Series(_logit(p) - _logit(prior), index=p.index)


@pytest.mark.parametrize("smooth", [10.0, "auto"])
def test_matches_hand_computed(smooth):
    X, y = make_binary(n=800, k=8, seed=0)
    enc = TargetEncoder(cols=["g"], stats=["woe"], smooth=smooth, output="numpy").fit(X, y)
    out = enc.transform(X).ravel()
    ref = X["g"].map(_hand_woe(X, y, smooth)).to_numpy()
    assert np.allclose(out, ref, rtol=1e-9, atol=1e-12)
    assert list(enc.get_feature_names_out()) == ["g__woe"]


def test_sign_convention_positive_class():
    # a category enriched in the positive class (classes_[1]) must get positive WOE
    X = pd.DataFrame({"g": ["hi"] * 20 + ["lo"] * 20})
    y = np.array([1] * 18 + [0] * 2 + [1] * 2 + [0] * 18)
    enc = TargetEncoder(cols=["g"], stats=["woe"], smooth=1.0).fit(X, y)
    out = enc.transform(pd.DataFrame({"g": ["hi", "lo"]})).to_numpy().ravel()
    assert out[0] > 0 > out[1]


def _fast(X, y, **kw):
    return np.asarray(TargetEncoder(**kw).fit_transform(X, y), dtype=float)


def _slow(X, y, **kw):
    saved = base._ADDITIVE_STATS
    base._ADDITIVE_STATS = frozenset()
    try:
        return np.asarray(TargetEncoder(**kw).fit_transform(X, y), dtype=float)
    finally:
        base._ADDITIVE_STATS = saved


@pytest.mark.parametrize("smooth", [20.0, "auto"])
@pytest.mark.parametrize("handle_missing", ["value", "return_nan"])
@pytest.mark.parametrize("handle_unknown", ["value", "return_nan"])
def test_fast_oof_equals_slow(smooth, handle_missing, handle_unknown):
    rng = np.random.RandomState(1)
    n = 400
    g = np.array([f"a{v}" for v in rng.randint(0, 50, n)], dtype=object)
    g[rng.rand(n) < 0.1] = np.nan
    X, y = pd.DataFrame({"g": g}), pd.Series((rng.rand(n) < 0.4).astype(int))
    kw = dict(
        cols=["g"],
        stats=["mean", "woe"],
        smooth=smooth,
        cv=5,
        shuffle=True,
        random_state=0,
        handle_missing=handle_missing,
        handle_unknown=handle_unknown,
    )
    np.testing.assert_allclose(
        _fast(X, y, **kw), _slow(X, y, **kw), rtol=1e-7, atol=1e-9, equal_nan=True
    )


def test_oof_noise_trap_and_asymmetry():
    rng = np.random.default_rng(2)
    n, k = 4000, 200
    X = pd.DataFrame({"g": rng.choice([f"n{i}" for i in range(k)], size=n)})
    y = (rng.uniform(size=n) < 0.5).astype(int)  # independent of g
    # fixed m>0 keeps every fold's WOE finite (auto leaves pure complements unshrunk -> +-inf)
    kw = dict(
        cols=["g"], stats=["woe"], smooth=20.0, cv=5, random_state=0, shuffle=True, output="numpy"
    )
    oof = np.asarray(TargetEncoder(**kw).fit_transform(X, y)).ravel()
    leaky = np.asarray(TargetEncoder(**kw).fit(X, y).transform(X)).ravel()
    assert not np.allclose(oof, leaky)  # fit_transform is out-of-fold
    corr = float(np.corrcoef(oof, y)[0, 1])
    assert abs(corr) < 0.06  # no target information leaks through the OOF WOE


def test_rejected_for_regression_and_multiclass():
    X = pd.DataFrame({"g": ["a", "b"] * 30})
    with pytest.raises(ValueError, match="binary target"):
        TargetEncoder(cols=["g"], stats=["woe"]).fit(X, np.linspace(0, 1, 60))
    with pytest.raises(ValueError, match="binary target"):
        TargetEncoder(cols=["g"], stats=["woe"]).fit(X, np.arange(60) % 3)


def test_rejected_for_loo_and_ordered():
    X, y = make_binary(seed=3)
    for scheme in ("loo", "ordered"):
        with pytest.raises(ValueError, match="mean only"):
            TargetEncoder(cols=["g"], stats=["woe"], scheme=scheme).fit(X, y)


def test_unknown_and_missing_fallbacks():
    X, y = make_binary(n=400, k=6, seed=4)
    enc = TargetEncoder(cols=["g"], stats=["woe"]).fit(X, y)
    assert enc.global_stats_["g__woe"] == 0.0
    assert enc.transform(pd.DataFrame({"g": ["UNSEEN"]})).iloc[0, 0] == 0.0

    Xm = X.copy()
    Xm.loc[Xm.index[:60], "g"] = np.nan
    enc_v = TargetEncoder(cols=["g"], stats=["woe"], handle_missing="value", output="numpy")
    out = enc_v.fit(Xm, y).transform(Xm).ravel()
    miss = Xm["g"].isna().to_numpy()
    assert np.isfinite(out[miss]).all()
    assert len(set(out[miss])) == 1  # missing is its own learned level

    enc_n = TargetEncoder(cols=["g"], stats=["woe"], handle_missing="return_nan", output="numpy")
    out_n = enc_n.fit(Xm, y).transform(Xm).ravel()
    assert np.isnan(out_n[miss]).all()


def test_pure_category_inf_behavior():
    """Honesty rule: no hidden clipping. A pure category is +-inf under smooth=0 AND under the
    default smooth='auto' (EB shrinks by within-category variance, which is zero for a pure
    category -> no shrinkage); a fixed m-estimate m>0 guarantees finite WOE."""
    X = pd.DataFrame({"g": ["pure1"] * 5 + ["pure0"] * 5 + ["mix"] * 10})
    y = np.array([1] * 5 + [0] * 5 + [0, 1] * 5)
    for smooth in (0.0, "auto"):
        enc = TargetEncoder(cols=["g"], stats=["woe"], smooth=smooth).fit(X, y)
        out = enc.transform(pd.DataFrame({"g": ["pure1", "pure0", "mix"]})).to_numpy().ravel()
        assert out[0] == np.inf and out[1] == -np.inf and np.isfinite(out[2])
    enc_m = TargetEncoder(cols=["g"], stats=["woe"], smooth=1.0).fit(X, y)
    assert np.isfinite(enc_m.transform(X).to_numpy()).all()


def test_alongside_other_stats_and_determinism():
    X, y = make_binary(n=600, k=10, seed=5)
    kw = dict(cols=["g"], stats=["mean", "woe", "count"], cv=5, random_state=0, output="numpy")
    a = np.asarray(TargetEncoder(**kw).fit_transform(X, y))
    b = np.asarray(TargetEncoder(**kw).fit_transform(X, y))
    assert np.array_equal(a, b)
    enc = TargetEncoder(**kw).fit(X, y)
    assert list(enc.get_feature_names_out()) == ["g__te_mean", "g__woe", "g__count"]
