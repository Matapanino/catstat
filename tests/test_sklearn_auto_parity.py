"""KI-010: catstat's smooth='auto' must match sklearn TargetEncoder(smooth='auto') exactly.

Both implement the same empirical-Bayes blend -- lambda = n/(n + sigma2_pop_i/tau2_pop), i.e.
sklearn's `lambda = n*tau2 / (n*tau2 + SS/n)` -- so the full-data encodings agree to fp rounding.
Compared per category against sklearn's ``encodings_`` (their fit(), catstat's fit(): both the
"leaky"/inference tables; the CV machinery differs by design and is out of scope here).
Requires scikit-learn >= 1.4 (repo convention; TargetEncoder landed in 1.3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

sklearn = pytest.importorskip("sklearn", minversion="1.4")
from sklearn.preprocessing import TargetEncoder as SkTE  # noqa: E402

from catstat import TargetEncoder as CsTE  # noqa: E402


def _catstat_encoding_for(cs, cats, class_label=None):
    tab = cs._fit_tables[("g", "mean", class_label)]
    order = [list(tab.index).index(c) for c in cats]
    return tab.values[order]


def _compare(X, y, target_type):
    sk = SkTE(smooth="auto", target_type=target_type, shuffle=False).fit(X[["g"]], y)
    cats = list(sk.categories_[0])
    cs = CsTE(cols=["g"], stats=["mean"], smooth="auto", target_type=target_type).fit(X, y)
    if target_type == "multiclass":
        # sklearn: one encoding array per class (single feature); catstat: one table per class
        for i, c in enumerate(sk.classes_):
            np.testing.assert_allclose(
                np.asarray(sk.encodings_[i], dtype=float),
                _catstat_encoding_for(cs, cats, class_label=c),
                rtol=1e-12, atol=1e-14,
            )
        return
    np.testing.assert_allclose(
        np.asarray(sk.encodings_[0], dtype=float),
        _catstat_encoding_for(cs, cats),
        rtol=1e-12, atol=1e-14,
    )


def _mixed_g():
    # mixed sizes incl. a pair and a singleton: the shrinkage-weight edge cases
    return np.array(["a"] * 50 + ["b"] * 7 + ["c"] * 2 + ["solo"] + ["d"] * 140)


def test_continuous_parity_incl_singletons_and_offset():
    rng = np.random.default_rng(0)
    g = _mixed_g()
    y = rng.normal(3.0, 2.0, size=len(g))
    _compare(pd.DataFrame({"g": g}), y, "continuous")
    _compare(pd.DataFrame({"g": g}), 1e9 + y, "continuous")  # shift-stable path stays exact


def test_constant_category_and_constant_target():
    rng = np.random.default_rng(1)
    g = _mixed_g()
    y = rng.normal(size=len(g))
    y[:50] = 5.0  # zero within-category variance -> lambda = 1 (no shrinkage) on both sides
    _compare(pd.DataFrame({"g": g}), y, "continuous")
    _compare(pd.DataFrame({"g": g}), np.full(len(g), 2.5), "continuous")


def test_binary_and_multiclass_parity():
    rng = np.random.default_rng(2)
    g = _mixed_g()
    yb = (rng.uniform(size=len(g)) < 0.3).astype(int)
    _compare(pd.DataFrame({"g": g}), yb, "binary")
    ymc = rng.integers(0, 4, size=len(g))
    _compare(pd.DataFrame({"g": g}), ymc, "multiclass")


def test_high_cardinality_parity():
    rng = np.random.default_rng(3)
    g = rng.integers(0, 500, size=20_000).astype(str)
    y = rng.normal(size=20_000)
    _compare(pd.DataFrame({"g": g}), y, "continuous")
