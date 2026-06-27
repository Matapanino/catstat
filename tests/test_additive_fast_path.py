"""PR-C: the single-pass additive (var/std) OOF kernel must equal the per-fold slow path.

The fast path reconstructs each category's out-of-fold var/std from the same complement
``(count, sum, sumsq)`` the mean kernel already computes, with a per-fold complement-global
fallback when ``n < min_samples_category`` or ``n < 2`` (sample variance of a singleton is NaN,
ddof=1). This asserts it matches catstat's own per-fold group-by path -- the reference -- at
allclose (not bitwise: the one-pass moment formula reassociates sums; CLAUDE.md invariant #2)
across the fallback matrix {var, std} x {min_samples 1/2/5} x {missing, unknown} x
{single, combination}, plus the hybrid mixed additive+non-additive case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import catstat._base as base
from catstat import TargetEncoder


def _data(n=400, seed=0, missing=True):
    """Frequent + rare + singleton categories (and optional missing) so the per-fold complements
    exercise the global-stat fallback (low count / undefined) and the unknown path (category absent
    from a fold's complement)."""
    rng = np.random.RandomState(seed)
    a = np.array([f"a{v}" for v in rng.randint(0, 60, n)], dtype=object)
    b = np.array([f"b{v}" for v in rng.randint(0, 8, n)], dtype=object)
    if missing:
        a[rng.rand(n) < 0.1] = np.nan
    return pd.DataFrame({"a": a, "b": b}), pd.Series(rng.randn(n))


def _fast(X, y, **kw):
    return np.asarray(TargetEncoder(**kw).fit_transform(X, y), dtype=float)


def _slow(X, y, **kw):
    """fit_transform with the additive fast path disabled -> the per-fold group-by loop (the
    reference). Patches the module-level gate set so ``_kfold_oof`` takes the slow branch."""
    saved = base._ADDITIVE_STATS
    base._ADDITIVE_STATS = frozenset()
    try:
        return np.asarray(TargetEncoder(**kw).fit_transform(X, y), dtype=float)
    finally:
        base._ADDITIVE_STATS = saved


@pytest.mark.parametrize("stat", ["var", "std"])
@pytest.mark.parametrize("min_samples", [1, 2, 5])
@pytest.mark.parametrize("handle_missing", ["value", "return_nan"])
@pytest.mark.parametrize("handle_unknown", ["value", "return_nan"])
@pytest.mark.parametrize("mode", ["independent", "combination"])
def test_fast_var_std_equals_slow(stat, min_samples, handle_missing, handle_unknown, mode):
    X, y = _data()
    kw = dict(
        cols=["a", "b"],
        stats=[stat],
        cv=5,
        shuffle=True,
        random_state=0,
        min_samples_category=min_samples,
        handle_missing=handle_missing,
        handle_unknown=handle_unknown,
        multi_feature_mode=mode,
    )
    np.testing.assert_allclose(
        _fast(X, y, **kw), _slow(X, y, **kw), rtol=1e-7, atol=1e-9, equal_nan=True
    )


def test_hybrid_mixed_stats_equals_slow():
    """Additive + non-additive together: the fast hybrid (single-pass kernel for mean/var/std, the
    per-fold loop only for median) must equal the full per-fold loop for every column."""
    X, y = _data()
    kw = dict(
        cols=["a", "b"],
        stats=["mean", "var", "std", "median"],
        cv=5,
        shuffle=True,
        random_state=0,
        min_samples_category=3,
        handle_missing="value",
        handle_unknown="value",
    )
    np.testing.assert_allclose(
        _fast(X, y, **kw), _slow(X, y, **kw), rtol=1e-7, atol=1e-9, equal_nan=True
    )


def test_singleton_var_falls_back_to_complement_global():
    """A category that is a singleton within a fold's complement has undefined sample variance
    (ddof=1) and must fall back to that fold's complement-global var -- never NaN, never leaking."""
    rng = np.random.RandomState(1)
    X = pd.DataFrame({"a": np.array([f"c{i}" for i in range(50)] * 2, dtype=object)})  # ~2 each
    y = pd.Series(rng.randn(100))  # continuous target
    out = _fast(X, y, cols=["a"], stats=["var"], cv=5, shuffle=True, random_state=0)
    assert np.isfinite(out).all()  # no undefined encodings escape the fallback
