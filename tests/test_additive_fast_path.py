"""The single-pass additive (var/std/skew/kurt) OOF kernel must equal the per-fold slow path.

The fast path reconstructs each category's out-of-fold statistic from complement power sums (the
``(count, sum, sumsq)`` the mean kernel computes; shape stats add the ``y'**3``/``y'**4`` sums,
shifted by the global mean for stability), with a per-fold complement-global fallback when
``n < min_samples_category`` or ``n`` is below the stat's min-n (var/std 2, skew 3, kurt 4). This
asserts it matches catstat's own per-fold group-by path -- the reference -- at allclose (not
bitwise: the one-pass moment formula reassociates sums; CLAUDE.md invariant #2) across the
fallback matrix {var, std, skew, kurt} x {min_samples 1/2/5} x {missing, unknown} x
{single, combination}, plus the hybrid mixed additive+non-additive case and a large-offset
stability case.
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


@pytest.mark.parametrize("stat", ["var", "std", "skew", "kurt"])
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
    """Additive + non-additive together: the fast hybrid (single-pass kernel for
    mean/var/std/skew/kurt, the per-fold loop only for median) must equal the full per-fold loop
    for every column. Includes the shape stats so the shared pass runs at order 4 with a nonzero
    shift while mean/var/std ride the same shifted sums."""
    X, y = _data()
    kw = dict(
        cols=["a", "b"],
        stats=["mean", "var", "std", "skew", "kurt", "median"],
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


def test_shape_stats_large_offset_fast_equals_slow():
    """y ~ 1e9 +- 1: the shifted power sums must stay allclose to the (two-pass, per-fold) slow
    path and produce finite values -- the raw (unshifted) sums would cancel catastrophically."""
    X, y = _data(seed=7)
    y = 1e9 + y
    kw = dict(
        cols=["a", "b"],
        stats=["mean", "var", "skew", "kurt"],
        cv=5,
        shuffle=True,
        random_state=0,
        handle_missing="value",
        handle_unknown="value",
    )
    fast = _fast(X, y, **kw)
    assert np.isfinite(fast).all()
    # mean columns carry the 1e9 scale -> compare at relative tolerance only
    np.testing.assert_allclose(fast, _slow(X, y, **kw), rtol=1e-6, atol=1e-6, equal_nan=True)


def test_small_complement_shape_fallback_finite():
    """Categories of size 3-4: some fold complements drop below skew/kurt's min-n and must take
    the per-fold complement-global fallback -- finite everywhere, never NaN from an undefined
    statistic."""
    rng = np.random.RandomState(2)
    X = pd.DataFrame({"a": np.array([f"c{i}" for i in range(30)] * 4, dtype=object)})  # ~4 each
    y = pd.Series(rng.randn(120))
    for stat in ("skew", "kurt"):
        out = _fast(X, y, cols=["a"], stats=[stat], cv=5, shuffle=True, random_state=0)
        assert np.isfinite(out).all()


def test_singleton_var_falls_back_to_complement_global():
    """A category that is a singleton within a fold's complement has undefined sample variance
    (ddof=1) and must fall back to that fold's complement-global var -- never NaN, never leaking."""
    rng = np.random.RandomState(1)
    X = pd.DataFrame({"a": np.array([f"c{i}" for i in range(50)] * 2, dtype=object)})  # ~2 each
    y = pd.Series(rng.randn(100))  # continuous target
    out = _fast(X, y, cols=["a"], stats=["var"], cv=5, shuffle=True, random_state=0)
    assert np.isfinite(out).all()  # no undefined encodings escape the fallback
