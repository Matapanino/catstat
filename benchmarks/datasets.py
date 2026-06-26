"""Seeded synthetic dataset generators for the catstat benchmark + correctness harness.

Each generator returns ``(X, y, meta)`` (``y`` may be ``None`` for unsupervised stress cases).
``meta`` records the knobs so a ledger row is self-describing. See
docs/proposals/evaluation-harness-design.md §2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_SIZES = {"small": 10_000, "medium": 100_000, "large": 1_000_000}


def _n(size):
    return _SIZES.get(size, size if isinstance(size, int) else 10_000)


def make_high_cardinality(size="small", cardinality=None, seed=0):
    n = _n(size)
    k = cardinality or max(2, n // 20)
    rng = np.random.default_rng(seed)
    g = rng.integers(0, k, size=n).astype(str)
    eff = rng.normal(size=k)
    y = eff[g.astype(int)] + rng.normal(0, 0.3, n)
    return pd.DataFrame({"g": g}), y, {"case": "high_cardinality", "n": n, "cardinality": k}


def make_rare_categories(size="small", seed=0):
    n = _n(size)
    rng = np.random.default_rng(seed)
    # Zipf-ish: a few common levels, a long tail of singletons.
    k = max(4, n // 5)
    probs = 1.0 / np.arange(1, k + 1)
    probs /= probs.sum()
    g = rng.choice([f"c{i}" for i in range(k)], size=n, p=probs)
    eff = {f"c{i}": rng.normal() for i in range(k)}
    y = np.array([eff[c] for c in g]) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"g": g}), y, {"case": "rare_categories", "n": n, "cardinality": k}


def make_with_missing(size="small", nan_frac=0.1, seed=0):
    X, y, meta = make_high_cardinality(size, seed=seed)
    rng = np.random.default_rng(seed + 1)
    mask = rng.uniform(size=len(X)) < nan_frac
    X = X.copy()
    X.loc[mask, "g"] = np.nan
    meta = {**meta, "case": "with_missing", "nan_frac": nan_frac}
    return X, y, meta


def make_regression(size="small", cardinality=50, seed=0):
    n = _n(size)
    rng = np.random.default_rng(seed)
    g = rng.integers(0, cardinality, size=n).astype(str)
    eff = rng.normal(size=cardinality)
    y = eff[g.astype(int)] + rng.normal(0, 0.5, n)
    return pd.DataFrame({"g": g}), y, {"case": "regression", "n": n, "cardinality": cardinality}


def make_binary(size="small", cardinality=50, pos_rate=0.3, seed=0):
    n = _n(size)
    rng = np.random.default_rng(seed)
    g = rng.integers(0, cardinality, size=n)
    base = rng.uniform(0.05, 0.95, size=cardinality)
    p = base[g] * (pos_rate / base.mean())
    y = (rng.uniform(size=n) < np.clip(p, 0, 1)).astype(int)
    return (
        pd.DataFrame({"g": g.astype(str)}),
        y,
        {"case": "binary", "n": n, "cardinality": cardinality, "pos_rate": pos_rate},
    )


def make_multiclass(size="small", cardinality=50, classes=5, seed=0):
    n = _n(size)
    rng = np.random.default_rng(seed)
    g = rng.integers(0, cardinality, size=n).astype(str)
    y = rng.integers(0, classes, size=n)
    return (
        pd.DataFrame({"g": g}),
        y,
        {"case": "multiclass", "n": n, "cardinality": cardinality, "classes": classes},
    )


def make_multi_column(size="small", n_cols=4, seed=0):
    n = _n(size)
    rng = np.random.default_rng(seed)
    cols = {f"g{j}": rng.integers(0, 20, size=n).astype(str) for j in range(n_cols)}
    X = pd.DataFrame(cols)
    # an interaction only a joint encoding would capture
    y = (X["g0"].astype(int) * X["g1"].astype(int)).to_numpy() / 100.0 + rng.normal(0, 0.3, n)
    return X, y, {"case": "multi_column", "n": n, "n_cols": n_cols}


def make_leakage_trap(size="small", n_levels=None, seed=0):
    n = _n(size)
    k = n_levels or max(2, n // 2)
    rng = np.random.default_rng(seed)
    g = rng.integers(0, k, size=n).astype(str)
    y = rng.normal(size=n)  # independent of g
    return pd.DataFrame({"g": g}), y, {"case": "leakage_trap", "n": n, "cardinality": k}


def make_mixed_dtypes(size="small", seed=0):
    n = _n(size)
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "obj": rng.choice(list("abcde"), size=n),
            "cat": pd.Categorical(rng.choice(["x", "y", "z"], size=n)),
            "int_code": rng.integers(0, 7, size=n),
        }
    )
    y = rng.normal(size=n)
    return X, y, {"case": "mixed_dtypes", "n": n}


# Registry the runner iterates over.
GENERATORS = {
    "regression": make_regression,
    "binary": make_binary,
    "multiclass": make_multiclass,
    "high_cardinality": make_high_cardinality,
    "rare_categories": make_rare_categories,
    "with_missing": make_with_missing,
    "leakage_trap": make_leakage_trap,
}
