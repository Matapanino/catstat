"""Shared fixtures + GPU-marker auto-skip."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_regression(n=400, k=6, seed=0):
    rng = np.random.default_rng(seed)
    cats = [f"c{i}" for i in range(k)]
    g = rng.choice(cats, size=n)
    eff = {c: rng.normal() for c in cats}
    y = np.array([eff[c] for c in g]) + rng.normal(0, 0.2, n)
    X = pd.DataFrame({"g": g, "x_num": rng.normal(size=n)})
    return X, y


def make_binary(n=400, k=6, seed=1, pos_rate=0.4):
    rng = np.random.default_rng(seed)
    cats = [f"c{i}" for i in range(k)]
    g = rng.choice(cats, size=n)
    base = {c: rng.uniform(0.1, 0.9) for c in cats}
    p = np.array([base[c] for c in g])
    y = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame({"g": g}), y


def make_multiclass(n=600, k=6, classes=3, seed=2):
    rng = np.random.default_rng(seed)
    cats = [f"c{i}" for i in range(k)]
    g = rng.choice(cats, size=n)
    y = rng.integers(0, classes, size=n)
    return pd.DataFrame({"g": g}), y


def make_leakage_trap(n=2000, n_levels=1000, seed=3):
    """High-cardinality category that is independent of the (random) target."""
    rng = np.random.default_rng(seed)
    g = rng.choice([f"n{i}" for i in range(n_levels)], size=n)
    y = rng.normal(size=n)
    return pd.DataFrame({"g": g}), y


@pytest.fixture
def reg_data():
    return make_regression()


@pytest.fixture
def bin_data():
    return make_binary()


@pytest.fixture
def mc_data():
    return make_multiclass()


def pytest_collection_modifyitems(config, items):
    try:
        import cudf  # noqa: F401
        import cupy  # noqa: F401

        have_gpu = True
    except Exception:
        have_gpu = False
    if not have_gpu:
        skip_gpu = pytest.mark.skip(reason="no GPU/RAPIDS available (CPU-only / CI)")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
