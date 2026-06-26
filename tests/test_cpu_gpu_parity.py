"""CPU/GPU parity -- runs only on a GPU/RAPIDS box (auto-skipped on CPU-only / CI).

Because catstat owns fold assignment, the same ``random_state`` yields the same folds on both
backends, so ``transform`` and ``fit_transform`` agree to **allclose** (not bitwise -- GPU
reduction order differs). Locally these are skipped by the ``gpu`` marker (see conftest); the
real run is the Colab loop, ``scripts/colab_gpu_parity.sh``.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("stats", [["mean"], ["var"]])
def test_cpu_gpu_parity(stats):  # pragma: no cover - GPU only
    import pandas as pd

    from catstat import TargetEncoder

    rng = np.random.default_rng(0)
    n, k = 200_000, 5_000
    g = rng.integers(0, k, size=n).astype(str)
    y = rng.normal(size=n)
    X = pd.DataFrame({"g": g})

    kw = dict(cols=["g"], stats=stats, cv=5, random_state=0, output="numpy")
    a_t = np.asarray(TargetEncoder(**kw, backend="cpu").fit(X, y).transform(X))
    b_t = np.asarray(TargetEncoder(**kw, backend="gpu").fit(X, y).transform(X))
    assert np.allclose(a_t, b_t, rtol=1e-5, atol=1e-8)

    a_ft = np.asarray(TargetEncoder(**kw, backend="cpu").fit_transform(X, y))
    b_ft = np.asarray(TargetEncoder(**kw, backend="gpu").fit_transform(X, y))
    assert np.allclose(a_ft, b_ft, rtol=1e-5, atol=1e-8, equal_nan=True)


@pytest.mark.parametrize("numeric", ["auto", "bin"])
def test_cpu_gpu_parity_numeric(numeric):  # pragma: no cover - GPU only
    """Numeric TE parity: bin edges are host-side numpy (deterministic), so the bin ids are
    identical on both backends and the per-bin encodings must agree at allclose. ``"auto"`` also
    exercises the direct path (the low-cardinality column routes to direct)."""
    import pandas as pd

    from catstat import TargetEncoder

    rng = np.random.default_rng(1)
    n = 200_000
    lc = rng.integers(0, 8, size=n)  # low cardinality -> "direct" under "auto"
    hc = rng.normal(size=n)  # high cardinality continuous -> "bin"
    y = rng.normal(size=n)
    X = pd.DataFrame({"lc": lc, "hc": hc})

    kw = dict(cols=["lc", "hc"], numeric=numeric, n_bins=20, cv=5, random_state=0, output="numpy")
    a_t = np.asarray(TargetEncoder(**kw, backend="cpu").fit(X, y).transform(X))
    b_t = np.asarray(TargetEncoder(**kw, backend="gpu").fit(X, y).transform(X))
    assert np.allclose(a_t, b_t, rtol=1e-5, atol=1e-8)

    a_ft = np.asarray(TargetEncoder(**kw, backend="cpu").fit_transform(X, y))
    b_ft = np.asarray(TargetEncoder(**kw, backend="gpu").fit_transform(X, y))
    assert np.allclose(a_ft, b_ft, rtol=1e-5, atol=1e-8, equal_nan=True)
