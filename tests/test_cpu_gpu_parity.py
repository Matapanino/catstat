"""CPU/GPU parity -- runs only on a GPU/RAPIDS box (auto-skipped on CPU-only / CI).

Because catstat owns fold assignment, the same ``random_state`` yields the same folds on both
backends, so ``transform`` and ``fit_transform`` agree to **allclose** (not bitwise -- GPU
reduction order differs). Locally these are skipped by the ``gpu`` marker (see conftest); the
real run is the Colab loop, ``scripts/colab_gpu_parity.sh``.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("stats", [["mean"], ["var"], ["skew"], ["kurt"]])
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


def test_cpu_gpu_parity_binary_woe():  # pragma: no cover - GPU only
    """WOE rides the mean's GPU reduce (binarized y), so CPU/GPU agree at allclose."""
    import pandas as pd

    from catstat import TargetEncoder

    rng = np.random.default_rng(4)
    n, k = 200_000, 5_000
    g = rng.integers(0, k, size=n).astype(str)
    y = (rng.uniform(size=n) < 0.4).astype(int)
    X = pd.DataFrame({"g": g})

    kw = dict(cols=["g"], stats=["mean", "woe"], cv=5, random_state=0, output="numpy")
    a_t = np.asarray(TargetEncoder(**kw, backend="cpu").fit(X, y).transform(X))
    b_t = np.asarray(TargetEncoder(**kw, backend="gpu").fit(X, y).transform(X))
    assert np.allclose(a_t, b_t, rtol=1e-5, atol=1e-8)

    a_ft = np.asarray(TargetEncoder(**kw, backend="cpu").fit_transform(X, y))
    b_ft = np.asarray(TargetEncoder(**kw, backend="gpu").fit_transform(X, y))
    assert np.allclose(a_ft, b_ft, rtol=1e-5, atol=1e-8, equal_nan=True)


@pytest.mark.parametrize("stats", [["mean"], ["var"]])
def test_cpu_gpu_parity_combination(stats):  # pragma: no cover - GPU only
    """Combination parity: the int64 mixed-radix joint codes are host-built (identical on both
    backends), so only the device group-by differs -> allclose. Also asserts the combination now
    actually runs on the GPU (no longer forced host-only)."""
    import pandas as pd

    from catstat import TargetEncoder

    rng = np.random.default_rng(2)
    n = 200_000
    a = rng.integers(0, 200, size=n).astype(str)
    b = rng.integers(0, 200, size=n).astype(str)
    y = rng.normal(size=n)
    X = pd.DataFrame({"a": a, "b": b})

    kw = dict(
        cols=["a", "b"],
        multi_feature_mode="combination",
        stats=stats,
        cv=5,
        random_state=0,
        output="numpy",
    )
    gpu = TargetEncoder(**kw, backend="gpu").fit(X, y)
    assert gpu.backend_ == "gpu"  # combination is no longer forced to CPU
    a_t = np.asarray(TargetEncoder(**kw, backend="cpu").fit(X, y).transform(X))
    b_t = np.asarray(gpu.transform(X))
    assert np.allclose(a_t, b_t, rtol=1e-5, atol=1e-8)

    a_ft = np.asarray(TargetEncoder(**kw, backend="cpu").fit_transform(X, y))
    b_ft = np.asarray(TargetEncoder(**kw, backend="gpu").fit_transform(X, y))
    assert np.allclose(a_ft, b_ft, rtol=1e-5, atol=1e-8, equal_nan=True)


def test_cpu_gpu_parity_combination_missing():  # pragma: no cover - GPU only
    """Combination with a missing component (handle_missing='value'): the missing-combo is folded
    into an ordinary int64 code on the host, so the device group-by needs no MISSING-sentinel
    machinery and still matches CPU."""
    import pandas as pd

    from catstat import TargetEncoder

    rng = np.random.default_rng(3)
    n = 200_000
    a = rng.integers(0, 200, size=n).astype(object)
    a[rng.uniform(size=n) < 0.1] = np.nan  # 10% missing in one component
    b = rng.integers(0, 200, size=n).astype(str)
    y = rng.normal(size=n)
    X = pd.DataFrame({"a": a, "b": b})

    kw = dict(
        cols=["a", "b"],
        multi_feature_mode="combination",
        stats=["mean"],
        handle_missing="value",
        cv=5,
        random_state=0,
        output="numpy",
    )
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
