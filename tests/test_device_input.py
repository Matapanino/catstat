"""Device-resident (cuDF) input: dispatch rules locally; full parity on a GPU box (Colab).

The gpu-marked tests fit on cuDF/cupy input and compare against the CPU path on ``to_pandas()``
at allclose -- fold parity is implied because catstat owns fold assignment (host RNG on both
paths). Locally only the dispatch/contract tests run (no RAPIDS -> gpu tests auto-skip).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from catstat import TargetEncoder
from catstat.backends._dispatch import is_device_frame, select_backend


def test_is_device_frame_false_without_cudf():
    # must not import cudf; a plain DataFrame / array is never a device frame
    assert not is_device_frame(pd.DataFrame({"g": ["a"]}))
    assert not is_device_frame(np.array([["a"]]))


def test_select_backend_device_input_contract():
    with pytest.raises(ValueError, match="to_pandas"):
        select_backend("cpu", 10, 1, True, device_input=True)
    try:
        import cudf  # noqa: F401

        have = True
    except Exception:
        have = False
    if not have:
        # gpu/auto with device input demand RAPIDS loudly (never a silent fallback)
        for backend in ("gpu", "auto"):
            with pytest.raises(ImportError, match="RAPIDS"):
                select_backend(backend, 10, 1, True, device_input=True)


# ---- everything below runs only on a GPU/RAPIDS box --------------------------------------------


def _device_data(n=100_000, k=2_000, seed=0, binary=False, missing=0.0):
    import cudf

    rng = np.random.default_rng(seed)
    g = rng.integers(0, k, size=n).astype(str).astype(object)
    if missing:
        g[rng.uniform(size=n) < missing] = None
    b = rng.integers(0, 50, size=n).astype(str)
    y = (rng.uniform(size=n) < 0.4).astype(int) if binary else rng.normal(size=n)
    pdf = pd.DataFrame({"g": g, "b": b})
    return cudf.from_pandas(pdf), pdf, y


def _parity(kw, Xg, Xp, y, y_dev=None):
    a = np.asarray(TargetEncoder(**kw, backend="cpu").fit_transform(Xp, y))
    gpu_enc = TargetEncoder(**kw, backend="gpu")
    b = np.asarray(gpu_enc.fit_transform(Xg, y if y_dev is None else y_dev))
    assert gpu_enc.backend_ == "gpu"
    assert np.allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=True)
    return gpu_enc


@pytest.mark.gpu
@pytest.mark.parametrize(
    "stats", [["mean"], ["mean", "var", "skew", "kurt"], ["mean", "count", "frequency"]]
)
def test_device_input_continuous_parity(stats):  # pragma: no cover - GPU only
    Xg, Xp, y = _device_data()
    kw = dict(cols=["g"], stats=stats, cv=5, random_state=0, output="numpy")
    _parity(kw, Xg, Xp, y)


@pytest.mark.gpu
@pytest.mark.parametrize("handle_missing", ["value", "return_nan"])
def test_device_input_missing_parity(handle_missing):  # pragma: no cover - GPU only
    Xg, Xp, y = _device_data(missing=0.1)
    kw = dict(
        cols=["g"], stats=["mean", "var"], handle_missing=handle_missing, cv=5,
        random_state=0, output="numpy",
    )
    _parity(kw, Xg, Xp, y)


@pytest.mark.gpu
def test_device_input_binary_woe_and_multiclass():  # pragma: no cover - GPU only
    Xg, Xp, y = _device_data(binary=True)
    kw = dict(cols=["g"], stats=["mean", "woe"], cv=5, random_state=0, output="numpy")
    _parity(kw, Xg, Xp, y)

    rng = np.random.default_rng(1)
    ymc = rng.integers(0, 3, size=len(Xp))
    kw = dict(cols=["g"], stats=["mean"], cv=5, random_state=0, output="numpy")
    _parity(kw, Xg, Xp, ymc)


@pytest.mark.gpu
@pytest.mark.parametrize("missing", [0.0, 0.1])
def test_device_input_combination_and_interactions(missing):  # pragma: no cover - GPU only
    Xg, Xp, y = _device_data(k=200, missing=missing)
    kw = dict(
        cols=["g", "b"], multi_feature_mode="combination", stats=["mean", "var"],
        handle_missing="value", cv=5, random_state=0, output="numpy",
    )
    _parity(kw, Xg, Xp, y)
    kw = dict(
        cols=["g", "b"], interactions=[["g", "b"]], stats=["mean"],
        handle_missing="value", cv=5, random_state=0, output="numpy",
    )
    _parity(kw, Xg, Xp, y)


@pytest.mark.gpu
def test_device_fit_then_pandas_transform():  # pragma: no cover - GPU only
    Xg, Xp, y = _device_data()
    kw = dict(cols=["g"], stats=["mean", "var"], cv=5, random_state=0, output="numpy")
    cpu = TargetEncoder(**kw, backend="cpu").fit(Xp, y)
    dev = TargetEncoder(**kw, backend="gpu").fit(Xg, y)
    a = np.asarray(cpu.transform(Xp))
    b = np.asarray(dev.transform(Xp))  # device-fitted, host transform: shared host machinery
    assert np.allclose(a, b, rtol=1e-5, atol=1e-8)
    # unseen + missing fallbacks flow through the same host tables
    probe = pd.DataFrame({"g": ["UNSEEN", None], "b": ["0", "1"]})
    pa, pb = np.asarray(cpu.transform(probe)), np.asarray(dev.transform(probe))
    assert np.allclose(pa, pb, rtol=1e-5, atol=1e-8, equal_nan=True)


@pytest.mark.gpu
def test_device_input_y_containers():  # pragma: no cover - GPU only
    import cudf
    import cupy as cp

    Xg, Xp, y = _device_data()
    kw = dict(cols=["g"], stats=["mean"], cv=5, random_state=0, output="numpy")
    ref = np.asarray(TargetEncoder(**kw, backend="cpu").fit_transform(Xp, y))
    for y_in in (y, cp.asarray(y), cudf.Series(y)):
        out = np.asarray(TargetEncoder(**kw, backend="gpu").fit_transform(Xg, y_in))
        assert np.allclose(ref, out, rtol=1e-5, atol=1e-8)


@pytest.mark.gpu
def test_device_input_fences():  # pragma: no cover - GPU only
    Xg, _Xp, y = _device_data(n=1_000, k=50)
    with pytest.raises(ValueError, match="to_pandas"):
        TargetEncoder(cols=["g"], backend="cpu").fit(Xg, y)
    with pytest.raises(NotImplementedError, match="output"):
        TargetEncoder(cols=["g"], output="pandas").fit(Xg, y)
    with pytest.raises(NotImplementedError, match="median"):
        TargetEncoder(cols=["g"], stats=["median"], output="numpy").fit(Xg, y)
    with pytest.raises(ValueError, match="CPU-only"):
        TargetEncoder(
            cols=["g"], stats=[("q9", lambda v: np.quantile(v, 0.9))], output="numpy"
        ).fit(Xg, y)
    with pytest.raises(NotImplementedError, match="scheme"):
        TargetEncoder(cols=["g"], scheme="loo", output="numpy").fit(Xg, y)
    with pytest.raises(NotImplementedError, match="numeric"):
        TargetEncoder(cols=["g"], numeric="auto", output="numpy").fit(Xg, y)
    with pytest.raises(NotImplementedError, match="cuDF"):
        TargetEncoder(cols=["g"], output="numpy").fit(Xg, y).transform(Xg)


@pytest.mark.gpu
def test_device_input_determinism():  # pragma: no cover - GPU only
    Xg, _Xp, y = _device_data()
    kw = dict(cols=["g"], stats=["mean"], cv=5, random_state=0, output="numpy", backend="gpu")
    a = np.asarray(TargetEncoder(**kw).fit_transform(Xg, y))
    b = np.asarray(TargetEncoder(**kw).fit_transform(Xg, y))
    assert np.allclose(a, b, rtol=1e-12, atol=1e-12)
