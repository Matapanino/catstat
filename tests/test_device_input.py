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


def test_output_cudf_without_rapids_raises():
    try:
        import cudf  # noqa: F401

        pytest.skip("RAPIDS present; the host-side ImportError contract is CPU-box-only")
    except ImportError:
        pass
    X = pd.DataFrame({"g": ["a", "b"] * 20})
    y = np.arange(40, dtype=float)
    with pytest.raises(ImportError, match="RAPIDS"):
        TargetEncoder(cols=["g"], output="cudf").fit(X, y)


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
    with pytest.raises(ValueError, match="CPU-only"):
        TargetEncoder(
            cols=["g"], stats=[("q9", lambda v: np.quantile(v, 0.9))], output="numpy"
        ).fit(Xg, y)
    with pytest.raises(NotImplementedError, match="scheme"):
        TargetEncoder(cols=["g"], scheme="loo", output="numpy").fit(Xg, y)
    with pytest.raises(NotImplementedError, match="numeric"):
        TargetEncoder(cols=["g"], numeric="auto", output="numpy").fit(Xg, y)


@pytest.mark.gpu
def test_device_output_matrix():  # pragma: no cover - GPU only
    """Output containers: device in -> cudf by default; numpy/pandas via one D2H; host in +
    output='cudf' via one explicit H2D. Values identical across containers."""
    import cudf

    Xg, Xp, y = _device_data(n=20_000, k=500)
    base = dict(cols=["g"], stats=["mean", "var"], cv=5, random_state=0, backend="gpu")
    ref = np.asarray(TargetEncoder(**base, output="numpy").fit_transform(Xg, y))

    out_auto = TargetEncoder(**base, output="auto").fit_transform(Xg, y)
    assert isinstance(out_auto, cudf.DataFrame)
    assert list(out_auto.columns) == ["g__te_mean", "g__te_var"]
    assert np.allclose(out_auto.to_pandas().to_numpy(), ref, equal_nan=True)

    out_cudf = TargetEncoder(**base, output="cudf").fit_transform(Xg, y)
    assert isinstance(out_cudf, cudf.DataFrame)
    assert np.allclose(out_cudf.to_pandas().to_numpy(), ref, equal_nan=True)

    out_pd = TargetEncoder(**base, output="pandas").fit_transform(Xg, y)
    assert isinstance(out_pd, pd.DataFrame)
    assert np.allclose(out_pd.to_numpy(), ref, equal_nan=True)

    # host input + output='cudf': explicit H2D on the way out
    host_cudf = TargetEncoder(
        cols=["g"], stats=["mean", "var"], cv=5, random_state=0, output="cudf"
    ).fit_transform(Xp, y)
    assert isinstance(host_cudf, cudf.DataFrame)
    assert np.allclose(host_cudf.to_pandas().to_numpy(), ref, rtol=1e-5, atol=1e-8,
                       equal_nan=True)


@pytest.mark.gpu
def test_device_set_output_pandas_wins():  # pragma: no cover - GPU only
    """sklearn set_output(transform='pandas') overrides output='auto' -> pandas even for cuDF
    input (sklearn semantics win; its wrapper cannot wrap a cuDF frame)."""
    Xg, _Xp, y = _device_data(n=20_000, k=500)
    enc = TargetEncoder(cols=["g"], stats=["mean"], cv=5, random_state=0, backend="gpu")
    enc.set_output(transform="pandas")
    out = enc.fit_transform(Xg, y)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["g__te_mean"]


@pytest.mark.gpu
def test_transform_cudf_input_parity():  # pragma: no cover - GPU only
    """transform on cuDF input (device gather) == CPU transform on to_pandas(), for both a
    host-fitted and a device-fitted encoder, incl. unknown/missing fallbacks and combination."""
    import cudf

    Xg, Xp, y = _device_data(k=300, missing=0.1)
    kw = dict(cols=["g", "b"], stats=["mean", "var"], handle_missing="value", cv=5,
              random_state=0, output="numpy")
    cpu = TargetEncoder(**kw, backend="cpu").fit(Xp, y)
    dev = TargetEncoder(**kw, backend="gpu").fit(Xg, y)
    ref = np.asarray(cpu.transform(Xp))
    for enc in (cpu, dev):
        got = np.asarray(enc.transform(Xg))
        assert np.allclose(ref, got, rtol=1e-5, atol=1e-8, equal_nan=True)

    probe_pd = pd.DataFrame({"g": ["UNSEEN", None, "0"], "b": ["0", "1", "UNSEEN"]})
    probe = cudf.from_pandas(probe_pd)
    ref_p = np.asarray(cpu.transform(probe_pd))
    got_p = np.asarray(dev.transform(probe))
    assert np.allclose(ref_p, got_p, rtol=1e-5, atol=1e-8, equal_nan=True)

    comb = dict(cols=["g", "b"], multi_feature_mode="combination", stats=["mean"],
                handle_missing="value", cv=5, random_state=0, output="numpy")
    cpu_c = TargetEncoder(**comb, backend="cpu").fit(Xp, y)
    dev_c = TargetEncoder(**comb, backend="gpu").fit(Xg, y)
    ref_c = np.asarray(cpu_c.transform(probe_pd))
    got_c = np.asarray(dev_c.transform(probe))
    assert np.allclose(ref_c, got_c, rtol=1e-5, atol=1e-8, equal_nan=True)


@pytest.mark.gpu
@pytest.mark.parametrize("stats", [["median"], ["min", "max"], ["mean", "median"]])
def test_device_input_order_stats_parity(stats):  # pragma: no cover - GPU only
    """median/min/max on device: per-fold device group-by OOF (no per-fold H2D of row data)
    must match the CPU slow path at allclose, incl. the hybrid mean+median case and small
    categories exercising the complement-global fallback."""
    Xg, Xp, y = _device_data(n=50_000, k=2_000, seed=6)  # ~25 rows/cat: some tiny complements
    kw = dict(cols=["g"], stats=stats, cv=5, random_state=0, output="numpy")
    _parity(kw, Xg, Xp, y)


@pytest.mark.gpu
def test_device_input_order_stats_missing_and_unknown():  # pragma: no cover - GPU only
    Xg, Xp, y = _device_data(n=50_000, k=2_000, seed=7, missing=0.1)
    for hm in ("value", "return_nan"):
        kw = dict(
            cols=["g"], stats=["median"], handle_missing=hm, cv=5, random_state=0,
            output="numpy",
        )
        _parity(kw, Xg, Xp, y)


@pytest.mark.gpu
def test_transform_lut_cache_is_correct_and_invalidated():  # pragma: no cover - GPU only
    """Repeated transform(cuDF) reuses the device LUT cache with identical results; refit
    invalidates it; pickling drops it (rebuilt on the next call)."""
    import pickle

    Xg, Xp, y = _device_data(n=30_000, k=1_000, missing=0.1)
    kw = dict(cols=["g", "b"], stats=["mean"], multi_feature_mode="combination",
              handle_missing="value", cv=5, random_state=0, output="numpy")
    enc = TargetEncoder(**kw, backend="gpu").fit(Xg, y)
    a = np.asarray(enc.transform(Xg))  # builds the cache
    assert hasattr(enc, "_device_transform_luts") and enc._device_transform_luts
    b = np.asarray(enc.transform(Xg))  # served from the cache
    assert np.array_equal(a, b)
    ref = np.asarray(TargetEncoder(**kw, backend="cpu").fit(Xp, y).transform(Xp))
    assert np.allclose(ref, b, rtol=1e-5, atol=1e-8, equal_nan=True)

    # refit on different data must not serve stale LUTs
    Xg2, Xp2, y2 = _device_data(n=30_000, k=1_000, seed=9, missing=0.1)
    enc.fit(Xg2, y2)
    got = np.asarray(enc.transform(Xg2))
    ref2 = np.asarray(TargetEncoder(**kw, backend="cpu").fit(Xp2, y2).transform(Xp2))
    assert np.allclose(ref2, got, rtol=1e-5, atol=1e-8, equal_nan=True)

    # pickle round-trip drops the cache and still transforms (pandas path shares the tables)
    enc2 = pickle.loads(pickle.dumps(enc))
    assert not hasattr(enc2, "_device_transform_luts")
    assert np.allclose(np.asarray(enc2.transform(Xp2)), ref2, rtol=1e-5, atol=1e-8,
                       equal_nan=True)


@pytest.mark.gpu
def test_device_input_determinism():  # pragma: no cover - GPU only
    Xg, _Xp, y = _device_data()
    kw = dict(cols=["g"], stats=["mean"], cv=5, random_state=0, output="numpy", backend="gpu")
    a = np.asarray(TargetEncoder(**kw).fit_transform(Xg, y))
    b = np.asarray(TargetEncoder(**kw).fit_transform(Xg, y))
    assert np.allclose(a, b, rtol=1e-12, atol=1e-12)
