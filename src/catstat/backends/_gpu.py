"""GPU backend (cuDF / CuPy).

Mirrors ``_cpu``'s primitives. The heavy per-category group-by runs on the GPU (the part that
scales); the small per-category result is returned as a **pandas** object so the downstream
smoothing / fallback / mapping logic is byte-for-byte identical to the CPU path (CPU and GPU
therefore agree to ``allclose`` -- float reduction order differs). This is the cuML pattern:
device group-by, host orchestration.

Missing values: ``_validation.normalize_keys`` replaces NaN with the host ``MISSING`` sentinel
(a Python object cuDF can't hold). For single-column (object) keys we map that sentinel to a cuDF
**null**, group with ``dropna=False`` so the missing level is its own category, then map the null
result-index entry back to ``MISSING`` so the host ``.map`` lines up. **Combination/interaction
units** arrive as **int64 joint codes** (host-built; a missing component is already folded into an
ordinary integer code, so there is no sentinel) and group directly on the device.

Imports cleanly on CPU-only boxes (``AVAILABLE`` is then ``False`` and nothing here runs).
Validated on Colab (``scripts/colab_gpu_parity.sh``); there is no local GPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "gpu"


def _detect() -> bool:
    try:
        import cudf  # noqa: F401
        import cupy  # noqa: F401

        return True
    except Exception:
        return False


AVAILABLE = _detect()


def ensure_available() -> None:
    """Raise a clear error if RAPIDS/GPU is not importable (never a silent fallback)."""
    if not AVAILABLE:
        raise ImportError(
            "backend='gpu' requires RAPIDS (cudf + cupy), which is not importable here. "
            "Install the 'gpu' extra on an NVIDIA/CUDA-12 box, or use backend='cpu'/'auto'."
        )


def _to_nullable(keys: np.ndarray):
    """Return ``(keys_or_copy, had_missing)`` with the host MISSING sentinel -> None (cuDF null).

    Only object key arrays (single-column / numeric string keys) can carry the sentinel; int64
    joint-code keys (combination/interaction units) never do, so they pass straight through.
    """
    keys = np.asarray(keys)
    if keys.dtype != object:
        return keys, False
    from .._validation import MISSING

    mask = np.asarray(keys == MISSING, dtype=bool)
    if not mask.any():
        return keys, False
    arr = np.asarray(keys, dtype=object).copy()
    arr[mask] = None
    return arr, True


def _remap_missing_index(result, had_missing):
    """Map the cuDF null group (NaN after to_pandas) in the result index back to MISSING."""
    if not had_missing:
        return result
    from .._validation import MISSING

    result.index = pd.Index(
        [MISSING if pd.isna(i) else i for i in result.index], dtype=object
    )
    return result


def category_reduce(keys: np.ndarray, y: np.ndarray | None = None) -> pd.DataFrame:
    """GPU group-by; returns a pandas DataFrame (small per-category result copied to host)."""
    import cudf
    import cupy as cp

    key_arr, had_missing = _to_nullable(keys)

    if y is None:
        vc = cudf.Series(key_arr).value_counts(dropna=False)
        res = cudf.DataFrame({"count": vc.astype("float64")}).to_pandas()
        return _remap_missing_index(res, had_missing)

    gdf = cudf.DataFrame({"k": cudf.Series(key_arr), "y": cp.asarray(y, dtype="float64")})
    gdf["y2"] = gdf["y"] * gdf["y"]
    g = gdf.groupby("k", sort=False, dropna=False)
    out = cudf.DataFrame(
        {
            "count": g["y"].count().astype("float64"),
            "sum": g["y"].sum(),
            "mean": g["y"].mean(),
            "sumsq": g["y2"].sum(),
        }
    ).to_pandas()
    return _remap_missing_index(out, had_missing)


# ---- device-input primitives (cuDF X / cupy y) -------------------------------------------------
# Everything below serves catstat._device (the device-resident fit/OOF orchestration). Only this
# module touches cudf/cupy; _device works with the returned arrays through operators alone.


def to_host_1d(y) -> np.ndarray:
    """Any 1-D target container -> host numpy (one explicit D2H for device containers)."""
    import cudf
    import cupy as cp

    if isinstance(y, cudf.Series):
        return y.to_pandas().to_numpy()
    if isinstance(y, cp.ndarray):
        return cp.asnumpy(y)
    return np.asarray(y)


def to_device(arr):
    """Host (or device) array -> cupy array (no copy if already resident)."""
    import cupy as cp

    return cp.asarray(arr)


def to_device_float(y):
    """Continuous target -> float64 cupy array (accepts cudf.Series / cupy / host numpy)."""
    import cudf
    import cupy as cp

    if isinstance(y, cudf.Series):
        return y.astype("float64").to_cupy()
    return cp.asarray(y, dtype="float64")


def binarize_device(y, pos):
    """One-vs-rest indicator ``(y == pos)`` as float64 cupy (labels may be host strings)."""
    import cudf
    import cupy as cp

    if isinstance(y, cudf.Series):
        return (y == pos).astype("float64").to_cupy()
    if isinstance(y, cp.ndarray):
        return (y == pos).astype(cp.float64)
    return cp.asarray((np.asarray(y) == pos).astype(float))


def factorize_column(ser, handle_missing):
    """cuDF factorize of one column: ``(codes int64 cupy, uniques host pd.Index, missing mask)``.

    Nulls factorize to ``-1``. Under ``handle_missing='value'`` they are remapped to their own
    trailing level whose host key is the ``MISSING`` sentinel -- exactly what
    ``_validation.normalize_keys`` produces on the host path; under ``'return_nan'`` the ``-1``
    stays (those rows are inactive); ``'error'`` is the caller's job (it has the unit name).
    ``uniques`` is a value-stable host index, so a device-fitted encoder transforms pandas input
    with the ordinary host machinery.
    """
    import cupy as cp

    from .._validation import MISSING

    codes, uniq = ser.factorize()
    codes = cp.asarray(codes).astype(cp.int64)
    missing = codes < 0
    uniques = pd.Index(np.asarray(uniq.to_pandas(), dtype=object))
    if handle_missing == "value" and bool(missing.any()):
        codes = cp.where(missing, cp.int64(len(uniques)), codes)
        uniques = uniques.append(pd.Index([MISSING], dtype=object))
    return codes, uniques, missing


def joint_codes_device(radices, comp_codes):
    """Mixed-radix int64 joint code on device (twin of ``_cross_fit.joint_codes``).

    Any negative component code (a null under ``handle_missing='return_nan'``) forces the row's
    joint code to ``-1`` -- the inactive sentinel, exactly like the host builder.
    """
    import cupy as cp

    joint = cp.zeros(comp_codes[0].shape[0], dtype=cp.int64)
    neg = cp.zeros(comp_codes[0].shape[0], dtype=cp.bool_)
    for radix, codes in zip(radices, comp_codes):
        neg = neg | (codes < 0)
        joint = joint * cp.int64(radix) + codes
    if bool(neg.any()):
        joint[neg] = -1
    return joint


def dense_codes(joint, has_negative):
    """Re-encode sparse mixed-radix joint codes densely: ``(codes cupy, uniques Int64 Index)``.

    The observed joint codes become the unit's canonical (host) index -- the device counterpart
    of the host path's group-by-observed index (order differs; encodings are per-category so the
    output is unaffected). Rows with ``joint == -1`` are nulled first so they stay ``-1``.
    """
    import cudf
    import cupy as cp

    s = cudf.Series(joint)
    if has_negative:
        s = s.where(cudf.Series(joint >= 0), None)
    codes, uniq = s.factorize()
    return cp.asarray(codes).astype(cp.int64), pd.Index(uniq.to_pandas())


def code_moments(codes, y, n_cat, order: int = 2):
    """Per-code count + raw power sums on device -> small host float64 arrays.

    ``codes`` must be dense in ``[0, n_cat)`` (the caller pre-filters negatives). ``y`` of
    ``None`` returns ``(count,)`` only (count/frequency). The caller pre-shifts ``y`` when it
    wants stable shape/dispersion reconstruction.
    """
    import cupy as cp

    codes = cp.asarray(codes)
    cnt = cp.bincount(codes, minlength=n_cat).astype(cp.float64)
    if y is None:
        return (cp.asnumpy(cnt),)
    yv = cp.asarray(y, dtype="float64")
    s1 = cp.bincount(codes, weights=yv, minlength=n_cat)
    y2 = yv * yv
    s2 = cp.bincount(codes, weights=y2, minlength=n_cat)
    if order < 4:
        return cp.asnumpy(cnt), cp.asnumpy(s1), cp.asnumpy(s2)
    s3 = cp.bincount(codes, weights=y2 * yv, minlength=n_cat)
    s4 = cp.bincount(codes, weights=y2 * y2, minlength=n_cat)
    return cp.asnumpy(cnt), cp.asnumpy(s1), cp.asnumpy(s2), cp.asnumpy(s3), cp.asnumpy(s4)


def category_agg_codes(codes, y, stat: str, n_cat: int) -> np.ndarray:
    """Per-code median/min/max on device -> host float64 array of length ``n_cat``.

    Codes absent from ``codes`` stay NaN (the caller's fallback trigger). Used by the device
    order-stat fit and the per-fold device OOF loop (only the small per-code table leaves the
    GPU; the per-fold row data never does).
    """
    import cudf

    gdf = cudf.DataFrame({"k": codes, "y": y})
    g = gdf.groupby("k", sort=False)["y"]
    if stat == "median":
        res = g.median()
    elif stat == "min":
        res = g.min()
    elif stat == "max":
        res = g.max()
    else:
        raise ValueError(f"Unknown order stat {stat!r}.")
    out = np.full(n_cat, np.nan, dtype=float)
    rp = res.to_pandas()
    out[rp.index.to_numpy()] = rp.to_numpy(dtype=float)
    return out


def global_agg(y, stat: str) -> float:
    """Device-wide median/min/max of ``y`` -> host scalar (the order stats' global fallback)."""
    import cupy as cp

    y = cp.asarray(y, dtype="float64")
    if y.size == 0:
        return float("nan")
    if stat == "median":
        return float(cp.median(y))
    if stat == "min":
        return float(y.min())
    if stat == "max":
        return float(y.max())
    raise ValueError(f"Unknown order stat {stat!r}.")


def gather_cells(values, codes):
    """Device gather ``values[codes]`` (values H2D'd if host); ``codes < 0`` -> NaN."""
    import cupy as cp

    v = cp.asarray(values, dtype=cp.float64)
    codes = cp.asarray(codes)
    neg = codes < 0
    out = v[cp.where(neg, 0, codes)]
    if bool(neg.any()):
        out[neg] = cp.nan
    return out


def scatter_active(vals, active, n: int):
    """Full-length device column: ``vals`` at the active positions, NaN elsewhere."""
    import cupy as cp

    if active is None:
        return cp.asarray(vals, dtype=cp.float64)
    out = cp.full(n, cp.nan, dtype=cp.float64)
    out[cp.asarray(active)] = vals
    return out


def stack_to_host(cols) -> np.ndarray:
    """Stack device columns into an (n, k) matrix and copy to host once (output='numpy')."""
    import cupy as cp

    return cp.asnumpy(cp.stack([cp.asarray(c, dtype=cp.float64) for c in cols], axis=1))


def wrap_cudf(data, columns, index=None):
    """Assemble a cuDF DataFrame from device columns (or a host matrix -- one explicit H2D).

    ``index`` may be a cuDF index (device input: mirrored as-is) or a pandas index (host input
    with ``output='cudf'``).
    """
    ensure_available()
    import cudf
    import cupy as cp

    if isinstance(data, (list, tuple)):
        df = cudf.DataFrame(
            {str(name): cp.asarray(c, dtype=cp.float64) for name, c in zip(columns, data)}
        )
    else:
        mat = cp.asarray(data, dtype=cp.float64)
        df = cudf.DataFrame({str(name): mat[:, i] for i, name in enumerate(columns)})
    if index is not None:
        # duck-type: cudf renamed its index base class across versions (BaseIndex/Index)
        if type(index).__module__.startswith("cudf"):
            df.index = index
        else:
            df.index = cudf.from_pandas(pd.Index(index))
    return df


def codes_from_uniques(uniques_host, ser):
    """Device codes of a cuDF column against a fit-time host category index.

    Returns ``(codes int64 cupy, missing bool cupy)``: nulls map to the index position of the
    ``MISSING`` sentinel when the fit-time index learned one, else ``-1``; unknown values map to
    ``-1``. Implemented as a device hash join against the (small, H2D'd) unique table -- robust
    across cudf versions, unlike ``Index.get_indexer``.
    """
    import cudf
    import cupy as cp

    from .._validation import MISSING

    missing_code = -1
    vals, codes_of_vals = [], []
    for i, v in enumerate(uniques_host):
        if v is MISSING:
            missing_code = i
        else:
            vals.append(v)
            codes_of_vals.append(i)
    lut = cudf.DataFrame(
        {
            "k": cudf.Series(pd.Series(vals)),  # let pandas re-infer the concrete dtype
            "code": cp.asarray(np.asarray(codes_of_vals, dtype=np.int64)),
        }
    )
    n = len(ser)
    left = cudf.DataFrame({"k": ser.reset_index(drop=True), "ord": cp.arange(n)})
    merged = left.merge(lut, on="k", how="left").sort_values("ord")
    codes = merged["code"].fillna(-1).astype("int64").to_cupy()
    missing = ser.isnull().to_cupy() if ser.has_nulls else cp.zeros(n, dtype=cp.bool_)
    if missing_code >= 0 and bool(missing.any()):
        codes = cp.where(cp.asarray(missing), cp.int64(missing_code), codes)
    return codes, cp.asarray(missing)


def codes_from_int_index(index_host, joint_codes):
    """Canonical positions of device int64 joint codes against the fit-time Int64 index.

    Same device hash join as :func:`codes_from_uniques`, but over int64 keys (no MISSING
    sentinel: a missing component is already folded into the joint code or is ``-1``). Unknown
    joint codes -> ``-1``.
    """
    import cudf
    import cupy as cp

    lut = cudf.DataFrame(
        {
            "k": cp.asarray(np.asarray(index_host, dtype=np.int64)),
            "code": cp.arange(len(index_host), dtype=cp.int64),
        }
    )
    left = cudf.DataFrame({"k": joint_codes, "ord": cp.arange(joint_codes.shape[0])})
    merged = left.merge(lut, on="k", how="left").sort_values("ord")
    return merged["code"].fillna(-1).astype("int64").to_cupy()


def oof_moment_tables(comp, y, size, order):
    """GPU twin of ``_cpu.oof_moment_tables``: per-(fold, key) sums via ``cupy.bincount``.

    ``comp``/``y`` may be host numpy (one H2D copy each -- the whole per-unit transfer on the
    host-origin path) or already-resident cupy arrays (``cp.asarray`` is then a no-op). The
    returned tables are small (``n_folds * n_cat``) host float64 arrays, so the finalizers above
    the seam stay backend-blind. fp64 atomics in ``bincount`` reorder additions vs numpy -->
    parity with CPU holds at allclose, not bitwise (CLAUDE.md invariant #2).
    """
    import cupy as cp

    comp_d = cp.asarray(comp)
    y_d = cp.asarray(y, dtype="float64")
    fc = cp.bincount(comp_d, minlength=size).astype(cp.float64)
    fs = cp.bincount(comp_d, weights=y_d, minlength=size)
    y2 = y_d * y_d
    fss = cp.bincount(comp_d, weights=y2, minlength=size)
    if order < 4:
        return cp.asnumpy(fc), cp.asnumpy(fs), cp.asnumpy(fss), None, None
    fs3 = cp.bincount(comp_d, weights=y2 * y_d, minlength=size)
    fs4 = cp.bincount(comp_d, weights=y2 * y2, minlength=size)
    return cp.asnumpy(fc), cp.asnumpy(fs), cp.asnumpy(fss), cp.asnumpy(fs3), cp.asnumpy(fs4)


def category_moments(keys: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """GPU per-category count + raw power sums ``S1..S4`` (caller pre-shifts ``y``).

    Twin of ``_cpu.category_moments``: only plain group-by sums, which cuDF supports, so the
    shape stats (skew/kurt) reconstructed from these sums are GPU-supported. Small per-category
    result returned as pandas (host).
    """
    import cudf
    import cupy as cp

    key_arr, had_missing = _to_nullable(keys)
    yd = cp.asarray(y, dtype="float64")
    y2 = yd * yd
    gdf = cudf.DataFrame(
        {"k": cudf.Series(key_arr), "y": yd, "y2": y2, "y3": y2 * yd, "y4": y2 * y2}
    )
    g = gdf.groupby("k", sort=False, dropna=False)
    out = cudf.DataFrame(
        {
            "count": g["y"].count().astype("float64"),
            "s1": g["y"].sum(),
            "s2": g["y2"].sum(),
            "s3": g["y3"].sum(),
            "s4": g["y4"].sum(),
        }
    ).to_pandas()
    return _remap_missing_index(out, had_missing)


def category_agg(keys: np.ndarray, y: np.ndarray, stat: str) -> pd.Series:
    """GPU dispersion/order group-by; returns a pandas Series (host)."""
    import cudf
    import cupy as cp

    key_arr, had_missing = _to_nullable(keys)
    gdf = cudf.DataFrame({"k": cudf.Series(key_arr), "y": cp.asarray(y, dtype="float64")})
    g = gdf.groupby("k", sort=False, dropna=False)["y"]
    if stat == "var":
        res = g.var(ddof=1)
    elif stat == "std":
        res = g.std(ddof=1)
    elif stat == "median":
        res = g.median()
    elif stat == "min":
        res = g.min()
    elif stat == "max":
        res = g.max()
    else:
        raise ValueError(f"Unknown non-mean stat {stat!r}.")
    out = res.to_pandas()
    return _remap_missing_index(out.to_frame("v"), had_missing)["v"]
