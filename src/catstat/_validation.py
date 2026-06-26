"""Input validation, target-type inference, and missing-key normalization.

Backend-agnostic: everything here is pandas/numpy and has no statistics or leakage logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target


class _MissingType:
    """Singleton sentinel used as the dict key for missing (NaN) categories.

    Using a unique, hashable object (rather than ``np.nan``) lets a missing level be a
    first-class category key without tripping over ``nan != nan`` during lookups.
    """

    _instance: _MissingType | None = None

    def __new__(cls) -> _MissingType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "MISSING"

    def __reduce__(self):  # keep the singleton identity across pickling
        return (_MissingType, ())


MISSING = _MissingType()

_VALID_HANDLE = ("value", "return_nan", "error")


def prepare_X(X) -> tuple[pd.DataFrame, bool, list]:
    """Return ``(dataframe, was_dataframe, column_labels)``.

    numpy / list input is wrapped in a DataFrame with ``x0, x1, ...`` column names so the
    rest of the pipeline can work uniformly on a DataFrame.
    """
    if isinstance(X, pd.DataFrame):
        return X, True, list(X.columns)
    arr = np.asarray(X, dtype=object) if not isinstance(X, np.ndarray) else X
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D input, got array with {arr.ndim} dimensions.")
    cols = [f"x{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols), False, cols


def _is_categorical_like(dtype) -> bool:
    """Whether an auto-selected column dtype counts as categorical for encoding.

    Recognizes object, pandas ``Categorical``, and pandas ``StringDtype``. pandas >= 3.0 types
    string columns as ``StringDtype`` (repr ``str``) rather than ``object``, so handling it keeps
    ``cols="auto"`` working across pandas versions (KI-022).
    """
    if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
        return True
    string_dtype = getattr(pd, "StringDtype", None)
    return string_dtype is not None and isinstance(dtype, string_dtype)


def _is_numeric_like(dtype) -> bool:
    """Whether a column dtype is a (non-boolean) numeric dtype eligible for numeric encoding.

    bool is excluded: it is already effectively categorical (two levels) and is not meaningfully
    binned. datetimes are not numeric here either (``is_numeric_dtype`` is False for them).
    """
    is_num = pd.api.types.is_numeric_dtype(dtype)
    return bool(is_num) and not bool(pd.api.types.is_bool_dtype(dtype))


def select_cols(Xdf: pd.DataFrame, cols, numeric_mode: str = "ignore") -> list:
    """Resolve the ``cols`` parameter to a concrete list of column labels.

    ``"auto"``/``None`` selects object, pandas ``category``, and pandas string-dtype columns -- and,
    when ``numeric_mode`` is not ``"ignore"``, (non-boolean) numeric columns as well, preserving the
    input column order. A list may hold column labels or positional integers (handy for numpy
    input); explicit lists are taken verbatim regardless of ``numeric_mode``.
    """
    if cols == "auto" or cols is None:

        def _keep(dtype) -> bool:
            if _is_categorical_like(dtype):
                return True
            return numeric_mode != "ignore" and _is_numeric_like(dtype)

        selected = [c for c in Xdf.columns if _keep(Xdf[c].dtype)]
        if not selected:
            raise ValueError(
                "cols='auto' found no columns to encode. Pass cols=[...] explicitly "
                "(object/category/string columns are auto-selected; numeric columns are "
                "auto-selected only when numeric is enabled)."
            )
        return selected

    resolved = []
    for c in cols:
        if c in Xdf.columns:
            resolved.append(c)
        elif isinstance(c, (int, np.integer)) and 0 <= int(c) < Xdf.shape[1]:
            resolved.append(Xdf.columns[int(c)])
        else:
            raise ValueError(f"Column {c!r} not found in input.")
    return resolved


def infer_target_type(y, target_type: str) -> str:
    """Resolve ``target_type`` ('auto' uses sklearn's ``type_of_target``)."""
    if target_type != "auto":
        if target_type not in ("continuous", "binary", "multiclass"):
            raise ValueError(f"Unknown target_type={target_type!r}.")
        return target_type
    tt = type_of_target(y)
    if tt in ("continuous", "binary", "multiclass"):
        return tt
    raise ValueError(
        f"Could not handle target of type {tt!r}. Supported: continuous, binary, multiclass. "
        "Pass target_type=... explicitly if inference is ambiguous (e.g. integer regression)."
    )


def normalize_keys(values) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(keys, missing_mask)``.

    ``keys`` is an object array where missing entries are replaced by the :data:`MISSING`
    sentinel; ``missing_mask`` flags those positions.
    """
    arr = np.asarray(values, dtype=object)
    missing_mask = pd.isna(arr.astype(object))
    # pd.isna on object arrays returns an object/array; coerce to bool
    missing_mask = np.asarray(missing_mask, dtype=bool)
    keys = arr.copy()
    if missing_mask.any():
        keys[missing_mask] = MISSING
    return keys, missing_mask


def check_handle(name: str, value: str) -> None:
    if value not in _VALID_HANDLE:
        raise ValueError(f"{name}={value!r} must be one of {_VALID_HANDLE}.")
