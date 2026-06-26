"""Backend selection: auto resolves to CPU here; explicit gpu errors loudly (no silent fallback)."""

import pytest
from tests.conftest import make_regression

from catstat import TargetEncoder
from catstat.backends import _gpu


def test_auto_resolves_to_cpu_without_gpu():
    X, y = make_regression()
    enc = TargetEncoder(cols=["g"], backend="auto").fit(X, y)
    assert enc.backend_ == "cpu"


def test_explicit_gpu_raises_without_rapids():
    if _gpu.AVAILABLE:  # pragma: no cover - only on a RAPIDS box
        pytest.skip("RAPIDS present; explicit gpu would not raise")
    X, y = make_regression()
    with pytest.raises(ImportError, match="RAPIDS"):
        TargetEncoder(cols=["g"], backend="gpu").fit(X, y)


def test_invalid_backend_raises():
    X, y = make_regression()
    with pytest.raises(ValueError, match="backend"):
        TargetEncoder(cols=["g"], backend="bogus").fit(X, y)
