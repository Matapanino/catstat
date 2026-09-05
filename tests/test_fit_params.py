"""Unsupported training metadata must fail before fitting or device dispatch."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from catstat import CountEncoder, FrequencyEncoder, TargetEncoder


@pytest.mark.parametrize("encoder", [TargetEncoder, CountEncoder, FrequencyEncoder])
@pytest.mark.parametrize(
    "params",
    [{"sample_weight": np.ones(6)}, {"groups": [0] * 6}, {"sample_weight": None, "typo": 1}],
)
@pytest.mark.parametrize("device", [False, True])
def test_fit_params_raise_before_any_fitting_or_dispatch(encoder, params, device, monkeypatch):
    enc = encoder(cols=["g"], cv=2) if encoder is TargetEncoder else encoder(cols=["g"])
    X = pd.DataFrame({"g": ["a", "b"] * 3})
    from catstat import _base

    monkeypatch.setattr(_base, "is_device_frame", lambda X: device)

    def forbidden(*args, **kwargs):
        pytest.fail("fit metadata must be rejected before fitting or device work")

    monkeypatch.setattr(enc, "fit", forbidden)
    monkeypatch.setattr(enc, "_fit_transform_device", forbidden)
    with pytest.raises(TypeError, match="Unsupported fit parameters") as exc:
        enc.fit_transform(X, np.arange(6, dtype=float), **params)
    for name in params:
        assert name in str(exc.value)
    assert not hasattr(enc, "n_features_in_")


def test_pipeline_cannot_silently_drop_encoder_weights():
    X = pd.DataFrame({"g": ["a", "b"] * 3})
    pipeline = Pipeline([("enc", TargetEncoder(cv=2)), ("model", LinearRegression())])
    with pytest.raises(TypeError, match="sample_weight"):
        pipeline.fit(X, np.arange(6, dtype=float), enc__sample_weight=np.ones(6))
