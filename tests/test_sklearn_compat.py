import pandas as pd
import pytest
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.fixes import parse_version
from tests.conftest import make_regression

from catstat import CountEncoder, TargetEncoder

SKLEARN_GE_16 = parse_version(sklearn.__version__) >= parse_version("1.6")


def test_clone_and_get_set_params():
    enc = TargetEncoder(cols=["g"], smooth=3.0, stats=["mean", "count"])
    cloned = clone(enc)
    assert cloned.get_params()["smooth"] == 3.0
    assert cloned.get_params()["stats"] == ["mean", "count"]
    cloned.set_params(smooth=9.0)
    assert cloned.get_params()["smooth"] == 9.0


def test_fitted_attributes_have_trailing_underscore():
    X, y = make_regression()
    enc = TargetEncoder(cols=["g"]).fit(X, y)
    for attr in ("classes_", "n_features_in_", "target_type_", "feature_names_out_", "backend_"):
        assert hasattr(enc, attr)


def test_pipeline_with_regressor():
    X, y = make_regression()
    pipe = Pipeline(
        [("enc", TargetEncoder(cols=["g"], output="numpy")), ("lr", LinearRegression())]
    )
    pipe.fit(X[["g"]], y)
    preds = pipe.predict(X[["g"]])
    assert preds.shape == (len(X),)


def test_column_transformer_passthrough():
    X, y = make_regression()
    ct = ColumnTransformer(
        [("te", TargetEncoder(cols="auto"), ["g"])],
        remainder="passthrough",
    )
    Xt = ct.fit_transform(X, y)
    assert Xt.shape == (len(X), 2)  # encoded g + passthrough x_num


def test_set_output_pandas_dataframe_input():
    X, y = make_regression()
    enc = TargetEncoder(cols=["g"]).set_output(transform="pandas")
    out = enc.fit_transform(X, y)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["g__te_mean"]


def test_set_output_pandas_numpy_input():
    X, y = make_regression()
    Xnp = X[["g"]].to_numpy()
    enc = TargetEncoder().set_output(transform="pandas")
    out = enc.fit_transform(Xnp, y)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["x0__te_mean"]


def test_unsupervised_encoder_works_without_y():
    X, _ = make_regression()
    ce = CountEncoder(cols=["g"])
    out = ce.fit_transform(X)
    assert out.shape == (len(X), 1)


def test_target_encoder_requires_y():
    X, _ = make_regression()
    with pytest.raises(ValueError, match="requires y"):
        TargetEncoder(cols=["g"]).fit(X)


def test_more_tags_reflect_supervision():
    # _more_tags is the scikit-learn < 1.6 tag API (kept for those versions; ignored by newer).
    assert TargetEncoder(cols=["g"])._more_tags()["requires_y"] is True
    assert CountEncoder(cols=["g"])._more_tags()["requires_y"] is False
    assert TargetEncoder(cols=["g"])._more_tags()["allow_nan"] is True


@pytest.mark.skipif(not SKLEARN_GE_16, reason="__sklearn_tags__ requires scikit-learn>=1.6")
def test_sklearn_tags_categorical_and_requires_y():
    t = TargetEncoder(cols=["g"]).__sklearn_tags__()
    assert t.target_tags.required is True
    assert t.input_tags.categorical and t.input_tags.string and t.input_tags.allow_nan
    c = CountEncoder(cols=["g"]).__sklearn_tags__()
    assert c.target_tags.required is False
    assert c.input_tags.allow_nan is True
