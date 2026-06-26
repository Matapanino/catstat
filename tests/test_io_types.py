import numpy as np
import pandas as pd
from tests.conftest import make_regression

from catstat import TargetEncoder


def test_dataframe_in_dataframe_out_auto():
    X, y = make_regression()
    out = TargetEncoder(cols=["g"]).fit_transform(X, y)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["g__te_mean"]


def test_numpy_object_array_in_ndarray_out_auto():
    X, y = make_regression()
    Xnp = X[["g"]].to_numpy()  # object array of strings
    enc = TargetEncoder()  # cols='auto' selects the object column x0
    out = enc.fit_transform(Xnp, y)
    assert isinstance(out, np.ndarray)
    assert out.shape == (len(X), 1)
    assert list(enc.get_feature_names_out()) == ["x0__te_mean"]


def test_category_dtype_column():
    rng = np.random.default_rng(0)
    g = pd.Categorical(rng.choice(list("abc"), size=300))
    X = pd.DataFrame({"g": g})
    y = rng.normal(size=300)
    out = TargetEncoder(cols="auto").fit_transform(X, y)
    assert out.shape == (300, 1)


def test_int_categorical_requires_explicit_cols():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"g": rng.integers(0, 5, size=300)})  # int dtype
    y = rng.normal(size=300)
    enc = TargetEncoder(cols=["g"])  # explicit: ints are not auto-selected
    out = enc.fit_transform(X, y)
    assert out.shape == (300, 1)
    # unseen integer category -> global mean
    tr = enc.transform(pd.DataFrame({"g": [999]}))
    assert tr.iloc[0, 0] == enc.target_mean_


def test_output_numpy_and_pandas():
    X, y = make_regression()
    arr = TargetEncoder(cols=["g"], output="numpy").fit_transform(X, y)
    assert isinstance(arr, np.ndarray)
    df = TargetEncoder(cols=["g"], output="pandas").fit_transform(X, y)
    assert isinstance(df, pd.DataFrame)
