"""The crown-jewel tests: prove fit_transform is out-of-fold and leaks no target."""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tests.conftest import make_leakage_trap, make_regression

from catstat import CountEncoder, TargetEncoder


def test_oof_reconstruction_is_exact():
    X, y = make_regression(n=800, k=20, seed=1)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = TargetEncoder(cols=["g"], smooth=0.0, cv=kf, output="numpy").fit_transform(X, y).ravel()

    # Independently reconstruct: each row = mean(y over its category in the OTHER folds).
    g = X["g"].to_numpy()
    gmean = y.mean()
    recon = np.empty(len(y))
    for tr, te in kf.split(X, y):
        means = pd.DataFrame({"g": g[tr], "y": y[tr]}).groupby("g")["y"].mean()
        for i in te:
            recon[i] = means.get(g[i], gmean)
    assert np.nanmax(np.abs(oof - recon)) == 0.0


def test_noise_category_does_not_leak():
    X, y = make_leakage_trap(n=2000, n_levels=1000, seed=3)
    oof = TargetEncoder(
        cols=["g"], smooth=0.0, cv=5, random_state=0, output="numpy"
    ).fit_transform(X, y).ravel()
    leaky = (
        TargetEncoder(cols=["g"], smooth=0.0, random_state=0, output="numpy")
        .fit(X, y)
        .transform(X)
        .ravel()
    )
    oof_corr = abs(np.corrcoef(oof, y)[0, 1])
    leaky_corr = abs(np.corrcoef(leaky, y)[0, 1])
    assert oof_corr < 0.1  # OOF encoding of a noise category carries no target signal
    assert leaky_corr > 0.4  # the leaky (fit-then-transform-on-train) path over-fits hard
    assert leaky_corr > 3 * oof_corr


def test_fit_transform_differs_from_fit_then_transform_with_signal():
    X, y = make_regression(n=500, k=30, seed=9)  # many small categories -> OOF != full
    enc = TargetEncoder(cols=["g"], smooth=0.0, cv=5, random_state=0, output="numpy")
    ft = np.asarray(enc.fit_transform(X, y)).ravel()
    leaky = np.asarray(enc.fit(X, y).transform(X)).ravel()
    assert not np.allclose(ft, leaky)


def test_unsupervised_fit_transform_equals_fit_then_transform():
    X, _ = make_regression(seed=2)
    ce = CountEncoder(cols=["g"], output="numpy")
    a = np.asarray(ce.fit_transform(X))
    b = np.asarray(ce.fit(X).transform(X))
    assert np.allclose(a, b)  # no target -> nothing to cross-fit
