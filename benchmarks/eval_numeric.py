"""Empirical quality eval for opt-in numeric-column target encoding.

Answers: does cardinality-aware numeric target encoding improve a downstream linear model's
cross-validated R^2 versus passing the raw numeric columns, and which ``n_bins`` / cardinality
threshold should be the default? Synthetic "playground"-style regression where the numeric signal
is non-linear (a high-cardinality continuous feature with a non-monotone effect) and categorical
(a low-cardinality integer code with arbitrary per-value offsets) -- exactly the regime a single
linear model cannot exploit from raw numbers but binned/direct target encoding can.

Methodology is leakage-honest: per outer CV fold, the encoder is ``fit_transform``-ed on the
training rows (out-of-fold encodings) and ``transform``-ed on the held-out rows; the linear model
is fit on the encoded training rows and scored on the held-out encoded rows. Reported over 5 seeds
(median + min/max spread). Run: ``PYTHONPATH=src python3 benchmarks/eval_numeric.py``.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from catstat import TargetEncoder

SEEDS = [0, 1, 2, 3, 4]


def make_playground(n=4000, seed=0):
    """High-card continuous feature with a non-monotone effect + low-card categorical-coded int."""
    rng = np.random.default_rng(seed)
    x_hc = rng.uniform(-3.0, 3.0, size=n)  # continuous, high cardinality
    c_lc = rng.integers(0, 8, size=n)  # 8 distinct integer codes (categorical, not ordinal)
    offsets = rng.normal(0, 2.0, size=8)  # arbitrary per-code effect
    signal = np.sin(2.5 * x_hc) + 0.6 * x_hc**2 - 1.0 + offsets[c_lc]  # non-linear + categorical
    y = signal + rng.normal(0, 0.5, size=n)
    return pd.DataFrame({"x_hc": x_hc, "c_lc": c_lc}), y


def _cv_r2(build, X, y, seed):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for tr, va in kf.split(X):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        enc = build()
        if enc is None:  # raw numeric passthrough
            Ztr, Zva = Xtr.to_numpy(float), Xva.to_numpy(float)
        else:
            Ztr = enc.fit_transform(Xtr, y[tr])
            Zva = enc.transform(Xva)
        model = Ridge(alpha=1.0).fit(Ztr, y[tr])
        scores.append(r2_score(y[va], model.predict(Zva)))
    return float(np.mean(scores))


def _builder(**kw):
    return lambda: TargetEncoder(smooth="auto", random_state=0, output="numpy", **kw)


def main():
    strategies = {
        "raw_numeric": lambda: None,
        "direct": _builder(numeric="direct"),
        "auto(n_bins=5)": _builder(numeric="auto", n_bins=5, cardinality_threshold=20),
        "auto(n_bins=10)": _builder(numeric="auto", n_bins=10, cardinality_threshold=20),
        "auto(n_bins=20)": _builder(numeric="auto", n_bins=20, cardinality_threshold=20),
        "auto(n_bins=40)": _builder(numeric="auto", n_bins=40, cardinality_threshold=20),
    }

    rows = {}
    for name, build in strategies.items():
        vals = [_cv_r2(build, *make_playground(seed=s), s) for s in SEEDS]
        rows[name] = vals
        print(
            f"{name:>18}  R2 median={np.median(vals):+.4f}  "
            f"min={min(vals):+.4f}  max={max(vals):+.4f}"
        )

    # routing sanity: which strategy does cardinality_threshold=20 pick per column?
    X, y = make_playground(seed=0)
    strat = TargetEncoder(numeric="auto", cardinality_threshold=20).fit(X, y).numeric_strategy_
    print("routing @threshold=20:", strat)

    out = {
        "date": str(date.today()),
        "metric": "downstream Ridge 5-fold CV R^2 (higher is better)",
        "seeds": SEEDS,
        "results": {k: {"median": float(np.median(v)), "values": v} for k, v in rows.items()},
        "routing_threshold20": strat,
    }
    path = f"benchmarks/results/{date.today()}-numeric-te-eval.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
