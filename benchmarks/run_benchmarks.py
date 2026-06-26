"""Run the catstat benchmark suite: time fit / transform / fit_transform separately, persist.

Usage:
    PYTHONPATH=src python3 benchmarks/run_benchmarks.py --size small --backend cpu --reps 5 \
        --out benchmarks/results/run.json

Writes a summary JSON (the baseline format) and appends one row per case to the JSONL ledger.
Correctness/leakage are covered by the test suite; this measures cost + a coarse quality number.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from catstat import CountEncoder, TargetEncoder

from . import datasets, ledger


def _time(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return {"median": statistics.median(ts), "spread": (max(ts) - min(ts)) / 2.0}


def _cases(size, seed):
    yield "regression", *datasets.make_regression(size, seed=seed), TargetEncoder(
        cols=["g"], cv=5, random_state=0, output="numpy"
    )
    yield "binary", *datasets.make_binary(size, seed=seed), TargetEncoder(
        cols=["g"], cv=5, random_state=0, output="numpy"
    )
    yield "multiclass", *datasets.make_multiclass(size, seed=seed), TargetEncoder(
        cols=["g"], cv=5, random_state=0, output="numpy"
    )
    yield "high_cardinality", *datasets.make_high_cardinality(size, seed=seed), TargetEncoder(
        cols=["g"], cv=5, random_state=0, output="numpy"
    )
    yield "regression_std", *datasets.make_regression(size, seed=seed), TargetEncoder(
        cols=["g"], stats=["std"], cv=5, random_state=0, output="numpy"
    )
    yield "combination", *datasets.make_multi_column(size, seed=seed), TargetEncoder(
        cols="auto", multi_feature_mode="combination", cv=5, random_state=0, output="numpy"
    )
    yield "count", *datasets.make_high_cardinality(size, seed=seed), CountEncoder(
        cols=["g"], output="numpy"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="small")
    ap.add_argument("--backend", default="cpu", choices=["cpu", "gpu", "auto"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="benchmarks/results/run.json")
    args = ap.parse_args()

    meta = ledger.run_meta(args.backend)
    cases = {}
    rows = []
    for name, X, y, gmeta, enc in _cases(args.size, args.seed):
        enc.set_params(backend=args.backend)
        supervised = y is not None and isinstance(enc, TargetEncoder)

        def _fit():
            enc.fit(X, y) if supervised else enc.fit(X)

        def _transform():
            enc.transform(X)

        def _fit_transform():
            enc.fit_transform(X, y) if supervised else enc.fit_transform(X)

        enc.fit(X, y) if supervised else enc.fit(X)  # warm + quality
        out = np.asarray(enc.fit_transform(X, y) if supervised else enc.fit_transform(X))
        quality = {}
        if name == "regression":  # OOF mean predicts y; RMSE is a meaningful quality signal here
            quality["oof_rmse"] = float(np.sqrt(np.mean((out.ravel() - y) ** 2)))

        rec = {
            "fit_s": _time(_fit, args.reps),
            "transform_s": _time(_transform, args.reps),
            "fit_transform_s": _time(_fit_transform, args.reps),
            "n_out_cols": int(out.shape[1]),
            "quality": quality,
            **gmeta,
        }
        cases[name] = rec
        rows.append({**meta, "case_name": name, **rec})
        ft_ms = rec["fit_transform_s"]["median"] * 1e3
        print(f"{name:16s} ft={ft_ms:7.1f}ms cols={rec['n_out_cols']}")

    payload = {"meta": meta, "size": args.size, "cases": cases}
    ledger.write_json(args.out, payload)
    ledger.append_rows(rows)
    print(f"wrote {args.out} and appended {len(rows)} ledger rows")


if __name__ == "__main__":
    main()
