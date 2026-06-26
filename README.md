# catstat

Unified CPU/GPU **statistical categorical encoding**: leakage-safe target encoding generalized to
arbitrary statistics, behind one scikit-learn-compatible API.

> Status: **M0 (alpha)** — CPU-only (pandas/numpy). Mean target encoding for regression / binary /
> multiclass, plus count / frequency encoding. GPU (cuDF/CuPy) and more statistics are planned
> (see [`docs/roadmap.md`](docs/roadmap.md)).

```python
from catstat import TargetEncoder, CountEncoder, FrequencyEncoder

enc = TargetEncoder(cols="auto", stats=["mean"], smooth="auto", cv=5, random_state=42)
X_train_enc = enc.fit_transform(X_train, y_train)   # out-of-fold (leakage-safe)
X_test_enc  = enc.transform(X_test)                 # full-data encodings
```

## Why
- **sklearn** `TargetEncoder` is CPU + mean-only; **cuML** is GPU-only (RAPIDS-locked, few stats);
  **category_encoders** has no cross-fitting (leakage risk). `catstat` is the union: one API, CPU
  today and GPU when it pays off, generalized statistics, always leakage-safe.

## Guarantees
- `fit_transform` is **out-of-fold** (leakage-safe for training data); `fit().transform()` learns
  full-data encodings for *new* data.
- Deterministic given `random_state`.
- sklearn-compatible: `Pipeline`, `ColumnTransformer`, `set_output`, `get_feature_names_out`.

## Develop
```bash
bash scripts/check.sh        # ruff + pytest + examples (the green gate)
PYTHONPATH=src python3 -m pytest tests/ -q
PYTHONPATH=src python3 -m benchmarks.run_benchmarks --size small --backend cpu --reps 5 \
    --out benchmarks/results/run.json
```
See [`CLAUDE.md`](CLAUDE.md) for the development rules and [`docs/`](docs/) for the design.

## License
MIT
