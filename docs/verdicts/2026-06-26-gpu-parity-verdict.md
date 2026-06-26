# Verdict: GPU backend CPU/GPU parity validated on Colab T4

- Date: 2026-06-26
- Branch: `main`
- Backend: gpu (Colab T4) vs cpu
- Artifacts:
  - `docs/verdicts/2026-06-26-gpu-parity-report.md` (harness-generated table)
  - `benchmarks/results/2026-06-26-T4-gpu-parity.jsonl` (raw)
- Roadmap target: `docs/roadmap.md` → Phase 2 GPU validation · Related: `docs/known_issues.md` KI-001

## Question
Does `backends/_gpu.py` (cuDF/CuPy, host-orchestrated) produce the **same** encodings as the CPU
backend — `transform` and out-of-fold `fit_transform` — under the same `random_state`, to
`allclose`?

## Evidence
Ran `scripts/colab_gpu_parity.sh --gpu T4` (Colab, Python 3.12.13, RAPIDS via `cudf-cu12`),
n=200k rows × 5k categories, `cv=5`, `random_state=0`:

| case | transform allclose | max\|Δ\| | fit_transform allclose | max\|Δ\| | gpu ft (s) |
|------|:------------------:|---------:|:----------------------:|---------:|-----------:|
| regression mean | ✅ | 2.2e-15 | ✅ | 2.7e-15 | 1.36* |
| regression var  | ✅ | 6.9e-15 | ✅ | 1.1e-14 | 0.14 |
| binary mean     | ✅ | 0.0     | ✅ | 0.0     | 0.12 |
| multiclass mean | ✅ | 0.0     | ✅ | 0.0     | 0.35 |

`backend_cpu="cpu"`, `backend_gpu="gpu"` confirmed. Max divergence ~1e-14 is float
reduction-order noise (GPU atomic ordering), well within `rtol=1e-5`. *First case includes
cuDF/CuPy JIT warmup.

## Decision
**KEEP** — the GPU backend is correct: CPU and GPU agree to machine precision for mean and var
across regression/binary/multiclass, for both `transform` and leakage-safe `fit_transform`. The
"device group-by, host orchestration" design (identical downstream math, catstat-owned folds)
delivers parity as intended. `backend="gpu"`/`"auto"` are now validated for single-column
numeric/string keys.

## Follow-ups
- Untested on device (KI-018): missing-as-value + cuDF nulls, and `combination` (tuple keys,
  currently forced to CPU). Add to a future Colab run before claiming general GPU support.
- Perf: this run measured correctness, not a CPU-vs-GPU crossover. Add a conversion-overhead /
  crossover benchmark to calibrate the `backend="auto"` cell threshold.
