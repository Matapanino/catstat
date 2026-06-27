# Verdict: GPU crossover re-measured after complement-subtraction (T4) — auto stays off

- Date: 2026-06-26
- Branch: `feat/perf-gpu-crossover-5m` (stacked on `feat/perf-complement-subtraction-mean`)
- Backend: gpu (Colab T4) vs cpu
- Artifacts: `docs/verdicts/2026-06-26-gpu-parity-report.md`, `benchmarks/results/2026-06-26-T4-gpu-parity.jsonl`
- Roadmap: Phase 2 GPU perf · Related: `docs/known_issues.md` KI-020 · Supersedes the crossover half of
  `docs/verdicts/2026-06-26-gpu-crossover-verdict.md` (re-measured post-PR-B; parity stance unchanged).

## Question
(1) Does the host-side complement-subtraction mean OOF (PR-B) **preserve CPU/GPU parity**?
(2) Extending the crossover to **5M / 10M** rows (the prior verdict's own follow-up), does GPU now
win enough to **re-enable `backend="auto"`**?

## Evidence (Colab T4, RAPIDS, cudf-cu12)

### Parity — all pass; mean `fit_transform` now bitwise-identical
n=200k × 5k cats, cv=5, CPU vs GPU `allclose` (rtol 1e-5):

| case | transform max\|Δ\| | fit_transform max\|Δ\| | allclose |
|------|---:|---:|:--:|
| regression_mean | 3.3e-16 | **0.0** | ✅ |
| regression_var | 1.3e-15 | 1.6e-15 | ✅ |
| binary_mean | 0.0 | **0.0** | ✅ |
| multiclass_mean | 0.0 | **0.0** | ✅ |
| regression_mean_missing | 2.2e-16 | **0.0** | ✅ |
| numeric_auto / numeric_bin | ~1e-17 | **0.0** | ✅ |

→ PR-B is **GPU-safe**. Because the fast mean OOF runs on the shared host kernel for both backends,
CPU/GPU `fit_transform` for mean is now **exact** (max|Δ|=0.0). KI-030 numeric parity re-confirmed.

### Crossover — GPU scales better, but the win is marginal and noisy
mean encoder, fit_transform median seconds, speedup = cpu/gpu (>1 ⇒ GPU faster):

| n | cardinality | cpu_ft_s | gpu_ft_s | speedup |
|---|------------:|---------:|---------:|--------:|
| 10k | 250 | 0.010 | 0.049 | 0.20 |
| 100k | 2.5k | 0.067 | 0.108 | 0.62 |
| 1M | 25k | 0.872 | 1.309 | **0.67** |
| 5M | 125k | 5.43 | 4.90 | **1.11** |
| 10M | 250k | 12.46 | 11.79 | **1.06** |

GPU scales **sublinearly** (1M→10M: cpu ×14.3, gpu ×9.0) and crosses ~parity at **5M**. But:
- The win at 5–10M is **marginal** (≤1.11×) on 5–12 s jobs.
- It is **noisy across runs**: an earlier identical run (same code, same T4 class) measured 1M at
  **0.98** vs this run's **0.67** (GPU times swung ~50%) — so ~1.1× at 5M is within run-to-run noise
  of parity.
- Below 5M (the common case) GPU is **0.2–0.67×**.

## Decision
**KEEP `backend="auto"` GPU OFF** — `_AUTO_GPU_ENABLED` stays `False` (no `_dispatch.py` change). A
marginal (≤1.11×), noisy win only at ≥5M rows does not justify penalizing the overwhelmingly common
<5M case, and the maintainer's stated precondition for re-enabling (keys/folds **on-device**) is met
only host-side, not on the device. Target encoding on host-origin data is **transfer/memory-bound,
not compute-bound**, so a complex on-device OOF kernel is **not warranted by the data** (the residual
GPU gap is the full-data device group-by, not the OOF — which PR-B already made host-fast for both
backends). Explicit `backend="gpu"` stays validated (allclose) for users with ≥5M-row, device-resident
pipelines. This is the honest stopping point for the GPU lever.

## Follow-ups
- The retained `_GPU_CELL_THRESHOLD = 5_000_000` (cells) already matches the ~5M crossover, so re-enabling
  is a **one-line flip** (`_AUTO_GPU_ENABLED = True`) **if** the maintainer accepts a marginal/noisy win
  at extreme scale — but the data does not recommend it.
- A true device-resident path (inputs originating on GPU; factorize/folds/aggregate on device, zero host
  round-trips) could lower the crossover, but the marginal payoff makes it a niche, low-priority lever.
- Extend to 20M+ only if a concrete large-scale use case appears (watch T4 memory).
