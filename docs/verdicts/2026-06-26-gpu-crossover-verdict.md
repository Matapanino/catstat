# Verdict: GPU crossover — disable auto-GPU (data-driven); missing-on-device validated

- Date: 2026-06-26
- Branch: `main`
- Backend: gpu (Colab T4) vs cpu
- Artifacts:
  - `docs/verdicts/2026-06-26-gpu-parity-report.md` (harness table, run 2)
  - `benchmarks/results/2026-06-26-T4-gpu-parity.jsonl`
- Roadmap target: Phase 2 GPU validation + `backend="auto"` calibration · `docs/known_issues.md` KI-001/018

## Question
(1) Does the GPU path handle a **missing-as-value** category correctly (cuDF nulls)? (2) Where is
the **CPU↔GPU crossover** — i.e. what should the `backend="auto"` threshold be?

## Evidence (Colab T4, Python 3.12.13, RAPIDS)

### Parity — all pass, incl. missing
n=200k × 5k cats, cv=5, seed=0; CPU vs GPU, `allclose` (rtol 1e-5):

| case | transform | fit_transform | max\|Δ\| |
|------|:---------:|:-------------:|---------:|
| regression_mean | ✅ | ✅ | 3.3e-16 |
| regression_var | ✅ | ✅ | 1.6e-15 |
| binary_mean | ✅ | ✅ | 0.0 |
| multiclass_mean | ✅ | ✅ | 0.0 |
| **regression_mean_missing** (10% NaN, handle_missing="value") | ✅ | ✅ | 3.3e-16 |

→ The MISSING→cuDF-null→back mapping in `_gpu.py` is correct; a missing level is encoded on-device
and the host `.map` lines up. **KI-018 (missing) closed.**

### Crossover — GPU is slower everywhere measured
mean encoder, fit_transform median seconds, speedup = cpu/gpu (>1 ⇒ GPU faster):

| n | cardinality | cpu_ft_s | gpu_ft_s | speedup |
|---|-------------|---------:|---------:|--------:|
| 10k | 250 | 0.032 | 0.116 | 0.28 |
| 100k | 2.5k | 0.159 | 0.587 | 0.27 |
| 1M | 25k | 1.88 | 2.19 | **0.86** |

GPU never wins up to 1M rows (it trends toward 1.0 but does not cross). The host-orchestrated path
pays a host↔device round-trip for **each** of the 6 group-bys per `fit_transform` (5 OOF folds +
the full refit), plus host key-normalization — and that dominates the device group-by gain at these
sizes.

## Decision
**CHANGE DEFAULT** — set `_AUTO_GPU_ENABLED = False` in `backends/_dispatch.py` so `backend="auto"`
**never selects GPU** for now. Picking the slower backend by default would be a regression; the old
`1e6`-cell threshold would have wrongly chosen GPU at n≥1M. Explicit `backend="gpu"` stays available
and validated (allclose, incl. missing) for users with device-resident pipelines or much larger
data. This reverts to the repleafgbm stance ("auto never picks GPU") — the design's auto-GPU
hypothesis (§4) is corrected by measurement.

## Follow-ups
- Re-enable + calibrate `_GPU_CELL_THRESHOLD` once the device path is optimized: keep binned keys
  and fold-ids **on-device**, do the OOF group-bys without per-fold host↔device transfers, and avoid
  the host MISSING-scan. Then re-run this crossover (and extend to n=5M/10M).
- Combination on GPU (tuple keys) still forced to CPU (KI-018 remainder) — fine; revisit with the
  on-device redesign.
