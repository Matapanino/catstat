# Verdict: GPU device-resident path (PR-D) — fresh three-lane crossover (feat/shape-stats-moments)

- Date: 2026-07-02
- Branch: `feat/shape-stats-moments`
- Backend: gpu (Colab T4, python 3.12.13, preinstalled RAPIDS cu12 26.2.1 / cupy 14.0.1, RMM pool)
- Artifacts:
  - `benchmarks/results/2026-07-02-T4-gpu-parity.jsonl` (parity + three-lane crossover rows)
  - `docs/verdicts/2026-07-02-gpu-parity-report.md` (tables; incl. the post-run fix note)
  - T4 test-suite runs: 358→360 passed (all gpu-marked device/kernel/parity tests)
- Roadmap target: PR-D / B1–B5 · Related: KI-020 (crossover), KI-001, KI-013

## Question
With the OOF kernel on device (B1), cuDF input device-resident end-to-end (B2/B3), and the
order-stat per-fold loop on device (B4): (1) does the **host-origin** lane now clear the bar for
re-enabling `_AUTO_GPU_ENABLED`? (2) how much does the **device-resident** lane win, which is
where `backend="gpu"` was always aimed?

## Evidence

### Correctness / leakage / parity
- T4 full suite **360 passed** (24+ gpu-marked tests): B1 kernel parity, skew/kurt/woe parity,
  the whole device-input matrix (missing/unknown/combination/interactions/y-containers/output
  containers/set_output/fences/determinism), order-stat OOF parity, and the large-offset case.
- Parity table (n=200k, 5k cats): 16/17 ok at first run; `shape_offset_1e9` (transform side)
  exposed **unshifted fit-path reductions** cancelling differently on cuDF (one-pass var) vs
  pandas (two-pass) at `|mean| >> sd` — fixed the same day (shift-stable fit reductions,
  `2872a76`; binarized targets exempt to preserve WOE's exact ±inf contract), covered by
  `test_cpu_gpu_parity_large_offset` in the 360-pass run.
- Leakage audits re-passed after every stage (OOF reconstruction exact/allclose, noise traps ≈0).

### Performance (fit_transform median seconds; reps=5 at ≥1M, 3 below; T4)
| n | profile | cpu | gpu **host-origin** | gpu **device-resident** | speedup host | speedup dev |
|---|---|---|---|---|---|---|
| 10k | mean | 0.009 | 0.0385 | 0.0099 | 0.23 | 0.91 |
| 100k | mean | 0.053 | 0.096 | 0.0207 | 0.55 | **2.56** |
| 1M | mean | 0.845 | 0.774 | 0.145 | 1.09 | **5.84** |
| 5M | mean | 4.54 | 4.01 | 0.771 | 1.13 | **5.89** |
| 10M | mean | 10.13 | 9.23 | 1.59 | 1.10 | **6.37** |
| 1M | mean+var+skew+kurt | 1.40 | 1.55 | 0.166 | 0.90 | **8.42** |
| 5M | mean+var+skew+kurt | 9.30 | 8.15 | 1.02 | 1.14 | **9.13** |
| 10M | mean+var+skew+kurt | 20.69 | 17.42 | 1.92 | 1.19 | **10.77** |
| 1M | median (per-fold loop) | 2.52 | 2.91 | 0.250 | 0.87 | **10.07** |
| 5M | median | 16.40 | 15.98 | 1.32 | 1.03 | **12.43** |

Transform-only (fitted encoder): 1M ×6.6, 10M ×**13.2** (device gather vs host map).

## Decision
**KEEP `_AUTO_GPU_ENABLED = False`** for host-origin data — the flip criterion (≥1.25× at ≥2
adjacent sizes, ≥0.95× everywhere above the threshold) is **not met**: host-origin peaks at
1.19×@10M (mvsk) and is ≤1.14× elsewhere ≥1M; the H2D copies of pandas-origin data still eat the
win, consistent with all previous crossovers (KI-020). `_GPU_CELL_THRESHOLD` unchanged.

**The device-resident lane is the deliverable**: cuDF input runs **2.6× (100k) to 5.8–12.4×
(1M–10M)** faster than CPU across mean / multi-stat / median profiles and **6.6–13.2×** at
transform, returning cuDF with no host round-trip. This path routes to the GPU *categorically*
(`is_device_frame` → `select_backend(device_input=True)`), independent of the auto flag, so no
default changes — and no host-origin user pays anything.

## Follow-ups
- KI-020 updated (host-origin numbers refreshed; device-resident story recorded).
- Optional: cache device uniques on the fitted estimator for repeated `transform(cuDF)` calls
  (currently re-H2D'd per call; transform is already ×6–13 despite it).
- Optional later: re-run the parity table for a pristine post-fix report when the Colab
  session-assignment quota resets (the fix is already suite-validated on T4).
