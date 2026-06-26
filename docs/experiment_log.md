# `catstat` — Experiment Log (append-only)

One line (or short block) per experiment, **including null and negative results**, so no future
session retries a dead end. Newest at the top. Each entry links its verdict when one exists.

**Format:**
```
## YYYY-MM-DD — <topic>
- Hypothesis: <what we expected>
- Setup: <dataset(s), seeds, reps, backend, git SHA>
- Result: <KEEP | REJECT | NULL> — <one-line evidence>
- Verdict: docs/verdicts/YYYY-MM-DD-<topic>-verdict.md (if any)
```

---

## 2026-06-26 — project bootstrap (design phase)
- Hypothesis: a unified CPU/GPU, statistically-general, leakage-safe encoder fills a real gap left
  by sklearn (CPU/mean-only), cuML (GPU/RAPIDS-only), and category_encoders (no cross-fit).
- Setup: research pass over the three libraries' docs + source; no code run.
- Result: KEEP (design) — gap confirmed; design recorded in `docs/proposals/`.
- Verdict: n/a (design, not a measured change).

## 2026-06-26 — M0 bootstrap (CPU mean encoder + count/frequency)
- Hypothesis: a CPU mean `TargetEncoder` with out-of-fold `fit_transform` plus unsupervised
  `Count`/`Frequency` encoders can be implemented leakage-safe, sklearn-compatible, and green.
- Setup: pandas 1.5.2 / numpy 1.23.5 / sklearn 1.2.0; `scripts/check.sh`; size=small benchmark, 5 reps.
- Result: KEEP — 46 passed / 1 GPU-skipped; OOF reconstruction exact (`max |Δ|=0.0`); noise-trap
  OOF corr ≈ -0.006 vs leaky 0.66; coverage 85.87%; baseline written.
- Verdict: docs/verdicts/2026-06-26-m0-bootstrap-verdict.md

## 2026-06-26 — Phase 2 (dispersion/order stats, combination, GPU scaffold)
- Hypothesis: var/std/median/min/max can be added as cross-fitted, continuous-only stats; a joint
  combination mode and a cuDF/CuPy backend fit behind the existing structure without regressing CPU.
- Setup: pandas 1.5.2 / numpy 1.23.5 / sklearn 1.2.0; `scripts/check.sh`; size=small benchmark.
- Result: KEEP — 67 passed / 2 GPU-skipped; coverage 88.17%; new stats cross-fitted & correct vs
  pandas groupby; combination joint encoding correct; CPU path byte-unchanged after backend
  threading. GPU path written, **Colab-validation pending** (no local GPU).
- Verdict: docs/verdicts/2026-06-26-phase2-stats-gpu-verdict.md

## 2026-06-26 — GPU backend CPU/GPU parity validated on Colab T4
- Hypothesis: `backends/_gpu.py` (cuDF/CuPy, host-orchestrated, catstat-owned folds) produces the
  same encodings as CPU to allclose.
- Setup: Colab T4, Python 3.12.13, RAPIDS (cudf-cu12); n=200k × 5k cats, cv=5, seed=0;
  `scripts/colab_gpu_parity.sh`.
- Result: KEEP — all 4 cases allclose (mean/var × reg/bin/mc), transform + fit_transform,
  max|Δ|~1e-14. backend_gpu="gpu" confirmed.
- Verdict: docs/verdicts/2026-06-26-gpu-parity-verdict.md (+ harness report + JSONL artifact).

## 2026-06-26 — GPU missing-on-device + CPU/GPU crossover (T4)
- Hypothesis: (1) GPU encodes missing-as-value correctly via cuDF nulls; (2) GPU beats CPU above
  some size, so `backend="auto"` should switch over at a calibrated threshold.
- Setup: Colab T4, Python 3.12.13; parity n=200k incl. 10%-missing case; crossover n=10k/100k/1M.
- Result: (1) KEEP — missing-as-value allclose (max|Δ|~3e-16); MISSING→cuDF-null→back works.
  (2) **NEGATIVE/CHANGE** — GPU is *slower* than CPU at all sizes up to 1M (speedup 0.28/0.27/0.86);
  the per-fold host↔device round-trip dominates. → disabled auto-GPU (`_AUTO_GPU_ENABLED=False`);
  explicit `backend="gpu"` retained.
- Verdict: docs/verdicts/2026-06-26-gpu-crossover-verdict.md (KI-020).

## 2026-06-26 — Phase 3a (skew + custom-callable aggregations)
- Hypothesis: skew (built-in) and arbitrary custom aggregations (quantiles, IQR, ...) fit the
  registry as cross-fitted, continuous-only, CPU stats without disturbing the GPU/CPU paths.
- Setup: pandas/numpy/sklearn; `scripts/check.sh`.
- Result: KEEP — 75 passed / 2 GPU-skipped, coverage 89.37%. skew matches pandas groupby; custom
  q90/IQR correct; custom forces CPU + is cross-fitted; `stats=["quantile"]` gives a helpful hint.
- Verdict: docs/verdicts/2026-06-26-phase3a-skew-custom-verdict.md

## 2026-06-26 — Phase 3b (leave-one-out + ordered/CatBoost schemes)
- Hypothesis: LOO and ordered target statistics can be added as a `scheme` param (mean-only,
  leakage-safe alternatives to k-fold) without changing default behavior.
- Setup: pandas/numpy/sklearn; `scripts/check.sh`.
- Result: KEEP — 86 passed / 2 GPU-skipped, coverage 90.64%. LOO exact-value check passes; both
  schemes leakage-safe (noise OOF corr <0.1 vs leaky >0.4); ordered deterministic per seed;
  non-mean+scheme raises. Bug found+fixed: ordered with smooth=0 gave a=0 → 0/0 nan; default a=1.
- Verdict: docs/verdicts/2026-06-26-phase3b-loo-ordered-verdict.md

<!-- Append new experiments below this line. Never edit or delete prior entries. -->
