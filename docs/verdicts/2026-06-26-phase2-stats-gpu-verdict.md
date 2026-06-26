# Verdict: Phase 2 — dispersion/order stats, combination mode, GPU backend scaffold

- Date: 2026-06-26
- Branch: `main`
- Backend: cpu (GPU path written, Colab-validation pending — no local GPU)
- Artifacts:
  - `benchmarks/results/baseline-cpu.json` (refreshed: now 7 cases incl. `regression_std`, `combination`)
  - `benchmarks/results/ledger.jsonl`
- Roadmap target: `docs/roadmap.md` → **Phase 2** · Related: `docs/known_issues.md` KI-001/002/003

## Question
Does Phase 2 add (a) the dispersion/order statistics var/std/median/min/max, (b)
`multi_feature_mode="combination"`, and (c) a real GPU backend + Colab parity loop + CI — without
regressing the green CPU path or the leakage guarantees?

## Evidence

### Correctness / leakage
- `bash scripts/check.sh`: **ruff clean · 67 passed, 2 skipped (GPU) · examples run**. Coverage
  **88.17%** (up from 85.87%).
- New stats verified against pandas groupby on full data (`test_stats.py`); var/std/median/min/max
  are **target-dependent → cross-fitted** (`test_dispersion_stats_are_cross_fitted`), so the
  leakage invariant extends to them. Singletons/undefined → **global** fallback (no blending).
- Dispersion/order stats are **continuous-target only**; classification raises a clear error.
- `multi_feature_mode="combination"` produces a single joint column whose encoding equals the
  groupby mean over the joint key; unseen combos → global (`test_multi_feature.py`).
- Backend selection: `auto` resolves to CPU here; explicit `backend="gpu"` raises `ImportError`
  (no silent fallback); invalid backend raises (`test_backend.py`).

### GPU backend (written; Colab-validation pending)
- `backends/_gpu.py` implements the cuDF/CuPy group-by primitives, returning **pandas** small
  per-category results so downstream smoothing/mapping is identical to CPU ⇒ CPU/GPU agree to
  `allclose`. Threaded through `_smoothing`/`_aggregations` via a selectable backend; the CPU path
  is byte-identical (the 67 tests confirm no regression).
- `scripts/colab_gpu_parity.{sh,py}` mirror the repleafgbm Colab loop (pack → `colab new --gpu T4`
  → upload → exec with watchdog → download → stop) and run `test_cpu_gpu_parity.py`-style checks on
  device. Combination (tuple keys) forces CPU.

### Performance (size=small=10k rows, cpu, 5 reps, median fit_transform)
| case | ft (ms) | cols |
|------|--------:|-----:|
| regression (mean) | 12.2 | 1 |
| binary | 13.1 | 1 |
| multiclass (5) | 46.6 | 5 |
| high_cardinality | 12.7 | 1 |
| regression_std | 11.9 | 1 |
| combination | 58.3 | 1 |
| count | 1.6 | 1 |

## Decision
**KEEP** — Phase 2 CPU features (stats, combination) are complete, green, and leakage-safe; GPU
backend + Colab loop + CI are in place. Baseline refreshed to include the new cases.

## Follow-ups
- **Colab**: run `scripts/colab_gpu_parity.sh` to validate/harden the GPU path (string keys, nulls,
  missing-as-value on device); record GPU baselines. Until then `backend="gpu"` is unverified.
- `combination` builds joint keys with a Python loop (58ms case) — vectorize for large N.
- KI-010 still open: verify `smooth="auto"` vs sklearn `_target_encoder_fast.pyx` (needs sklearn≥1.4).
- Phase 3: quantile/skew/custom aggregations, ordered/LOO encoders, `output="polars"`.
