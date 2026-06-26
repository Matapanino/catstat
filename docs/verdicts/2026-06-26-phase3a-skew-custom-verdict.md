# Verdict: Phase 3a — skew + custom-callable aggregations

- Date: 2026-06-26
- Branch: `main`
- Backend: cpu
- Artifacts: `tests/test_phase3.py`
- Roadmap target: `docs/roadmap.md` → Phase 3 · Closes `docs/known_issues.md` KI-002

## Question
Can `skew` and arbitrary **custom aggregations** (quantiles, IQR, …) be added as cross-fitted,
continuous-only CPU statistics — honoring the smoothing-honesty rule — without disturbing the
mean/count/dispersion paths or the GPU/CPU split?

## Evidence
- `bash scripts/check.sh`: **ruff clean · 75 passed, 2 skipped (GPU) · examples run**.
  Coverage **89.37%**.
- `skew` (built-in, `g__te_skew`) matches `pandas.groupby.skew()`; **continuous-only** (raises on
  classification); n<3 → global skew fallback.
- Custom aggregations via `stats=[("q90", fn)]` or `stats={"p10": fn}`: `q90` matches
  `groupby.quantile(0.9)`; unseen → global value of the callable; **cross-fitted** (OOF ≠ leaky).
- Custom/skew are **CPU-only** → they force `backend_="cpu"` even under `backend="auto"`.
- `stats=["quantile"]` raises a helpful error pointing at the custom-callable form.

## Decision
**KEEP** — Phase 3a complete and green. Quantiles are expressed as custom callables (more general
than a fixed `quantile` built-in, matching the design's example). No new smoothing was added (order
/shape/custom stats never blend — only a small-n/undefined → global fallback), preserving the
honesty rule.

## Follow-ups
- Custom aggregations use `groupby.apply` (slow for many groups) and are assumed order-independent
  (documented, not enforced).
- Next Phase 3: ordered (CatBoost) + leave-one-out modes; then `set_output("polars")` + PyPI.
