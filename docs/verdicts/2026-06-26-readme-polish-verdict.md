# Verdict: README polish — honest status, badges, feature table (main)

- Date: 2026-06-26
- Branch: `main`
- Backend: cpu (docs change — no compute path touched)
- Artifacts:
  - `README.md` (rewritten)
  - `src/catstat/__init__.py` (module docstring honesty — pdoc landing page)
- Roadmap target: `docs/roadmap.md` Phase 3 (docs) · Related: KI-020 (GPU honesty), KI-012 (sklearn)

## Question
Does the public README accurately and credibly represent `catstat` at 0.1.1?

## Evidence

### Correctness / leakage / parity
No behavior changed. Green gate green: `bash scripts/check.sh` → "All checks passed" (88 passed /
2 GPU-skipped). `python -m build` + `twine check` **PASSED** — the new README renders as the PyPI
long-description. The statistics table was cross-checked against the `src/catstat/_stats.py`
registry (mean/count/frequency/var/std/median/min/max/skew + `(name, callable)` custom; skew and
custom are CPU-only; only mean is smoothed; `quantile` via custom). The GPU claim matches KI-020
(parity-validated; `auto` = CPU).

### Performance (≥5 reps, median + spread)
N/A — no compute path changed.

## Decision
**KEEP** — replaces the stale "M0 (alpha) — CPU-only" marker with an honest 0.1.x status and adds
CI/PyPI/Python/license badges, install + extras, the stat/feature table, a "leakage-safe by design"
note, the scikit-learn-compat subset pointer, and the API-docs link (goes live with Commit 3's docs
site). The PyPI version/Python badges populate automatically once the package is on PyPI. Touches no
invariant, default, or public API.

## Follow-ups
- Commit 3: API docs via `pdoc` + a GitHub Pages workflow — makes the linked docs URL live.
