# Verdict: project hygiene — CONTRIBUTING, SECURITY, issue/PR templates (main)

- Date: 2026-06-26
- Branch: `main`
- Backend: cpu (project meta — no code)
- Artifacts:
  - `CONTRIBUTING.md`, `SECURITY.md`
  - `.github/ISSUE_TEMPLATE/{bug_report.md, feature_request.md, config.yml}`
  - `.github/PULL_REQUEST_TEMPLATE.md`
- Roadmap target: `docs/roadmap.md` Phase 3 (hygiene)

## Question
Does the repo have the standard contributor-facing hygiene files for a credible public release?

## Evidence

### Correctness
Added `CONTRIBUTING.md` (dev setup, the green gate, the non-negotiable invariants, PR expectations),
`SECURITY.md` (supported versions + private reporting), GitHub issue templates (bug / feature +
a config with a private-security contact link), and a PR template (green-gate / invariants /
CHANGELOG checklist). Docs/meta only — outside ruff's lint paths and pytest — so the green gate is
unaffected (`bash scripts/check.sh` green); `config.yml` parses.

### Performance (≥5 reps, median + spread)
N/A.

## Decision
**KEEP** — standard hygiene; no code or behavior change.

## Follow-ups
- Optional: `CODE_OF_CONDUCT.md` (Contributor Covenant).
- The release-polish arc (0.1.1) is complete on `main` with CI green. Remaining steps are the
  maintainer's: configure the PyPI Trusted Publisher and enable GitHub Pages, then tag `v0.1.1` to
  auto-publish. Optional larger work: KI-020 GPU on-device perf (needs a fresh crossover verdict).
