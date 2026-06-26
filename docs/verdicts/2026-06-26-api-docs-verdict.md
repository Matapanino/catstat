# Verdict: API docs — pdoc + GitHub Pages (main)

- Date: 2026-06-26
- Branch: `main`
- Backend: cpu (CI/docs infra — no compute path touched)
- Artifacts:
  - `scripts/build_docs.sh` (new)
  - `.github/workflows/docs.yml` (new)
  - `.gitignore` (`site/` build output)
- Roadmap target: `docs/roadmap.md` Phase 3 (docs)

## Question
Can the API reference be generated reproducibly (pdoc) and published to GitHub Pages on each push?

## Evidence

### Correctness / leakage / parity
No library change. `scripts/build_docs.sh` builds the reference with `pdoc` into `site/` — verified
locally with `pdoc>=14`: produced `index.html`, `catstat.html`, and `search.js` with **no import
errors** (pdoc skips the private `_gpu` module, so there is no RAPIDS import on the CPU box).
`docs.yml` parses (`yaml.safe_load`); the green gate is unaffected.

### Performance (≥5 reps, median + spread)
N/A — no compute path changed.

## Decision
**KEEP** — a pure build script (errors with a hint if `pdoc` is missing; makes the src-layout
package importable via `PYTHONPATH`) plus a standard Pages workflow (build → `upload-pages-artifact@v3`
→ `deploy-pages@v4`) with a `pages` concurrency group and Pages/OIDC permissions. The README already
links the site. No invariant, default, or public API touched.

## Follow-ups
- **Maintainer (one-time):** enable Pages — repo Settings → Pages → Source: **GitHub Actions**.
  Until then the `deploy` job fails while `build` succeeds; the first push after enabling deploys to
  https://matapanino.github.io/catstat/.
- Commit 4: scikit-learn estimator-check hardening (tags + a documented `check_estimator` subset).
