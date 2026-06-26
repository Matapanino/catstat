# Verdict: Release automation — tag-driven PyPI Trusted Publishing (main)

- Date: 2026-06-26
- Branch: `main`
- Backend: cpu (CI/infra change — no compute path touched)
- Artifacts:
  - `.github/workflows/release.yml` (new)
  - `docs/publishing_checklist.md` (rewritten: automated path + manual fallback)
  - `pyproject.toml` / `src/catstat/__init__.py` (version 0.1.0 → 0.1.1)
  - `CHANGELOG.md` (`## [0.1.1]` opened)
- Roadmap target: `docs/roadmap.md` Phase 3 (release) · Related: `docs/publishing_checklist.md`

## Question
Can the manual `twine upload` release step be replaced by a tag-driven, tokenless PyPI publish, and
is the workflow safe to keep on `main`?

## Evidence

### Correctness / leakage / parity
No library behavior changed — this is CI/infra plus a version string. Green gate is green before and
after the bump: `bash scripts/check.sh` → "All checks passed" (88 passed / 2 GPU-skipped). Both
workflow YAML files parse (`yaml.safe_load`). 0.1.1 packages cleanly: `python -m build` produced
`catstat-0.1.1.tar.gz` + `catstat-0.1.1-py3-none-any.whl`, and `twine check` **PASSED** on both.

### Release workflow design
- **Trigger:** push of a `v*` tag. The `build` job verifies the tag equals the `pyproject` *and*
  `__init__` version (fails the run on mismatch), builds sdist+wheel, runs `twine check`, and
  uploads the `dist/` artifact.
- **Publish:** the `publish` job (`needs: build`) uses `pypa/gh-action-pypi-publish@release/v1` with
  `permissions: id-token: write` + `environment: pypi` — OIDC **Trusted Publishing, no stored
  token**. Least-privilege: top-level `contents: read`; `id-token` scoped to the publish job only.

### Performance (≥5 reps, median + spread)
N/A — no compute path changed.

## Decision
**KEEP** — the workflow is additive, off by default (fires only on a `v*` tag), tokenless, and
guarded against a tag/version mismatch. It touches no core invariant, default, or backend threshold.
Per the maintainer's decision, the post-tag-v0.1.0 polish is versioned **0.1.1**; v0.1.0 keeps its
separate manual upload built from the existing `v0.1.0` tag.

## Follow-ups
- **Maintainer (one-time, outside the sandbox):** add a PyPI Trusted Publisher for `catstat`
  (owner `Matapanino`, repo `catstat`, workflow `release.yml`, environment `pypi`); use a *pending
  publisher* if `catstat` is not yet on PyPI. Then `git tag v0.1.1 && git push origin v0.1.1`
  publishes 0.1.1 and exercises the pipeline end-to-end.
- Next polish commit: README (badges, honest status, install, stat/feature table, API-docs link).
