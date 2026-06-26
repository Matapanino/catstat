---
name: release-prep
description: >-
  Prepare a catstat PyPI release: confirm the green gate, bump the version in both pyproject.toml
  and __init__.py (kept in sync), update CHANGELOG.md, build sdist+wheel, and run twine check +
  a clean-venv smoke install. Follows docs/publishing_checklist.md. Prepares and verifies artifacts
  only — never uploads or tags (the maintainer holds PyPI/GitHub credentials).
---

You prepare a release and **verify it builds and installs**, stopping short of publishing.

## When to use
- Cutting a new `catstat` version (after features have landed and the gate is green).

## When NOT to use
- Mid-development; or to change library behavior (that's a normal PR, not a release).

## Required inputs
- The target version (SemVer). Confirm what changed since the last tag (read `git log` + the
  unreleased CHANGELOG section).

## Steps / commands
1. `bash scripts/check.sh` must be green; coverage ≥ floor.
2. Bump version in **both** `pyproject.toml` `project.version` and `src/catstat/__init__.py`
   `__version__` — they MUST match (a mismatch is a release bug).
3. Move the CHANGELOG `[Unreleased]` items under a dated `[X.Y.Z]` heading; keep it honest
   (Added / Changed / Fixed / Known limitations).
4. Build + verify:
   ```bash
   python3 -m build
   python3 -m twine check dist/*
   python3 -m pip install dist/catstat-*.whl   # in a fresh venv
   python3 -c "import catstat; print(catstat.__version__)"
   ```
5. Hand off: report that artifacts are ready; the maintainer runs `twine upload` + `git tag`.

## Files to inspect
`pyproject.toml`, `src/catstat/__init__.py`, `CHANGELOG.md`, `docs/publishing_checklist.md`,
`README.md`, `LICENSE`.

## Failure modes to catch
- Version mismatch between pyproject and `__init__`.
- `twine check` warnings (README won't render, missing metadata).
- Wheel missing `py.typed` or the package data.
- Building from a dirty tree (uncommitted changes) or a non-green gate.
- Uploading from this environment (don't — no credentials; it's the maintainer's step).

## Final report format
Version old→new, the CHANGELOG section, `twine check` result, the smoke-install import line, and an
explicit "ready to `twine upload` + tag vX.Y.Z" hand-off (or the blocking issue).
