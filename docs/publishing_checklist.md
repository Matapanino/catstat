# Publishing checklist (PyPI)

Steps to cut a `catstat` release. Run from a clean `main`. The routine path is **tag-driven**: a
`v*` tag push triggers `.github/workflows/release.yml`, which builds and publishes to PyPI via
**Trusted Publishing** (OIDC — no stored token). A manual `twine` fallback is documented below.

## Pre-flight
- [ ] `bash scripts/check.sh` is green (ruff + pytest + examples).
- [ ] Coverage ≥ floor: `PYTHONPATH=src python3 -m pytest --cov=catstat`.
- [ ] `docs/roadmap.md`, `docs/known_issues.md`, and `CHANGELOG.md` are up to date and honest.
- [ ] Version bumped in **both** `pyproject.toml` (`project.version`) and
      `src/catstat/__init__.py` (`__version__`) — they must match (the workflow re-checks this).
- [ ] The `CHANGELOG.md` `## [X.Y.Z]` heading is finalized (replace `— unreleased` with the date).
- [ ] `LICENSE`, `README.md`, `py.typed` present; `pyproject` metadata/classifiers/URLs correct.

## Build & verify (local — recommended before tagging)
```bash
python3 -m pip install --upgrade build twine
python3 -m build                      # -> dist/catstat-X.Y.Z{.tar.gz,-py3-none-any.whl}
python3 -m twine check dist/*         # metadata + README render check
```
- [ ] Both sdist and wheel build; `twine check` passes.
- [ ] Smoke-install the wheel in a fresh venv and import:
      `pip install dist/catstat-*.whl && python -c "import catstat; print(catstat.__version__)"`.

> The `release-prep` skill automates this build + `twine check` + clean-venv smoke install.

## Release (automated — preferred)
Push a version tag; GitHub Actions builds and publishes to PyPI for you.
```bash
git tag -a vX.Y.Z -m "catstat X.Y.Z"
git push origin main
git push origin vX.Y.Z
```
- The `release.yml` workflow verifies the tag matches the package version, builds sdist + wheel,
  runs `twine check`, then publishes to PyPI with OIDC (no token).
- [ ] Workflow run is green; the new version appears on https://pypi.org/project/catstat/.
- [ ] Create a GitHub release from the tag with the `CHANGELOG.md` section as the notes.
- [ ] Install from PyPI in a clean venv and run an example.

### One-time: configure the PyPI Trusted Publisher
On PyPI → project `catstat` → **Manage → Publishing** → add a GitHub publisher:
- Owner `Matapanino`, repository `catstat`, workflow `release.yml`, environment `pypi`.
- If `catstat` is not on PyPI yet, register a **pending publisher** (same form) before the first
  upload, or do the first upload manually (below) and add the publisher afterwards.

## Manual publish (fallback / first upload)
If automation is unavailable — or for the initial `0.1.0` upload built from the `v0.1.0` tag:
```bash
python3 -m twine upload --repository testpypi dist/*   # rehearse on TestPyPI first
python3 -m twine upload dist/*                          # real PyPI (needs an API token)
```
- [ ] Install from (Test)PyPI in a clean venv and run an example.

## Post-release
- [ ] Bump to the next version; open a new `## [Unreleased]` (or `## [X.Y.Z] — unreleased`)
      CHANGELOG section.
- [ ] Note the release in `docs/experiment_log.md`.

> Credentials are the maintainer's: the maintainer holds the PyPI account, configures the Trusted
> Publisher, and runs any manual `twine upload`. Claude prepares and verifies artifacts and the
> automation, but never uploads or tags.
