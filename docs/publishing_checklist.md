# Publishing checklist (PyPI)

Steps to cut a `catstat` release. Run from a clean `main`.

## Pre-flight
- [ ] `bash scripts/check.sh` is green (ruff + pytest + examples).
- [ ] Coverage ≥ floor: `PYTHONPATH=src python3 -m pytest --cov=catstat`.
- [ ] `docs/roadmap.md`, `docs/known_issues.md`, and `CHANGELOG.md` are up to date and honest.
- [ ] Version bumped in **both** `pyproject.toml` (`project.version`) and
      `src/catstat/__init__.py` (`__version__`) — they must match.
- [ ] `LICENSE`, `README.md`, `py.typed` present; `pyproject` metadata/classifiers/URLs correct.

## Build
```bash
python3 -m pip install --upgrade build twine
python3 -m build                      # -> dist/catstat-X.Y.Z{.tar.gz,-py3-none-any.whl}
python3 -m twine check dist/*         # metadata + README render check
```
- [ ] Both sdist and wheel build; `twine check` passes.
- [ ] Smoke-install the wheel in a fresh venv and import:
      `pip install dist/catstat-*.whl && python -c "import catstat; print(catstat.__version__)"`.

## Publish
```bash
python3 -m twine upload --repository testpypi dist/*   # rehearse on TestPyPI first
python3 -m twine upload dist/*                          # real PyPI (needs API token)
```
- [ ] Install from (Test)PyPI in a clean venv and run an example.

## Tag
```bash
git tag -a v0.1.0 -m "catstat 0.1.0"
git push origin main --tags
```
- [ ] GitHub release created from the tag with the CHANGELOG section as notes.

## Post-release
- [ ] Bump to the next dev version; add an `## [Unreleased]` CHANGELOG section.
- [ ] Note the release in `docs/experiment_log.md`.

> Credentials (PyPI token, GitHub auth) are the maintainer's — Claude prepares artifacts and
> verifies the build, but does not upload.
