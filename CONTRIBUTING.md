# Contributing to catstat

Thanks for your interest in `catstat`! It's a small, focused library; contributions that keep it
honest, leakage-safe, and well-tested are very welcome.

## Development setup

```bash
python -m pip install -e ".[dev]"   # tests + lint + build; add ,gpu / ,polars / ,docs as needed
```

## The green gate (run before every PR)

```bash
bash scripts/check.sh               # ruff + pytest + runnable examples
```

This is the single source of truth for "is it green?". CI runs the same checks on Python
3.10–3.12. New behavior needs tests: encode correctness, **out-of-fold / no-leakage**,
unknown/missing fallback, feature names, and determinism.

- GPU and CPU↔GPU parity tests carry the `gpu` marker and are skipped without RAPIDS; they run on
  Colab via `scripts/colab_gpu_parity.sh`.
- The scikit-learn `check_estimator` subset test needs scikit-learn ≥ 1.6.
- Build the API docs locally with `bash scripts/build_docs.sh`.

## Non-negotiable invariants

Please read [`CLAUDE.md`](CLAUDE.md) before changing behavior. In short:

- **Leakage safety** — `fit_transform` is out-of-fold; `smooth="auto"` variance is computed per
  fold; folds flow only through `random_state` (never the global numpy RNG).
- **Smoothing honesty** — only mean/probability statistics are smoothed; the rest fall back to the
  global value for small/unseen categories and never blend.
- **CPU/GPU parity** at allclose; catstat owns its fold assignment.
- **Public API stability** — additive, backward-compatible changes only; bump SemVer and update
  `CHANGELOG.md` for anything user-visible.

## Pull requests

- Keep changes PR-sized and focused — one logical change per PR.
- The green gate passes and coverage stays ≥ the configured floor (85%).
- Update `CHANGELOG.md` for user-visible changes, and `docs/` where relevant.
- Write clear commit messages.

## Reporting bugs / requesting features

Open a GitHub issue using the templates. For security issues, see [`SECURITY.md`](SECURITY.md).
