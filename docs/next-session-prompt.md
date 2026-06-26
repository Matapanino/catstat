# Next-session kickoff prompt — release polish

Paste the block below to start the next Claude Code session. It assumes `catstat` is at **0.1.0**,
published on GitHub (Matapanino/catstat), with the **PyPI upload still pending** the maintainer's
token. Goal: polish toward a clean, credible public release — not new encoding features.

---

You are continuing development of **`catstat`** (`~/dev/masaTE`) — a scikit-learn-compatible library
for **leakage-safe statistical categorical encoding** across CPU (pandas/numpy) and optional GPU
(cuDF/CuPy). It is feature-complete at **v0.1.0** and already published on GitHub
(`Matapanino/catstat`, tag `v0.1.0` + release). The **PyPI upload is the only remaining publish step
and is the maintainer's** (their token; this sandbox can't do interactive `twine` auth).

**Your goal this session: raise catstat to a polished, credible public release** — release
automation, docs, README, and scikit-learn-compliance hardening. This is *release quality*, not new
encoding behavior.

## Read first (in this order — this bootstraps your context; do not skip)
1. `CLAUDE.md` — the operating contract: invariants, code map, the one-command green gate, workflows.
2. `docs/roadmap.md` — current status and the "Next" pointer.
3. `docs/known_issues.md` — open items (esp. KI-010 sklearn-auto parity, KI-018/KI-020 GPU).
4. `docs/experiment_log.md` + the newest `docs/verdicts/*` — what's been decided and *why* (note
   especially **why `auto`-GPU is disabled** — KI-020 — and do not undo it without new evidence).
5. `benchmarks/results/baseline-cpu.json` — the committed CPU perf baseline.

Then run `bash scripts/check.sh` to confirm you start green (expect **~88 passed, 2 GPU-skipped**).

## Non-negotiable invariants (breaking any one is a regression — re-read CLAUDE.md)
- **Leakage safety**: `fit_transform` is out-of-fold / loo / ordered; `smooth="auto"` variance is
  computed **per fold**; fold assignment flows only through `random_state` (never the global numpy
  RNG).
- **CPU/GPU parity** at allclose: catstat **owns fold assignment** — never delegate it to a backend.
- **Smoothing honesty rule**: only mean/probability are smoothed; every other statistic falls back
  to the global value for small/unseen categories and **never blends**.
- **Public API stability**: 0.1.0 is released. Make **additive, backward-compatible** changes only;
  bump the version per **SemVer** and update `CHANGELOG.md` for anything user-visible.
- **Do not re-enable `auto`-GPU** (`_AUTO_GPU_ENABLED` in `backends/_dispatch.py`) without a fresh
  Colab crossover verdict showing GPU actually wins. **Never change a default without a
  results-backed verdict.**

## Prioritized work (PR-sized; do in order; one verdict per change)
1. **Release automation** — add `.github/workflows/release.yml` that builds the sdist+wheel and
   publishes to PyPI **on a `v*` tag via PyPI Trusted Publishing (OIDC)** — no stored token. Update
   `docs/publishing_checklist.md` to say the maintainer configures the trusted publisher on PyPI
   once; thereafter `git tag vX.Y.Z && git push --tags` ships it. (The pending **0.1.0** upload is
   separate — the maintainer does it manually this once, or re-runs the workflow.)
2. **README polish** — badges (CI status, PyPI version, supported Python, license), a tight
   quickstart, a feature/stat table, and a short "leakage-safe by design" section. Stay **honest
   about GPU** (validated for parity, not yet faster → `auto` is CPU; explicit `backend="gpu"`
   available).
3. **API docs** — `scripts/build_docs.sh` using `pdoc` (the `docs` extra) + a GitHub Pages workflow
   to publish, linked from the README. Ensure public docstrings render well.
4. **sklearn-compat hardening** — add a `check_estimator`-subset test: run the checks that apply to a
   supervised, multi-output transformer; `xfail`/skip the rest with a one-line reason each. Fix any
   cheap compliance gaps. Document the supported subset (closes part of KI-012).
5. **Project hygiene** — `CONTRIBUTING.md` (point at `bash scripts/check.sh`), `SECURITY.md`, issue/
   PR templates.
6. **(Optional, larger — only after 1–5)** KI-020 GPU perf: prototype keeping binned keys + fold-ids
   **on-device** to remove the per-fold host↔device round-trips, re-run `scripts/colab_gpu_parity.sh`
   on a T4, and only then consider re-enabling `auto`-GPU **with a new crossover verdict**.

## How to work here (reiterated from CLAUDE.md + the self-improvement loop)
- One small change at a time. Finish each with `bash scripts/check.sh` **green** and coverage ≥ the
  pyproject floor.
- End every change with: a `docs/verdicts/YYYY-MM-DD-<topic>-verdict.md`; updates to
  `docs/roadmap.md` + `docs/experiment_log.md`; a SemVer + `CHANGELOG.md` bump if user-visible; and a
  commit whose message ends with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Push to
  `origin main`.
- Keep **harness/CI changes in their own commit**, separate from `src/` behavior changes.
- **GPU runs only via `scripts/colab_gpu_parity.sh`** (no local GPU; the user runs it and pastes
  output). **Never attempt an interactive `twine upload`** in the sandbox — it can't prompt; the
  token is the maintainer's.
- Use a subagent **only** for a long detached benchmark or an external-docs lookup; otherwise work
  inline (you already hold the context). See CLAUDE.md "When to / not to use subagents".

## Environment facts
- Local: macOS, CPU-only. **scikit-learn 1.2** locally (no `TargetEncoder` → sklearn-parity tests
  skip; they need ≥1.4). The built wheel installs/imports cleanly on sklearn **1.9**.
- `gh` is authenticated as **Matapanino**; `git remote origin` → `github.com/Matapanino/catstat`.
- `dist/` is gitignored. `polars` is installed locally (so `output="polars"` tests run).

## First action
Read the loop docs above, confirm green, then implement the single highest-ROI change (start with
task 1 or 2), prove it green, write its verdict, update the ledgers, commit + push, and **stop with a
one-paragraph summary and the next suggested task**. Keep the public repo's docs accurate at every
step.
