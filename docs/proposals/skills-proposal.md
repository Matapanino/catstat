# `catstat` — Skills Proposal

> Status: **proposal**. Covers design section 15. Goal: a **small, high-signal** skill set that
> removes repeated reasoning — *not* a skill per task. Skills live in `.claude/skills/<name>/SKILL.md`.
> **Build the first 3 now; defer the rest** until the work that needs them exists.

## The set (and build order)
| # | skill | build | one-line purpose |
|---|---|---|---|
| 1 | `leakage-audit` | **now** | Prove `fit_transform` is out-of-fold and no target leaks via any path. |
| 2 | `sklearn-compat` | **now** | Verify estimator-protocol / Pipeline / set_output / feature-name compliance. |
| 3 | `benchmark-harness` | **now** | Run the harness, persist results, compare to baseline, draft a verdict. |
| 4 | `cpu-gpu-parity` | Phase 2 | Drive the Colab loop; assert CPU/GPU allclose parity. |
| 5 | `target-encoder-research` | as needed | Standardized external lookup (sklearn/cuML/category_encoders/papers) → dated note. |
| 6 | `release-prep` | Phase 3 | Version bump, CHANGELOG, docs, PyPI checklist. |
| 7 | `target-encoder-implementation` | optional | Scaffolding helper for a new stat/encoder following the invariants. |

Why only 3 now: the MVP is CPU correctness. **Leakage** is the dominant risk (skill 1), **sklearn
compatibility** is the compatibility burden the user called out (skill 2), and the **harness**
powers the self-improvement loop (skill 3). The rest map to work that doesn't exist yet (GPU,
release) — adding them early would be dead weight.

---

## Skill specs

### 1. `leakage-audit` (build now)
- **Purpose:** Establish, with evidence, that no target information leaks into a `fit_transform`
  output, and that the OOF contract holds after any change to the cross-fit / smoothing path.
- **When to use:** any diff touching `_cross_fit.py`, `_smoothing.py`, `_base.py`'s transform path,
  or fold assignment; before keeping any such change.
- **When NOT to use:** pure docs/benchmark/naming changes; unsupervised `Count/Frequency` logic
  (no target ⇒ run only the "fit_transform == fit().transform()" check).
- **Required inputs:** the diff/PR scope; a seeded dataset with known signal and a noise-trap.
- **Commands:** `PYTHONPATH=src python3 -m pytest tests/test_cross_fit_no_leakage.py -q`;
  ad-hoc: recompute each fold's encoding from its complement and assert equality.
- **Files to inspect:** `_cross_fit.py`, `_smoothing.py`, `_base.py`, `tests/test_cross_fit_no_leakage.py`.
- **Failure modes to catch:** per-fold stats that secretly include the held fold; auto-smoothing
  variance computed on full data; row-order scrambled on merge; unknown fallback drawn from the
  transformed set; using `fit().transform()` on train in an example.
- **Final report format:** PASS/FAIL + the OOF-reconstruction result + noise-trap correlation +
  which traps were checked; on FAIL, the exact offending line/path.

### 2. `sklearn-compat` (build now)
- **Purpose:** Confirm `catstat` behaves as a well-mannered sklearn transformer.
- **When to use:** changes to the public classes, constructor params, fitted attrs, feature names,
  or output handling; before a release.
- **When NOT to use:** internal backend/perf changes with no API surface effect.
- **Required inputs:** the class under test; installed `scikit-learn` (note version; full
  `check_estimator` only meaningfully on ≥1.4).
- **Commands:** `PYTHONPATH=src python3 -m pytest tests/test_sklearn_compat.py -q`; spot-checks:
  `clone`, `get_params`/`set_params`, `Pipeline`, `ColumnTransformer`, `set_output("pandas")`,
  `get_feature_names_out`.
- **Files to inspect:** `target_encoder.py`, `_base.py`, `_feature_names.py`, the compat test.
- **Failure modes to catch:** params not stored verbatim on `__init__`; fitted attrs missing
  trailing underscore; `get_feature_names_out` length ≠ output width; `set_output` not honored;
  silent failure inside `ColumnTransformer` (cuML's historical bug).
- **Final report format:** PASS/FAIL per check + the sklearn version + the **documented** subset of
  `check_estimator` that is (in)applicable and why.

### 3. `benchmark-harness` (build now)
- **Purpose:** Run benchmarks reproducibly, persist to the ledger, compare to baseline, and draft a
  verdict — the engine of the self-improvement loop.
- **When to use:** any perf-relevant change; establishing/refreshing a baseline; the benchmark step
  of the loop.
- **When NOT to use:** correctness-only changes (use the test skills); never to change a default by
  itself (that needs a verdict).
- **Required inputs:** target `--size`/`--backend`/`--reps`; a committed baseline JSON.
- **Commands:** `python3 benchmarks/run_benchmarks.py --backend cpu --reps 5 --out
  benchmarks/results/<run>.jsonl`; `python3 benchmarks/compare_results.py <run>.jsonl
  benchmarks/results/baseline-cpu.json`; `python3 scripts/summarize_benchmark_results.py`.
- **Files to inspect:** `benchmarks/{datasets,run_benchmarks,ledger,compare_results}.py`,
  `benchmarks/results/baseline-cpu.json`, `docs/verdicts/TEMPLATE-verdict.md`.
- **Failure modes to catch:** <5 reps; comparing across different seeds/versions/SHA; reporting a
  microbenchmark as end-to-end; updating the committed baseline without a verdict; bundling a
  harness change with a behavior change.
- **Final report format:** before/after table (median+spread), regressions/improvements vs
  threshold, and a filled `docs/verdicts/YYYY-MM-DD-<topic>-verdict.md`.

---

## Deferred skill stubs (one line each; flesh out when needed)
- **`cpu-gpu-parity` (P2):** pack tree → `colab new --gpu T4` → `colab exec` parity+bench → pull
  artifacts → assert allclose → verdict. Watchdog the `colab exec`.
- **`target-encoder-research` (as needed):** read primary sources (sklearn/cuML/category_encoders
  source, papers) → dated note in `docs/`; don't re-confirm what proposals already capture.
- **`release-prep` (P3):** green gate → version bump → CHANGELOG → docs build → PyPI checklist.
- **`target-encoder-implementation` (optional):** scaffold a new stat/encoder honoring the
  invariants (registry entry + smoothing policy + fallback + tests).

## Anti-patterns
- A skill per file or per test (skills should remove *reasoning*, not narrate commands).
- Skills that duplicate `scripts/check.sh` — the gate is one command; skills add judgment around it.
- Building Phase 2/3 skills before the code they audit exists.
