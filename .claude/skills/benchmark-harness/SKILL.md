---
name: benchmark-harness
description: >-
  Run catstat's benchmark harness reproducibly, persist results to the ledger, compare against the
  committed baseline, and draft a verdict. Invoke for any perf-relevant change, to establish or
  refresh a baseline, or as the benchmark step of the self-improvement loop. Enforces >=5 reps,
  median+spread, pinned seeds/versions/SHA, and never changes a default by itself. Outputs a
  before/after table and a filled docs/verdicts/ entry.
---

You run the **measurement harness** and turn numbers into a decision. You never change a default
on your own — that requires a written verdict backed by repeated runs.

## When to use
- Any perf-relevant change (backend, group-by, dispatch, conversion path).
- Establishing or refreshing a committed baseline.
- The benchmark step of the self-improvement loop.

## When NOT to use
- Correctness-only changes (use `leakage-audit` / `sklearn-compat`).
- To justify a default change from a single run or a microbenchmark presented as end-to-end.

## Required inputs
- `--size {small,medium,large}`, `--backend {cpu,gpu}`, `--reps N` (≥5), and a committed baseline
  JSON to compare against.

## Commands
```bash
python3 benchmarks/run_benchmarks.py --backend cpu --reps 5 --out benchmarks/results/<run>.jsonl
python3 benchmarks/compare_results.py benchmarks/results/<run>.jsonl benchmarks/results/baseline-cpu.json
python3 scripts/summarize_benchmark_results.py benchmarks/results/<run>.jsonl
```
GPU runs go through `scripts/colab_gpu_parity.sh` (Phase 2), not local.

## Files to inspect
`benchmarks/datasets.py`, `benchmarks/run_benchmarks.py`, `benchmarks/ledger.py`,
`benchmarks/compare_results.py`, `benchmarks/results/baseline-cpu.json`,
`docs/verdicts/TEMPLATE-verdict.md`, and the harness design `docs/proposals/evaluation-harness-design.md`.

## Failure modes to catch
- Fewer than 5 reps; reporting mean without spread.
- Comparing across different seeds, package versions, or git SHAs.
- Reporting a microbenchmark win as an end-to-end win.
- Updating the committed baseline without a verdict.
- Bundling a harness change with a behavior change in one diff (result becomes un-attributable —
  keep harness changes in a separate commit).
- Timing only `fit_transform` wall time without separating `fit` / `transform` / conversion.

## Final report format
A before/after table (per case: `fit_s`, `transform_s`, `fit_transform_s` as median+spread, peak
memory), regressions/improvements vs the §8 thresholds (harness doc), parity status if relevant,
and a filled `docs/verdicts/YYYY-MM-DD-<topic>-verdict.md` with a keep/change/revert decision.
Update the committed baseline **only** if the verdict says so.
