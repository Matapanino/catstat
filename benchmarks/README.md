# `catstat` benchmarks

> Status: **contract only** — no harness code exists yet. This README is the spec the harness is
> built to (design: `docs/proposals/evaluation-harness-design.md`).

The harness is a first-class part of the library: it powers correctness/leakage/parity testing and
the self-improvement loop, not just one-off timing.

## Planned layout
```
benchmarks/
  README.md            # this file
  datasets.py          # seeded synthetic generators (FIRST file to build)
  run_benchmarks.py    # CLI: --size {small,medium,large} --backend {cpu,gpu} --reps N --out PATH
  ledger.py            # append-only JSONL writer + schema
  compare_results.py   # diff a run vs a baseline JSON; non-zero exit on regression
  results/
    .gitkeep
    baseline-cpu.json  # committed; updated only via a docs/verdicts/ entry
```

## How to run (once built)
```bash
python3 benchmarks/run_benchmarks.py --backend cpu --reps 5 --out benchmarks/results/<run>.jsonl
python3 benchmarks/compare_results.py benchmarks/results/<run>.jsonl benchmarks/results/baseline-cpu.json
```
GPU runs go through `scripts/colab_gpu_parity.sh` (Phase 2) — there is no local GPU.

## Rules
- **≥5 reps**, report median + spread; never change a default on one run or a microbenchmark.
- Record git SHA + package versions in every ledger row; never compare across mismatched versions.
- Separate `fit` / `transform` / `fit_transform` / conversion time — never just end-to-end.
- The committed baseline changes **only** via a verdict in `docs/verdicts/`.
- Correctness/leakage/parity regressions are absolute blockers; perf gate at the thresholds in
  `docs/proposals/evaluation-harness-design.md` §8.

## Stress cases (`datasets.py`)
high cardinality · rare/singleton categories · missing (NaN) · unseen-at-transform · regression ·
binary (incl. imbalanced) · multiclass (many/imbalanced classes) · multi-column with interaction
(for `combination` mode) · **leakage trap** (category independent of `y`) · mixed dtypes
(object/category/int). Each is seeded and carries analytic ground-truth statistics for exact
correctness assertions.
