# Verdict: <short title> (<PR/branch>)

- Date: YYYY-MM-DD
- Branch: `<branch>`
- Backend: cpu | gpu (T4 via Colab dev loop)
- Artifacts:
  - `benchmarks/results/<run>.jsonl` (AFTER)
  - `benchmarks/results/baseline-cpu.json` (BEFORE)
  - (parity) `benchmarks/results/<parity>.jsonl`
- Roadmap target: `docs/roadmap.md` <tier/PR> · Related: `docs/known_issues.md` <KI-id>

## Question
<The single question this change answers. What did we expect to move, and is it worth keeping?>

## Evidence

### Correctness / leakage / parity
<Test results. OOF reconstruction exact? noise-trap correlation ≈ 0? sklearn-parity within tol?
CPU/GPU allclose? State pass/fail with the numbers.>

### Performance (≥5 reps, median + spread)
| case | metric | before | after | delta |
|------|--------|-------:|------:|------:|
| <case> | fit_s | | | |
| <case> | transform_s | | | |
| <case> | fit_transform_s | | | |
| <case> | peak_mem_mb | | | |

## Decision
**KEEP | CHANGE-DEFAULT | REVERT** — <one-paragraph justification tied to the evidence and the
acceptance thresholds. If changing a committed baseline, say so explicitly and why.>

## Follow-ups
- <next task, new known-issue, or experiment-log entry to append>
