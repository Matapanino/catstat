# `catstat` — Self-Improvement Loop Design

> Status: **proposal**. Covers design section 13. Defines how *future* Claude Code sessions move
> `catstat` forward in disciplined, small, reversible steps — without re-discovering context or
> repeating failed experiments. Inspired by the repleafgbm perf/research loops.

## 0. Why this exists
Multi-session projects waste their budget two ways: (1) **re-deriving context** already written
down, and (2) **re-running experiments** that already failed. This loop fixes both with a fixed
read order, an append-only experiment ledger, and a hard rule that every change ends in a written
verdict and a roadmap update.

## 1. The loop (one iteration = one PR-sized change)
1. **Read current state** (in this order, stop when you have enough):
   `CLAUDE.md` → `docs/roadmap.md` → `docs/known_issues.md` → `docs/experiment_log.md` →
   newest 1–2 files in `docs/verdicts/` → `benchmarks/results/baseline-*.json`.
2. **Inspect prior logs** — check `experiment_log.md` for anything already tried (esp. *null/
   negative* results) so you don't repeat it.
3. **Choose the next improvement** — highest ROI from `roadmap.md` "next" + `known_issues.md`,
   skipping anything the log marks as tried-and-failed. Prefer small, high-confidence steps.
4. **Establish the baseline** — run `scripts/check.sh` (must be green to start) and, for
   perf-relevant work, `run_benchmarks.py` to capture the *before* numbers.
5. **Implement one small change** — a single PR-sized diff with a tight scope; add/adjust tests.
6. **Run tests + benchmarks** — `scripts/check.sh` green; `run_benchmarks.py` for the *after*.
7. **Compare with baseline** — `compare_results.py after.jsonl baseline.json`. Correctness/leakage
   regressions are absolute blockers; perf deltas judged against §8 thresholds of the harness doc.
8. **Write a verdict** — `docs/verdicts/YYYY-MM-DD-<topic>-verdict.md` from the template:
   question, evidence (before/after, ≥5 reps, median+spread), parity status, keep/change/revert.
9. **Update the ledgers** — append the outcome (incl. nulls) to `experiment_log.md`; move the item
   in `roadmap.md`; add/clear entries in `known_issues.md`. Update the committed baseline JSON
   **only** if the verdict says so.
10. **Stop clean** — end with a short summary and a single suggested next task. Do not start a
    second change in the same session unless asked.

## 2. Decision rules
- **Keep a change** only if: all correctness/leakage/parity tests pass **and** the verdict shows a
  real, repeated improvement (or it's a correctness/feature win with no perf regression beyond
  threshold).
- **Revert** if a perf gain costs correctness, parity, or readability — every time.
- **Never change a default** (smoothing, `cv`, `auto` thresholds, fallbacks) without a verdict
  backed by ≥5 reps across more than one dataset/seed.
- **Leakage invariant is non-negotiable.** Any diff touching `_cross_fit.py`, `_smoothing.py`, or
  the OOF path requires the `leakage-audit` skill to sign off (§ skills doc).

## 3. When to use subagents
- **Use** a research subagent to survey an external method/library *before* implementing something
  new (target: a dated note, not code) — analogous to repleafgbm's `literature-scout`.
- **Use** a runner subagent for long/detached multi-seed benchmark runs so the main session keeps
  its context for reasoning (`experiment-runner` analog).
- **Use** an analysis subagent to turn a finished run into a verdict (`results-analyst` analog).
- **Fan out** independent benchmarks in parallel; aggregate with one analysis pass.
- *(Whether these live as `.claude/agents/` files or are spun ad-hoc is a project choice; the skills
  set in [`skills-proposal.md`](./skills-proposal.md) is the MVP mechanism.)*

## 4. When NOT to use subagents
- Small, local edits where you already hold the context (most MVP PRs) — just do it inline.
- Anything needing the *current* in-context state of files you've already read — a cold subagent
  re-derives it (the expensive path).
- Final correctness judgment on leakage — that stays with the main session + `leakage-audit`.

## 5. When to research externally
- Before adding a new statistic's smoothing rule, a new encoder variant (LOO/CatBoost/GLMM), or a
  GPU primitive — confirm the method against primary sources (sklearn/cuML/category_encoders
  source, papers) and write a dated note. **Don't** browse to re-confirm something already
  captured in the proposal docs or a prior note.

## 6. Artifact & decision preservation
- **Never delete** benchmark cases or results; the ledger is append-only.
- **Every** default change, kept or rejected, leaves a verdict in `docs/verdicts/`.
- **Every** experiment outcome (incl. "no effect") goes in `experiment_log.md` so it's never
  retried blindly.
- Raw run artifacts live under `benchmarks/results/` (and, for GPU, are pulled back from Colab);
  committed baselines are the only ones updated in place, and only via a verdict.

## 7. Separation of modes (and their gates)
| mode | what it touches | gate to finish |
|---|---|---|
| **research** | `docs/` notes only | a dated note; no code |
| **planning** | `docs/proposals/`, `roadmap.md` | reviewed proposal |
| **implementation** | `src/`, `tests/` | `scripts/check.sh` green + tests added |
| **benchmarking** | `benchmarks/`, `docs/verdicts/` | verdict written; baseline updated only if warranted |
| **release** | version, `CHANGELOG`, docs, PyPI | green gate + reviewed `CHANGELOG` + tag |

Keep these in **separate commits**. Harness/benchmark changes never ride along with `src/` changes
(a measurement change and a behavior change in one diff make the result un-attributable).

## 8. Concrete artifacts this loop relies on
- `docs/roadmap.md` — what's done / next / later (honest status).
- `docs/known_issues.md` — current limitations and bugs, with severity.
- `docs/experiment_log.md` — append-only ledger of every experiment + outcome (incl. nulls).
- `docs/verdicts/YYYY-MM-DD-<topic>-verdict.md` — one per kept/rejected change.
- `benchmarks/results/*.json` / `*.jsonl` — raw + baseline numbers.
- `CLAUDE.md` rules — the operating contract that makes step 1's read order reliable.

## 9. Anti-patterns (how the loop fails)
- Starting to code before reading `experiment_log.md` (⇒ repeating a failed idea).
- Changing a default on a single run, or on a microbenchmark presented as end-to-end.
- Bundling a harness change with a behavior change (un-attributable result).
- "Improving" something `known_issues.md` already marks as intentionally deferred.
- Leaving an iteration without a verdict + roadmap update (the next session re-discovers it).
