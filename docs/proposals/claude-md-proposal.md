# `catstat` — CLAUDE.md Proposal (rationale)

> Status: **proposal**. Covers design section 14. The repo had **no existing `CLAUDE.md`**, so this
> proposes a **full initial version**, which is committed live at [`/CLAUDE.md`](../../CLAUDE.md).
> This doc explains *why* each section exists; the live file is the source of truth.

## Why a CLAUDE.md at all
`catstat` is a multi-session, harness-driven project. CLAUDE.md is the operating contract that lets
any session start correct from cold: what the project is, the invariants it must never break, where
the code lives, the one command that proves green, which skills to invoke, and what *not* to touch
without explicit instruction. It mirrors the proven structure of the sibling repo
`repleafgbm/CLAUDE.md`.

## Section-by-section rationale (maps to the live file)
1. **Project summary** — one paragraph so a cold session knows the thesis (unified CPU/GPU
   statistical categorical encoding, leakage-safe).
2. **Core invariants** — the three things that must always hold:
   - **Leakage safety:** `fit_transform` is out-of-fold; `fit().transform()` on train is the leaky
     path; auto-smoothing variance is per-fold; never touch global numpy RNG.
   - **CPU/GPU parity:** `catstat` owns fold assignment so both devices produce the same OOF
     encodings; parity is asserted at allclose (not bitwise).
   - **sklearn compatibility:** `BaseEstimator`/`TransformerMixin`, `get_feature_names_out`,
     `set_output`, Pipeline/ColumnTransformer.
   These are the "do not break" list and are called out again under "Do not change without
   instruction."
3. **Smoothing honesty rule** — only mean/probability smooth; other stats fall back to global; order
   stats never blend. Stated in CLAUDE.md so no session "helpfully" adds bogus smoothing.
4. **Code map** — the `src/catstat/` module table from the design doc, so edits land in the right
   place and the backend-agnostic vs backend-specific boundary is explicit.
5. **Testing** — *one command*: `bash scripts/check.sh` (ruff + pytest + examples,
   `OMP_NUM_THREADS=1`). GPU tests skip locally; run on Colab.
6. **Skills** — a short table of the high-signal skills and when to invoke each (start: 3).
7. **Workflows** — research / implementation / benchmark / release loops, and the self-improvement
   loop read order.
8. **When to / not to use subagents** — keep the costly path rare; inline for local edits.
9. **Context-saving rules** — read the proposal docs/roadmap first; don't re-browse what's already
   captured; pass file paths, not pasted dumps.
10. **Do not change without instruction** — the invariants, public API shape, default smoothing,
    `auto` thresholds, and the leakage path.

## "Patch vs replace" policy (for future edits)
Because a live `CLAUDE.md` now exists, future changes should be **patches** to it (preserve the
structure, update the affected section), never blind rewrites — exactly the policy applied when
adopting it here.

## Divergences from repleafgbm (called out on purpose)
- repleafgbm: "`auto` never selects the GPU" (GPU is explicit-only). **`catstat`: `backend="auto"`
  *may* select GPU** when it pays off — auto-acceleration is the value prop — while keeping the
  explicit-beats-silent rule (explicit `"gpu"` errors loudly if RAPIDS/GPU is missing).
- repleafgbm centers on a tiered `.claude/agents/` fleet; `catstat` starts with a **small skills
  set** (the agent fleet is an optional later addition).
