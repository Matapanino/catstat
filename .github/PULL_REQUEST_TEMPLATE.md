## Summary

<!-- What does this PR change, and why? -->

## Checklist

- [ ] `bash scripts/check.sh` is green (ruff + pytest + examples).
- [ ] Tests cover the new behavior (encode correctness, **OOF / no-leakage**, unknown/missing
      fallback, feature names, determinism — as applicable).
- [ ] Core invariants intact (leakage safety, smoothing honesty, CPU/GPU parity, public API).
- [ ] `CHANGELOG.md` updated for user-visible changes; SemVer considered.
- [ ] Docs updated (`README.md` / `docs/`) where relevant.

## Notes

<!-- Anything reviewers should know: trade-offs, follow-ups, benchmark/verdict links. -->
