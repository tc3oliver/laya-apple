# Staged handoff: PB-ASYNC behind a request-count guard (1.5 execution closure)

**Status: preregistered, not run.** Issue #101. The criteria, the gates and the outcome rules are
in [`criteria.md`](criteria.md), committed before any formal run.

- **Question.** Can PB-ASYNC become a production-safe GPU+ANE path that is never worse than today's
  production path during a hetero transition, and keeps GPU completion isolation in steady state?
- **Hypothesis: staged handoff.** The first N real short requests of each hetero episode take
  today's synchronous path, and PB-ASYNC takes over after that. N = 32 (H32) or 64 (H64). The state
  machine is [`scripts/handoff.py`](scripts/handoff.py).
- **Phase 0,** existing #96 / #99 data only: [`phase0.md`](phase0.md).

## Run

```sh
# Idle machine on AC power, local LLM server stopped. Resumable.
LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-staged-handoff/scripts/run_all.sh 1
```

Unit tests: `tests/unit/test_staged_handoff.py`.
