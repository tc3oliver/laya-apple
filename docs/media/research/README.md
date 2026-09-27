# Research figures

Five figures for the 1.4 → 1.5 research line summarized in
[`research/README.md`](../../../research/README.md). Each PNG is 2400 px wide (a 1200 px page
at 2×). Every number in them is copied from the study it names; nothing was recomputed, and
no research data was changed.

| File | Shows | Evidence |
|---|---|---|
| `fig1-gil-stall.png` | The GPU result waits for the GIL while a synchronous Core ML `predict` holds it | observation (#43, #44), intervention (#46) |
| `fig2-closed-loop-confound.png` | The −13.5% ANE regression was the closed-loop GPU sending more work; at a fixed offered load it is +0.1% | intervention (#46, #51) |
| `fig3-host-slow-state.png` | Host CPU work grows in the slow state while the native Core ML call does not; E-core residency correlates with it | post-hoc observation (#88), observation (#96) |
| `fig4-static-handoff.png` | One real episode: a normal H64 handoff, then the slow state about 4.7 s later | preregistered gate (#103), post-hoc replay onset (#104) |
| `fig5-adaptive-execution.png` | The 1.5 state machine, with recovery and production validation shown as separate evidence | shipped code, preregistered (#104, #105) |

## Sources

### Figure 1

| Number | Source |
|---|---|
| service 11.98 → 12.70 ms, return 0.05 → 7.41 ms, and the other legs (mean per request, laya-typed-decisions, thread placement) | [`gpu-ane-interference/ledger/tables.md`](../../../research/gpu-ane-interference/ledger/tables.md), GPU alone against the matrix cell |
| +7.36 ms, 91.1% of the added occupancy | the same table, "share of Δ occupancy" |
| GPU return P50 7.67 → 0.14 ms (A → C, matrix cell) | [`coreml-gil-completion-path/tables.md`](../../../research/coreml-gil-completion-path/tables.md) |
| 872 against 0 samples in `take_gil` | the same file, native stack samples of the GPU dispatcher |

The two measurements come from different campaigns: the ledger re-run reports means over
25 s windows, #46 reports P50 over 20 s windows. Panel A is a schematic and is labelled so.

### Figure 2

| Number | Source |
|---|---|
| GPU 48.1 → 69.5 req/s (+44%), ANE 95.9 → 83.0 req/s (−13.5%) | [`coreml-gil-completion-path/tables.md`](../../../research/coreml-gil-completion-path/tables.md), matrix cell, A against C |
| one arrival trace (same SHA-256), 46.65 req/s offered, 46.35 / 46.60 completed | [`coreml-placement-deconfounding/README.md`](../../../research/coreml-placement-deconfounding/README.md) |
| ANE 97.5 → 97.6 req/s (+0.1%) | the same file |

+0.1% holds at that load only. One run per configuration; the higher-load comparison was not
re-tested.

### Figure 3

| Number | Source |
|---|---|
| each dot: short-stream P99 of one GPU + ANE window | [`coreml-prebind-full-protocol/raw/`](../../../research/coreml-prebind-full-protocol/raw/), `part_a.windows[]` where `condition` is `hetero`, `streams.short.p99_ms` |
| normal 10.4–12.3 ms, slow 13.6–21.2 ms, 13 ms split, slow counts | [`coreml-prebind-full-protocol/README.md`](../../../research/coreml-prebind-full-protocol/README.md) |
| native Core ML call 9.62 ms (A, whole `predict`) against 9.67 ms (PB, native call), P50 per forward | [`coreml-prebind-full-protocol/tables.md`](../../../research/coreml-prebind-full-protocol/tables.md) |
| GPU-thread CPU per forward 1.8 against 6.1 ms | the same file |
| E share 0.91–1.00 in the transient, 0.00 in steady state, and the listed exceptions | [`coreml-async-transient/README.md`](../../../research/coreml-async-transient/README.md), [`tables.md`](../../../research/coreml-async-transient/tables.md) |

Panel B compares PB with A, not slow windows with normal ones. Panel C comes from a separate
study (#96) of PB-ASYNC, which did not measure #88's windows. E-core residency is a
correlation; causality was not established.

### Figure 4

Every dot is one ANE request of `windows[5]` in
[`coreml-staged-handoff/raw-eval/mix-laya-P-r5.json.gz`](../../../research/coreml-staged-handoff/raw-eval/)
(SHA-256 `a91a471a6f81b10f38a64252ee7b620ebe22290f31db9804f0252c0031600c2a`): client
latency from `streams.short.latency_ms`, and `prepare` as `prepared_ns − submit_ns`. The
handoff is the window's first async decision.

| Number | Source |
|---|---|
| handoff at 0.73 s | [`coreml-staged-handoff/eval_tables.md`](../../../research/coreml-staged-handoff/eval_tables.md) (0.734) |
| onset about 4.7 s after the handoff | [`coreml-adaptive-breaker/replay.md`](../../../research/coreml-adaptive-breaker/replay.md) (4.70), post-hoc |
| H5 30 consecutive host-slow bins, H6 15 consecutive slow spans | `eval_tables.md` |
| OrbStack Helper at 92% CPU before the run; A slow in 0 of 12 episodes | `raw-eval/mix-laya-P-r5.before.json`; `eval_tables.md` |

### Figure 5

| Number | Source |
|---|---|
| 64 forwards, C3 = 3 consecutive `prepare_ms` > 0.3 ms, rest of the episode on 1.4, re-arm | [`laya_apple/handoff.py`](../../../laya_apple/handoff.py) |
| 12 of 12 recovered, 215 / 364 / 414 ms, limit 1.0 s, 0.999× | [`coreml-adaptive-breaker/phase1_tables.md`](../../../research/coreml-adaptive-breaker/phase1_tables.md) |
| 154 episodes stayed async, 0 trips, GPU return 0.035–0.043 against 4.28–8.60 ms, 1.038–1.042× | [`coreml-adaptive-breaker/val_tables.md`](../../../research/coreml-adaptive-breaker/val_tables.md) |
| 0 mismatches, routing failures, lost requests or crashes | [`coreml-adaptive-breaker/README.md`](../../../research/coreml-adaptive-breaker/README.md) |

Phase 1 used the research breaker, armed 1 s into the episode, without the 64-forward guard.
Production validation had no natural slow state, so the shipped trip path did not run.

## How they were made

The figures were drawn as SVG by a standalone Python script outside this repository, which
reads the files above without changing them, and rasterized at 2× with headless Chrome.
