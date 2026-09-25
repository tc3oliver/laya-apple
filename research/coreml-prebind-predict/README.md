# One prediction crossing on the ANE request path

**Status: run.**
- **Verdict:** PB PASS on laya and laya-typed-decisions, INCONCLUSIVE on laya-multilingual.
  So no production change follows, and a larger preregistered replication decides
  multilingual.
- **An unplanned observation limits what the passes mean.** Under this protocol, C also
  passes on laya and laya-typed-decisions. See [Results](#results).
- **Where things are:**
  - the criteria: [`criteria.md`](criteria.md), committed before any campaign data, in its own
    commit;
  - the numbers: [`tables.md`](tables.md), [`results.json`](results.json);
  - the raw data: [`raw/`](raw/).

Machine: Mac Studio M4 Max, macOS 26, Python 3.12.14, coremltools 9.0, MLX 0.32.2,
PyObjC 12.2.2.

## Results

18 runs, in the order P C PB PB C P per model: 6 matched hetero-window pairs per candidate per
model. Every run finished in 78–81 s, with 0 mismatches in every window. PB was bit-identical
to coremltools on every bucket of all three models (`raw/check.json`).

**The laya block was re-run** under the addendum in `criteria.md`: an unrelated package install
overlapped the first laya run. The re-run is the primary laya result. The original six laya
runs are kept unchanged in `raw/contaminated/`. They are reported alongside in
[`tables-contaminated-laya.md`](tables-contaminated-laya.md), outside the verdict. They show
the same pattern: PB and C both pass on laya, with PB short P99 0.954 [0.933, 0.977] and C
0.991 [0.967, 1.016].

### Preregistered verdict (PB vs production, paired gate, 95% t-interval)

| model | aggregate throughput | short P99 | long P99 | GPU return P50 | verdict |
|---|---|---|---|---|---|
| laya (vs A) | 1.060 [1.057, 1.062] | 0.957 [0.953, 0.962] | 0.862 [0.857, 0.867] | 0.034 ms, ×129 | **PASS** |
| laya-typed-decisions (vs A) | 1.049 [1.047, 1.051] | 0.963 [0.952, 0.974] | 0.882 [0.881, 0.883] | 0.039 ms, ×225 | **PASS** |
| laya-multilingual (vs B) | 1.171 [0.905, 1.515] | 0.851 [0.524, 1.384] | 0.952 [0.812, 1.116] | 0.032 ms | **INCONCLUSIVE** |

Per the preregistered outcome, **no production change follows**, and a larger preregistered
replication (for example 9 cycles, n = 18 pairs) decides laya-multilingual. The bootstrap
sensitivity check agrees with every verdict above except multilingual's aggregate, where its
interval is [0.961, 1.364]. It does not change the verdict.

### Unplanned observation: C also passes under this protocol

Under this hetero-only protocol, C passes on laya and laya-typed-decisions:
- laya: short P99 1.004 [1.000, 1.008], aggregate 1.032;
- laya-typed-decisions: short P99 0.996 [0.990, 1.002], aggregate 1.022.

#77, which had solo and gpu_only windows between its hetero windows, recorded C as a FAIL on
both models. The same C binding, the same shapes and the same request are used here.

**What differs.** The protocol differences are the removed solo_short, solo_long and gpu_only
windows (with the gpu_only instance) and the fixed 2 s warm-up before each hetero window. The
paired gate also replaced #77's pooled point estimate, but #77's C short P99 regressions were
+12.9% and +83.0% on those models, far beyond either rule.

**What this means:**
- #77's regression depends on the protocol or on what ran before each hetero window.
- PB's pass on laya and laya-typed-decisions **cannot be attributed to handoff reduction
  alone**.
- #77's FAIL is not rewritten: it stands as recorded under its own protocol.
- Why the regression depends on the protocol is a new question, not answered here.

On laya-multilingual, C fails under this protocol too: short P99 1.423 [1.063, 1.904].

### PB vs C, same campaign (not gating)

- **laya and laya-typed-decisions.** PB is better than C, with non-overlapping intervals:
  - short P99 by about 3–5%: PB 0.957 and 0.963 against C 1.004 and 0.996;
  - aggregate throughput by about 3%: PB 1.060 and 1.049 against C 1.032 and 1.022.
- **ANE thread CPU per forward** fell from 0.64 ms (C) to 0.34 ms (PB) on both models. PB's
  pre and post stages are 0.014 ms and 0.005–0.006 ms, against C's 0.13–0.15 ms and
  0.08–0.09 ms.
- **laya-multilingual.** C fails, while PB's point estimates are better than production B:
  - short P99 4.41 ms against B's 5.85 ms;
  - aggregate 293 req/s against 243.

  PB's ANE dispatcher thread used 1431 ms of CPU per window, against C's 6505 ms. PB's
  intervals are wide, so this is not a verdict.
- **Crossings per forward:** PB 4 (predict 1, pool 3) with 0 re-entries. C 108 with 62
  re-entries.

### Diagnostics and their limits

- **Slow-CPU flag: not informative in this campaign.** It needs the `client-short` thread's CPU
  in both the candidate window and P's matched window. That thread is only in a window's
  closing snapshot when the short client finishes last, and in P's windows it never did. So
  no pair has a ratio on any model, and `tables.md` marks every cell "not computable". The
  per-window `client-short` column is also missing wherever the long client finished last. It
  appears only for laya C and PB.
- **The GIL probe:**
  - Its P99 lateness in A windows is about 10 ms, against about 0.6 ms for C and PB. That fits
    coremltools holding the GIL through each `predict`.
  - In laya-multilingual, C's probe P99 (2.2 ms) is above B's (1.6 ms) and PB's (0.65 ms).
- **Collision diagnostics** (fixed exposure window) are in `tables.md`. They are association
  only.

## Purpose and question

**Purpose.** Test whether reducing the ANE request path's many PyObjC/GIL handoffs to the
minimum restores ANE short-tail latency while keeping GPU completion isolation.

**Question.** If the ANE request path's many PyObjC/GIL handoffs are collapsed into a single
prediction crossing, can production-level ANE short-tail latency be restored while GPU
completion isolation is kept?

The experiment changes one thing, the number of bridge handoffs per ANE forward. It does not
test which mechanism behind the handoffs matters.

## Why

- **The earlier work.**
  - #45 and #46 removed the GPU completion wait by releasing the GIL around a thread-placed
    Core ML `predict` (`research/coreml-gil-completion-path/`).
  - #51 found that the ANE cost of doing so was confounded with the GPU's offered load
    (`research/coreml-placement-deconfounding/`).
  - #57 is the production gate for ANE placement (`benchmarks/ane-process-isolation/`).
- **#77 (issue #66).** On the v1.0 product mix, #77 (`research/coreml-nogil-product-mix/`)
  tested C: a thread-placed ANE whose `predict` releases the GIL through PyObjC, with the GPU
  worker kept as a process.
  - C kept GPU completion isolation (GPU return P50 0.05–0.19 ms).
  - C failed the gate on all three models: short (ANE) P99 rose by 13–83%.
  - **That FAIL stands.** This experiment does not re-score it.
- **The handoffs.** C builds its inputs and reads its outputs through PyObjC, and PyObjC
  releases and re-acquires the GIL around every Objective-C call. That is dozens of handoffs
  per forward, each a point where another Python thread can take the GIL.
- **PB** keeps C's topology and makes the ANE request path cross the bridge once for the
  prediction.

## Configurations

| | ANE | Core ML `predict` | GPU (MLX) |
|---|---|---|---|
| **P** | production: A (thread, coremltools, GIL held) for laya and laya-typed-decisions; B (worker process) for laya-multilingual | as in production | worker process |
| **C** | thread | #77's binding, unchanged (`research/coreml-nogil-product-mix/scripts/nogil.py`) | worker process |
| **PB** | thread | prebound: one prediction crossing per forward (`scripts/prebind.py`) | worker process |

These were fixed for all three models before any data. No model or configuration was chosen
from #77's results.

### How PB makes one crossing

**At model load, once per bucket.** PyObjC builds everything the forward needs:
- the input MLMultiArrays and a NumPy view of each;
- the MLFeatureValues, and one MLDictionaryFeatureProvider over them;
- the output MLMultiArrays and a NumPy view of each, passed to Core ML as
  `MLPredictionOptions.outputBackings`;
- the MLPredictionOptions object itself.

Strong references to all of these are kept on the model object.

Everything Core ML reads during a prediction is a Foundation object: the shapes are NSArrays
of NSNumbers, and the dictionaries are NSDictionaries with NSString keys. A Python list or dict
handed to PyObjC becomes a Python-backed proxy. Core ML then calls back into it during
`predict`, and each callback takes the GIL inside the crossing. PB refuses to load if any such
proxy reaches Core ML. The raw pointers and the ctypes buffers for the shim are also taken
once, at load.

**Per forward:**
1. **Write the inputs.** NumPy copies the features into the input views. There is no
   Objective-C call.
2. **Predict.** Inside an autorelease pool, one `predictionFromFeatures:options:error:` is
   sent through `scripts/predict_options_stamped.m`, called with `ctypes.CDLL`:
   - The shim is #77's `predict_stamped.m` with one extra argument, the options.
   - `ctypes.CDLL` releases the GIL once, for exactly that call.
   - The shim builds nothing and copies nothing. It only timestamps the send, so the GIL
     re-acquire wait can be measured the same way for C and PB.
   - When the shim is compiled, a check confirms it references no Python C-API symbols
     (`nm -u`). It cannot call back into Python, so the call is a single GIL crossing.
3. **Read the outputs.** NumPy copies them out of the backed views, again with no
   Objective-C call.

**The resulting count.** In backed mode, a forward makes:
- one prediction crossing;
- the autorelease pool's push, which is two sends (`alloc`, `init`);
- the pool's pop.

That is 4 crossings, with no other sends and no re-entries into Python. The pool cannot be
dropped: the prediction's autoreleased objects would leak (#46), and C has the same pool.

**How the count is taken.** `scripts/crossings.py` counts this with `sys.monitoring` rather
than assuming it, for both C and PB. It counts the forward's own crossings. It also counts
**re-entries**: Python code starting while a crossing's native call is still in flight, which
is a GIL acquisition hidden inside the crossing. The count is recorded in every run and in
the bit-identity check.

**Measured before the campaign** (`check_prebind.py`, every bucket of all three models):

| | top-level crossings | of which sends | predict | pool | re-entries (nested sends) |
|---|---|---|---|---|---|
| PB | 4 | 0 | 1 | 3 | 0 (0) |
| C (#77) | 108 | 104 | 1 | 3 | 62 (112) |

The first PB draft passed the array shapes as Python lists. The count found 20 re-entries
(40 nested `objCType` / `longLongValue` sends from PyObjC's `numberWrapper`) inside its one
prediction. That was fixed before any campaign run. C's re-entries come from the same
Python-backed shapes and dictionaries.

**Risks, and how they are handled:**
- **Output backings.** Each output's backing is verified at load. A sentinel is written into
  the backing, one prediction is run, and the backing must then hold exactly the returned
  output.
  - An output whose backing passes is read from its view: mode `backed`.
  - An output whose backing fails is read from the returned provider with three extra sends:
    mode `read`.
  - The mode of each output is recorded.
- **Buffer lifetime.** Each bucket's objects are kept as strong references for as long as
  the backend exists.
- **Buffer reuse.** Each bucket has a non-blocking lock, so an overlapping forward raises
  instead of overwriting a buffer.

**Scope.** Nothing in `laya_apple/` or its packaging changes, and the package imports nothing
from `research/`. PyObjC is added with `uv run --with pyobjc-framework-CoreML==12.2.2`, for
research only. No binary is committed: the shim is compiled with `xcrun clang` into a
temporary directory when a run starts.

## Workload

**Hetero-only.** This is a protocol change from #57 and #77, made before any data. The
workload is #57's Part A reduced to the windows the gate uses. `scripts/run_config.py` drives
`scripts/bench_concurrency.py`'s own closed loop, statistics, request construction and inline
references.
- **Removed:** the solo_short, solo_long and gpu_only windows, the gpu_only instance and
  Part B.
- **Kept:** the 3 `hetero` windows of 20 s per run, one per cycle, which give 6 matched pairs
  per candidate per model.
- **Each cycle:** 2.0 s idle, then a fixed 2.0 s warm-up of both streams (closed-loop, the
  same instance, not measured, the same in every configuration), then a 0.5 s lead, then the
  measured `hetero` window.
- **Identical request streams:**
  - Each stream repeats the one request bench_concurrency builds with `make_request(seed=0)`.
    It is the same request #57 and #77 used.
  - Window k therefore has the same request content in every configuration.
  - The clients are closed-loop, so there is no arrival schedule to seed.
- **Other settings:** one closed-loop client per stream. Answers are checked against the inline
  references.
- **Comparisons with #57 and #77.** The results are compared within this experiment only,
  because #57 and #77 had solo windows between their hetero windows.

The shapes are the v1.0 commands:

| model | short | long |
|---|---|---|
| laya | L128 | L512 |
| laya-typed-decisions | L128 | L1024 |
| laya-multilingual | L96 | L1024 |

**Order per model:** P C PB PB C P. That is 18 runs in total.

## Gate and interpretation

The full rules are in [`criteria.md`](criteria.md).

- **Budgets.** They are #57's, unchanged: short and long P99 ≤ 1.05×, aggregate throughput
  ≥ 0.95×, 0 mismatches, and GPU return P50 ≤ 1 ms. Against A, GPU return must also be ≥ 5×
  better; against B, only ≤ 1 ms applies. Each candidate is compared with P, per model.
- **Paired gate.** From this preregistration on, the P99 and throughput budgets are judged by
  a paired statistical gate. It is not applied to #77.
  - Each candidate `hetero` window is paired with P's window of the same round and cycle,
    giving 6 pairs.
  - The statistic is the geometric mean of the pair ratios, with a 95% t-interval on the log
    ratios.
  - Each criterion's verdict is PASS, FAIL or INCONCLUSIVE.
  - A seeded bootstrap over the pairs is reported as a sensitivity check only.
- **Power.** The power estimate from #77's windows is in `criteria.md`. With 6 pairs, laya's
  short P99 is expected to be INCONCLUSIVE unless the effect is large. laya-multilingual and
  aggregate throughput can resolve 5%.
- **Interpretation, per model:**
  - **PB PASS:** supports "the many PyObjC/GIL handoffs are a significant execution-layer
    cost". It is not attributed to GPU-completion collisions alone.
  - **PASS on all three models:** a production binding is considered next, in its own PR.
  - **INCONCLUSIVE:** no production change. A larger preregistered replication decides it.
  - **FAIL:** the paused equal-load experiment (`research/coreml-nogil-equal-load`) runs next.
    If it passes, workload and SoC contention come next. If it fails, the CPU slow state, host
    scheduling and Python runtime contention come next.
- **Out of scope:** QoS, a Swift worker and a native full-forward path.

**Not gating.** These are recorded identically in P, C and PB:
- crossings per forward;
- GIL re-acquire wait;
- ANE stage timings (features, pre, native, re-acquire, post, tail);
- GPU completion collisions, using the fixed exposure window from #77's observational
  analysis, and overlap counts. Both `received_ns` and `response_ns` are kept;
- a 1 ms GIL probe thread;
- per-thread CPU per window, from Mach `thread_info`;
- executing-thread CPU per forward;
- a slow-CPU flag per window.

These diagnostics differ from #77's instrumentation. Numbers are compared within this
campaign, and any comparison with #77 says so.

## Run

Environment: `uv sync --extra dev --extra ane`. The Xcode command-line tools must be present.

```sh
# ~30 min: the bit-identity check (~15 s), then 18 runs x ~85 s. Idle machine on AC
# power, the local LLM server and other GPU/ANE services stopped. Resumable: finished runs are
# skipped. The campaign does not start unless PB is bit-identical to coremltools everywhere.
LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-prebind-predict/scripts/run_all.sh
uv run python research/coreml-prebind-predict/scripts/analyze.py      # --check to verify
```

To run the check or a single run by hand:

```sh
LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run --with pyobjc-framework-CoreML==12.2.2 python \
  research/coreml-prebind-predict/scripts/check_prebind.py
LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run --with pyobjc-framework-CoreML==12.2.2 python \
  research/coreml-prebind-predict/scripts/run_config.py --config PB --model laya \
  --short 128 --long 512 --output research/coreml-prebind-predict/raw/laya-PB-r1.json.gz
```

**Time estimate.**
- **The check:** `check_prebind.py` took 14 s for all three models, C included.
- **One run:** 3 cycles × 24.5 s (2 s idle, 2 s warm-up, 0.5 s lead, 20 s window), plus about
  8 s of load and references (#77 measured that part), plus uv start-up. That is about 85 s.
- **The campaign:** 18 runs, about 26 min, so about 30 min in all.

Measured: each campaign run took 78–81 s, and the 18 runs took 24 min in total.

The contaminated laya block is analysed separately with:

```sh
uv run python research/coreml-prebind-predict/scripts/analyze.py \
  --raw research/coreml-prebind-predict/raw/contaminated --models laya --tag contaminated-laya
```

This writes `results-contaminated-laya.json` and `tables-contaminated-laya.md`. Add `--check`
to verify them.

| file | what |
|---|---|
| `criteria.md` | question, configurations, paired gate, power, interpretation, non-gating records, protocol |
| `scripts/prebind.py` | PB: the prebound binding |
| `scripts/predict_options_stamped.m` | the one `predictionFromFeatures:options:error:` send, with clock stamps |
| `scripts/crossings.py` | counts bridge crossings per call (`sys.monitoring`) |
| `scripts/probe.py` | the 1 ms GIL probe thread and Mach per-thread CPU snapshots |
| `scripts/derive.py` | pure derivations: collisions, overlaps, stages, odds ratios, slow-CPU flag |
| `scripts/gate.py` | the paired gate: pairing, t-interval, bootstrap, three-way verdicts |
| `scripts/check_prebind.py` | the pre-campaign bit-identity check and crossing counts (`raw/check.json`) |
| `scripts/run_config.py` | one run: the hetero-only mix, the configuration's injection, and the records above (adapted from #77's) |
| `scripts/run_all.sh` | the check, then the campaign in order |
| `scripts/analyze.py` | `results.json`, `tables.md` (`--check`) |

Unit tests: `tests/unit/test_prebind_predict.py`.
