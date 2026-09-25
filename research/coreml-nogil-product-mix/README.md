# GIL-released Core ML predict on the product mix

**Status: preregistered, not run.** The criteria are in [`criteria.md`](criteria.md), which
was committed before any campaign data. `raw/`, `results.json` and `tables.md` do not exist
yet.

Machine: Mac Studio M4 Max, macOS 26, Python 3.12.14, coremltools 9.0, MLX 0.32.2,
PyObjC 12.2.2.

## Question

Can a thread-placed ANE whose Core ML `predict` releases the GIL pass #57's production gate
on the v1.0 closed-loop mix? Two variants are tested: one with the GPU in its worker process,
and one with the GPU on a thread in the caller, with no IPC at all.

## Why

- **The GIL wait, and a fix for it (#45, #46).** A thread-placed coremltools `predict`
  holds the GIL for the whole call. The GPU dispatcher then waits in `take_gil` with the
  worker's reply already read. Releasing the GIL around `predict` removed the wait (GPU
  return P50 7.67 → 0.14 ms, `research/coreml-gil-completion-path/`).
- **The ANE cost of that fix was confounded.** It cost the ANE 13.5% throughput in #46's
  two-stream matrix, where the closed-loop GPU ran 44% more requests once it was no longer
  blocked. At an identical GPU offered load it cost nothing measurable: +0.1% ANE throughput
  (#51, `research/coreml-placement-deconfounding/`).
- **Process isolation fails the production gate.** On the product mix, process placement
  removes the wait but fails #57 (`benchmarks/ane-process-isolation/`):
  - laya: short P99 +17.5%;
  - laya-typed-decisions: aggregate −16.1% and short P99 +73.3%.

  #52 rejected it for production. #58 found the regression dominated by host-side CPU time
  and parent-side queue delay, not by the ANE.
- **So the GIL-release path is the remaining in-process candidate.** It has never been run
  on the product mix (C).
- **Why D: the GPU might not need a process either.**
  - The GPU moved to a worker process because two backends in one interpreter interfered:
    the GPU stream lost 9–11% (`research/v0.2-concurrency/`, step 1). That was measured
    while Core ML held the GIL for every `predict`.
  - MLX's forward does not block other threads: another thread wakes on time during an MLX
    forward loop (`research/gpu-ane-interference/`, "GIL probe").
  - With both devices busy, an IPC-fed device runs its host-side work 4–6× slower
    (`research/v0.2-concurrency/`). #54 and #58 saw the same CPU inflation in process
    workers.
  - If the interference came from the GIL, D removes both causes at once: the GIL hold and
    the IPC.

## Configurations

| | ANE | Core ML `predict` | GPU (MLX) | role |
|---|---|---|---|---|
| **A** | thread | coremltools, GIL held | worker process | production: laya, laya-typed-decisions |
| **B** | worker process | coremltools, in the worker | worker process | production: laya-multilingual |
| **C** | thread | GIL released (`scripts/nogil.py`) | worker process | candidate |
| **D** | thread | GIL released (`scripts/nogil.py`) | thread in the caller, no IPC | candidate |

**How C and D are built.** Everything is injected from `scripts/run_config.py`. Nothing in
`laya_apple/` changes, and the package imports nothing from `research/`:
- **GIL-released `predict`** (`--ane-predict nogil`, C and D):
  - The ANE backend loads and verifies through the runtime as usual. Then #46's PyObjC
    binding is installed on it, the same way #46 installed it (`nogil.install`), and each
    bucket is warmed again.
  - One change from #46: the Objective-C message is sent from a 20-line native shim
    (`scripts/predict_stamped.m`, compiled with the system clang into a temporary
    directory), called through `ctypes.CDLL`, rather than through PyObjC's method call.
  - Both release the GIL for the call. The shim is only there to stamp the clock when
    `predict` returns natively, which is what makes the re-acquire wait measurable.
  - Checked before any run: outputs bit-identical to coremltools on 30 inputs (laya L64,
    L96, L128), and the stamps are ordered (Python before ≤ native before ≤ native after ≤
    Python after).
- **GPU on a thread** (`--gpu-placement thread`, D only): the heterogeneous instance's GPU
  `DeviceWorker` is built with `placement="thread"`. This is the executor's existing
  in-process mode, the same code path a thread-placed ANE uses.
- **Only the heterogeneous instance changes.** The inline references and the `gpu_only`
  instance are the product's in every configuration.
- **All configurations are measured the same way.** Every process worker starts through
  `scripts/worker_entry.py`, which is `python -m laya_apple.executor` plus two
  `thread_time_ns()` reads per forward. In-process forwards get the same two reads. So every
  configuration pays the same instrumentation.

**The PyObjC dependency (research only, not in `pyproject.toml`).** `uv run --with
pyobjc-framework-CoreML==12.2.2` adds three MIT-licensed wheels:

| wheel | size |
|---|---|
| pyobjc-core | 25.3 MB |
| pyobjc-framework-Cocoa | 2.6 MB |
| pyobjc-framework-CoreML | 0.1 MB |

Together they are 28.0 MB installed. Importing `CoreML` and `objc` takes 31 ms. The shim
also needs the Xcode command-line tools (`xcrun clang`) when a run starts. No binary is
committed.

## Workload

It is #57's, unchanged: `benchmarks/ane-process-isolation/run_mix.py`, which runs
`scripts/bench_concurrency.py --part a`:
- windows of solo_short, solo_long, hetero and gpu_only;
- 20 s windows, 3 cycles per run, in alternating order;
- one closed-loop client per stream;
- answers checked against inline references computed per device.

The shapes are the v1.0 commands:

| model | short | long |
|---|---|---|
| laya | L128 | L512 |
| laya-typed-decisions | L128 | L1024 |
| laya-multilingual | L96 | L1024 |

**Order per model:** P C D D C P, where P is the model's production configuration (A, or B
for laya-multilingual). That is 2 runs × 3 cycles = 6 hetero windows per configuration,
matching #57.

## Gate and stop rules

See [`criteria.md`](criteria.md): #57's five criteria, copied verbatim. Each of C and D is
compared with the model's production configuration. For laya-multilingual, the "≥ 5×
better" half of the GPU completion criterion does not apply. A candidate passes only if it
passes on all three models.
- **C and D both fail:** FAIL is recorded, and the Swift-worker fallback research is
  preregistered next.
- **D passes:** D is the candidate production design.
- **Only C passes:** C is the candidate, and the GPU worker stays a process.

**Not gating:**
- the ANE thread's GIL re-acquire wait after `predict` returns (C and D);
- CPU time per request on the GPU and ANE executing threads;
- the GPU service span.

## Run

```sh
# ~1 h 30 min (18 runs × ~4.8 min): idle machine on AC power, oMLX and other GPU/ANE
# services stopped. Resumable: finished runs are skipped.
LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 sh research/coreml-nogil-product-mix/scripts/run_all.sh
uv run python research/coreml-nogil-product-mix/scripts/analyze.py      # --check to verify
```

One run by hand:

```sh
LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run --with pyobjc-framework-CoreML==12.2.2 python \
  research/coreml-nogil-product-mix/scripts/run_config.py --config D --model laya \
  --short 128 --long 512 --output research/coreml-nogil-product-mix/raw/laya-D-r1.json.gz
```

**Time estimate.** A run has 12 windows of 20 s, with 2.5 s before each, plus about 10 s
for loading, references and warm-up. That is about 4.8 min, so the campaign's 18 runs take
about 1 h 30 min. The estimate comes from the harness smoke checks at 5 s windows, scaled
to 20 s.

| file | what |
|---|---|
| `criteria.md` | the gate, its reading for this experiment, stop rules, valid runs, protocol |
| `scripts/run_config.py` | one run: #57's `run_mix.py` + the configuration's injection + extra records |
| `scripts/nogil.py` | #46's GIL-releasing binding with the stamped `predict` call |
| `scripts/predict_stamped.m` | the native `predictionFromFeatures:error:` call with clock stamps |
| `scripts/worker_entry.py` | process-worker entry: the executor unchanged, plus per-forward thread CPU |
| `scripts/run_all.sh` | the campaign, in order |
| `scripts/analyze.py` | `results.json`, `tables.md` (`--check`) |
