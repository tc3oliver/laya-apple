# Correctness: fast is not the same as correct

A Core ML model that loads, runs without error and returns quickly can still return
different decisions from the model it was converted from. laya-apple therefore treats
correctness as a property of an exact configuration, measured on the machine that uses
it, and not as a property of a model.

## The finding

Platform: Apple M4 Max, macOS 26.6.2 (25G83), coremltools 9.0. Reference: unmodified
upstream Laya on PyTorch CPU FP32, over the golden rows in `laya_apple/parity/goldens`.

On this platform, the ordinary Core ML export of Laya (BxLxC layout, SDPA attention,
fixed shapes, FP16) produced the following on `CPU_AND_NE` and `ALL`. It raised no error
in any of these runs.

| Model | Compute units | Hard mismatches | Near-tie flips | Max prob error | Verdict |
|---|---|---:|---:|---:|---|
| laya | `CPU_AND_NE` | 12 | 5 | 0.56 | ❌ fail |
| laya | `ALL` | 6 | 2 | 0.322 | ❌ fail |
| laya | `CPU_AND_GPU` | 0 | 0 | 0.00659 | ✅ pass |
| laya-multilingual | `CPU_AND_NE` | 85 | 5 | 1 | ❌ fail |
| laya-multilingual | `ALL` | 42 | 1 | 1 | ❌ fail |
| laya-multilingual | `CPU_AND_GPU` | 0 | 0 | 0.00585 | ✅ pass |
| laya-typed-decisions | `CPU_AND_NE` | 19 | 4 | 0.418 | ❌ fail |
| laya-typed-decisions | `ALL` | 3 | 2 | 0.418 | ❌ fail |
| laya-typed-decisions | `CPU_AND_GPU` | 0 | 0 | 0.00283 | ✅ pass |

Source: `scripts/v1_report.py parity`, from `benchmarks/v1.0/parity/`. The same export
passes on `CPU_AND_GPU`: the decisions change with the compute units it runs on.

It was also fast. Forward P50 on laya-typed-decisions, one question
(`scripts/v1_report.py latency`):

| Implementation | L64 | L128 | L256 | L512 | L1024 |
|---|---:|---:|---:|---:|---:|
| Core ML ordinary · `CPU_AND_NE` ❌ | 7.3 | 9.7 | 23.3 | 65.1 | 2201.8 |
| Core ML ordinary · `CPU_AND_GPU` | 8.5 | 12.6 | 20.1 | 36.2 | 71.0 |
| laya-apple MLX FP16 | 9.3 | 12.2 | 19.2 | 35.3 | 71.0 |
| laya-apple ANE | 8.0 | 9.9 | — | — | — |

Two more results on the same platform:
- **L1024 runs on the CPU without notice.** For laya-typed-decisions at L1024 on
  `CPU_AND_NE`, the ANE compiler fails and Core ML runs the model on the CPU instead. The
  call returns normally and takes 2201.8 ms
  ([`research/phase-0-feasibility/typed-decisions-ane.md`](../research/phase-0-feasibility/typed-decisions-ane.md)).
- **The enumerated-shape export never reaches an accelerator.** With flexible
  (enumerated) shapes, the ordinary export runs 100% on the CPU under every compute-unit
  setting (same document). Its v1.0 forward P50 on laya is 216.8 ms at L128 with
  `CPU_AND_GPU` requested.

laya-apple instead uses a channel-first (BC1S) rewrite of the graph for the ANE. Its v1.0
results against the same reference: laya 0 hard mismatches, 1 near-tie flip, max
probability error 0.0122; laya-multilingual 0 / 0, 0.0128; laya-typed-decisions 0 / 0,
0.00767.

## The production rule

Correctness is proven per exact configuration, and only a proven configuration becomes a
production capability:

```
model + revision + graph + shape + compute target
                    │
                    ▼
      parity validation (+ placement: 100% ANE, 0 transitions)
                    │
                    ▼
          production capability (manifest)
```

- **Configuration.** Pinned checkpoint revision and weight hash, BC1S graph, one fixed
  length bucket at batch 1, FP16 on `CPU_AND_NE`.
- **Validation.** `laya-apple artifacts build` converts the bucket, checks the Core ML
  compute plan (100% Neural Engine, 0 device transitions) and runs the parity gate on
  this machine. A build that fails is never registered.
- **Capability.** The manifest records the configuration, platform profile, parity
  result and file hash. The runtime re-checks it on load (`laya_apple/artifacts.py`).

A change to any part of the configuration is a different configuration and needs its
own validation.

## Definitions

Fixed before measurement
([methodology](../research/phase-0-feasibility/methodology.md), `laya_apple/parity/`):

- **Probability error:** the largest |Δ| between a backend's calibrated option
  probabilities and those of upstream Laya on PyTorch CPU FP32, over the golden rows.
- **Hard mismatch:** the chosen option differs from upstream, *and* upstream's top-1 /
  top-2 probability margin is at least 2× the tolerance (0.04 for FP16, 2e-4 for FP32).
- **Near-tie flip:** the chosen option differs inside that band. It is listed row by
  row, never hidden, and does not fail the gate.
- **FP16 gate:** probability error ≤ 0.02, action-probability error ≤ 0.02, 0 hard
  mismatches, all outputs finite, and repeated calls bit-identical.
- **FP32 gate:** the same, at a 1e-4 tolerance.

### Golden set

The goldens are recorded from upstream Laya 0.3.20 (`scripts/make_reference.py`, raw files
in `research/upstream-reference/raw/laya-0.3.20/`). They hold every Phase -1 case, whose
token ids, decision logits and action logits are identical to the Phase -1 record made with
upstream 0.3.5. They also hold five cases per model for the upstream prompt changes after
0.3.5:

- a left-truncated conversation;
- non-ASCII structured instructions;
- noul `labels`;
- mixed-case noul criteria keys;
- option text over the 48-token cap.

These add 13 rows per model. The v1.0 parity tables were measured before these cases
existed, over 29 cases for laya and 32 for each of the other two models. A parity run on the
current goldens covers more rows and is not the same measurement.

## Checks after validation

- **On import.** `laya-apple artifacts import` re-runs the full parity gate, the compute
  plan check and the platform check on the importing machine before it registers an
  artifact built elsewhere (`lifecycle.import_artifact`).
- **On load.** Every loaded bucket passes the runtime placement probe: the loaded model
  is timed against a `CPU_ONLY` instance of the same artifact, and a model that is not
  clearly faster raises `ComputeUnitMismatchError`
  ([`architecture.md`](architecture.md#runtime-placement-probe)).
- **On demand.** `laya-apple parity MODEL --device gpu|ane` runs the gate
  against the shipped goldens.

## Option order

`choice` decisions can depend on the order of the options. This is a property of
upstream Laya: on a 40-case set, upstream PyTorch changes its decision under permutation
in 22.5% of cases for laya and 35.0% for laya-typed-decisions. laya-apple MLX FP32
reproduces upstream's decision for every permutation. Details:
[`research/option-order/`](../research/option-order/).
