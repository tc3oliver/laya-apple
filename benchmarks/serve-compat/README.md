# `laya-apple serve` against upstream `laya.serve`

**Question.** Does `laya-apple serve` answer the same requests the same way as unmodified
upstream `laya.serve` (Laya v0.3.20), when both serve the same pinned weights?

**Answer [Measured].** Yes, on every case below. Across 792 requests:
- the wire shape matched;
- the language routing matched;
- every value was within the FP16 parity gate: probabilities, score, noul, confidence,
  `answer_confidence` and `act_probability` all within 0.02, with the same `choice`.

365 of those answers came from the Neural Engine.

## Setup

- **Hardware and software:** Apple M4 Max, macOS 26.6.2 (25G83), Python 3.12.14, MLX
  0.32.2, coremltools 9.0.
- **Date:** 2026-09-25.
- **Ours:** `laya-apple serve --preload typed-decisions` on 127.0.0.1:8642, from the
  `feat/serve` branch, which became 1.3.0. The default `--model auto` loads `laya` and
  `laya-multilingual`. The server uses GPU + ANE workers, and every ANE artifact was
  validated and ready.
- **Upstream:** `laya[serve]==0.3.20` with torch 2.7.0 and transformers 5.17.0, on the CPU
  in FP32, at 127.0.0.1:8643.
  - It runs through `scripts/upstream_serve_pinned.py`, which passes upstream's own
    `create_app` a `Router` whose three checkpoints are the local snapshots laya-apple pins:
    - `laya` c5d78730;
    - `laya-multilingual` 052592a1;
    - `laya-typed-decisions` f9ab0b22.
  - Nothing in upstream is patched.
- **Requests:** the shipped parity golden cases for each model: state plus questions, as
  in `laya_apple/parity/goldens/`.
- **Comparison:** `scripts/compare_serve_upstream.py` (its docstring lists every check).
  `routing.repo` is not compared: ours names the standalone repository of the pinned
  weights, and the pinned upstream launcher reports a local path.

## Results

| File | Requests | Mode | Answered on | Pass |
|---|---|---|---|---:|
| `english.json` | 34 golden cases | `model: english` | GPU 34 | 34 / 34 |
| `multilingual.json` | 37 | `model: multilingual` | GPU 37 | 37 / 37 |
| `typed-decisions.json` | 37 | `model: typed-decisions` | GPU 37 | 37 / 37 |
| `auto-english-goldens.json` | 34 | no `model`: language routing on both sides; routing model, reason, detection and workflow compared | GPU 34 | 34 / 34 |
| `auto-multilingual-goldens.json` | 37 | no `model` | GPU 37 | 37 / 37 |
| `auto-typed-decisions-goldens.json` | 37 | no `model` | GPU 37 | 37 / 37 |
| `english-split.json` | 176 questions, one per request | `model: english`, `--split` | ANE 122, GPU 54 | 176 / 176 |
| `multilingual-split.json` | 200 | `model: multilingual`, `--split` | ANE 121, GPU 79 | 200 / 200 |
| `typed-decisions-split.json` | 200 | `model: typed-decisions`, `--split` | ANE 122, GPU 78 | 200 / 200 |

- **Why most golden cases run on the GPU.** They carry several questions each, and
  laya-apple routes those to the GPU.
- **What `--split` covers.** It sends each question alone. Short single questions then take
  the validated Neural Engine path, so the split runs compare the ANE's answers over HTTP.

## Reproduce

```bash
laya-apple download                     # prints the three snapshot directories
uv run --no-project --with 'laya[serve]==0.3.20' --with torch==2.7.0 --with transformers==5.17.0 \
    python scripts/upstream_serve_pinned.py --port 8643 english=DIR multilingual=DIR typed-decisions=DIR
laya-apple serve --preload typed-decisions
uv run python scripts/compare_serve_upstream.py --ours http://127.0.0.1:8642 \
    --upstream http://127.0.0.1:8643 --model english [--auto] [--split] --out FILE
```

## Not covered

- **Deliberate differences:** the status codes laya-apple returns differently from upstream
  (`docs/serve.md`, "Status codes").
- **Language hints:** upstream's opt-in typed-decisions routing (`LAYA_AUTO_TASK`) and caller
  language hints.
- **Performance:** latency and throughput.
