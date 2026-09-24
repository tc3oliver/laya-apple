# Switchyard launch media

`switchyard-launch.mp4` (1920×1080, 30 fps, 20 s, for social posts) and
`switchyard-launch.gif` (800×450, 10 fps, 10 s, the README figure) come from one recorded
Switchyard run on an Apple M4 Max with macOS 26.6.2:
[`benchmarks/switchyard/v1-m4-max/raw/run-003/`](../../benchmarks/switchyard/v1-m4-max/raw/run-003/).
They are made with Remotion outside this repository, so the runtime has no Node dependency.

| Input | SHA-256 |
|---|---|
| `result.json` | `d638d28e41700e7664aeee7b43609b81fd9a009831cbec241cdfa44fa5de4146` |
| `replay.html` | `4df6fa38e0f8e63d2031ba9d7d9eca35874687c6de1f380430519549c9aeee19` |

`replay.html` is not committed. The committed code rebuilds it byte for byte from the run:

```bash
uv run laya-apple switchyard --replay benchmarks/switchyard/v1-m4-max/raw/run-003 --no-open
```

The run was picked by rule, not by eye: of the campaign's three runs, the one whose GPU-only
P99 decision latency is the median (3107.72, **3137.24**, 3170.55 ms). This is a different run
from the screenshots in [`switchyard.md`](switchyard.md), which use run-001.

## Timing

MP4:

| Time | Shows |
|---|---|
| 0–2 s | "1,407 LATE TRAINS → 0", "Same Mac. Same model.", and "MLX GPU → MLX GPU + Apple Neural Engine" |
| 2–8 s | GPU-only round, data time 29.75–31.25 s: from just before the rush hour that starts at 30 s ("Rush hour starts…") |
| 8–14 s | Hard cut to the GPU + ANE round, the same data window on the same timetable |
| 14–18 s | Result card: trains late and P99 decision latency for both rounds |
| 18–20 s | "Try it on your Mac", `uvx laya-apple switchyard` and the repository link |

GIF (no command; the README shows it above the GIF):

| Time | Shows |
|---|---|
| 0–3 s | GPU-only round, data time 30.0–30.75 s: the start of the rush hour |
| 3–6 s | GPU + ANE round, the same 30.0–30.75 s |
| 6–10 s | Result card |

## How it was made

- **Footage.** The committed `replay.html` above, opened unmodified in headless Chrome with a
  1600×900 stage, the timeline drawer closed and the page's own 0.25× speed (4 s of video =
  1 s of data). `requestAnimationFrame` was replaced by a virtual clock advanced exactly one
  video frame per screenshot, so every frame is deterministic. Each screenshot is clipped to
  the rail yard (CSS px x 234–1600, y 196–836.3, all six lines, queues, signals, junctions
  and platforms) at 1920×900. The header, status row, scrubber, column labels, line numbers
  and legend are outside the clip. The capture is 180 frames per round from data time
  29.75 s. The MP4 plays frames 0–179 of both rounds, the GIF frames 30–119 of both, 1:1.
  So each pair of shots shows the same data window.
- **Overlays.** Over each shot, a 180 px band in the page's colours names the configuration
  (GPU amber, ANE teal, late red), shows that round's whole-run result from the first frame,
  and says what the train colour means: red for late (no answer in 100 ms), teal for answered
  by the ANE. A clock in the top left of the yard shows the replay's data time
  (`window_start_s + frame / 30 × 0.25`), so the two shots visibly cover the same window. A
  caption box sits over the empty track at the bottom left for the
  first 2 s (MP4) or 1.2 s (GIF). The hook, result card and ending are drawn by the
  composition. Every number in them is read from `result.json` at capture time; none is
  typed in by hand.
- **MP4.** H.264 High, yuv420p, CRF 16, 30 fps.
- **GIF.** A separate 10 s composition with the same shots and card, rendered at
  800/1920 scale. Every 3rd frame of the 30 fps render is kept (10 fps, 100 frames) and
  encoded with one global 128-colour palette, no dithering.

## Where the numbers come from

Every number is from `run-003/result.json`. The total, 1,422 trains, is
`workload.offered.trains`, and the 100 ms deadline is `workload.deadline_ms`.

| Shown | Value in `result.json` | Field |
|---|---|---|
| GPU only: 1,407 / 1,422 late | 1407 | `configs.gpu_only.summary.late.count` |
| GPU only: P99 ~3.1 s | 3137.24 ms | `configs.gpu_only.summary.decision_latency.p99_ms` |
| GPU + ANE: 0 / 1,422 late | 0 | `configs.hybrid.summary.late.count` |
| GPU + ANE: P99 ~55 ms | 54.68 ms | `configs.hybrid.summary.decision_latency.p99_ms` |

The late counts are exact. The two P99 values are rounded for display: to one decimal
place in seconds at or above 1 s, and to whole milliseconds below. The README table keeps the
exact ranges over all three runs.

The page's running counters (trains late so far, P99 so far) are outside the clip. The only
other numbers on screen are the replay's own "N waiting" labels under each queue, which count
the trains waiting at that moment.
