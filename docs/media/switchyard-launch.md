# Switchyard launch video

`switchyard-launch.mp4` (1920×1080, 30 fps, 15 s) and `switchyard-launch.gif` (800×450,
10 fps, 10 s: the two replay shots and the first 2 s of the result card) come from one
recorded Switchyard run on an Apple M4 Max with macOS 26.6.2:
[`benchmarks/switchyard/v1-m4-max/raw/run-003/`](../../benchmarks/switchyard/v1-m4-max/raw/run-003/).
They are made with Remotion outside this repository, so the runtime has no Node dependency.

| Input | SHA-256 |
|---|---|
| `result.json` | `d638d28e41700e7664aeee7b43609b81fd9a009831cbec241cdfa44fa5de4146` |
| `replay.html`, result card frames | `4df6fa38e0f8e63d2031ba9d7d9eca35874687c6de1f380430519549c9aeee19` |
| `replay.html`, all other frames | `6ba8ccc41da3414ed7677b9d28ae3200aa242d56207654de4986c36a7d04127c` |

The `6ba8…` page was a pre-release build of the replay page and cannot be rebuilt from
committed code. The result card frames come from the `4df6…` page, which is the committed
replay page built from this run (see [Result card frames](#result-card-frames)).

The run was picked by rule, not by eye: of the campaign's three runs, the one whose GPU-only
P99 decision latency is the median (3107.72, **3137.24**, 3170.55 ms). This is a different run
from the screenshots in [`switchyard.md`](switchyard.md), which use run-001.

| Time | Shows |
|---|---|
| 0–2 s | "Have a Mac?", then `uvx laya-apple switchyard` |
| 2–6 s | The GPU-only round, data time 30.0–31.0 s: the rush hour that starts at 30 s |
| 6–10 s | Hard cut to the GPU + ANE round, the same data window on the same timetable |
| 10–13 s | The result card. The P99 queue wait and "Slower than" rows are dimmed, not removed |
| 13–15 s | `uvx laya-apple switchyard` and the repository link |

## How it was made

Apart from the result card, the frames come from the run's `replay.html` at the time (the
`6ba8…` build), opened unmodified in headless Chrome at 1600×900, device
scale factor 1.2, with the Engines drawer open and the page's own speed of 0.25× real time
(4 s of video = 1 s of data). `requestAnimationFrame` was replaced by a virtual clock advanced
exactly one video frame per screenshot, so every frame is deterministic.

The video covers the page's header bar (brand, round tabs, playback buttons) with a bar in the
page's own colours holding the configuration label, and adds a light vignette. Those overlays
contain no numbers.

## Result card frames

The card frames are MP4 frames 300–389 (10.0–13.0 s) and GIF frames 80–99. They show the
PNG that the committed replay page's **Save PNG** button produces. That page was built with

```bash
uv run laya-apple switchyard --replay benchmarks/switchyard/v1-m4-max/raw/run-003 --no-open
```

and has SHA-256 `4df6…`. These frames replace the card of the pre-release build, which
rounded the miss rates differently: it showed the GPU-only 25 ms and 50 ms rates, 0.997 and
0.994 in `result.json`, as "100%" and "99%". The committed page shows 99.7% and 99.4%.
Every other number on the card is the same in both builds.

How the frames were redone:
- Both cards were rendered at 2×: the committed page's, and one from a scratch copy of the page with only the old rounding restored. Both were scaled to the frame size. They differ only around those two figures.
- In each card frame, a per-channel gain and offset was fitted between the old card and the frame over each differing region. This captures that frame's dimming and vignette.
- That scaled difference between the new and old card was added to the frame there. All other pixels are unchanged.
- The GIF's frames 0–79 decode pixel-identical to the previous GIF.
- The MP4 was re-encoded (H.264 High, yuv420p, CRF 16, 30 fps, faststart). Outside the card frames it matches a plain re-encode of the previous MP4 with the same settings, a mean difference of 0.007 levels. So those frames differ from the previous file only by encoding noise.

## Where the numbers come from

- **Replay shots:** the status row shows running values: trains late so far, and the P99 of
  the answers received so far. The page computes them from the trains in its replay data,
  so they are not the final round totals.
- **Result card:** every number comes from `run-003/result.json`, as described for the card in
  [`switchyard.md`](switchyard.md#where-the-numbers-come-from). For this run: 1,407 of 1,422
  trains late and P99 decision latency 3,137 ms GPU-only; 0 late and 54.7 ms GPU + ANE.
  GPU-only miss rates at 25, 50, 100, 250 and 500 ms show as 99.7%, 99.4%, 99%, 97% and 89%.
