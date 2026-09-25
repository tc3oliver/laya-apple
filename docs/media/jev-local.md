# Jev-local launch media

`jev-local.mp4` (1920×1080, 30 fps, 24 s) and `jev-local.gif` (960×540, 15 fps, 24 s) show
a Jev client pointed at `laya-apple serve` on an Apple M4 Max, then a replay of the
serve-under-LLM-load benchmark. They are made with Remotion outside this repository, so the
runtime has no Node dependency. Every number and data string on screen is read from the
files below at tag `v1.4.0`; none is typed in by hand.

| Input (at `v1.4.0`) | SHA-256 |
|---|---|
| [`docs/readme/serve-demo.json`](../readme/serve-demo.json) | `88f94299745bda73f78515cff1d5027a753d067067bf928fb3803677443ea073` |
| [`integrations/jev-plugins/README.md`](../../integrations/jev-plugins/README.md) | `f315026c2d11b237ecb331db7115549f9ee11954b0f9c6ac657bd56e210f2dbb` |
| [`benchmarks/serve/m4-max-r2/raw/002-decisions_llm-gpu.json`](../../benchmarks/serve/m4-max-r2/raw/002-decisions_llm-gpu.json) | `04757fbb3b06196007e0706eab5c739ab93d572407cbee2c2cf40f2d1d59ef2d` |
| [`benchmarks/serve/m4-max-r2/raw/004-decisions_llm-auto.json`](../../benchmarks/serve/m4-max-r2/raw/004-decisions_llm-auto.json) | `0804fc060f1e03ef6d8a1f2472f5f95c18bbf189d7be0575d6a3c66f5edba15f` |
| [`benchmarks/serve/m4-max-r2/results.json`](../../benchmarks/serve/m4-max-r2/results.json) | `9511be6f9e6332c1802735e8fdcfe40312b17a9f843a031de234dc49248187c5` |

The replayed windows are picked by rule, not by eye: the lowest-index `decisions_llm`
window of each configuration, and its first 15 s.

## Timing

| Time | Shows |
|---|---|
| 0–4 s | A terminal: `export TYPESAFE_BASE_URL=http://127.0.0.1:8642`; "Point your Jev client at your Mac." |
| 4–9 s | The recorded serve session: `python next_step.py` → `fix_code`, then a card: answered on the Neural Engine, 8.0 ms server-side, `laya` |
| 9–13 s | The 7 Jev clients tested unmodified, with their versions |
| 13–21 s | The two replayed windows, `--device gpu` and `--device auto`, each beside the busy local LLM, at 3× speed (printed on screen); then the full-run short-decision P99 and LLM tok/s change |
| 21–24 s | Install, start and `export TYPESAFE_BASE_URL=…`; non-affiliation; the repository |

## Where the numbers come from

| Shown | Value | Source |
|---|---|---|
| `fix_code {'fix_code': 0.602, 'run_tests': 0.2346, 'commit': 0.1634}` | as shown | `serve-demo.json`, `session` and `response.answers` |
| Neural Engine, 8.0 ms server-side, `laya` | `ane`, 7.967 ms, `laya` | `serve-demo.json`, `response.laya_apple` |
| 7 clients and versions | as shown | `integrations/jev-plugins/README.md`, the client table; every row says Pass |
| Decision marks and LLM stream ticks | send time, client-measured latency, device; chunk times | the two raw window files |
| Short-decision P99 (full run): 122 ms (`gpu`), 47 ms (`auto`) | 122.31, 47.18 ms | `results.json`, `cells.{gpu,auto}/decisions_llm.classes.short_1q.p99_ms_median` |
| LLM tok/s vs LLM alone (full run): −5.3%, −4.0% | 0.0535, 0.0404 | `results.json`, `llm_tok_s_drop` |
| 63 of 1,572 `auto` short decisions on the GPU | `{ane: 1509, gpu: 63}` | `results.json`, `cells.auto/decisions_llm.classes.short_1q.devices` |
| 4 windows per config | 4 | `results.json`, `cells.*.windows` |

The "answered in this replay" counters count the marks in the replayed 15 s only; they
are not full-run numbers.

## Rules the video keeps

- Colour is the device that answered, never the configuration: the `--device auto` short
  decisions that spilled to the GPU are amber. Outlined marks are multi-question requests,
  which always run on the GPU and are not part of the short-decision P99.
- Window data and full-run numbers are kept apart: the dashed P99 lines and the result table
  are labelled "full run" and appear only after the replay ends.
- The export command is the recorded one, shown as two lines (base URL, then placeholder key)
  so it fits the terminal. The typing cadence is synthetic.
- No Jev or TypeSafe branding, and the hosted Jev URL never appears. The end card says the
  API is Jev-compatible and powered by Laya, not affiliated with TypeSafe or Jev, and that
  the answers are Laya's, not Jev's.
