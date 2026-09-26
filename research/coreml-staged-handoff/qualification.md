# Environment qualification before the H64 production evaluation

This file is committed before any qualification run. Its purpose is to separate machine jitter
from H64 itself before the production evaluation. It changes no earlier verdict:
- the screen (`criteria.md`) stays RESEARCH-CLOSED;
- the H64 confirmation (`confirmation.md`) stays FAILED on its preregistered outlier guard.

## Environment

- **Machine state:**
  - oMLX / the local LLM server stopped;
  - no other GPU / ANE workload started by us;
  - no pytest, build, benchmark or agent work while a run is in progress;
  - AC power.
- **Recorded, not changed:**
  - OrbStack runs its long-lived services, and its CPU use is recorded;
  - the desktop's Aerial wallpaper decodes video continuously (WallpaperAerialsExtension,
    VTDecoderXPCService). It was present in every earlier run of this directory.
- **Snapshot per run.** `scripts/machine_snapshot.py` runs right before and right after each run.
  It uses only public tools and records:
  - uptime and load average;
  - CPU user / sys / idle;
  - PhysMem and memory_pressure free %;
  - swap usage and the vm_stat swapin / swapout / pageout counters;
  - pmset thermal and performance warnings;
  - power source;
  - Time Machine running;
  - the top CPU consumers.

## Runs

**3 fresh-process runs of production A:** `scripts/prod_run.py --cell A --model laya --short 128
--long 512 --schedule product`.
- **A is the default Laya.** It runs with `execution="workers"`, `device="auto"` and no
  `ane_handoff` argument, on the unmodified main code (74c6842). This is 1.4's production path.
- **The product schedule** is idle 10 s, then: hetero, solo_short, hetero, solo_long, hetero,
  gpu_only, hetero, solo_long, solo_short, hetero, hetero.
  - The hetero windows are 20 s and the single-device windows 10 s.
  - That gives 7 hetero transitions per run, 21 in all.
- **No H64 run is made here.**
- Raw output goes to `raw-qual/`.

## Reported per run

- **Hetero short requests:** median, P95, P99 and P99.9, per window and pooled.
- **Aggregate req/s** per hetero window.
- **GPU return** (received − service_end) P50 / P95 / P99 over the hetero windows.
- **Correctness:** mismatches, routing and crash.
- **Host-slow:** prepare > 0.3 ms, in 0.5 s bins.
- **The machine snapshots.**

## Rule (fixed before the runs)

**Environment PASS** needs every one of the following, for all 3 runs:
1. **Correctness:** 0 mismatches and 0 routing failures in every window, no crash, and both
   workers alive at the end.
2. **Throughput:** every hetero window's aggregate req/s is in #94's A range, [109.9, 135.0].
3. **No persistent host-slow:** no hetero window has 2 or more consecutive 0.5 s bins in
   [t0 + 1 s, end) with a host-slow share ≥ 0.10.
4. **No sustained slow state:** no hetero window has 2 or more consecutive 1 s spans in
   [t0 + 1 s, end) whose short median latency is > 1.2 × that run's pooled hetero short median.
5. **Machine, in both snapshots of every run:**
   - no thermal and no performance warning;
   - AC power;
   - memory free ≥ 50%;
   - Time Machine not running;
   - CPU idle ≥ 80% in the snapshot before the run;
   - no process other than the benchmark's own python above 50% CPU;
   - swapins + swapouts during the run (after − before) ≤ 10,000 pages.

**Not a reason for INVALID:** a single high P99 on its own, without a failure of items 1–5.

**Environment INVALID:** any item fails. The campaign then stops, the environment is reported, and
no H64 run starts. The runs are not repeated until the numbers look good.

Analysis: `scripts/qual_analyze.py`.
