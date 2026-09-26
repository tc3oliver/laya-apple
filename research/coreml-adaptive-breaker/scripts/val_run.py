"""One release-validation run of 1.5's adaptive execution (../validation.md).

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run --extra ane python \
        research/coreml-adaptive-breaker/scripts/val_run.py --cell P --model laya --short 128 --long 512 \
        --schedule mix --output research/coreml-adaptive-breaker/raw-val/mix-laya-P-r1.json.gz

It runs the released code path: laya_apple.Laya with execution="workers" and device="auto".
- **Cell P** passes no ane_handoff: the production default, adaptive execution wherever
  eligible.
- **Cell A** passes ane_handoff=False: 1.4's path.
- The GPU-only instance of the mix is device="gpu", execution="workers", as in #92's part_a.

This is research/coreml-staged-handoff/scripts/prod_run.py (#103's runner) with:
- the cells above;
- a `bursty` schedule;
- the breaker's trips, recorded by wrapping StagedHandoff.observe.

**The workload** is #92's:
- one closed-loop client per stream (scripts/bench_concurrency.closed_loop, unchanged), one short
  (1 question) and one long request;
- answers checked against inline GPU / ANE references;
- 5 warm-up predicts per stream and instance.

Windows start 0.5 s after they are scheduled, after the schedule's gap.

**Schedules:**
- **mix:** #92's part_a order (solo_short, solo_long, hetero, gpu_only), reversed on the odd cycle;
  2 cycles, 20 s windows, 2.0 s gaps.
- **product:** idle 10 s, then hetero, solo_short, hetero, solo_long, hetero, gpu_only, hetero,
  solo_long, solo_short, hetero, hetero; hetero windows 20 s, single-device windows 10 s, 2.0 s
  gaps.
- **bursty:** idle 5 s, then 8 hetero_bursty windows of 12 s with 2.0 s gaps.
  - The long stream is continuous.
  - The short stream runs in bursts: closed loop for 0.4 s, then 0.2 s with no request.
  - The pauses are shorter than the 1.0 s episode gap, so each window is one episode.
- **productsoak** (validation.md addendum 1): 48 units, each "predecessor window, 1.0 s gap,
  hetero 8 s". The predecessors cycle through idle 3 s, solo_short 4 s, solo_long 4 s, gpu_only
  4 s, hetero 8 s and hetero_bursty 8 s. Every hetero window is an episode: 64 per run (56 hetero,
  8 bursty).
- **soak55:** 55 units per run, each "predecessor window, 1.0 s gap, hetero 8 s".
  - The predecessors cycle through idle 3 s, solo_short 4 s, solo_long 4 s, gpu_only 4 s and a
    hetero 8 s window.
  - That gives 66 hetero episodes per run.

**Recorded** (harness only; nothing is added to laya_apple):
- **per window:** condition, index, start and end (time.monotonic_ns); per stream the client
  latencies, devices, mismatches and req/s;
- **every request of the auto instance,** through its RequestTrace callback: target, submit,
  prepared, service start and end, received, response;
- **cell P:**
  - every StagedHandoff.decide() of the auto instance (decision_ns, path, state, count);
  - every StagedHandoff.observe() that opened the breaker (t_ns);
  - the handoff's snapshot() at the end;
- the auto instance's info() at start and end, and both workers' liveness at the end.
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import bench_concurrency  # noqa: E402  #92's closed loop and stats, unchanged

CELLS = ("A", "P")
SCHEDULES = ("mix", "product", "bursty", "soak55", "productsoak")
TRACE_COLUMNS = [
    "target",
    "submit_ns",
    "prepared_ns",
    "service_start_ns",
    "service_end_ns",
    "received_ns",
    "response_ns",
    "request_id",
]
PATH_CODE = {"sync": 0, "async": 1}


def schedule(name: str, seconds: float = 20.0) -> list[dict]:
    """[{condition, seconds, gap_s}] in order; condition in solo_short, solo_long, hetero,
    gpu_only, idle."""
    if name == "mix":
        conds = ["solo_short", "solo_long", "hetero", "gpu_only"]
        out = []
        for k in range(2):
            for c in conds if k % 2 == 0 else list(reversed(conds)):
                out.append({"condition": c, "seconds": seconds, "gap_s": 2.0, "cycle": k})
        return out
    if name == "product":
        seq = ["hetero", "solo_short", "hetero", "solo_long", "hetero", "gpu_only", "hetero"]
        seq += ["solo_long", "solo_short", "hetero", "hetero"]
        out = [{"condition": "idle", "seconds": 10.0, "gap_s": 2.0}]
        for c in seq:
            out.append({"condition": c, "seconds": 20.0 if c == "hetero" else 10.0, "gap_s": 2.0})
        return out
    if name == "bursty":
        out = [{"condition": "idle", "seconds": 5.0, "gap_s": 2.0}]
        for _ in range(8):
            out.append({"condition": "hetero_bursty", "seconds": 12.0, "gap_s": 2.0})
        return out
    if name == "productsoak":  # validation.md addendum 1: the combined product-mix soak
        before = [
            ("idle", 3.0),
            ("solo_short", 4.0),
            ("solo_long", 4.0),
            ("gpu_only", 4.0),
            ("hetero", 8.0),
            ("hetero_bursty", 8.0),
        ]
        out = []
        for i in range(48):
            c, s = before[i % len(before)]
            out.append({"condition": c, "seconds": s, "gap_s": 1.0, "role": "before"})
            out.append({"condition": "hetero", "seconds": 8.0, "gap_s": 1.0, "role": "episode"})
        return out
    if name == "soak55":  # evaluation.md: 55 hetero episodes per run, 11 of each predecessor kind
        before = [("idle", 3.0), ("solo_short", 4.0), ("solo_long", 4.0), ("gpu_only", 4.0), ("hetero", 8.0)]
        out = []
        for i in range(55):
            c, s = before[i % len(before)]
            out.append({"condition": c, "seconds": s, "gap_s": 1.0, "role": "before"})
            out.append({"condition": "hetero", "seconds": 8.0, "gap_s": 1.0, "role": "episode"})
        return out
    raise ValueError(name)


STREAMS = {
    "solo_short": ("auto", ["short"]),
    "solo_long": ("auto", ["long"]),
    "hetero": ("auto", ["short", "long"]),
    "hetero_bursty": ("auto", ["short", "long"]),
    "gpu_only": ("gpu", ["short", "long"]),
    "idle": (None, []),
}


def parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", choices=CELLS, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--short", type=int, required=True)
    ap.add_argument("--long", type=int, required=True)
    ap.add_argument("--schedule", choices=SCHEDULES, required=True)
    ap.add_argument("--seconds", type=float, default=20.0, help="mix window length (20 in the campaign)")
    ap.add_argument("--output", required=True, help=".json.gz")
    ap.add_argument(
        "--expect-rejected",
        action="store_true",
        help="validation.md's multilingual smoke: first check that ane_handoff=True raises ValueError",
    )
    return ap.parse_args()


BURST_ON_S, BURST_OFF_S = 0.4, 0.2


def bursty_loop(laya, req, start_at, end_at, out, reference):
    """closed_loop in bursts: BURST_ON_S of back-to-back requests, then BURST_OFF_S idle."""
    lat, devices, mismatches = [], {}, 0
    t = start_at
    while t < end_at:
        seg: dict = {}
        bench_concurrency.closed_loop(laya, req, t, min(t + BURST_ON_S, end_at), seg, reference)
        lat += seg["latency_ms"]
        for k, v in seg["devices"].items():
            devices[k] = devices.get(k, 0) + v
        mismatches += seg["mismatches"]
        t += BURST_ON_S + BURST_OFF_S
    while time.monotonic() < end_at:
        time.sleep(0.005)
    out.update(latency_ms=lat, wall_s=time.monotonic() - start_at, devices=devices, mismatches=mismatches)


def runtime_revision() -> dict:
    """The laya-apple code under test: HEAD, and whether the runtime, the lock or the harness scripts
    differ from it."""
    import subprocess

    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()

    return {
        "head": git("rev-parse", "HEAD"),
        "laya_apple_tree": git("rev-parse", "HEAD:laya_apple"),
        "pyproject_blob": git("rev-parse", "HEAD:pyproject.toml"),
        "uv_lock_blob": git("rev-parse", "HEAD:uv.lock"),
        "dirty": bool(
            git(
                "status",
                "--porcelain",
                "--",
                "laya_apple",
                "pyproject.toml",
                "uv.lock",
                "scripts",
                "research/coreml-adaptive-breaker/scripts",
            )
        ),
    }


def main():
    a = parse()
    t_start = time.monotonic()
    import laya_apple
    from laya_apple import Laya
    from laya_apple.workload import make_request

    decisions: list = []
    trips: list = []
    if a.cell == "P":
        from laya_apple import handoff as handoff_mod

        base_decide = handoff_mod.StagedHandoff.decide
        base_observe = handoff_mod.StagedHandoff.observe

        def decide(self):
            out = base_decide(self)
            decisions.append((time.monotonic_ns(), PATH_CODE[out[0]], out[1], out[2]))
            return out

        def observe(self, prepare_ms):
            opened = base_observe(self, prepare_ms)
            if opened:
                trips.append(time.monotonic_ns())
            return opened

        handoff_mod.StagedHandoff.decide = decide
        handoff_mod.StagedHandoff.observe = observe

    trace_rows: list = []
    tlock = threading.Lock()

    def record(t):
        row = (
            t.target,
            t.submit_ns,
            t.prepared_ns,
            t.service_start_ns,
            t.service_end_ns,
            t.received_ns,
            t.response_ns,
            t.request_id,
        )
        with tlock:
            trace_rows.append(row)

    # P: the production default (no argument); A: 1.4's path, explicitly
    kw = {} if a.cell == "P" else {"ane_handoff": False}
    rejected = None
    if a.expect_rejected:
        try:
            Laya.from_pretrained(a.model, device="auto", execution="workers", local_files_only=True, ane_handoff=True)
            rejected = {"raised": None}
        except Exception as e:  # recorded: evaluation.md requires ValueError, nothing else
            rejected = {"raised": type(e).__name__, "message": str(e)}
    laya_auto = Laya.from_pretrained(
        a.model, device="auto", execution="workers", local_files_only=True, trace=record, **kw
    )
    laya_gpu = Laya.from_pretrained(a.model, device="gpu", execution="workers", local_files_only=True)
    ref_gpu = Laya.from_pretrained(a.model, device="gpu", local_files_only=True)
    ref_ane = Laya.from_pretrained(a.model, device="ane", local_files_only=True)
    tok, cfg = laya_auto.tokenizer, laya_auto.config

    def req(length):
        state, qs = make_request(tok, cfg, length, n_questions=1, seed=0)
        return {"state": state, "questions": qs, "length": length, "q": 1}

    reqs = {"short": req(a.short), "long": req(a.long)}
    refs = {}
    for k, r in reqs.items():
        refs[k] = {"gpu": ref_gpu.predict(context=r["state"], questions=r["questions"]).answers}
        try:  # as #92's bench_concurrency: a length beyond the ANE buckets has no ANE reference
            refs[k]["ane"] = ref_ane.predict(context=r["state"], questions=r["questions"]).answers
        except laya_apple.LayaAppleError:
            refs[k]["ane"] = None
    ref_gpu.close()
    ref_ane.close()
    for laya in (laya_auto, laya_gpu):  # warm both worker paths, as #92
        for r in reqs.values():
            for _ in range(5):
                laya.predict(context=r["state"], questions=r["questions"])
    info_start = laya_auto.info()
    with tlock:
        trace_rows.clear()  # the warm-up is not measured
    decisions.clear()
    trips.clear()

    instances = {"auto": laya_auto, "gpu": laya_gpu}
    windows = []
    for i, w in enumerate(schedule(a.schedule, a.seconds)):
        time.sleep(w["gap_s"])
        start_at = time.monotonic() + 0.5
        end_at = start_at + w["seconds"]
        inst, streams = STREAMS[w["condition"]]
        outs = {s: {} for s in streams}
        ts = [
            threading.Thread(
                target=bursty_loop
                if (w["condition"] == "hetero_bursty" and s == "short")
                else bench_concurrency.closed_loop,
                args=(instances[inst], reqs[s], start_at, end_at, outs[s], refs[s]),
                name=f"client-{s}",
            )
            for s in streams
        ]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        if not streams:
            while time.monotonic() < end_at:
                time.sleep(0.05)
        rec = {
            "index": i,
            "cycle": w.get("cycle"),
            "condition": w["condition"],
            "instance": inst,
            "start_ns": int(start_at * 1e9),
            "end_ns": int(end_at * 1e9),
            "streams": {},
        }
        for s, o in outs.items():
            rec["streams"][s] = {
                **{k: v for k, v in bench_concurrency.stats(o["latency_ms"]).items()},
                "req_s": len(o["latency_ms"]) / o["wall_s"],
                "devices": o["devices"],
                "mismatches": o["mismatches"],
                "latency_ms": o["latency_ms"],
            }
        windows.append(rec)
        if a.schedule != "soak55" or i % 20 == 1:
            print(
                i,
                w["condition"],
                {
                    s: (round(v["req_s"], 1), round(v.get("p99_ms", 0), 2), v["mismatches"])
                    for s, v in rec["streams"].items()
                },
                flush=True,
            )

    alive = {k: w.alive for k, w in laya_auto._workers.items()}
    info_end = laya_auto.info()
    snapshot = None
    ane_w = laya_auto._workers.get("ane")
    backend = getattr(ane_w, "_backend", None)
    h = getattr(backend, "handoff", None)
    if h is not None and hasattr(h, "snapshot"):
        snapshot = h.snapshot()
    laya_auto.close()
    laya_gpu.close()

    with tlock:
        rows = sorted(trace_rows, key=lambda r: r[1])
    record_out = {
        "experiment": "coreml-adaptive-breaker validation",
        "runtime": runtime_revision(),
        "expect_rejected": rejected,
        "args": vars(a),
        "laya_apple": laya_apple.__version__,
        "python": platform.python_version(),
        "info_start": info_start,
        "info_end": info_end,
        "workers_alive_at_end": alive,
        "handoff_snapshot": snapshot,
        "windows": windows,
        "trace_columns": TRACE_COLUMNS,
        "trace": [list(r) for r in rows],
        "decisions_columns": ["decision_ns", "path", "state", "count"],
        "decisions": decisions,
        "trips": trips,
        "run_wall_s": time.monotonic() - t_start,
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    with gzip.open(part, "wt") as fh:
        json.dump(record_out, fh, default=str)
    part.rename(out)
    print("wrote", out, f"({record_out['run_wall_s']:.0f} s)")


if __name__ == "__main__":
    main()
