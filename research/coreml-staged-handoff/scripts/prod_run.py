"""One run of the production staged handoff (Phases 4-7; ../criteria.md, addendum 2).

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run --extra ane python \
        research/coreml-staged-handoff/scripts/prod_run.py --cell P --model laya --short 128 --long 512 \
        --schedule mix --output research/coreml-staged-handoff/raw-prod/mix-laya-P-r1.json.gz

It runs the released code path, not a research model swap: laya_apple.Laya with execution="workers",
device="auto" and ane_handoff=True (cell P, the prototype's staged handoff, opt-in; addendum 3),
or with ane_handoff=False (cell A, today's production path and the default). The GPU-only instance of the mix is
device="gpu", execution="workers", as in #92's part_a.

The workload is #92's: one closed-loop client per stream (scripts/bench_concurrency.closed_loop,
unchanged), one short (1 question) and one long request, answers checked against inline GPU / ANE
references, 5 warm-up predicts per stream and instance. Windows start 0.5 s after they are scheduled,
after the schedule's gap. Schedules:

  mix      #92's part_a order: solo_short, solo_long, hetero, gpu_only, reversed on the odd cycle;
           2 cycles, 20 s windows, 2.0 s gaps (Phases 4 and 5)
  product  idle 10 s, then: hetero, solo_short, hetero, solo_long, hetero, gpu_only, hetero,
           solo_long, solo_short, hetero, hetero; hetero 20 s, single-device 10 s, 2.0 s gaps
           (the second-last pair is hetero -> two single-device windows -> hetero; the last pair is
           hetero -> 2 s gap -> hetero) (Phase 6)
  soak     60 hetero windows of 8 s; before each one a gap window cycling idle 3 s, solo_short 4 s,
           solo_long 4 s, gpu_only 4 s; 1.0 s gaps (Phase 7)

Recorded (harness only; nothing is added to laya_apple):
- per window: condition, cycle / index, start and end (time.monotonic_ns), per stream the client
  latencies, devices, mismatches, req/s;
- every request of the auto instance through its RequestTrace callback: target, submit, prepared,
  service start and end, received, response (time.monotonic_ns);
- cell P: every StagedHandoff.decide() of the auto instance (decision_ns, path, state, count),
  by wrapping the class method; and the handoff's snapshot() at the end;
- the auto instance's info() at start and end, both workers' liveness at the end.
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
SCHEDULES = ("mix", "product", "soak")
TRACE_COLUMNS = [
    "target",
    "submit_ns",
    "prepared_ns",
    "service_start_ns",
    "service_end_ns",
    "received_ns",
    "response_ns",
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
    if name == "soak":
        gaps = [("idle", 3.0), ("solo_short", 4.0), ("solo_long", 4.0), ("gpu_only", 4.0)]
        out = []
        for i in range(60):
            c, s = gaps[i % len(gaps)]
            out.append({"condition": c, "seconds": s, "gap_s": 1.0})
            out.append({"condition": "hetero", "seconds": 8.0, "gap_s": 1.0})
        return out
    raise ValueError(name)


STREAMS = {
    "solo_short": ("auto", ["short"]),
    "solo_long": ("auto", ["long"]),
    "hetero": ("auto", ["short", "long"]),
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
    return ap.parse_args()


def main():
    a = parse()
    t_start = time.monotonic()
    import laya_apple
    from laya_apple import Laya
    from laya_apple.workload import make_request

    decisions: list = []
    if a.cell == "P":
        from laya_apple import handoff as handoff_mod

        base_decide = handoff_mod.StagedHandoff.decide

        def decide(self):
            out = base_decide(self)
            decisions.append((time.monotonic_ns(), PATH_CODE[out[0]], out[1], out[2]))
            return out

        handoff_mod.StagedHandoff.decide = decide

    trace_rows: list = []
    tlock = threading.Lock()

    def record(t):
        row = (t.target, t.submit_ns, t.prepared_ns, t.service_start_ns, t.service_end_ns, t.received_ns, t.response_ns)
        with tlock:
            trace_rows.append(row)

    kw = {"ane_handoff": a.cell == "P"}  # addendum 3: the prototype's handoff is opt-in
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
        refs[k] = {
            "gpu": ref_gpu.predict(context=r["state"], questions=r["questions"]).answers,
            "ane": ref_ane.predict(context=r["state"], questions=r["questions"]).answers,
        }
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
                target=bench_concurrency.closed_loop,
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
        if a.schedule != "soak" or w["condition"] == "hetero" and i % 20 == 1:
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
        "experiment": "coreml-staged-handoff production",
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
