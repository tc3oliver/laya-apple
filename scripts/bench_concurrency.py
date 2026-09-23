"""v0.2 release benchmark: heterogeneous execution through the product runtime.

    LAYA_APPLE_CACHE=... uv run python scripts/bench_concurrency.py --model laya-typed-decisions \
        --short 128 --long 1024 --output benchmarks/v0.2/concurrency-typed.json

Part A, exit gate: the Phase -1 short/long mix, closed-loop, one
client thread per stream, all through ONE `Laya(execution="workers")` instance.
  solo_short / solo_long   each stream alone (device="auto")
  hetero                   both streams, device="auto" (short -> ANE, long -> MLX by routing)
  gpu_only                 both streams, device="gpu" (the best single device, Phase -1)
Gate: aggregate(hetero) >= 2.5x aggregate(gpu_only); each stream's P99 within 10% of solo;
every answer identical to the inline-mode answer for the same request and device.

Part B, open-loop mixed workload: Poisson arrivals (and an on/off bursty variant) of a mix
of request classes (short 1q, medium 1q, long 1q, short 4q), for device="auto" workers
vs device="gpu" workers at the same offered load. Per class: completed req/s, P50/P95/P99
latency from arrival (queueing included), device and routing-reason counts.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import wait
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def stats(values) -> dict:
    a = np.asarray(values, np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ms": float(a.mean()),
        "max_ms": float(a.max()),
    }


def conditions() -> dict:
    """Background load before a window: other processes share this machine's GPU/ANE/CPU."""
    import os
    import subprocess

    env = dict(os.environ, LC_ALL="C")  # ps formats %cpu per locale, force period decimals
    top = subprocess.run(["ps", "-Ao", "%cpu=,comm="], capture_output=True, text=True, env=env).stdout.splitlines()
    busy = sorted((line.split(None, 1) for line in top if line.strip()), key=lambda x: -float(x[0]))[:5]
    return {"loadavg": os.getloadavg(), "top_cpu": [(float(c), n.rsplit("/", 1)[-1]) for c, n in busy]}


def closed_loop(laya, req, start_at, end_at, out, reference):
    while time.monotonic() < start_at:
        time.sleep(0.0005)
    lat, devices, mismatches = [], Counter(), 0
    while True:
        t0 = time.monotonic()
        if t0 >= end_at:
            break
        r = laya.predict(context=req["state"], questions=req["questions"])
        lat.append((time.monotonic() - t0) * 1e3)
        devices[r.runtime.device] += 1
        if r.answers != reference[r.runtime.device]:
            mismatches += 1
    out.update(latency_ms=lat, wall_s=time.monotonic() - start_at, devices=dict(devices), mismatches=mismatches)


def part_a(args, laya_auto, laya_gpu, reqs, refs):
    windows = []
    conds = [
        ("solo_short", laya_auto, ["short"]),
        ("solo_long", laya_auto, ["long"]),
        ("hetero", laya_auto, ["short", "long"]),
        ("gpu_only", laya_gpu, ["short", "long"]),
    ]
    for cycle in range(args.cycles):
        for name, laya, streams in conds if cycle % 2 == 0 else list(reversed(conds)):
            time.sleep(2.0)
            cond_before = conditions()
            start_at = time.monotonic() + 0.5
            end_at = start_at + args.seconds
            outs = {s: {} for s in streams}
            ts = [
                threading.Thread(target=closed_loop, args=(laya, reqs[s], start_at, end_at, outs[s], refs[s]))
                for s in streams
            ]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            w = {"cycle": cycle, "condition": name, "streams": {}, "conditions_before": cond_before}
            for s, o in outs.items():
                w["streams"][s] = {
                    **stats(o["latency_ms"]),
                    "req_s": len(o["latency_ms"]) / o["wall_s"],
                    "devices": o["devices"],
                    "mismatches": o["mismatches"],
                    "latency_ms": o["latency_ms"],
                }
            windows.append(w)
            print(
                "A",
                cycle,
                name,
                {
                    s: (round(v["req_s"], 1), round(v["p99_ms"], 2), v["devices"], v["mismatches"])
                    for s, v in w["streams"].items()
                },
                flush=True,
            )
    med = defaultdict(lambda: defaultdict(list))
    for w in windows:
        for s, v in w["streams"].items():
            med[w["condition"]][s].append(v)
    summary = {}
    for c, streams in med.items():
        summary[c] = {
            s: {
                "req_s": float(np.median([v["req_s"] for v in vs])),
                "p50_ms": float(np.median([v["p50_ms"] for v in vs])),
                "p99_ms": float(np.median([v["p99_ms"] for v in vs])),
                "mismatches": int(sum(v["mismatches"] for v in vs)),
            }
            for s, vs in streams.items()
        }
    agg = {c: sum(v["req_s"] for v in s.values()) for c, s in summary.items()}
    gate = {
        "aggregate_hetero_req_s": agg["hetero"],
        "aggregate_gpu_only_req_s": agg["gpu_only"],
        "ratio": agg["hetero"] / agg["gpu_only"],
        "p99_vs_solo": {
            "short": summary["hetero"]["short"]["p99_ms"] / summary["solo_short"]["short"]["p99_ms"] - 1,
            "long": summary["hetero"]["long"]["p99_ms"] / summary["solo_long"]["long"]["p99_ms"] - 1,
        },
        "mismatches": sum(v["mismatches"] for s in summary.values() for v in s.values()),
    }
    gate["passed"] = bool(
        gate["ratio"] >= 2.5 and all(v <= 0.10 for v in gate["p99_vs_solo"].values()) and gate["mismatches"] == 0
    )
    print("A gate", json.dumps(gate), flush=True)
    return {"windows": windows, "summary": summary, "gate": gate}


def arrivals(rate, seconds, seed, bursty):
    rng = random.Random(seed)
    t, out = 0.0, []
    while t < seconds:
        r = rate
        if bursty:  # 1 s on at 3x the rate, 2 s off: same mean offered load
            r = rate * 3 if (t % 3.0) < 1.0 else 1e-9
        gap = rng.expovariate(r)
        if bursty and (t % 3.0) >= 1.0:
            t = (t // 3.0 + 1) * 3.0
            continue
        t += gap
        out.append(t)
    return [x for x in out if x < seconds]


def part_b(args, lays, classes, refs_b, rate, bursty):
    names = [c for c, _ in classes]
    weights = [w for _, w in classes]
    results = {}
    for label, laya in lays.items():
        # identical arrival times and classes for every configuration
        times = arrivals(rate, args.open_seconds, seed=11, bursty=bursty)
        rng = random.Random(7)
        picks = [rng.choices(names, weights)[0] for _ in times]
        records = []
        start = time.monotonic() + 0.5
        futs = []
        for at, cls in zip(times, picks):
            while time.monotonic() < start + at:
                time.sleep(0.0002)
            sent = time.monotonic()
            f = laya.submit(context=args._reqs_b[cls]["state"], questions=args._reqs_b[cls]["questions"])
            futs.append((cls, sent, f))
        wait([f for _, _, f in futs], timeout=600)
        end = time.monotonic()
        for cls, sent, f in futs:
            r = f.result()
            records.append(
                {
                    "class": cls,
                    "device": r.runtime.device,
                    "reason": r.runtime.routing_reason,
                    "latency_ms": r.runtime.latency_ms,  # from submit (= arrival) to result, queueing included
                    "queue_wait_ms": r.runtime.queue_wait_ms,
                    "match": r.answers == refs_b[cls][r.runtime.device],
                }
            )
        per = {}
        for cls in names:
            rs = [x for x in records if x["class"] == cls]
            per[cls] = {
                **stats([x["latency_ms"] for x in rs]),
                "devices": dict(Counter(x["device"] for x in rs)),
                "reasons": dict(Counter(x["reason"] for x in rs)),
                "mismatches": sum(not x["match"] for x in rs),
            }
        results[label] = {
            "offered_req_s": len(times) / args.open_seconds,
            "completed": len(records),
            "makespan_s": end - start,
            "completed_req_s": len(records) / (end - start),
            "classes": per,
        }
        print(
            "B",
            "bursty" if bursty else "poisson",
            label,
            round(results[label]["completed_req_s"], 1),
            {
                c: (round(v.get("p50_ms", 0), 1), round(v.get("p99_ms", 0), 1), v["devices"], v["mismatches"])
                for c, v in per.items()
            },
            flush=True,
        )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laya-typed-decisions")
    ap.add_argument("--short", type=int, default=128)
    ap.add_argument("--long", type=int, default=1024)
    ap.add_argument("--medium", type=int, default=512)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--open-seconds", type=float, default=20)
    ap.add_argument("--rates", type=float, nargs="+", default=[20, 40])
    ap.add_argument("--output", required=True)
    ap.add_argument("--part", choices=["a", "b", "ab"], default="ab")
    ap.add_argument("--ane-placement", choices=["auto", "thread", "process"], default="auto")
    args = ap.parse_args()

    import laya_apple
    from laya_apple import Laya
    from laya_apple.artifacts import platform_profile
    from laya_apple.workload import make_request

    t = time.perf_counter()
    laya_auto = Laya.from_pretrained(
        args.model, device="auto", execution="workers", local_files_only=True, ane_placement=args.ane_placement
    )
    laya_gpu = Laya.from_pretrained(args.model, device="gpu", execution="workers", local_files_only=True)
    load_s = time.perf_counter() - t
    # inline references per device, for correctness under load
    ref_gpu = Laya.from_pretrained(args.model, device="gpu", local_files_only=True)
    ref_ane = Laya.from_pretrained(args.model, device="ane", local_files_only=True)
    tok, cfg = laya_auto.tokenizer, laya_auto.config

    def req(L, q, seed=0):
        state, qs = make_request(tok, cfg, L, n_questions=q, seed=seed)
        return {"state": state, "questions": qs, "length": L, "q": q}

    def refs_for(r):
        out = {"gpu": ref_gpu.predict(context=r["state"], questions=r["questions"]).answers}
        try:
            out["ane"] = ref_ane.predict(context=r["state"], questions=r["questions"]).answers
        except laya_apple.LayaAppleError:
            out["ane"] = None
        return out

    reqs = {"short": req(args.short, 1), "long": req(args.long, 1)}
    refs = {k: refs_for(v) for k, v in reqs.items()}
    for laya in (laya_auto, laya_gpu):  # warm both worker paths
        for r in reqs.values():
            for _ in range(5):
                laya.predict(context=r["state"], questions=r["questions"])

    record = {
        "experiment": "v0.2 heterogeneous execution release benchmark",
        "args": vars(args),
        "laya_apple": laya_apple.__version__,
        "platform": platform_profile() | {"python": platform.python_version()},
        "time": datetime.now(timezone.utc).isoformat(),
        "load_s_two_workers_instances": load_s,
        "auto_info": laya_auto.info(),
    }
    if "a" in args.part:
        record["part_a"] = part_a(args, laya_auto, laya_gpu, reqs, refs)

    classes = [("short_1q", 0.6), ("medium_1q", 0.2), ("long_1q", 0.1), ("short_4q", 0.1)]
    args._reqs_b = {
        "short_1q": req(args.short, 1, seed=1),
        "medium_1q": req(args.medium, 1, seed=2),
        "long_1q": req(args.long, 1, seed=3),
        "short_4q": req(args.short, 4, seed=4),
    }
    refs_b = {k: refs_for(v) for k, v in args._reqs_b.items()}
    record["part_b"] = {}
    for rate in args.rates if "b" in args.part else []:
        for bursty in (False, True):
            key = f"{'bursty' if bursty else 'poisson'}@{rate:g}"
            record["part_b"][key] = part_b(
                args, {"auto": laya_auto, "gpu_only": laya_gpu}, classes, refs_b, rate, bursty
            )
    del args._reqs_b
    record["args"] = {k: v for k, v in vars(args).items()}
    for laya in (laya_auto, laya_gpu):
        laya.close()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
