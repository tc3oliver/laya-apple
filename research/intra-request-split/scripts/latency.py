"""One latency run of laya-apple's arms (split, gpu, ane) on every workload of one model.

    LAYA_APPLE_CACHE=<cache> HF_HUB_OFFLINE=1 uv run --extra ane --extra convert python \\
        research/intra-request-split/scripts/latency.py --model laya --run-id laya-b1-p1

Timing-sensitive: exclusive machine, AC power, the local LLM server and every other GPU/ANE
consumer stopped. Protocol (criteria.json, "protocol.latency"):

1. Open the product instance (common.open_laya): Laya(device="auto", execution="workers") with the
   workloads' ANE buckets loaded, the 1.5.0 tie band, adaptive execution as shipped.
2. Correctness pass: every fixture request once per arm, answers compared with the FP32
   reference (raw/reference-<model>.json) by the FP16 gate. Not timed.
3. Warm-up: `run_warmup_requests` per workload and arm, discarded.
4. `cycles` cycles. Each cycle runs every workload in a fixed order; within a workload the three
   arms run in an order rotated by (cycle + workload index) mod 3. A window is `idle_before_s` of
   idle, `window_warmup_requests` discarded requests, then `window_requests` measured requests,
   closed loop (the next request is sent when the previous answer is back), cycling the
   workload's request variants (seeds).

Latency is caller-observed: from the call that submits the request (before tokenisation) to the
answers in the calling thread. Every measured answer is also compared with the reference.
The run file is written once, at the end; a crashed run leaves nothing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import split  # noqa: E402


def call(laya, sub: split.SplitSubmitter, arm: str, req: dict):
    """(latency_ms, answers, record) for one request of one arm, caller-observed."""
    t = time.perf_counter_ns()
    if arm == "gpu":  # today's product path for these requests: Laya.submit, routed by the product
        res = laya.submit(context=req["state"], questions=req["questions"]).result()
        ms = (time.perf_counter_ns() - t) / 1e6
        rt = res.runtime
        return ms, res.answers, {"device": rt.device, "reason": rt.routing_reason, "device_ms": rt.device_ms}
    oc = sub.submit(req["state"], req["questions"], mode=arm).result()
    ms = (time.perf_counter_ns() - t) / 1e6
    p = oc.plan
    rec = {
        "k": p.k,
        "ane_rows": list(p.ane_rows),
        "buckets": list(p.buckets),
        "predicted_ms": round(p.makespan_ms, 3),
        "predicted_gpu_only_ms": round(p.gpu_only_ms, 3),
        "gpu_backlog_ms": round(p.gpu_backlog_ms, 3),
        "ane_backlog_ms": round(p.ane_backlog_ms, 3),
        "devices": {
            d: {
                "rows": v["rows"],
                "wait_ms": (v["dispatch_ns"] - v["queue_enter_ns"]) / 1e6,
                "device_ms": (v["end_ns"] - v["start_ns"]) / 1e6,
            }
            for d, v in oc.devices.items()
        },
    }
    return ms, oc.answers, rec


def check(ref: dict, got: dict) -> dict:
    c = common.compare_answers(ref, got)
    return {"prob_err": c["prob_err"], "act_err": c["act_err"], "hard": c["hard"], "flips": c["flips"]}


def summary(lat) -> dict:
    x = np.asarray(lat, np.float64)
    return {
        "n": len(x),
        "p50_ms": float(np.percentile(x, 50)),
        "p90_ms": float(np.percentile(x, 90)),
        "p99_ms": float(np.percentile(x, 99)),
        "mean_ms": float(x.mean()),
        "min_ms": float(x.min()),
        "max_ms": float(x.max()),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--run-id", required=True, help="e.g. laya-b1-p1 (block 1, position 1)")
    p.add_argument("--workloads", nargs="*", help="default: every workload that fits the model")
    p.add_argument(
        "--screen",
        action="store_true",
        help="addendum 1's stop-only screen: its workloads, arms and cycles; written to raw/screen/",
    )
    a = p.parse_args(argv)

    crit = common.criteria()
    proto = dict(crit["protocol"]["latency"])
    arms = common.ARMS
    out_path = common.RAW / "latency" / f"{a.run_id}.json"
    if a.screen:
        scr = common.screen_criteria()
        proto["cycles"] = scr["cycles"]
        arms = tuple(scr["arms"])
        a.workloads = scr["workloads"]
        out_path = common.RAW / "screen" / f"{a.run_id}.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; raw data is never overwritten")
    fx = common.load_fixtures(a.model)
    ref_path = common.RAW / f"reference-{a.model}.json"
    if not ref_path.exists():
        raise SystemExit(f"{ref_path} is missing: run scripts/reference.py --model {a.model} first")
    import json

    ref = json.loads(ref_path.read_text())
    if ref["fixtures_sha256"] != common.sha256_file(common.fixtures_path(a.model)):
        raise SystemExit("the reference was computed from different fixtures")
    names = a.workloads or list(fx["workloads"])
    workloads = {n: fx["workloads"][n] for n in names}
    refs = {n: {r["seed"]: r["answers"] for r in ref["workloads"][n]} for n in names}
    needed = sorted({split.bucket_for(w["tokens"], crit["policy"]["ane_bucket_ladder"]) for w in workloads.values()})

    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    env_start = common.environment()
    t_load = time.perf_counter()
    laya = common.open_laya(a.model, extra_buckets=needed)
    load_s = time.perf_counter() - t_load
    try:
        sub = split.SplitSubmitter(laya)
        info_start = laya.info()

        # 2. correctness pass (not timed)
        correctness = []
        for name, w in workloads.items():
            for req in w["requests"]:
                for arm in arms:
                    _, answers, rec = call(laya, sub, arm, req)
                    correctness.append(
                        {
                            "workload": name,
                            "seed": req["seed"],
                            "arm": arm,
                            **check(refs[name][req["seed"]], answers),
                            **rec,
                        }
                    )
        # 3. run warm-up
        for name, w in workloads.items():
            for arm in arms:
                for i in range(proto["run_warmup_requests"]):
                    call(laya, sub, arm, w["requests"][i % len(w["requests"])])

        # 4. cycles
        windows = []
        for cycle in range(proto["cycles"]):
            for wi, (name, w) in enumerate(workloads.items()):
                shift = (cycle + wi) % len(arms)
                order = arms[shift:] + arms[:shift]
                for pos, arm in enumerate(order):
                    time.sleep(proto["idle_before_window_s"])
                    reqs = w["requests"]
                    for i in range(proto["window_warmup_requests"]):
                        call(laya, sub, arm, reqs[i % len(reqs)])
                    rows = []
                    t_window = time.monotonic_ns()
                    for i in range(proto["window_requests"]):
                        req = reqs[i % len(reqs)]
                        ms, answers, rec = call(laya, sub, arm, req)
                        rows.append({"i": i, "seed": req["seed"], "latency_ms": ms, **rec, "answers": answers})
                    t_end = time.monotonic_ns()
                    for r in rows:  # after the window: comparison work never sits between requests
                        r.update(check(refs[name][r["seed"]], r.pop("answers")))
                    windows.append(
                        {
                            "cycle": cycle + 1,
                            "workload": name,
                            "arm": arm,
                            "position": pos,
                            "start_ns": t_window,
                            "end_ns": t_end,
                            "summary": summary([r["latency_ms"] for r in rows]),
                            "requests": rows,
                        }
                    )
                    s = windows[-1]["summary"]
                    print(
                        f"{a.run_id} c{cycle + 1} {name:6s} {arm:5s} P50 {s['p50_ms']:8.2f} P90 {s['p90_ms']:8.2f} ms",
                        flush=True,
                    )
        info_end = laya.info()
    finally:
        laya.close()
    common.write_json(
        out_path,
        {
            "run_id": a.run_id,
            "kind": "laya-apple",
            "model": a.model,
            "started": started,
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "protocol": proto,
            "arms": list(arms),
            "screen": a.screen,
            "policy": crit["policy"],
            "buckets_loaded_for_split": needed,
            "load_s": round(load_s, 2),
            "environment_start": env_start,
            "environment_end": common.environment(),
            "laya_info_start": info_start,
            "laya_info_end": info_end,
            "fixtures_sha256": common.sha256_file(common.fixtures_path(a.model)),
            "correctness": correctness,
            "windows": windows,
        },
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
