"""Generate the synthetic Switchyard replay fixtures (UI development and tests only).

    uv run python tests/fixtures/switchyard/make_synthetic.py

Writes three files:
  replay-synthetic.json                 ANE ready: gpu_only + hybrid
  replay-synthetic-gpu-only.json        ANE setup_available: gpu_only only
  replay-synthetic-ane-unverified.json  the Neural Engine stops being used 20 s into the hybrid
                                        round: ane_verified false, ANE unavailable, comparison
                                        not available

The timetable is the real switchyard-v1 schedule, but every timing is SIMULATED: a FIFO
single-server queue per device with made-up service times. The simulated trace records go
through the real `result.build` and `replay.replay_data`, so the files have exactly the
shape a real run produces, and `result.synthetic` is true in every one. Never use these numbers
in docs, the README or a benchmark claim.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from pathlib import Path

from laya_apple.demos.switchyard import ane_state, driver, replay, result, world
from laya_apple.routing import RUNTIME_UNAVAILABLE

HERE = Path(__file__).parent
SEED = world.DEFAULT_SEED
MS = 1_000_000  # ns per ms

# Median service time (ms) per (class, device) and a log-normal spread. Made up; of the same
# order as (a little faster than) the v1.0 trace in docs/media/heterogeneous-serving-trace.json.
SERVICE = {
    ("train", "gpu"): 9.8,
    ("train", "ane"): 9.6,
    ("medium_1q", "gpu"): 27.5,
    ("long_1q", "gpu"): 55.0,
    ("short_4q", "gpu"): 26.0,
}
SPREAD = 0.12
HYBRID_TRAIN_ON_GPU = 0.02  # a few trains the router keeps on the GPU
MISROUTE = 0.002

MACHINE = {
    "platform": {
        "soc": "Apple M4 Max",
        "macos": "26.6.2",
        "macos_build": "25G83",
        "coremltools": "9.0",
        "machine": "arm64",
    },
    "memory_gb": 64,
    "python": "3.12.14",
    "mlx": "0.29.0",
    "routing_profile_validated": True,
    "calibrated_profile": False,
}
MODEL = {
    "name": world.MODEL,
    "revision": "0" * 40,  # synthetic, not a real revision
    "weights_sha256": "0" * 64,
    "ane_artifacts": {},
}


def simulate(items, config, index, rng, ane_stops_s=None):
    """One simulated round, with trace records in driver.join's shape. With `ane_stops_s`,
    a hybrid round stops using the Neural Engine at that offset: from then on the router
    reports the ANE runtime gone and sends trains to the GPU."""
    start = 10**12 + index * 10**11
    free_at = {"gpu": 0, "ane": 0}
    records = []
    for a in items:
        arrival = start + round(a.offset_s * 1e9)
        device, reason = "gpu", "device_gpu"
        if config == "hybrid":
            reason = "long_or_multi_question"
            if a.cls == world.TRAIN and ane_stops_s is not None and a.offset_s >= ane_stops_s:
                reason = RUNTIME_UNAVAILABLE
            elif a.cls == world.TRAIN:
                device, reason = (
                    ("ane", "short_validated") if rng.random() >= HYBRID_TRAIN_ON_GPU else ("gpu", "ane_busy")
                )
        submit = arrival + round(rng.uniform(0.1, 0.4) * MS)
        queue_enter = submit + round(rng.uniform(0.05, 0.2) * MS)
        dispatch = max(queue_enter, free_at[device]) + round(rng.uniform(0.05, 0.3) * MS)
        service_start = dispatch + round(rng.uniform(0.1, 0.4) * MS)
        service_end = service_start + round(SERVICE[(a.cls, device)] * math.exp(rng.gauss(0, SPREAD)) * MS)
        free_at[device] = service_end
        response = service_end + round(rng.uniform(0.2, 0.8) * MS)
        rec = {
            "request_id": f"r{index}-{a.index}",
            "target": device,
            "routing_reason": reason,
            "submit_ns": submit,
            "queue_enter_ns": queue_enter,
            "dispatch_ns": dispatch,
            "service_start_ns": service_start,
            "service_end_ns": service_end,
            "response_ns": response,
            "round": index,
            "index": a.index,
            "class": a.cls,
            "arrival_ns": arrival,
        }
        if a.train is not None:
            oracle = a.train.oracle
            answer = oracle
            if rng.random() < MISROUTE:
                answer = rng.choice([p for p in world.PLATFORMS if p != oracle])
            latency = (response - arrival) / MS
            rec |= {
                "line": a.train.line,
                "train_id": a.train.train_id,
                "pattern": a.train.pattern,
                "answer": answer,
                "probabilities": {p: (0.94 if p == answer else 0.03) for p in world.PLATFORMS},
                "oracle": oracle,
                "outcome": "late"
                if latency > world.DEADLINE_MS
                else ("delivered" if answer == oracle else "misrouted"),
            }
        records.append(rec)
    started = f"2026-09-24T10:0{index * 2}:00Z"
    conditions = {
        "start": {"utc": started, "loadavg": [2.1, 2.4, 2.6], "top_cpu": [[12.0, "python3.12"]], "pmset_therm": []},
        "end": {
            "utc": f"2026-09-24T10:0{index * 2 + 1}:10Z",
            "loadavg": [3.0, 2.6, 2.6],
            "top_cpu": [[96.0, "python3.12"]],
            "pmset_therm": [],
        },
        "other_laya_processes": [],
    }
    return driver.Round(index, config, started, start, records, conditions)


def build(configs, ane, ane_stops_s=None):
    items = world.schedule(SEED, world.DURATION_S)
    sequence = list(configs)
    if len(configs) == 2 and SEED % 2:
        sequence.reverse()  # seed parity: odd seeds run hybrid first
    run = driver.Run(SEED, world.DURATION_S, items, world.warmup_schedule(SEED), ane)
    meta = {
        "gpu_only": {"device": "gpu", "execution": "workers", "ane_placement": None},
        "hybrid": {"device": "auto", "execution": "workers", "ane_placement": "process"},
    }
    for index, config in enumerate(sequence):
        rng = random.Random(1000 + index + (0 if config == "gpu_only" else 7))
        rnd = simulate(items, config, index, rng, ane_stops_s)
        if config == "hybrid":  # as driver.run verifies a GPU + ANE round
            why = driver.ane_lost(rnd.records)
            rnd.ane_verified = why is None
            if why is not None:
                run.ane = replace(run.ane, state=ane_state.UNAVAILABLE, reason=ane_state.ANE_LOST, detail=why)
        run.rounds.append(rnd)
        run.configs[config] = meta[config]
    res = result.build(run, machine_info=MACHINE, model=MODEL, created_utc="2026-09-24T10:04:00Z")
    res["synthetic"] = True
    problems = result.validate(res)
    if problems:
        raise SystemExit("synthetic result does not match the schema: " + "; ".join(problems[:5]))
    return replay.replay_data(res, [r for rnd in run.rounds for r in rnd.records])


def write(name, data):
    path = HERE / name
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    late = {r["config"]: r["systems"]["late"]["count"] for r in data["result"]["rounds"]}
    print(f"{path.name}: {path.stat().st_size // 1024} KiB, late {late}")


if __name__ == "__main__":
    write(
        "replay-synthetic.json",
        build(("gpu_only", "hybrid"), ane_state.AneStatus(ane_state.READY, warmup_ane_requests=42)),
    )
    write(
        "replay-synthetic-gpu-only.json",
        build(
            ("gpu_only",),
            ane_state.AneStatus(ane_state.SETUP_AVAILABLE, ane_state.NO_ARTIFACTS, warmup_ane_requests=0),
        ),
    )
    write(
        "replay-synthetic-ane-unverified.json",
        build(
            ("gpu_only", "hybrid"),
            ane_state.AneStatus(ane_state.READY, warmup_ane_requests=42),
            ane_stops_s=20.0,
        ),
    )
