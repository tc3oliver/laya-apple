"""Phase 3 S2 / S4: mixed load through an unmodified inline Laya(device="auto"), A vs B.

Mix per request (fixed seeds): 80% L64 single question (routes to the ANE), 10% L64 with 3
questions and 10% L256 single question (both route to the GPU at the production routing defaults).
Arm B swaps the W8 CompiledMLModel into `laya.ane.models[64]` (research-side injection, not the
production path). Clients are threads sharing one inline Laya, which serialises requests on its
lock: the latency includes that queueing. Per concurrency: CYCLES cycles, each one window of
REQS requests per arm, order AB / BA, the same request sequence for both arms of a cycle.
Timing-sensitive: run alone.

    serve_mix.py --root <w8 root> --out <json> [--clients 1 2 4 8]
"""

import argparse
import json
import os
import random
import resource
import sys
import threading
import time
import warnings
from pathlib import Path

import coremltools as ct

from laya_apple.model import Laya
from laya_apple.registry import ANE_COMPUTE_UNITS
from laya_apple.workload import make_request

CYCLES, REQS, WARM, L = 10, 200, 40, 64
MIX = [(0.8, 64, 1), (0.1, 64, 3), (0.1, 256, 1)]
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--clients", type=int, nargs="+", default=[1, 2, 4, 8])
a = ap.parse_args()
out = Path(a.out)
if out.exists():
    sys.exit(f"{out} exists; refusing to overwrite")
warnings.simplefilter("ignore")

laya = Laya.from_pretrained("laya-typed-decisions", device="auto", local_files_only=True)
assert laya.ane is not None and 64 in laya.ane.models
spec = laya.spec
path = Path(a.root).expanduser() / spec.name / spec.revision[:12] / f"bc1s-masked-w8-pt-L{L}-B1/model.mlmodelc"
models = {
    "A": laya.ane.models[L],
    "B": ct.models.CompiledMLModel(str(path), compute_units=getattr(ct.ComputeUnit, ANE_COMPUTE_UNITS)),
}
pool = {
    (length, n): [make_request(laya.tokenizer, laya.config, length, n_questions=n, seed=s) for s in range(20)]
    for _, length, n in MIX
}


def sequence(seed, n):
    rng = random.Random(seed)
    classes = rng.choices([(l, q) for _, l, q in MIX], weights=[w for w, _, _ in MIX], k=n)
    return [(c, pool[c][rng.randrange(20)]) for c in classes]


def run_window(arm, seq, clients):
    laya.ane.models[L] = models[arm]
    res = [None] * len(seq)
    nxt = iter(range(len(seq)))
    lock = threading.Lock()

    def client():
        while True:
            with lock:
                i = next(nxt, None)
            if i is None:
                return
            (length, nq), (state, qs) = seq[i]
            t = time.perf_counter()
            try:
                r = laya.predict(context=state, questions=qs)
                res[i] = ((time.perf_counter() - t) * 1e3, r.runtime.device, json.dumps(
                    {k: v.get("choice", v.get("answer")) for k, v in r.answers.items()}, sort_keys=True), None)
            except Exception as e:
                res[i] = ((time.perf_counter() - t) * 1e3, None, None, f"{type(e).__name__}: {e}")

    t0, w0 = time.perf_counter(), time.time()
    ts = [threading.Thread(target=client) for _ in range(clients)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    wall, w1 = time.perf_counter() - t0, time.time()
    return {
        "arm": arm,
        "wall_s": wall,
        "t0_wall": w0,
        "t1_wall": w1,
        "lat_ms": [r[0] for r in res],
        "device": [r[1] for r in res],
        "answer": [r[2] for r in res],
        "errors": [r[3] for r in res if r[3]],
    }


rec = {"mix": MIX, "cycles": CYCLES, "reqs": REQS, "cells": {}}
for clients in a.clients:
    for arm in "AB":
        run_window(arm, sequence(999, WARM), clients)
    cell = []
    for c in range(CYCLES):
        seq = sequence(c, REQS)
        for arm in ("AB" if c % 2 == 0 else "BA"):
            cell.append({"cycle": c} | run_window(arm, seq, clients))
    rec["cells"][str(clients)] = cell
    print("clients", clients, "done", flush=True)
rec["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
rec["loadavg"] = os.getloadavg()
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rec))
laya.close()
