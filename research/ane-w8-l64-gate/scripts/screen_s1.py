"""Phase 3 screen S1: end-to-end laya.predict latency, L64, concurrency 1, device="ane", A vs B.

Research-side injection (not the production path): arm B swaps the W8 CompiledMLModel into
`laya.ane.models[64]` of an otherwise unmodified Laya. One process; windows alternate AB / BA.
Timing-sensitive: run alone.

    screen_s1.py --root <w8 root> --out <json>
"""

import argparse
import json
import os
import resource
import sys
import time
import warnings
from pathlib import Path

import coremltools as ct
import numpy as np

from laya_apple.model import Laya
from laya_apple.registry import ANE_COMPUTE_UNITS
from laya_apple.workload import make_request

WARMUP, CYCLES, WINDOW, L = 20, 10, 100, 64
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out)
if out.exists():
    sys.exit(f"{out} exists; refusing to overwrite")
warnings.simplefilter("ignore")

laya = Laya.from_pretrained("laya-typed-decisions", device="ane", local_files_only=True)
spec = laya.spec
path = Path(a.root).expanduser() / spec.name / spec.revision[:12] / f"bc1s-masked-w8-pt-L{L}-B1/model.mlmodelc"
models = {
    "A": laya.ane.models[L],
    "B": ct.models.CompiledMLModel(str(path), compute_units=getattr(ct.ComputeUnit, ANE_COMPUTE_UNITS)),
}
state, qs = make_request(laya.tokenizer, laya.config, L, n_questions=1, seed=0)


def use(arm):
    laya.ane.models[L] = models[arm]


def window(arm, n):
    use(arm)
    lat, errs, choices = [], 0, set()
    t_start = time.perf_counter()
    for _ in range(n):
        t = time.perf_counter()
        try:
            r = laya.predict(context=state, questions=qs)
            assert r.runtime.device == "ane" and r.runtime.sequence_length == L
            choices.add(r.answers["q0"]["choice"])
        except Exception:
            errs += 1
        lat.append((time.perf_counter() - t) * 1e3)
    return lat, errs, sorted(choices), time.perf_counter() - t_start


for arm in "AB":
    window(arm, WARMUP)
rec = {"seed": 0, "protocol": {"warmup": WARMUP, "cycles": CYCLES, "window": WINDOW}, "windows": []}
for c in range(CYCLES):
    for arm in ("AB" if c % 2 == 0 else "BA"):
        lat, errs, ch, wall = window(arm, WINDOW)
        rec["windows"].append({"cycle": c, "arm": arm, "e2e_ms": lat, "errors": errs, "choices": ch, "wall_s": wall})
rec["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
rec["loadavg"] = os.getloadavg()
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rec))
laya.close()
