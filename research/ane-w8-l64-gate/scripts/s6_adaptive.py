"""Phase 3 S6: adaptive execution / fallback / strict-ANE behaviour with arm A or B injected.

Fresh process per arm. Laya(device="auto", execution="workers") with the production adaptive
handoff (ANE on a thread). Arm B swaps the W8 CompiledMLModel into backend.models[64] and an
AsyncPrebindModel of the W8 artifact into backend.async_models[64], after the same output-identity
check the production loader applies to the async model (research-side injection).

  Q  quiet: sequential L64 requests, no GPU work
  G  GPU busy: sequential L64 requests while a thread keeps L256 (GPU) requests in flight
  R  re-arm: GPU load stopped, handoff must return to armed / sync
  F  fallback: the async predict is made to raise during an episode; the request must still be
     served and adaptive execution disabled (the production behaviour)
  X  strict device="ane" inline: an L64 request is served, an L96 request (no artifact in the
     isolated cache) raises

    s6_adaptive.py A|B --root <w8 root> --out <json>
"""

import argparse
import json
import sys
import threading
import time
import warnings
from pathlib import Path

import coremltools as ct
import numpy as np

from laya_apple.artifacts import COMPILED
from laya_apple.backends.coreml_async import AsyncPrebindModel
from laya_apple.model import Laya
from laya_apple.registry import ANE_COMPUTE_UNITS
from laya_apple.workload import make_request

L = 64
ap = argparse.ArgumentParser()
ap.add_argument("arm", choices=["A", "B"])
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out)
if out.exists():
    sys.exit(f"{out} exists; refusing to overwrite")
caught = []
warnings.simplefilter("default")
warnings.showwarning = lambda m, c, *x, **k: caught.append(f"{c.__name__}: {str(m)[:160]}")


def pct(v):
    return {f"p{p}": float(np.percentile(v, p)) for p in (50, 95, 99)}


def snap(laya):
    h = laya.info()["ane_handoff"]
    return {k: h[k] for k in ("enabled", "state", "episodes", "forwards", "disabled_reason")} | {"trips": h["breaker"]["trips"]}


rec = {"arm": a.arm}
laya = Laya.from_pretrained("laya-typed-decisions", device="auto", execution="workers", local_files_only=True)
spec = laya.spec
rec["placement"], rec["handoff_at_start"] = laya.ane_placement, laya.info().get("ane_handoff")
assert laya._handoff is not None and laya._handoff.enabled, rec
backend = laya._workers["ane"]._backend
assert backend.async_models, "async models did not load"
if a.arm == "B":
    path = Path(a.root).expanduser() / spec.name / spec.revision[:12] / f"bc1s-masked-w8-pt-L{L}-B1/model.mlmodelc"
    sync = ct.models.CompiledMLModel(str(path), compute_units=getattr(ct.ComputeUnit, ANE_COMPUTE_UNITS))
    asy = AsyncPrebindModel(path)
    ident = []
    for feats in backend._check_features(L):
        ref, o = sync.predict(feats), asy.predict(feats)
        ident.append(set(ref) == set(o) and all(ref[k].shape == o[k].shape and ref[k].dtype == o[k].dtype and np.array_equal(ref[k], o[k]) for k in ref))
    rec["b_async_identical_to_sync"] = ident
    assert all(ident)
    backend.models[L], backend.async_models[L] = sync, asy
else:
    ident = []
    for feats in backend._check_features(L):
        ref, o = backend.models[L].predict(feats), backend.async_models[L].predict(feats)
        ident.append(all(np.array_equal(ref[k], o[k]) for k in ref))
    rec["a_async_identical_to_sync"] = ident

state, qs = make_request(laya.tokenizer, laya.config, L, n_questions=1, seed=0)
gstate, gqs = make_request(laya.tokenizer, laya.config, 256, n_questions=1, seed=0)


def ane_req(n):
    lat, dev = [], set()
    for _ in range(n):
        t = time.perf_counter()
        r = laya.submit(context=state, questions=qs).result()
        lat.append((time.perf_counter() - t) * 1e3)
        dev.add(r.runtime.device)
    return lat, sorted(dev)


for _ in range(10):
    ane_req(1)
lat, dev = ane_req(300)
rec["Q"] = {"lat_ms": pct(lat), "devices": dev, "handoff": snap(laya)}

stop = threading.Event()
gpu_n = [0]


def gpu_load():
    while not stop.is_set():
        futs = [laya.submit(context=gstate, questions=gqs) for _ in range(2)]
        [f.result() for f in futs]
        gpu_n[0] += 2


th = threading.Thread(target=gpu_load)
th.start()
time.sleep(0.3)
states, lat = [], []
for i in range(800):
    t = time.perf_counter()
    r = laya.submit(context=state, questions=qs).result()
    lat.append((time.perf_counter() - t) * 1e3)
    states.append(laya._handoff.state)
transitions = [(i, s) for i, s in enumerate(states) if i == 0 or s != states[i - 1]]
rec["G"] = {"lat_ms": pct(lat), "lat_ms_after_guard": pct(lat[200:]), "state_transitions": transitions, "gpu_requests": gpu_n[0], "handoff": snap(laya), "lat_samples": lat}
stop.set()
th.join()
time.sleep(1.5)
ane_req(100)
rec["R"] = {"handoff": snap(laya)}

# F: new episode, then the async predict fails
th = threading.Thread(target=gpu_load)
stop.clear()
th.start()
time.sleep(0.3)
ane_req(70)  # past the guard: async path next
orig = backend.async_models[L].predict


def boom(feats):
    raise RuntimeError("injected async failure")


backend.async_models[L].predict = boom
try:
    lat, dev = ane_req(20)
    rec["F"] = {"served": True, "devices": dev, "handoff": snap(laya)}
except Exception as e:
    rec["F"] = {"served": False, "error": f"{type(e).__name__}: {e}", "handoff": snap(laya)}
backend.async_models[L].predict = orig
stop.set()
th.join()
laya.close()

x = Laya.from_pretrained("laya-typed-decisions", device="ane", local_files_only=True)
if a.arm == "B":
    x.ane.models[L] = ct.models.CompiledMLModel(str(path), compute_units=getattr(ct.ComputeUnit, ANE_COMPUTE_UNITS))
rec["X"] = {}
r = x.predict(context=state, questions=qs)
rec["X"]["L64"] = {"device": r.runtime.device, "reason": r.runtime.routing_reason}
s96, q96 = make_request(x.tokenizer, x.config, 96, n_questions=1, seed=0)
try:
    x.predict(context=s96, questions=q96)
    rec["X"]["L96"] = "served (unexpected)"
except Exception as e:
    rec["X"]["L96"] = f"{type(e).__name__}"
x.close()
rec["warnings"] = sorted(set(caught))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rec))
print(json.dumps({k: (v if k != "G" else {kk: vv for kk, vv in v.items() if kk != "lat_samples"}) for k, v in rec.items()}, default=str)[:3000])
