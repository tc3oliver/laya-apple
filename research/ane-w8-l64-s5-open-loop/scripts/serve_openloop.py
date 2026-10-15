"""Open-loop S5: fixed-arrival-rate mixed load with a continuous GPU LLM, A vs B (PREREGISTRATION.md).

One fresh process = one replicate. Same MIX, request pool, research-side W8 injection into an
unmodified inline Laya(device="auto") as ane-w8-l64-gate/scripts/serve_mix.py. A single dispatcher
(the main thread, monotonic clock) issues a seeded Poisson schedule into a fixed pool of worker
threads. The schedule and request sequence are pure functions of (rate, cycle); both arms of a cycle
replay them, and a hash of schedule + sequence is stored per window and asserted equal per pair.
Latency is later measured from the SCHEDULED arrival. Timing-sensitive: run alone.

    serve_openloop.py --root <w8 root> --out-dir <dir> [--cells eq hi null] [--cycles 8] [--dry-run]

Output (one directory per replicate, refuses to overwrite): meta.json, index.jsonl, llm.jsonl (the
unchanged llm_load.py output), c<cycle>-<cell>-<arm>.json per window (compact per-request arrays,
times in ms from the window's first scheduled instant), meta_end.json.
"""

import argparse
import hashlib
import json
import os
import queue
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

MIX = [(0.8, 64, 1), (0.1, 64, 3), (0.1, 256, 1)]
POOL, L, WORKERS, MAX_OUT = 20, 64, 32, 256
LLM_PY = "~/Developer/ai/omlx-production/releases/hpcp14-20261003/.venv/bin/python3"
LLM_MODEL = "/Volumes/Data/models/mlx-community/Qwen3.5-0.8B-MLX-4bit"
LLM_LOAD = Path(__file__).resolve().parents[2] / "ane-w8-l64-gate" / "scripts" / "llm_load.py"
CELLS = {"eq": ("eq", ("A", "B")), "hi": ("hi", ("A", "B")), "null": ("eq", ("A", "A2"))}


def schedule(rate, cycle, window_s):
    """Pure function of (rate, cycle): Poisson arrival times (s) and request classes (index into MIX, pool index)."""
    rng = random.Random(1_000_003 * cycle + round(rate * 100))
    arr, t = [], 0.0
    while True:
        t += rng.expovariate(rate)
        if t >= window_s:
            break
        arr.append(t)
    rq = random.Random(7_000_001 * cycle + round(rate * 100))
    classes = rq.choices(range(len(MIX)), weights=[w for w, _, _ in MIX], k=len(arr))
    return arr, [(c, rq.randrange(POOL)) for c in classes]


def sched_hash(arr, reqs):
    return hashlib.sha256(json.dumps([[repr(x) for x in arr], reqs]).encode()).hexdigest()


def plan(a):
    rates = {"eq": a.rate_eq, "hi": a.rate_hi}
    out = []
    for c in range(a.cycles):
        for cell in a.cells:
            rate_key, arms = CELLS[cell]
            for arm in (arms if c % 2 == 0 else arms[::-1]):
                out.append((c, cell, rates[rate_key], arm))
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, help="W8 research root (holds the B artifact)")
ap.add_argument("--out-dir", required=True)
ap.add_argument("--rate-eq", type=float, default=52.0)
ap.add_argument("--rate-hi", type=float, default=65.0)
ap.add_argument("--cycles", type=int, default=8)
ap.add_argument("--window-s", type=float, default=30.0)
ap.add_argument("--cells", nargs="+", choices=list(CELLS), default=["eq", "hi", "null"])
ap.add_argument("--drain-s", type=float, default=5.0)
ap.add_argument("--settle-s", type=float, default=2.0)
ap.add_argument("--llm-block-s", type=float, default=60.0)
ap.add_argument("--llm-python", default=LLM_PY)
ap.add_argument("--llm-model", default=LLM_MODEL)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
windows = plan(a)
if a.dry_run:
    per = a.window_s + a.settle_s
    print(f"{len(windows)} windows, ~{(len(windows) * per + 2 * a.llm_block_s) / 60:.1f} min, workers {WORKERS}, max outstanding {MAX_OUT}")
    seen = {}
    for c, cell, rate, arm in windows:
        arr, reqs = schedule(rate, c, a.window_s)
        h = sched_hash(arr, reqs)
        print(f"c{c} {cell:4s} {arm:2s} rate {rate:g} n {len(arr)} sched {h[:12]}")
        assert seen.setdefault((c, rate), h) == h
    sys.exit(0)

out_dir = Path(a.out_dir)
if out_dir.exists() and any(out_dir.iterdir()):
    sys.exit(f"{out_dir} exists and is not empty; refusing to overwrite")
cache = os.environ.get("LAYA_APPLE_CACHE", "")
if not cache or cache.startswith("/Volumes/Data"):
    sys.exit("LAYA_APPLE_CACHE must point at an isolated cache")
out_dir.mkdir(parents=True, exist_ok=True)

import importlib.metadata as md  # noqa: E402
import platform  # noqa: E402
import warnings  # noqa: E402

import coremltools as ct  # noqa: E402

from laya_apple.model import Laya  # noqa: E402
from laya_apple.registry import ANE_COMPUTE_UNITS  # noqa: E402
from laya_apple.workload import make_request  # noqa: E402

warnings.simplefilter("ignore")
sys.setswitchinterval(0.0002)  # the dispatcher must win the GIL back from 32 workers within the lag bound


def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


laya = Laya.from_pretrained("laya-typed-decisions", device="auto", local_files_only=True)
assert laya.ane is not None and L in laya.ane.models
spec = laya.spec
b_path = Path(a.root).expanduser() / spec.name / spec.revision[:12] / f"bc1s-masked-w8-pt-L{L}-B1/model.mlmodelc"
models = {"A": laya.ane.models[L]}
models["A2"] = models["A"]
models["B"] = ct.models.CompiledMLModel(str(b_path), compute_units=getattr(ct.ComputeUnit, ANE_COMPUTE_UNITS))
pool = {
    (length, n): [make_request(laya.tokenizer, laya.config, length, n_questions=n, seed=s) for s in range(POOL)]
    for _, length, n in MIX
}
mix_keys = [(length, n) for _, length, n in MIX]
meta = {
    "pid": os.getpid(),
    "args": vars(a),
    "mix": MIX,
    "macos_build": sh("sw_vers", "-buildVersion"),
    "python": platform.python_version(),
    "coremltools": ct.__version__,
    "mlx": md.version("mlx"),
    "model": spec.name,
    "revision": spec.revision,
    "b_artifact": str(b_path),
    "cache": cache,
    "loadavg_start": os.getloadavg(),
    "thermal_start": sh("pmset", "-g", "therm"),
    "switchinterval_s": sys.getswitchinterval(),
    "n_windows": len(windows),
}
(out_dir / "meta.json").write_text(json.dumps(meta, indent=1))

llm_out = out_dir / "llm.jsonl"
llm = subprocess.Popen(
    [os.path.expanduser(a.llm_python), str(LLM_LOAD), "--model", a.llm_model, "--out", str(llm_out), "--max-tokens", "128"],
    stdout=subprocess.PIPE, text=True, env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"},
)


def stop_llm():
    Path(str(llm_out) + ".stop").touch()
    try:
        llm.wait(timeout=60)
    except subprocess.TimeoutExpired:
        llm.terminate()
        llm.wait(timeout=30)
    Path(str(llm_out) + ".stop").unlink(missing_ok=True)


def run_window(arm, rate, cycle, cell):
    arr, reqs = schedule(rate, cycle, a.window_s)
    n = len(arr)
    seq = [(mix_keys[c], pool[mix_keys[c]][p]) for c, p in reqs]
    laya.ane.models[L] = models[arm]
    iss, sta, end = [-1.0] * n, [-1.0] * n, [-1.0] * n
    dev, ans, err = [None] * n, [None] * n, [None] * n
    qlen, outst = [0] * n, [0] * n
    st = {"arrived": 0, "started": 0, "ended": 0, "dropped": 0}
    lock = threading.Lock()
    q = queue.SimpleQueue()
    pc = time.perf_counter

    def worker():
        while True:
            i = q.get()
            if i is None:
                return
            (_, nq), (state, qs) = seq[i]
            with lock:
                st["started"] += 1
            sta[i] = pc()
            try:
                r = laya.predict(context=state, questions=qs)
                dev[i] = r.runtime.device
                ans[i] = json.dumps({k: v.get("choice", v.get("answer")) for k, v in r.answers.items()}, sort_keys=True)
            except Exception as e:
                err[i] = f"{type(e).__name__}: {e}"
            end[i] = pc()
            with lock:
                st["ended"] += 1

    ts = [threading.Thread(target=worker) for _ in range(WORKERS)]
    [t.start() for t in ts]
    T0 = pc() + 0.05
    w0 = time.time() + 0.05
    for i, at in enumerate(arr):
        while (rem := T0 + at - pc()) > 0:
            if rem > 0.0005:  # one long sleep overshoots by 4-7 ms on this machine (timer coalescing)
                time.sleep(0.0002)
        now = pc()
        with lock:
            st["arrived"] += 1
            out = st["arrived"] - 1 - st["ended"] - st["dropped"]
            drop = out > MAX_OUT
            if drop:
                st["dropped"] += 1
            qlen[i] = st["arrived"] - st["started"]
            outst[i] = out
        if drop:
            err[i] = "dropped"
        else:
            iss[i] = now - T0
            q.put(i)
    last = T0 + (arr[-1] if arr else 0.0)
    deadline = last + a.drain_s
    while pc() < deadline:
        with lock:
            if st["ended"] + st["dropped"] == n:
                break
        time.sleep(0.002)
    drained_s = pc() - last
    with lock:
        drained = st["ended"] + st["dropped"] == n
    w1 = time.time()
    for _ in ts:
        q.put(None)
    [t.join() for t in ts]
    f = lambda v: [round(x * 1e3, 2) if x >= 0 else -1 for x in v]  # noqa: E731
    tab = lambda v: (lambda u: (u, [u.index(x) if x is not None else -1 for x in v]))(sorted({x for x in v if x is not None}))  # noqa: E731
    dtab, dvals = tab(dev)
    atab, avals = tab(ans)
    etab, evals = tab(err)
    return {
        "cell": cell, "cycle": cycle, "arm": arm, "rate": rate, "n": n,
        "sched_hash": sched_hash(arr, reqs), "model": str(b_path) if arm == "B" else "production-A",
        "t0_wall": w0, "t1_wall": w1, "drained": drained, "drain_s": round(drained_s, 3),
        "dropped": st["dropped"], "cls": [c for c, _ in reqs],
        "sched_ms": f(arr), "issued_ms": f([x for x in iss]),
        "started_ms": f([x - T0 if x >= 0 else -1 for x in sta]), "ended_ms": f([x - T0 if x >= 0 else -1 for x in end]),
        "queue": qlen, "outstanding": outst,
        "device_tab": dtab, "device": dvals, "answer_tab": atab, "answer": avals, "error_tab": etab, "error": evals,
    }


try:
    line = llm.stdout.readline().strip()
    assert line == "ready", f"LLM subprocess said {line!r}"
    t = time.time()
    time.sleep(a.llm_block_s)
    meta_end = {"llm_only_pre": [t, time.time()]}
    # warm-up (not data): both arms, the full mix, LLM running
    wseq = random.Random(999)
    for arm in ("A", "B"):
        laya.ane.models[L] = models[arm]
        for _ in range(60):
            k = mix_keys[wseq.choices(range(len(MIX)), weights=[w for w, _, _ in MIX])[0]]
            state, qs = pool[k][wseq.randrange(POOL)]
            laya.predict(context=state, questions=qs)
    seen = {}
    for c, cell, rate, arm in windows:
        time.sleep(a.settle_s)
        w = run_window(arm, rate, c, cell)
        assert seen.setdefault((c, rate), w["sched_hash"]) == w["sched_hash"], "schedule differs between arms"
        name = f"c{c}-{cell}-{arm}.json"
        p = out_dir / name
        assert not p.exists()
        p.write_text(json.dumps(w, separators=(",", ":")))
        assert p.stat().st_size < 2**20, f"{name} is over 1 MiB"
        with open(out_dir / "index.jsonl", "a") as ix:
            ix.write(json.dumps({k: w[k] for k in ("cell", "cycle", "arm", "rate", "n", "sched_hash", "t0_wall", "t1_wall", "dropped", "drained", "drain_s")} | {"file": name}) + "\n")
        print(name, "n", w["n"], "drops", w["dropped"], "drained", w["drained"], flush=True)
    t = time.time()
    time.sleep(a.llm_block_s)
    meta_end["llm_only_post"] = [t, time.time()]
finally:
    stop_llm()
    laya.close()
meta_end |= {"loadavg_end": os.getloadavg(), "thermal_end": sh("pmset", "-g", "therm"), "llm_rc": llm.returncode}
(out_dir / "meta_end.json").write_text(json.dumps(meta_end, indent=1))
