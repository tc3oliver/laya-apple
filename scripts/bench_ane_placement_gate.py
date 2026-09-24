"""Acceptance gate for ANE process placement: resource, lifecycle and isolation cost (#52).

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run python scripts/bench_ane_placement_gate.py \
        smoke --model laya-typed-decisions --placement process --out benchmarks/ane-placement-gate/raw/...
    ... lifecycle --model M --placement P --iterations 10 --out ...
    ... orphan --model M --out ...

Every mode runs in a fresh interpreter (run it once per model and placement), so RSS and CPU
belong to one configuration. Only the public runtime is used: `Laya(execution="workers")` with
`trace=`. Worker processes are found as this process's children.

smoke      One GPU instance (device="gpu", MLX worker process) and one ANE instance
           (device="ane", ane_placement=P) serve closed-loop streams:
             gpu_alone   GPU L128 alone
             both        GPU L128 and ANE L<smallest bucket> at once
           for --cycles windows of --seconds each. Reported:
           - start-up wall time of each instance;
           - RSS per role (parent, GPU worker, ANE worker), sampled every 100 ms, and its peak;
             phys_footprint and its lifetime peak per role at the end of each window (macOS
             denies USS for child processes); tree totals;
           - CPU time per role over the windows, and tree CPU per request;
           - the runtime trace: GPU return leg (service_end -> received), GPU e2e, ANE req/s and
             e2e, ANE dispatch + return (IPC; ~0 on a thread);
           - answers against inline references (mismatches);
           - close() wall time, and whether every worker process has exited afterwards.
lifecycle  --iterations times: create one device="auto" instance, serve --requests requests
           (half ANE-eligible, half GPU-only), close. Per iteration: start-up, close time,
           parent RSS/USS, open file descriptors, threads, live child processes after close.
orphan     Worker cleanup when things go wrong, with ane_placement="process":
           - an exception inside `with Laya(...)`: the workers must exit;
           - a SIGKILLed parent: a child interpreter creates an instance, reports its worker
             pids and is killed; the workers must exit on their own (EOF on the connection).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

now = time.monotonic_ns
SELF = psutil.Process()


def pct(a) -> dict:
    a = np.asarray([x for x in a if x is not None], float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }


def mem(p: psutil.Process) -> dict:
    """RSS (readable for any own process) and USS (macOS denies it for other processes,
    children included, so it is None for the workers)."""
    out = {"rss_mb": None, "uss_mb": None}
    try:
        out["rss_mb"] = p.memory_info().rss / 2**20
        out["uss_mb"] = p.memory_full_info().uss / 2**20
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return out


def footprint(pid: int) -> dict:
    """macOS phys_footprint (the memory the OS charges the process: dirty + compressed, no
    shared clean pages) and its lifetime peak, from /usr/bin/footprint."""
    text = subprocess.run(["/usr/bin/footprint", "-p", str(pid)], capture_output=True, text=True).stdout
    out = {}
    for key in ("phys_footprint", "phys_footprint_peak"):
        for line in text.splitlines():
            if line.strip().startswith(key + ":"):
                v, unit = line.split(":", 1)[1].split()
                out[key + "_mb"] = float(v) * {"B": 2**-20, "KB": 2**-10, "MB": 1, "GB": 2**10}[unit]
    return out


def cpu_s(p: psutil.Process) -> float | None:
    try:
        t = p.cpu_times()
        return t.user + t.system
    except psutil.NoSuchProcess:
        return None


def environment() -> dict:
    import coremltools
    import mlx.core as mx

    import laya_apple
    from laya_apple.artifacts import platform_profile

    return {
        "laya_apple": laya_apple.__version__,
        "platform": platform_profile(),
        "python": platform.python_version(),
        "mlx": mx.__version__,
        "coremltools": coremltools.__version__,
        "machine": platform.machine(),
        "memory_gb": round(psutil.virtual_memory().total / 2**30),
    }


def roles(gpu, ane) -> dict:
    """role -> psutil.Process. A thread-placed ANE lives in the parent."""
    out = {"parent": SELF, "gpu_worker": psutil.Process(gpu._workers["gpu"].pid)}
    w = ane._workers["ane"]
    if w.placement == "process":
        out["ane_worker"] = psutil.Process(w.pid)
    return out


class MemSampler:
    def __init__(self, procs: dict, period: float = 0.1):
        self.procs, self.period = procs, period
        self.samples: dict = {k: [] for k in procs}
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            for k, p in self.procs.items():
                self.samples[k].append(mem(p))
            self._stop.wait(self.period)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    def summary(self) -> dict:
        out = {}
        for k, xs in self.samples.items():
            rss = [x["rss_mb"] for x in xs if x["rss_mb"] is not None]
            out[k] = {
                "rss_mb_steady": float(np.median(rss)) if rss else None,
                "rss_mb_peak": max(rss) if rss else None,
            }
        out["tree"] = {
            key: sum(v[key] for v in out.values() if v[key] is not None) for key in ("rss_mb_steady", "rss_mb_peak")
        }
        return out


def requests(laya, length: int, seeds: int = 4):
    from laya_apple.workload import make_request

    return [make_request(laya.tokenizer, laya.config, length, n_questions=1, seed=100 + s) + (s,) for s in range(seeds)]


_REFS = """
import json, sys
from laya_apple import Laya
from laya_apple.workload import make_request
model, gpu_len, ane_len = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out = {}
for dev, length in (("gpu", gpu_len), ("ane", ane_len)):
    laya = Laya.from_pretrained(model, device=dev, local_files_only=True)
    out[dev] = {s: laya.predict(context=c, questions=q).answers
                for s in range(4) for c, q in [make_request(laya.tokenizer, laya.config, length, n_questions=1, seed=100 + s)]}
    laya.close()
print(json.dumps(out))
"""


def references(model: str, gpu_len: int, ane_len: int):
    """Inline (execution="inline") answers per seed, computed in a child interpreter so this
    process's memory holds only the workers' client side."""
    out = json.loads(
        subprocess.run(
            [sys.executable, "-c", _REFS, model, str(gpu_len), str(ane_len)], capture_output=True, text=True, check=True
        ).stdout.splitlines()[-1]
    )
    return ({int(k): v for k, v in out["gpu"].items()}, {int(k): v for k, v in out["ane"].items()})


def closed_loop(laya, reqs, refs, stop_at, out: list):
    i = 0
    while now() < stop_at:
        state, questions, s = reqs[i % len(reqs)]
        r = laya.predict(context=state, questions=questions)
        out.append((r.runtime.request_id, json.loads(json.dumps(r.answers)) == refs[s]))
        i += 1


def smoke(args) -> dict:
    from laya_apple import Laya
    from laya_apple.registry import resolve

    spec = resolve(args.model)
    ane_len = min(spec.auto_ane_buckets)
    gpu_len = 128
    traces: list = []
    t = now()
    gpu = Laya.from_pretrained(
        args.model, device="gpu", execution="workers", local_files_only=True, trace=traces.append
    )
    gpu_start_s = (now() - t) / 1e9
    t = now()
    ane = Laya.from_pretrained(
        args.model,
        device="ane",
        execution="workers",
        ane_placement=args.placement,
        local_files_only=True,
        trace=traces.append,
    )
    ane_start_s = (now() - t) / 1e9
    assert ane._workers["ane"].placement == args.placement
    procs = roles(gpu, ane)
    worker_pids = [p.pid for k, p in procs.items() if k != "parent"]
    greqs, areqs = requests(gpu, gpu_len), requests(ane, ane_len)
    grefs, arefs = references(args.model, gpu_len, ane_len)
    for c, q, _ in greqs:  # warm both paths once more through the workers
        gpu.predict(context=c, questions=q)
    for c, q, _ in areqs:
        ane.predict(context=c, questions=q)

    windows = []
    with MemSampler(procs) as sampler:
        for cycle in range(args.cycles):
            for cell in ("gpu_alone", "both"):
                time.sleep(args.settle)
                del traces[:]
                cpu0 = {k: cpu_s(p) for k, p in procs.items()}
                g_out, a_out = [], []
                start = now()
                stop_at = start + int(args.seconds * 1e9)
                ths = [threading.Thread(target=closed_loop, args=(gpu, greqs, grefs, stop_at, g_out))]
                if cell == "both":
                    ths.append(threading.Thread(target=closed_loop, args=(ane, areqs, arefs, stop_at, a_out)))
                for th in ths:
                    th.start()
                for th in ths:
                    th.join()
                span = (now() - start) / 1e9
                cpu1 = {k: cpu_s(p) for k, p in procs.items()}
                fp = {k: footprint(p.pid) for k, p in procs.items()}
                by_id = {tr.request_id: tr for tr in traces}
                g_tr = [by_id[i] for i, _ in g_out if i in by_id]
                a_tr = [by_id[i] for i, _ in a_out if i in by_id]
                n = len(g_out) + len(a_out)
                cpu = {k: cpu1[k] - cpu0[k] for k in procs}
                windows.append(
                    {
                        "cell": cell,
                        "cycle": cycle,
                        "seconds": span,
                        "gpu_requests": len(g_out),
                        "ane_requests": len(a_out),
                        "gpu_mismatches": sum(not ok for _, ok in g_out),
                        "ane_mismatches": sum(not ok for _, ok in a_out),
                        "gpu_return_ms": [(x.received_ns - x.service_end_ns) / 1e6 for x in g_tr],
                        "gpu_e2e_ms": [(x.response_ns - x.submit_ns) / 1e6 for x in g_tr],
                        "gpu_service_ms": [(x.service_end_ns - x.service_start_ns) / 1e6 for x in g_tr],
                        "ane_e2e_ms": [(x.response_ns - x.submit_ns) / 1e6 for x in a_tr],
                        "ane_service_ms": [(x.service_end_ns - x.service_start_ns) / 1e6 for x in a_tr],
                        "ane_ipc_ms": [
                            (x.service_start_ns - x.dispatch_ns + x.received_ns - x.service_end_ns) / 1e6 for x in a_tr
                        ],
                        "cpu_s": cpu,
                        "footprint": fp,
                        "tree_cpu_ms_per_request": sum(cpu.values()) / n * 1e3 if n else None,
                    }
                )
                w = windows[-1]
                print(
                    args.model,
                    args.placement,
                    cell,
                    cycle,
                    f"gpu {w['gpu_requests']} ane {w['ane_requests']}",
                    "gpu return p50",
                    round(float(np.median(w["gpu_return_ms"])), 3),
                    "mism",
                    w["gpu_mismatches"] + w["ane_mismatches"],
                    flush=True,
                )
    memory = sampler.summary()
    t = now()
    ane.close()
    gpu.close()
    close_s = (now() - t) / 1e9
    time.sleep(0.5)
    alive = [pid for pid in worker_pids if psutil.pid_exists(pid) and psutil.Process(pid).status() != "zombie"]
    zombies = [pid for pid in worker_pids if psutil.pid_exists(pid) and psutil.Process(pid).status() == "zombie"]
    return {
        "mode": "smoke",
        "gpu_length": gpu_len,
        "ane_length": ane_len,
        "startup_s": {"gpu_instance": gpu_start_s, "ane_instance": ane_start_s},
        "memory": memory,
        "windows": windows,
        "close_s": close_s,
        "workers_alive_after_close": alive,
        "zombies_after_close": zombies,
    }


def lifecycle(args) -> dict:
    from laya_apple import Laya

    iters = []
    first = None
    for i in range(args.iterations):
        t = now()
        laya = Laya.from_pretrained(
            args.model, execution="workers", ane_placement=args.placement, local_files_only=True
        )
        start_s = (now() - t) / 1e9
        if first is None:
            first = requests(laya, 128) + requests(laya, min(512, int(laya.config.get("max_len", 512))))
        pids = [w.pid for w in laya._workers.values() if w.placement == "process"]
        devices = []
        for k in range(args.requests):
            state, questions, _ = first[k % len(first)]
            devices.append(laya.predict(context=state, questions=questions).runtime.device)
        t = now()
        laya.close()
        close_s = (now() - t) / 1e9
        del laya
        time.sleep(0.3)
        live_children = [c.pid for c in SELF.children(recursive=True) if c.status() != "zombie"]
        zombies = [c.pid for c in SELF.children(recursive=True) if c.status() == "zombie"]
        iters.append(
            {
                "iteration": i,
                "startup_s": start_s,
                "close_s": close_s,
                "devices": {d: devices.count(d) for d in set(devices)},
                "worker_pids": pids,
                "workers_alive_after_close": [p for p in pids if psutil.pid_exists(p)],
                "live_children_after_close": live_children,
                "zombie_children_after_close": zombies,
                **mem(SELF),
                "fds": SELF.num_fds(),
                "threads": SELF.num_threads(),
            }
        )
        print(
            args.model,
            args.placement,
            "iteration",
            i,
            {k: iters[-1][k] for k in ("startup_s", "close_s", "rss_mb", "fds", "threads")},
            flush=True,
        )
    return {"mode": "lifecycle", "requests_per_iteration": args.requests, "iterations": iters}


_CHILD = """
import json, sys, time
from laya_apple import Laya
laya = Laya.from_pretrained(sys.argv[1], execution="workers", ane_placement="process", local_files_only=True)
print(json.dumps([w.pid for w in laya._workers.values() if w.placement == "process"]), flush=True)
time.sleep(600)
"""


def orphan(args) -> dict:
    from laya_apple import Laya

    out = {"mode": "orphan"}
    # 1. an exception inside the context manager
    pids = []
    try:
        with Laya.from_pretrained(
            args.model, execution="workers", ane_placement="process", local_files_only=True
        ) as laya:
            pids = [w.pid for w in laya._workers.values() if w.placement == "process"]
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    time.sleep(0.5)
    out["exception_in_with"] = {"worker_pids": pids, "alive_after": [p for p in pids if psutil.pid_exists(p)]}
    # 2. the parent is SIGKILLed: nothing runs close()
    child = subprocess.Popen([sys.executable, "-c", _CHILD, args.model], stdout=subprocess.PIPE, text=True)
    pids = json.loads(child.stdout.readline())
    time.sleep(1.0)
    os.kill(child.pid, signal.SIGKILL)
    child.wait()
    t, alive = now(), pids
    while alive and now() - t < 30e9:
        time.sleep(0.1)
        alive = [p for p in pids if psutil.pid_exists(p) and psutil.Process(p).status() != "zombie"]
    out["parent_sigkill"] = {
        "worker_pids": pids,
        "alive_after_30s": alive,
        "seconds_to_exit": (now() - t) / 1e9 if not alive else None,
    }
    for p in alive:  # never leave a probe behind
        os.kill(p, signal.SIGKILL)
    print(out, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["smoke", "lifecycle", "orphan"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--placement", choices=["thread", "process"], default="process")
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--requests", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = {"smoke": smoke, "lifecycle": lifecycle, "orphan": orphan}[args.mode](args)
    result.update(model=args.model, placement=args.placement if args.mode != "orphan" else "process")
    result["args"] = vars(args)
    result["environment"] = environment()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
