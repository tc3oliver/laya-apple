"""One same-machine reference run of laya-fast (a separate public Apple-silicon Laya runtime) on
the 8x512 fixtures of `laya`.

Runs with laya-fast's own interpreter, in its own throwaway checkout and venv, never in
laya-apple's environment, and imports nothing from laya_apple:

    cd <scratch>/laya-fast && .venv/bin/python \\
        <repo>/research/intra-request-split/scripts/laya_fast_driver.py \\
        --laya-fast-dir . --fixtures <repo>/research/intra-request-split/raw/fixtures-laya.json \\
        --criteria <repo>/research/intra-request-split/criteria.json \\
        --out <repo>/research/intra-request-split/raw/laya-fast/<run-id>.json --run-id <run-id>

laya-fast is used unmodified, at the commit in criteria.json, through its public Python API
(`LayaFast(model_dir)` with its defaults, `system_one(state, questions)`). Its outputs are not
evaluated here; only caller-observed latency is recorded, with the same window protocol as
latency.py: idle, discarded warm-up requests, then closed-loop measured requests cycling the
request variants.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


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


def versions() -> dict:
    from importlib.metadata import version

    out = {}
    for pkg in ("mlx", "coremltools", "numpy", "torch", "tokenizers", "safetensors"):
        try:
            out[pkg] = version(pkg)
        except Exception:
            out[pkg] = None
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--laya-fast-dir", required=True)
    p.add_argument("--fixtures", required=True)
    p.add_argument("--criteria", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--model-dir", default="converted-fp16", help="laya-fast's converted checkpoint (its README)")
    a = p.parse_args(argv)

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; raw data is never overwritten")
    crit = json.loads(Path(a.criteria).read_text())
    lf = crit["protocol"]["laya_fast"]
    proto = crit["protocol"]["latency"]
    root = Path(a.laya_fast_dir).resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if commit != lf["commit"]:
        raise SystemExit(f"laya-fast checkout is at {commit}, criteria.json pins {lf['commit']}")
    # Its export step rewrites tracked build reports (ane/body*/report.json); its code must be unmodified.
    dirty = subprocess.check_output(["git", "diff", "--name-only", "HEAD", "--", "*.py"], cwd=root, text=True)
    if dirty.strip():
        raise SystemExit(f"laya-fast's Python sources are modified; it must run unmodified:\n{dirty}")
    os.chdir(root)
    sys.path.insert(0, str(root))
    fx = json.loads(Path(a.fixtures).read_text())
    if fx["model"] != "laya":
        raise SystemExit("laya-fast serves the English `laya` checkpoint only")
    workloads = {n: fx["workloads"][n] for n in lf["workloads"]}
    bodies = sorted(int(d.name[4:]) for d in (root / "ane").glob("body*") if (d / "model.mlmodelc").exists())

    from laya_fast import LayaFast  # laya-fast's public API

    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    t = time.perf_counter()
    agent = LayaFast(a.model_dir)
    load_s = time.perf_counter() - t
    ane_buckets = list(getattr(agent.ane, "buckets", []) or []) if agent.ane is not None else []
    if sorted(lf["ane_bodies"]) != sorted(set(ane_buckets) & set(lf["ane_bodies"])):
        raise SystemExit(f"laya-fast loaded ANE bodies {ane_buckets}; the protocol needs {lf['ane_bodies']}")

    def call(req):
        t0 = time.perf_counter_ns()
        agent.system_one(req["state"], req["questions"])
        return (time.perf_counter_ns() - t0) / 1e6

    tokens = {}
    for name, w in workloads.items():
        tokens[name] = []
        for req in w["requests"]:
            _, items, _ = agent.prepare(req["state"], req["questions"])
            tokens[name].append({"seed": req["seed"], "lengths": [len(it["ids"]) for it in items]})
    for name, w in workloads.items():
        for i in range(proto["run_warmup_requests"]):
            call(w["requests"][i % len(w["requests"])])
    windows = []
    for cycle in range(proto["cycles"]):
        for name, w in workloads.items():
            time.sleep(proto["idle_before_window_s"])
            reqs = w["requests"]
            for i in range(proto["window_warmup_requests"]):
                call(reqs[i % len(reqs)])
            rows = []
            t_window = time.monotonic_ns()
            for i in range(proto["window_requests"]):
                req = reqs[i % len(reqs)]
                rows.append({"i": i, "seed": req["seed"], "latency_ms": call(req)})
            windows.append(
                {
                    "cycle": cycle + 1,
                    "workload": name,
                    "arm": "laya_fast",
                    "start_ns": t_window,
                    "end_ns": time.monotonic_ns(),
                    "summary": summary([r["latency_ms"] for r in rows]),
                    "requests": rows,
                }
            )
            s = windows[-1]["summary"]
            print(
                f"{a.run_id} c{cycle + 1} {name} laya-fast P50 {s['p50_ms']:8.2f} P90 {s['p90_ms']:8.2f} ms", flush=True
            )
    therm = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout.strip().splitlines()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            {
                "run_id": a.run_id,
                "kind": "laya-fast",
                "model": "laya",
                "started": started,
                "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "laya_fast_commit": commit,
                "ane_bodies_present": bodies,
                "ane_buckets_loaded": ane_buckets,
                "load_s": round(load_s, 2),
                "python": platform.python_version(),
                "versions": versions(),
                "loadavg_end": list(os.getloadavg()),
                "pmset_therm_end": therm,
                "protocol": proto,
                "tokens": tokens,
                "windows": windows,
            },
            indent=1,
        )
        + "\n"
    )
    tmp.replace(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
