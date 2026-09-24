"""Summarise the equal-GPU-load CPU runs: results.json and tables.md.

    uv run python benchmarks/ane-equal-load-cpu/analyze.py [--check] [--rate 30]

Inputs: raw/<model>-{thread,process}-gpu<rate>.json.gz (run.sh). Every run goes through
research/coreml-placement-deconfounding/scripts/analyze.py (#51) unchanged: the request ledger,
in-window requests, the stability rule and the per-cell stats. This file adds the process
tree's CPU time.

Checked before any CPU comparison, per model: thread and process carry the same SHA-256 over
the GPU stream's scheduled arrival offsets and request seeds in every window, the same number
of scheduled arrivals, and the same offered in-window GPU rate; every cycle of the concurrent
cell passes #51's stability rule; 0 errors and 0 answer mismatches.

Criterion (from #54, applied here at equal GPU load): whole-tree CPU time per completed
request in the concurrent cell, process / thread ≤ 1.5. CPU time is user + system, interpolated
at each window's measurement bounds from 50 ms samples; completed = GPU + ANE requests whose
reply was received inside the measurement span.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
_spec = importlib.util.spec_from_file_location(
    "fixed_analyze", ROOT.parents[1] / "research" / "coreml-placement-deconfounding" / "scripts" / "analyze.py"
)
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)

MODELS = ("laya", "laya-typed-decisions")
PLACEMENTS = ("thread", "process")
LIMIT = 1.5
BOTH, GPU_ALONE = fx.BOTH, "fixed:gpu_M"


def at(samples: list, pid: str, t: float) -> float | None:
    xs = [(s, row[pid]) for s, row in samples if row.get(pid) is not None]
    if len(xs) < 2 or not xs[0][0] <= t <= xs[-1][0]:
        return None
    ts, vs = zip(*xs, strict=True)
    return float(np.interp(t, ts, vs))


def window_cpu(w: dict) -> dict:
    """CPU seconds per role over the measurement span; roles absent from the run are omitted."""
    tc = w["tree_cpu"]
    lo, hi = w["measure"]
    by_pid = {str(pid): role for role, pid in tc["roles"].items()}
    pids = {pid for _, row in tc["samples"] for pid in row}
    out = {"other": 0.0}
    for pid in pids:
        a, b = at(tc["samples"], pid, lo), at(tc["samples"], pid, hi)
        if a is None or b is None:
            if pid in by_pid:  # a worker's CPU must never go missing silently
                raise ValueError(f"{by_pid[pid]} (pid {pid}) not sampled across {w['cell']} c{w['cycle']}")
            continue  # a transient child, e.g. ps between windows
        role = by_pid.get(pid, "other")
        out[role] = out.get(role, 0.0) + (b - a)
    return out


def completed(recs: list, run: dict, cell: str, cycle: int, stream: str) -> int:
    w = next(w for w in run["windows"] if w["cell"] == cell and w["cycle"] == cycle)
    lo, hi = w["measure"][0] * 1e3, w["measure"][1] * 1e3
    return sum(
        lo <= r["t_ms"]["received"] < hi
        for r in recs
        if r["cell"] == cell and r["cycle"] == cycle and r["stream"] == stream
    )


def cpu_summary(run: dict, recs: list, cell: str, streams: tuple) -> dict:
    cpu, n = {}, {s: 0 for s in streams}
    for w in (w for w in run["windows"] if w["cell"] == cell):
        for role, v in window_cpu(w).items():
            cpu[role] = cpu.get(role, 0.0) + v
        for s in streams:
            n[s] += completed(recs, run, cell, w["cycle"], s)
    tree = sum(cpu.values())
    total = sum(n.values())
    gpu_worker = cpu.get("gpu:gpu", 0.0)
    ane_side = cpu.get("parent", 0.0) + cpu.get("ane:ane", 0.0)
    return {
        "cpu_s": {k: round(v, 4) for k, v in cpu.items()},
        "tree_cpu_s": tree,
        "completed": n,
        "tree_cpu_ms_per_request": tree / total * 1e3 if total else None,
        "gpu_worker_cpu_ms_per_gpu_request": gpu_worker / n["gpu_M"] * 1e3 if n.get("gpu_M") else None,
        "parent_plus_ane_worker_cpu_ms_per_ane_request": ane_side / n["ane_B"] * 1e3 if n.get("ane_B") else None,
    }


def run_summary(path: Path) -> dict:
    out = fx.config_summary(path)
    run = fx.load(path)
    records, _ = fx.build(run)
    recs = [r for r in records if r["in_window"]]
    out["tree_cpu"] = {
        "both": cpu_summary(run, recs, BOTH, ("gpu_M", "ane_B")),
        "gpu_alone": cpu_summary(run, recs, GPU_ALONE, ("gpu_M",)),
    }
    out["gpu_rate"] = run["fixed_load"]["gpu_rate"]
    return out


def model_verdict(t: dict, p: dict) -> dict:
    checks = {
        "arrivals_sha256_identical": t["arrivals_sha256"] == p["arrivals_sha256"],
        "scheduled_arrivals_identical": t["scheduled_arrivals"] == p["scheduled_arrivals"],
        "gpu_offered_identical": t["cells"]["gpu"]["offered_req_s"] == p["cells"]["gpu"]["offered_req_s"],
        "stable": all(c["stable"] for r in (t, p) for c in r["stability"].values()),
        "no_errors": all(r["cells"][k]["errors"] == 0 for r in (t, p) for k in r["cells"]),
        "no_mismatches": all(r["cells"][k]["mismatches"] == 0 for r in (t, p) for k in r["cells"]),
    }
    a = t["tree_cpu"]["both"]["tree_cpu_ms_per_request"]
    b = p["tree_cpu"]["both"]["tree_cpu_ms_per_request"]
    ratio = b / a
    return {
        "checks": checks,
        "valid": all(checks.values()),
        "tree_cpu_ratio": ratio,
        "cpu_pass": ratio <= LIMIT,
    }


def summarise(rate: int) -> dict:
    res = {"gpu_rate": rate, "limit": LIMIT, "models": {}}
    for m in MODELS:
        runs = {pl: run_summary(RAW / f"{m}-{pl}-gpu{rate}.json.gz") for pl in PLACEMENTS}
        res["models"][m] = {"runs": runs, "verdict": model_verdict(runs["thread"], runs["process"])}
    return res


def f(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def tables(res: dict) -> str:
    L = [f"# Equal GPU offered load ({res['gpu_rate']} req/s target): thread vs process ANE placement\n"]
    L += [
        "## Workload equality and stability\n",
        "| model | placement | GPU offered req/s | GPU completed req/s | arrival SHA-256 (c0) | stable cycles | errors | mismatches |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        for pl, r in x["runs"].items():
            g = r["cells"]["gpu"]
            sha = r["arrivals_sha256"][f"{BOTH}|c0"][:16]
            st = sum(c["stable"] for c in r["stability"].values())
            err = sum(r["cells"][k]["errors"] for k in r["cells"])
            mm = sum(r["cells"][k]["mismatches"] for k in r["cells"])
            L.append(
                f"| {m} | {pl} | {f(g['offered_req_s'])} | {f(g['completed_in_window_req_s'])} | `{sha}…` "
                f"| {st}/{len(r['stability'])} | {err} | {mm} |"
            )
    L += [
        "\n## CPU, both devices busy (3 cycles × 20 s)\n",
        "| model | placement | tree CPU ms / request | GPU worker ms / GPU req | parent + ANE worker ms / ANE req "
        "| GPU forward thread CPU mean | ANE req/s | GPU return P50 / P99 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        for pl, r in x["runs"].items():
            c, g, a = r["tree_cpu"]["both"], r["cells"]["gpu"], r["cells"]["ane"]
            L.append(
                f"| {m} | {pl} | {f(c['tree_cpu_ms_per_request'])} | {f(c['gpu_worker_cpu_ms_per_gpu_request'])} "
                f"| {f(c['parent_plus_ane_worker_cpu_ms_per_ane_request'])} | {f(g['cpu_ms']['mean'])} "
                f"| {f(a['completed_in_window_req_s'], 1)} | {f(g['return_ms']['p50'], 3)} / {f(g['return_ms']['p99'], 3)} |"
            )
    L += [
        "\n## CPU seconds per role, both devices busy (sum of 3 windows)\n",
        "| model | placement | parent | GPU worker | ANE worker | other | tree | GPU done | ANE done |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        for pl, r in x["runs"].items():
            c = r["tree_cpu"]["both"]
            s = c["cpu_s"]
            L.append(
                f"| {m} | {pl} | {f(s.get('parent'))} | {f(s.get('gpu:gpu'))} | {f(s.get('ane:ane'))} "
                f"| {f(s.get('other'))} | {f(c['tree_cpu_s'])} | {c['completed']['gpu_M']} | {c['completed']['ane_B']} |"
            )
    L += [
        "\n## GPU alone, same arrivals\n",
        "| model | placement | tree CPU ms / request | GPU worker ms / GPU req |",
        "|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        for pl, r in x["runs"].items():
            c = r["tree_cpu"]["gpu_alone"]
            L.append(
                f"| {m} | {pl} | {f(c['tree_cpu_ms_per_request'])} | {f(c['gpu_worker_cpu_ms_per_gpu_request'])} |"
            )
    L += [
        "\n## Verdict (criterion: process / thread tree CPU per request ≤ 1.5)\n",
        "| model | workload valid | ratio | CPU |",
        "|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        v = x["verdict"]
        L.append(f"| {m} | {v['valid']} | ×{f(v['tree_cpu_ratio'])} | {'pass' if v['cpu_pass'] else 'FAIL'} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=int, default=30)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    res = summarise(a.rate)
    js = json.dumps(res, indent=1, sort_keys=True, default=float) + "\n"
    md = tables(res)
    if a.check:
        stale = [p.name for p, s in ((ROOT / "results.json", js), (ROOT / "tables.md", md)) if p.read_text() != s]
        if stale:
            sys.exit(f"stale: {stale}")
        print("outputs are up to date")
        return
    (ROOT / "results.json").write_text(js)
    (ROOT / "tables.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
