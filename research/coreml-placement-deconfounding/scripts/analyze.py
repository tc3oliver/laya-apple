"""Summarise the fixed-GPU-load runs: results.json and tables.md.

    uv run python research/coreml-placement-deconfounding/scripts/analyze.py [--check] [--rate 45]

Inputs: raw/{A-thread,B-process,C-thread-nogil}-gpu<rate>.json.gz (run.sh). Each run is joined
into the canonical request ledger (research/gpu-ane-interference/scripts/ledger.py). Only
in-window requests count: a request is in the window if its scheduled arrival (GPU) or its
submission (ANE, closed loop) falls inside the 20 s measurement span.

Workload equality is checked before anything else. For every window, the three runs must
carry the same SHA-256 over the GPU stream's scheduled arrival offsets and request seeds.

Queue stability, fixed before the runs: in every cycle of the concurrent cell, the GPU stream
must have 0 errors, complete ≥98% of its offered in-window rate within the window, and have a
last-quarter mean queue delay ≤ max(2 × the first quarter's, the first quarter's + 20 ms).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
STUDIES = ROOT.parent
sys.path.insert(0, str(STUDIES / "gpu-ane-interference" / "scripts"))
from ledger import annotate_overlap, build, load, rows  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "gil_analyze", STUDIES / "coreml-gil-completion-path" / "scripts" / "analyze.py"
)
gil = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gil)
pct = gil.pct

CONFIGS = {
    "A": ("A-thread", "thread + coremltools (product)"),
    "B": ("B-process", "process + coremltools"),
    "C": ("C-thread-nogil", "thread + PyObjC, GIL released"),
}
BOTH = "fixed:gpu_M+ane_B"
CELLS = {
    "ane_alone": ("solo:ane_B", "ane_B"),
    "gpu_alone": ("fixed:gpu_M", "gpu_M"),
    "gpu": (BOTH, "gpu_M"),
    "ane": (BOTH, "ane_B"),
}


def seconds(run: dict, cell: str) -> float:
    return gil.seconds(run, cell)


def stability(run: dict, recs: list) -> dict:
    """The pre-set rule, per cycle of the concurrent cell."""
    out = {}
    windows = {w["cycle"]: w for w in run["windows"] if w["cell"] == BOTH}
    for cycle, w in sorted(windows.items()):
        lo, hi = w["measure"][0] * 1e3, w["measure"][1] * 1e3  # ms on the run's axis
        g = sorted(
            (r for r in recs if r["cell"] == BOTH and r["cycle"] == cycle and r["stream"] == "gpu_M"),
            key=lambda r: r["t_ms"]["arrival"],
        )
        q = np.array([r["timing_ms"]["queue"] for r in g])
        quarters = [float(x.mean()) for x in np.array_split(q, 4)]
        offered = len(g) / (hi - lo) * 1e3
        completed = sum(lo <= r["t_ms"]["received"] < hi for r in g) / (hi - lo) * 1e3
        errors = len(w["streams"]["gpu_M"]["errors"])
        ok = errors == 0 and completed >= 0.98 * offered and quarters[3] <= max(2 * quarters[0], quarters[0] + 20.0)
        out[f"c{cycle}"] = {
            "offered_req_s": offered,
            "completed_in_window_req_s": completed,
            "errors": errors,
            "queue_quarter_means_ms": quarters,
            "stable": bool(ok),
        }
    return out


def cell_stats(run: dict, recs: list, backend: dict, cell: str, stream: str) -> dict:
    rs = [r for r in recs if r["cell"] == cell and r["stream"] == stream]
    t = lambda k: [r["timing_ms"][k] for r in rs]  # noqa: E731
    s = seconds(run, cell)
    lo_hi = [(w["measure"][0] * 1e3, w["measure"][1] * 1e3) for w in run["windows"] if w["cell"] == cell]
    done_in = sum(any(lo <= r["t_ms"]["received"] < hi for lo, hi in lo_hi) for r in rs)
    errors = sum(len(w["streams"][stream]["errors"]) for w in run["windows"] if w["cell"] == cell)
    return {
        "requests": len(rs),
        "offered_req_s": len(rs) / s,
        "completed_in_window_req_s": done_in / s,
        "errors": errors,
        "mismatches": sum(r["match"] is False for r in rs),
        "queue_ms": pct(t("queue")),
        "return_ms": pct(t("return")),
        "e2e_ms": pct(t("e2e")),
        "service_ms": pct(t("service")),
        "device_exec_ms": pct(t("device_exec")),
        "host_ms": pct(t("host")),
        "ipc_ms": pct([r["timing_ms"]["dispatch"] + r["timing_ms"]["return"] for r in rs]),
        "generator_lag_ms": pct(t("generator_lag")),
        "depth_at_routing": pct([r["queue"]["gpu_depth" if stream.startswith("gpu") else "ane_depth"] for r in rs]),
        "backlog_at_routing_ms": pct(
            [r["routing"]["gpu_backlog_ms" if stream.startswith("gpu") else "ane_backlog_ms"] for r in rs]
        ),
        "cpu_ms": pct([backend[r["request_id"]]["cpu_us"] / 1e3 for r in rs if r["request_id"] in backend]),
        "overlap": pct([r.get("overlap") for r in rs]) if cell == BOTH else None,
    }


def config_summary(path: Path) -> dict:
    run = load(path)
    records, report = build(run)
    annotate_overlap(records)
    recs = [r for r in records if r["in_window"]]
    backend = {r["request_id"]: r for r in rows(run["backend"]) if r["request_id"] is not None}
    out = {
        "ane_predict": run["completion_path"]["ane_predict"],
        "ane_placement": run["args"]["ane_placement"],
        "join": {k: report[k] for k in ("traces", "traces_joined_1to1", "join_rate")},
        "arrivals_sha256": {k: v["sha256"] for k, v in sorted(run["fixed_load"]["arrivals"].items())},
        "scheduled_arrivals": {k: len(v["offsets_ns"]) for k, v in sorted(run["fixed_load"]["arrivals"].items())},
        "stability": stability(run, recs),
        "cells": {k: cell_stats(run, recs, backend, c, s) for k, (c, s) in CELLS.items()},
    }
    ane_ids = {r["request_id"] for r in records if r["cell"] == BOTH and r["device"] == "ane"}
    predicts = [(a / 1e3, b / 1e3) for i, e in backend.items() if i in ane_ids for a, b in e["spans"]]
    gpu = [r for r in recs if r["cell"] == BOTH and r["stream"] == "gpu_M"]
    out["gpu_reply_boundary"] = gil.boundary(run, gpu, predicts)
    return out


def compare(res: dict) -> dict:
    c = {k: v["cells"] for k, v in res["configs"].items()}
    shas = {k: v["arrivals_sha256"] for k, v in res["configs"].items()}
    out = {
        "arrival_trace_identical_across_configs": len({json.dumps(v, sort_keys=True) for v in shas.values()}) == 1,
        "all_cycles_stable": all(s["stable"] for v in res["configs"].values() for s in v["stability"].values()),
        "mismatches": {k: sum(x["mismatches"] for x in v.values()) for k, v in c.items()},
        "errors": {k: sum(x["errors"] for x in v.values()) for k, v in c.items()},
    }
    for k in ("B", "C"):
        out[f"{k}_vs_A"] = {
            "ane_req_s_change": c[k]["ane"]["offered_req_s"] / c["A"]["ane"]["offered_req_s"] - 1,
            "ane_e2e_p99_ratio": c[k]["ane"]["e2e_ms"]["p99"] / c["A"]["ane"]["e2e_ms"]["p99"],
            "ane_alone_req_s_change": c[k]["ane_alone"]["offered_req_s"] / c["A"]["ane_alone"]["offered_req_s"] - 1,
        }
    out["C_vs_B"] = {
        "ane_req_s_change": c["C"]["ane"]["offered_req_s"] / c["B"]["ane"]["offered_req_s"] - 1,
        "ane_e2e_p99_ratio": c["C"]["ane"]["e2e_ms"]["p99"] / c["B"]["ane"]["e2e_ms"]["p99"],
    }
    # the ANE's own slowdown from running beside the GPU, per configuration
    out["ane_both_vs_alone"] = {
        k: v["ane"]["offered_req_s"] / v["ane_alone"]["offered_req_s"] - 1 for k, v in c.items()
    }
    return out


def f(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def tables(res: dict) -> str:
    cfg = res["configs"]
    L = [f"# Fixed GPU load: GPU L128 open loop at {res['gpu_rate']:g} req/s + ANE L128 closed loop", ""]
    L += ["Generated by `scripts/analyze.py` from `raw/`. ms unless noted; P50 / P95 / P99.", ""]
    L += [
        "## Workload equality and stability",
        "",
        "| config | windows | scheduled GPU arrivals per window | arrival SHA-256 (first window) | cycles stable |",
        "|---|---|---|---|---|",
    ]
    for k, (_, label) in CONFIGS.items():
        v = cfg[k]
        first = next(iter(v["arrivals_sha256"].values()))
        L.append(
            f"| {k} {label} | {len(v['arrivals_sha256'])} | {sorted(set(v['scheduled_arrivals'].values()))} | "
            f"`{first[:16]}` | {sum(s['stable'] for s in v['stability'].values())}/{len(v['stability'])} |"
        )
    L += [
        "",
        "| config | cycle | offered req/s | completed in window req/s | errors | queue mean by quarter |",
        "|---|---|---|---|---|---|",
    ]
    for k in CONFIGS:
        for cyc, s in cfg[k]["stability"].items():
            L.append(
                f"| {k} | {cyc} | {f(s['offered_req_s'])} | {f(s['completed_in_window_req_s'])} | {s['errors']} | "
                + " / ".join(f(x, 1) for x in s["queue_quarter_means_ms"])
                + " |"
            )
    tri = lambda p, d=2: f"{f(p.get('p50'), d)} / {f(p.get('p95'), d)} / {f(p.get('p99'), d)}"  # noqa: E731
    L += [
        "",
        "## GPU",
        "",
        "| config | cell | offered req/s | completed req/s | queue | return | e2e | service mean / P99 | mx.eval mean / P99 | depth at routing P50 / P99 | backlog at routing P50 / P99 | generator lag P99 |",
        "|" + "---|" * 12,
    ]
    for k, (_, label) in CONFIGS.items():
        for cell in ("gpu_alone", "gpu"):
            x = cfg[k]["cells"][cell]
            L.append(
                f"| {k} | {'alone' if cell == 'gpu_alone' else 'with ANE'} | {f(x['offered_req_s'])} | "
                f"{f(x['completed_in_window_req_s'])} | {tri(x['queue_ms'], 1)} | {tri(x['return_ms'], 3)} | "
                f"{tri(x['e2e_ms'], 1)} | {f(x['service_ms']['mean'])} / {f(x['service_ms']['p99'])} | "
                f"{f(x['device_exec_ms']['mean'])} / {f(x['device_exec_ms']['p99'])} | "
                f"{f(x['depth_at_routing']['p50'], 0)} / {f(x['depth_at_routing']['p99'], 0)} | "
                f"{f(x['backlog_at_routing_ms']['p50'], 1)} / {f(x['backlog_at_routing_ms']['p99'], 1)} | "
                f"{f(x['generator_lag_ms']['p99'], 2)} |"
            )
    L += [
        "",
        "## ANE",
        "",
        "| config | cell | req/s | predict mean / P95 / P99 | e2e | host mean | CPU/request mean | dispatch+return mean |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k in CONFIGS:
        for cell in ("ane_alone", "ane"):
            x = cfg[k]["cells"][cell]
            d = x["device_exec_ms"]
            L.append(
                f"| {k} | {'alone' if cell == 'ane_alone' else 'with GPU'} | {f(x['offered_req_s'], 1)} | "
                f"{f(d['mean'])} / {f(d['p95'])} / {f(d['p99'])} | {tri(x['e2e_ms'])} | {f(x['host_ms']['mean'])} | "
                f"{f(x['cpu_ms']['mean'])} | {f(x['ipc_ms']['mean'], 3)} |"
            )
    L += [
        "",
        "## Joint (concurrent cell)",
        "",
        "| config | GPU completed req/s | ANE req/s | total req/s | GPU overlap mean | ANE overlap mean | mismatches |",
        "|---|---|---|---|---|---|---|",
    ]
    for k in CONFIGS:
        g, a = cfg[k]["cells"]["gpu"], cfg[k]["cells"]["ane"]
        L.append(
            f"| {k} | {f(g['completed_in_window_req_s'], 1)} | {f(a['offered_req_s'], 1)} | "
            f"{f(g['completed_in_window_req_s'] + a['offered_req_s'], 1)} | {f(g['overlap']['mean'], 3)} | "
            f"{f(a['overlap']['mean'], 3)} | {res['comparison']['mismatches'][k]} |"
        )
    L += ["", "## Comparison", "", "```", json.dumps(res["comparison"], indent=1), "```", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", default="45")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    res = {
        "gpu_rate": float(args.rate),
        "configs": {
            k: {"label": label, **config_summary(RAW / f"{stem}-gpu{args.rate}.json.gz")}
            for k, (stem, label) in CONFIGS.items()
        },
    }
    res["comparison"] = compare(res)
    outputs = {
        ROOT / "results.json": json.dumps(res, indent=1, sort_keys=True) + "\n",
        ROOT / "tables.md": tables(res),
    }
    if args.check:
        stale = [p.name for p, text in outputs.items() if not p.exists() or p.read_text() != text]
        if stale:
            raise SystemExit(f"out of date: {stale}; re-run analyze.py")
        print("outputs are up to date")
        return
    for p, text in outputs.items():
        p.write_text(text)
    print(outputs[ROOT / "tables.md"])


if __name__ == "__main__":
    main()
