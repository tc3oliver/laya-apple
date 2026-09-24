"""Summarise the completion-path runs: results.json and tables.md.

    uv run python research/coreml-gil-completion-path/scripts/analyze.py [--check]

Inputs (raw/, written by run.sh):
  A-thread.json.gz          ANE thread placement, coremltools predict (the product path)
  B-process.json.gz         ANE process placement (the control)
  C-thread-nogil.json.gz    ANE thread placement, predict through PyObjC (GIL released)
  completion-probe-*.json   the 2x2 probe (completion_probe.py)

A-C are runs of the gpu-ane-interference harness, joined into the canonical request ledger by
research/gpu-ane-interference/scripts/ledger.py. Only in-window requests count. Durations come
from the runtime trace (return = service_end -> received, occupancy = dispatch -> received),
the backend hooks (device_exec = mx.eval or Core ML predict spans) and boundary.py's parent-side
stamps on the GPU reply (header = the reply's read returned with the GIL held again).
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
sys.path.insert(0, str(ROOT.parent / "gpu-ane-interference" / "scripts"))
from ledger import annotate_overlap, build, load, rows  # noqa: E402

CONFIGS = {
    "A": ("A-thread.json.gz", "thread, coremltools predict (product)"),
    "B": ("B-process.json.gz", "process (control)"),
    "C": ("C-thread-nogil.json.gz", "thread, PyObjC predict (GIL released)"),
}
CELLS = {
    "solo_gpu": ("solo:gpu_M", "gpu_M"),
    "solo_ane": ("solo:ane_B", "ane_B"),
    "matrix_gpu": ("matrix:gpu_M+ane_B", "gpu_M"),
    "matrix_ane": ("matrix:gpu_M+ane_B", "ane_B"),
}


def pct(a) -> dict:
    a = np.asarray([x for x in a if x is not None], float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p5": float(np.percentile(a, 5)),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }


def seconds(run: dict, cell: str) -> float:
    return sum(w["measure"][1] - w["measure"][0] for w in run["windows"] if w["cell"] == cell)


def boundary(run: dict, recs: list, ane_predicts: list) -> dict:
    """GPU reply leg split at the parent's stamps, and its alignment to the ANE's predict ends."""
    st = run["completion_path"]["stamps"]
    by_id = {r["request_id"]: r for r in rows(st)}
    starts = sorted(ane_predicts)
    parts = defaultdict(list)
    inside = 0
    for r in recs:
        s = by_id.get(r["request_id"])
        if s is None:
            continue
        end = r["t_ms"]["service_end"]
        hdr, body, loaded = s["header_us"] / 1e3, s["body_us"] / 1e3, s["loaded_us"] / 1e3
        parts["service_end_to_header"].append(hdr - end)
        parts["header_to_body"].append(body - hdr)
        parts["body_to_loaded"].append(loaded - body)
        parts["loaded_to_received"].append(r["t_ms"]["received"] - loaded)
        i = bisect.bisect_right(starts, (end, float("inf"))) - 1  # the last predict started before
        if i >= 0 and starts[i][0] <= end < starts[i][1]:
            inside += 1
            parts["predict_end_to_header"].append(hdr - starts[i][1])
            parts["predict_left_at_service_end"].append(starts[i][1] - end)
            parts["_header_when_inside"].append(hdr - end)
    paired = parts.pop("_header_when_inside")
    out = {k: pct(v) for k, v in parts.items()}
    out["stamped"] = len(parts["service_end_to_header"])
    out["forwards_ending_inside_an_ane_predict"] = inside
    # does the wait track the predict time left when the forward ended?
    left = parts["predict_left_at_service_end"]
    out["corr_predict_left_vs_header_wait"] = float(np.corrcoef(left, paired)[0, 1]) if len(left) > 2 else None
    return out


def config_summary(path: Path) -> dict:
    run = load(path)
    records, report = build(run)
    annotate_overlap(records)
    recs = [r for r in records if r["in_window"]]
    backend = {r["request_id"]: r for r in rows(run["backend"]) if r["request_id"] is not None}
    out = {
        "join": {
            k: report[k] for k in ("traces", "traces_joined_1to1", "join_rate", "backend_events_without_request_id")
        },
        "load_s": run["load_s"],
        "cells": {},
    }
    for key, (cell, stream) in CELLS.items():
        rs = [r for r in recs if r["cell"] == cell and r["stream"] == stream]
        t = lambda k: [r["timing_ms"][k] for r in rs]  # noqa: E731
        s = seconds(run, cell)
        c = {
            "requests": len(rs),
            "req_s": len(rs) / s,
            "mismatches": sum(r["match"] is False for r in rs),
            "reasons": dict(Counter(r["routing"]["reason"] for r in rs)),
            "return_ms": pct(t("return")),
            "occupancy_ms": pct(t("occupancy")),
            "service_ms": pct(t("service")),
            "device_exec_ms": pct(t("device_exec")),
            "host_ms": pct(t("host")),
            "e2e_ms": pct(t("e2e")),
            "ipc_ms": pct([r["timing_ms"]["dispatch"] + r["timing_ms"]["return"] for r in rs]),
            "cpu_ms": pct([backend[r["request_id"]]["cpu_us"] / 1e3 for r in rs if r["request_id"] in backend]),
            "overlap": pct([r.get("overlap") for r in rs]) if cell.startswith("matrix") else None,
        }
        out["cells"][key] = c
    # the ANE's predict spans in the matrix cell, for the GPU reply alignment
    mat = "matrix:gpu_M+ane_B"
    ane_ids = {r["request_id"] for r in recs if r["cell"] == mat and r["device"] == "ane"}
    ane_ids |= {r["request_id"] for r in records if r["cell"] == mat and r["device"] == "ane"}
    predicts = [(a / 1e3, b / 1e3) for i, e in backend.items() if i in ane_ids for a, b in e["spans"]]
    gpu = [r for r in recs if r["cell"] == mat and r["stream"] == "gpu_M"]
    out["gpu_reply_boundary"] = boundary(run, gpu, predicts)
    return out


def probe_summary(path: Path) -> dict:
    p = json.loads(path.read_text())
    out = {"sleep_us": p["sleep_us"], "stack_samples": p["stack_samples"], "loads": {}}
    by = defaultdict(list)
    for r in p["results"]:
        by[r["load"]].append(r)
    for load, rs in by.items():
        out["loads"][load] = {
            "repeats": len(rs),
            "gpu_req_s": [r["gpu_req_s"] for r in rs],
            "mismatches": sum(r["mismatches"] for r in rs),
            **{
                m: {q: [r[m][q] for r in rs] for q in ("mean", "p50", "p95", "p99")}
                for m in ("return_ms", "service_end_to_header_ms", "header_to_received_ms", "gpu_service_ms")
            },
            "load_call_ms_p50": [r["load_call_ms"]["p50"] for r in rs] if rs[0]["load_call_ms"] else None,
            "load_call_end_to_header_ms_p50": [r["load_call_end_to_header_ms"].get("p50") for r in rs],
        }
    return out


def criteria(res: dict) -> dict:
    """The success criteria written in issue #45 before any run, evaluated as written."""
    A, B, C = (res["configs"][k]["cells"] for k in "ABC")
    ret = C["matrix_gpu"]["return_ms"]
    return {
        "gpu_return_mean_ms": {
            "A": A["matrix_gpu"]["return_ms"]["mean"],
            "B": B["matrix_gpu"]["return_ms"]["mean"],
            "C": ret["mean"],
            "C_le_0.5ms_mean": ret["mean"] <= 0.5,
            "C_le_0.5ms_p50": ret["p50"] <= 0.5,
        },
        "gpu_device_exec_mean_ms": {k: X["matrix_gpu"]["device_exec_ms"]["mean"] for k, X in zip("ABC", (A, B, C))},
        "ane_req_s_change_C_vs_A": {
            cell: C[cell]["req_s"] / A[cell]["req_s"] - 1 for cell in ("solo_ane", "matrix_ane")
        },
        "ane_e2e_p99_ratio_C_vs_A": {
            cell: C[cell]["e2e_ms"]["p99"] / A[cell]["e2e_ms"]["p99"] for cell in ("solo_ane", "matrix_ane")
        },
        "mismatches": {k: sum(c["mismatches"] for c in X.values()) for k, X in zip("ABC", (A, B, C))},
    }


def fmt(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def tables(res: dict) -> str:
    L = ["# Completion-path results", "", "Generated by `scripts/analyze.py` from `raw/`. ms unless noted.", ""]
    L += [
        "## GPU L128 with ANE L128 busy (matrix cell), and solo",
        "",
        "| config | cell | GPU req/s | return P50 | return P95 | return P99 | occupancy P50 | occupancy P95 | occupancy P99 "
        "| service mean | mx.eval mean | e2e P50 | e2e P95 | e2e P99 | mismatches |",
        "|" + "---|" * 15,
    ]
    for k, (_, label) in CONFIGS.items():
        for cell in ("solo_gpu", "matrix_gpu"):
            c = res["configs"][k]["cells"][cell]
            L.append(
                f"| {k} {label} | {cell} | {fmt(c['req_s'], 1)} | {fmt(c['return_ms']['p50'], 3)} | "
                f"{fmt(c['return_ms']['p95'], 3)} | {fmt(c['return_ms']['p99'], 3)} | {fmt(c['occupancy_ms']['p50'])} | "
                f"{fmt(c['occupancy_ms']['p95'])} | {fmt(c['occupancy_ms']['p99'])} | {fmt(c['service_ms']['mean'])} | "
                f"{fmt(c['device_exec_ms']['mean'])} | {fmt(c['e2e_ms']['p50'])} | {fmt(c['e2e_ms']['p95'])} | "
                f"{fmt(c['e2e_ms']['p99'])} | {c['mismatches']} |"
            )
    L += [
        "",
        "## ANE L128 with GPU L128 busy (matrix cell), and solo",
        "",
        "| config | cell | ANE req/s | predict mean | predict P99 | host mean | dispatch+return mean | occupancy P50 | e2e P50 "
        "| e2e P95 | e2e P99 | CPU/request mean | mismatches |",
        "|" + "---|" * 13,
    ]
    for k, (_, label) in CONFIGS.items():
        for cell in ("solo_ane", "matrix_ane"):
            c = res["configs"][k]["cells"][cell]
            L.append(
                f"| {k} {label} | {cell} | {fmt(c['req_s'], 1)} | {fmt(c['device_exec_ms']['mean'])} | "
                f"{fmt(c['device_exec_ms']['p99'])} | {fmt(c['host_ms']['mean'])} | {fmt(c['ipc_ms']['mean'], 3)} | "
                f"{fmt(c['occupancy_ms']['p50'])} | {fmt(c['e2e_ms']['p50'])} | "
                f"{fmt(c['e2e_ms']['p95'])} | {fmt(c['e2e_ms']['p99'])} | {fmt(c['cpu_ms']['mean'])} | {c['mismatches']} |"
            )
    L += [
        "",
        "## Both devices at once (matrix cell), and start-up",
        "",
        "| config | GPU req/s | ANE req/s | total req/s | GPU mean overlap with ANE busy | load_s (both instances) |",
        "|---|---|---|---|---|---|",
    ]
    for k, (_, label) in CONFIGS.items():
        x = res["configs"][k]
        g, a = x["cells"]["matrix_gpu"], x["cells"]["matrix_ane"]
        L.append(
            f"| {k} {label} | {fmt(g['req_s'], 1)} | {fmt(a['req_s'], 1)} | {fmt(g['req_s'] + a['req_s'], 1)} | "
            f"{fmt(g['overlap']['mean'], 3)} | {fmt(x['load_s'], 1)} |"
        )
    L += [
        "",
        "## GPU reply leg, parent side (matrix cell)",
        "",
        "| config | stamped | forwards ending inside an ANE predict | service_end→header P50 | P95 | header→body P50 "
        "| body→loaded P50 | loaded→received P50 | header − that predict's end P50 | P5..P95 | corr(predict left, wait) |",
        "|" + "---|" * 11,
    ]
    for k, (_, label) in CONFIGS.items():
        b = res["configs"][k]["gpu_reply_boundary"]
        al = b.get("predict_end_to_header", {"n": 0})
        L.append(
            f"| {k} {label} | {b['stamped']} | {b['forwards_ending_inside_an_ane_predict']} | "
            f"{fmt(b['service_end_to_header']['p50'], 3)} | {fmt(b['service_end_to_header']['p95'], 3)} | "
            f"{fmt(b['header_to_body']['p50'], 3)} | {fmt(b['body_to_loaded']['p50'], 3)} | "
            f"{fmt(b['loaded_to_received']['p50'], 3)} | {fmt(al.get('p50'), 3)} | "
            f"{fmt(al.get('p5'), 3)}..{fmt(al.get('p95'), 3)} | {fmt(b['corr_predict_left_vs_header_wait'])} |"
        )
    p = res["probe"]
    L += [
        "",
        f"## 2x2 probe (GPU L128 closed loop next to one load thread; sleep = {p['sleep_us']} µs)",
        "",
        "| load | Core ML | GIL | return P50 (per repeat) | return P99 (per repeat) | service_end→header P50 | GPU req/s | mismatches |",
        "|---|---|---|---|---|---|---|---|",
    ]
    kind = {
        "idle": ("–", "–"),
        "ct_predict": ("yes", "held"),
        "objc_predict": ("yes", "released"),
        "held_sleep": ("no", "held"),
        "free_sleep": ("no", "released"),
    }
    for load, v in p["loads"].items():
        j = lambda xs, d=3: " / ".join(fmt(x, d) for x in xs)  # noqa: E731
        L.append(
            f"| {load} | {kind[load][0]} | {kind[load][1]} | {j(v['return_ms']['p50'])} | {j(v['return_ms']['p99'])} | "
            f"{j(v['service_end_to_header_ms']['p50'])} | {j(v['gpu_req_s'], 1)} | {v['mismatches']} |"
        )
    L += [
        "",
        "Stack samples of the GPU dispatcher (1 ms, `sample`), under its `os_read` / `os_write`:",
        "",
        "| load | read syscall | read done, in take_gil | write syscall | write done, in take_gil |",
        "|---|---|---|---|---|",
    ]
    for load, s in p["stack_samples"].items():
        L.append(
            f"| {load} | {s['read_syscall']} | {s['read_then_take_gil']} | {s['write_syscall']} | {s['write_then_take_gil']} |"
        )
    c = res["criteria"]
    L += ["", "## Success criteria (issue #45, fixed before the runs)", "", "```", json.dumps(c, indent=1), "```", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    res = {
        "configs": {k: {"label": label, **config_summary(RAW / f)} for k, (f, label) in CONFIGS.items()},
        "probe": probe_summary(next(RAW.glob("completion-probe-*.json"))),
    }
    res["criteria"] = criteria(res)
    outputs = {ROOT / "results.json": json.dumps(res, indent=1, sort_keys=True) + "\n", ROOT / "tables.md": tables(res)}
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
