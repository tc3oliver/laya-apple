"""The environment qualification: qual_results.json and qual_tables.md from raw-qual/ (../qualification.md).

    uv run python research/coreml-staged-handoff/scripts/qual_analyze.py [--check] [--raw DIR] [--out DIR]

Inputs: raw-qual/product-laya-A-r{1,2,3}.json.gz (prod_run.py, cell A, product schedule), their
.before.json / .after.json machine snapshots (machine_snapshot.py) and raw-qual/failed/*.log.
The rule is qualification.md's, items 1-5; a single high P99 is reported, never a reason on its own.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RAW = EXP / "raw-qual"
RUNS = [f"product-laya-A-r{r}" for r in (1, 2, 3)]
S = 1_000_000_000
AGG = (109.9, 135.0)
HOST_SLOW_MS = 0.3
BIN_S, SHARE = 0.5, 0.10
SPAN_S, SLOW_RATIO = 1.0, 1.2
SKIP_S = 1.0
MACHINE = {"memory_free_pct": 50, "cpu_idle_pct": 80.0, "process_cpu_pct": 50.0, "swap_pages": 10_000}
ROUTES = {
    "solo_short": {"short": {"ane"}},
    "solo_long": {"long": {"gpu"}},
    "hetero": {"short": {"ane"}, "long": {"gpu"}},
    "gpu_only": {"short": {"gpu"}, "long": {"gpu"}},
}
COL = {c: i for i, c in enumerate(
    ["target", "submit_ns", "prepared_ns", "service_start_ns", "service_end_ns", "received_ns", "response_ns"]
)}  # fmt: skip


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


def consecutive(flags) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def window(run: dict, w: dict, pooled_median: float) -> dict:
    t0, end = w["start_ns"], w["end_ns"]
    tr = [r for r in run["trace"] if t0 <= r[COL["submit_ns"]] < end]
    ane = [r for r in tr if r[0] == "ane"]
    gpu = [r for r in run["trace"] if r[0] == "gpu" and t0 <= r[COL["received_ns"]] < end]
    lat = np.asarray(w["streams"]["short"]["latency_ms"], float)
    sub = np.asarray([r[COL["submit_ns"]] for r in ane], np.int64)
    prep = np.asarray([(r[COL["prepared_ns"]] - r[COL["submit_ns"]]) / 1e6 for r in ane])
    resp = np.asarray([(r[COL["response_ns"]] - r[COL["submit_ns"]]) / 1e6 for r in ane])
    lo = t0 + int(SKIP_S * S)
    nb = int(np.ceil((end - lo) / (BIN_S * S)))
    slow_bins = []
    for b in range(nb):
        m = (sub >= lo + int(b * BIN_S * S)) & (sub < lo + int((b + 1) * BIN_S * S))
        slow_bins.append(bool(m.any() and (prep[m] > HOST_SLOW_MS).mean() >= SHARE))
    ns = int(np.ceil((end - lo) / (SPAN_S * S)))
    slow_spans = []
    for k in range(ns):
        m = (sub >= lo + int(k * SPAN_S * S)) & (sub < lo + int((k + 1) * SPAN_S * S))
        slow_spans.append(bool(m.any() and np.median(resp[m]) > SLOW_RATIO * pooled_median))
    ret = np.asarray([(r[COL["received_ns"]] - r[COL["service_end_ns"]]) / 1e6 for r in gpu])
    agg = sum(s["req_s"] for s in w["streams"].values())
    return {
        "index": w["index"],
        "n_short": int(len(lat)),
        "short_ms": {q: pct(lat, v) for q, v in (("median", 50), ("p95", 95), ("p99", 99), ("p999", 99.9))},
        "aggregate_req_s": agg,
        "gpu_return_ms": {q: pct(ret, v) for q, v in (("p50", 50), ("p95", 95), ("p99", 99))},
        "host_slow_longest_bins": consecutive(slow_bins),
        "slow_state_longest_spans": consecutive(slow_spans),
        "response_matches_trace": len(resp) == len(lat),
    }


def snapshot_checks(before: dict | None, after: dict | None) -> list[str]:
    bad = []
    for tag, s in (("before", before), ("after", after)):
        if s is None:
            bad.append(f"{tag}: snapshot missing")
            continue
        if s.get("thermal_warning") or s.get("performance_warning"):
            bad.append(f"{tag}: thermal / performance warning")
        if "AC Power" not in (s.get("power") or ""):
            bad.append(f"{tag}: not on AC power")
        if (s.get("memory_free_pct") or 0) < MACHINE["memory_free_pct"]:
            bad.append(f"{tag}: memory free {s.get('memory_free_pct')}% < {MACHINE['memory_free_pct']}%")
        if s.get("time_machine_running"):
            bad.append(f"{tag}: Time Machine running")
        for c, n in s.get("top_cpu") or []:
            if c > MACHINE["process_cpu_pct"] and not n.startswith("python"):
                bad.append(f"{tag}: {n} at {c:.0f}% CPU")
    if before is not None:
        idle = (before.get("cpu_user_sys_idle_pct") or [0, 0, 0])[2]
        if idle < MACHINE["cpu_idle_pct"]:
            bad.append(f"before: CPU idle {idle:.1f}% < {MACHINE['cpu_idle_pct']}%")
    if before is not None and after is not None:
        vb, va = before.get("vm_counters") or {}, after.get("vm_counters") or {}
        d = sum(va.get(k, 0) - vb.get(k, 0) for k in ("Swapins", "Swapouts"))
        if d > MACHINE["swap_pages"]:
            bad.append(f"swapins + swapouts during the run: {d} pages > {MACHINE['swap_pages']}")
    return bad


def load_json(p: Path):
    if not p.exists():
        return None
    if p.suffix == ".gz":
        with gzip.open(p, "rt") as fh:
            return json.load(fh)
    return json.loads(p.read_text())


def analyse_run(raw: Path, name: str) -> dict:
    run = load_json(raw / f"{name}.json.gz")
    before, after = load_json(raw / f"{name}.before.json"), load_json(raw / f"{name}.after.json")
    crashed = any((raw / "failed").glob(f"{name}.*.log"))
    if run is None:
        return {"run": name, "present": False, "crashed": crashed, "snapshots": [before, after]}
    het = [w for w in run["windows"] if w["condition"] == "hetero"]
    pooled = np.concatenate([np.asarray(w["streams"]["short"]["latency_ms"], float) for w in het])
    med = float(np.median(pooled))
    wins = [window(run, w, med) for w in het]
    mism = sum(s["mismatches"] for w in run["windows"] for s in w["streams"].values())
    routing = [
        f"{w['index']}:{w['condition']}:{s}"
        for w in run["windows"]
        for s, want in ROUTES.get(w["condition"], {}).items()
        if set(w["streams"][s]["devices"]) != want
    ]
    alive = run.get("workers_alive_at_end") or {}
    fails = []
    if mism or routing or crashed or not alive or not all(alive.values()):
        fails.append("correctness")
    if any(not (AGG[0] <= w["aggregate_req_s"] <= AGG[1]) for w in wins):
        fails.append("throughput")
    if any(w["host_slow_longest_bins"] >= 2 for w in wins):
        fails.append("persistent host-slow")
    if any(w["slow_state_longest_spans"] >= 2 for w in wins):
        fails.append("sustained slow state")
    machine = snapshot_checks(before, after)
    if machine:
        fails.append("machine")
    gpu = np.asarray(
        [
            (r[COL["received_ns"]] - r[COL["service_end_ns"]]) / 1e6
            for w in het
            for r in run["trace"]
            if r[0] == "gpu" and w["start_ns"] <= r[COL["received_ns"]] < w["end_ns"]
        ]
    )
    return {
        "run": name,
        "present": True,
        "crashed": crashed,
        "mismatches": mism,
        "routing_failures": routing,
        "workers_alive_at_end": alive,
        "pooled_short_ms": {q: pct(pooled, v) for q, v in (("median", 50), ("p95", 95), ("p99", 99), ("p999", 99.9))},
        "pooled_gpu_return_ms": {q: pct(gpu, v) for q, v in (("p50", 50), ("p95", 95), ("p99", 99))},
        "windows": wins,
        "machine_findings": machine,
        "snapshots": {"before": before, "after": after},
        "failed": fails,
    }


def summarise(raw: Path) -> dict:
    runs = [analyse_run(raw, n) for n in RUNS]
    missing = [r["run"] for r in runs if not r["present"]]
    crashed = [r["run"] for r in runs if r["crashed"]]
    fails = [f"{r['run']}: {', '.join(r['failed'])}" for r in runs if r.get("failed")]
    if crashed or fails:
        outcome = "environment INVALID: " + "; ".join(fails or [f"{c}: crashed" for c in crashed])
    elif missing:
        outcome = "pending: " + ", ".join(missing)
    else:
        outcome = "environment PASS"
    return {"outcome": outcome, "rule": {"aggregate_req_s": AGG, **MACHINE}, "runs": runs}


def f(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def tables(res: dict) -> str:
    out = ["# Environment qualification: production A, product schedule (laya, L128 / L512)", ""]
    out += [f"Outcome: {res['outcome']}", ""]
    out += ["## Runs", "", "| run | mismatches | routing | crashed | workers alive | failed | machine findings |"]
    out += ["|---|---|---|---|---|---|---|"]
    for r in res["runs"]:
        if not r["present"]:
            out.append(f"| {r['run']} | – | – | {r['crashed']} | – | pending | – |")
            continue
        out.append(
            f"| {r['run']} | {r['mismatches']} | {len(r['routing_failures'])} | {r['crashed']} | "
            f"{r['workers_alive_at_end']} | {', '.join(r['failed']) or '–'} | {'; '.join(r['machine_findings']) or '–'} |"
        )
    out += ["", "## Hetero windows", ""]
    out += ["| run | window | n short | median | P95 | P99 | P99.9 | agg req/s | GPU return P50 / P95 / P99 | "
            "host-slow bins (longest) | slow spans (longest) |"]  # fmt: skip
    out += ["|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["runs"]:
        for w in r.get("windows") or []:
            s, g = w["short_ms"], w["gpu_return_ms"]
            out.append(
                f"| {r['run']} | {w['index']} | {w['n_short']} | {f(s['median'])} | {f(s['p95'])} | {f(s['p99'])} | "
                f"{f(s['p999'])} | {f(w['aggregate_req_s'], 1)} | {f(g['p50'], 3)} / {f(g['p95'], 3)} / {f(g['p99'], 3)} | "
                f"{w['host_slow_longest_bins']} | {w['slow_state_longest_spans']} |"
            )
    out += ["", "## Pooled per run", "", "| run | median | P95 | P99 | P99.9 | GPU return P50 / P95 / P99 |"]
    out += ["|---|---|---|---|---|---|"]
    for r in res["runs"]:
        if r["present"]:
            s, g = r["pooled_short_ms"], r["pooled_gpu_return_ms"]
            out.append(
                f"| {r['run']} | {f(s['median'])} | {f(s['p95'])} | {f(s['p99'])} | {f(s['p999'])} | "
                f"{f(g['p50'], 3)} / {f(g['p95'], 3)} / {f(g['p99'], 3)} |"
            )
    out += [
        "",
        "## Machine snapshots",
        "",
        "| run | when | load | CPU idle % | memory free % | swap | thermal | power | top CPU |",
    ]
    out += ["|---|---|---|---|---|---|---|---|---|"]
    for r in res["runs"]:
        snaps = r["snapshots"] if isinstance(r["snapshots"], dict) else dict(zip(("before", "after"), r["snapshots"]))
        for tag, s in snaps.items():
            if s is None:
                continue
            load = s["uptime"].split("load averages:")[-1].strip()
            top = ", ".join(f"{n} {c:.0f}%" for c, n in s["top_cpu"][:3])
            out.append(
                f"| {r['run']} | {tag} | {load} | {f((s.get('cpu_user_sys_idle_pct') or [None] * 3)[2], 1)} | "
                f"{s.get('memory_free_pct')} | {s.get('swapusage')} | "
                f"{'warning' if s.get('thermal_warning') or s.get('performance_warning') else 'none'} | "
                f"{'AC' if 'AC Power' in (s.get('power') or '') else s.get('power')} | {top} |"
            )
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=EXP)
    a = ap.parse_args()
    res = summarise(a.raw)
    js = json.dumps(res, indent=1, default=str) + "\n"
    md = tables(res)
    pj, pm = a.out / "qual_results.json", a.out / "qual_tables.md"
    if a.check:
        ok = pj.exists() and pm.exists() and pj.read_text() == js and pm.read_text() == md
        print("outputs are up to date" if ok else "outputs differ")
        sys.exit(0 if ok else 1)
    pj.write_text(js)
    pm.write_text(md)
    print(res["outcome"])


if __name__ == "__main__":
    main()
