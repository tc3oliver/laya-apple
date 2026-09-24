"""The production regression gate for ANE process isolation: results.json and tables.md.

    uv run python benchmarks/ane-process-isolation/analyze.py [--check]

Inputs: raw/<model>-{thread,process}-r{1,2}.json (run.sh). The criteria and their definitions
are in README.md and were fixed before the campaign ran; this file implements them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
MODELS = ("laya", "laya-typed-decisions")
PLACEMENTS = ("thread", "process")
ROUNDS = (1, 2)
LIMITS = {"aggregate_min_ratio": 0.95, "p99_max_ratio": 1.05, "return_p50_max_ms": 1.0, "return_min_gain": 5.0}


def hetero_bounds(run: dict) -> list[tuple[int, int]]:
    """(start_ns, end_ns) of the hetero windows: the auto instance running both streams at once."""
    auto = [w for w in run["windows_t"] if w["instance"] == "auto"]
    starts = {}
    for w in auto:
        starts.setdefault((w["start_ns"], w["end_ns"]), set()).add(w["stream"])
    return sorted(k for k, s in starts.items() if s == {"short", "long"})


def placement_summary(runs: list[dict]) -> dict:
    windows = [w for r in runs for w in r["part_a"]["windows"] if w["condition"] == "hetero"]
    per = {
        s: {
            "req_s": [w["streams"][s]["req_s"] for w in windows],
            "p99_ms": [w["streams"][s]["p99_ms"] for w in windows],
            "p50_ms": [w["streams"][s]["p50_ms"] for w in windows],
            "devices": [w["streams"][s]["devices"] for w in windows],
        }
        for s in ("short", "long")
    }
    returns = []
    for r in runs:
        bounds = hetero_bounds(r)
        g = r["gpu_return"]
        returns += [
            u / 1e3 for t, u in zip(g["received_ns"], g["return_us"], strict=True) if any(a <= t < b for a, b in bounds)
        ]
    mismatches = sum(v["mismatches"] for r in runs for w in r["part_a"]["windows"] for v in w["streams"].values())
    return {
        "windows": len(windows),
        "streams": {
            s: {
                "req_s_median": float(np.median(v["req_s"])),
                "p99_ms_median": float(np.median(v["p99_ms"])),
                "p50_ms_median": float(np.median(v["p50_ms"])),
                "req_s": v["req_s"],
                "p99_ms": v["p99_ms"],
                "devices": v["devices"],
            }
            for s, v in per.items()
        },
        "aggregate_req_s": float(sum(np.median(v["req_s"]) for v in per.values())),
        "gpu_return_ms": {
            "n": len(returns),
            "p50": float(np.percentile(returns, 50)) if returns else None,
            "p99": float(np.percentile(returns, 99)) if returns else None,
        },
        "mismatches_all_windows": int(mismatches),
        "hetero_windows_with_gpu_returns": sum(len(hetero_bounds(r)) for r in runs),
    }


def verdict(t: dict, p: dict) -> dict:
    agg = p["aggregate_req_s"] / t["aggregate_req_s"]
    short = p["streams"]["short"]["p99_ms_median"] / t["streams"]["short"]["p99_ms_median"]
    long_ = p["streams"]["long"]["p99_ms_median"] / t["streams"]["long"]["p99_ms_median"]
    rp, rt = p["gpu_return_ms"]["p50"], t["gpu_return_ms"]["p50"]
    checks = {
        "correctness": t["mismatches_all_windows"] == 0 and p["mismatches_all_windows"] == 0,
        "aggregate_throughput": agg >= LIMITS["aggregate_min_ratio"],
        "short_p99": short <= LIMITS["p99_max_ratio"],
        "long_p99": long_ <= LIMITS["p99_max_ratio"],
        "gpu_completion_isolation": rp is not None
        and rt is not None
        and rp <= LIMITS["return_p50_max_ms"]
        and rt / rp >= LIMITS["return_min_gain"],
    }
    return {
        "ratios": {
            "aggregate": agg,
            "short_p99": short,
            "long_p99": long_,
            "gpu_return_p50_gain": rt / rp if rp else None,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def summarise() -> dict:
    res = {"limits": LIMITS, "models": {}}
    for m in MODELS:
        runs = {pl: [json.loads((RAW / f"{m}-{pl}-r{i}.json").read_text()) for i in ROUNDS] for pl in PLACEMENTS}
        s = {pl: placement_summary(rs) for pl, rs in runs.items()}
        res["models"][m] = {
            "args": {k: runs["thread"][0]["args"][k] for k in ("short", "long", "seconds", "cycles")},
            "placements": s,
            "gpu_only_aggregate_req_s": {
                pl: float(np.median([r["part_a"]["gate"]["aggregate_gpu_only_req_s"] for r in rs]))
                for pl, rs in runs.items()
            },
            "verdict": verdict(s["thread"], s["process"]),
        }
    res["passed"] = all(x["verdict"]["passed"] for x in res["models"].values())
    return res


def f(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def pct(r):
    return f"{(r - 1) * 100:+.1f}%"


def tables(res: dict) -> str:
    L = ["# ANE process isolation: production regression gate (v1.0 closed-loop mix, hetero windows)\n"]
    L += [
        "| model | placement | aggregate req/s | short req/s | short P99 | long req/s | long P99 "
        "| GPU return P50 / P99 | mismatches |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        for pl, s in x["placements"].items():
            sh, lo, g = s["streams"]["short"], s["streams"]["long"], s["gpu_return_ms"]
            L.append(
                f"| {m} | {pl} | {f(s['aggregate_req_s'], 1)} | {f(sh['req_s_median'], 1)} | {f(sh['p99_ms_median'])} "
                f"| {f(lo['req_s_median'], 1)} | {f(lo['p99_ms_median'])} | {f(g['p50'], 3)} / {f(g['p99'], 3)} "
                f"| {s['mismatches_all_windows']} |"
            )
    L += [
        "\n## Criteria (process vs thread)\n",
        "| model | aggregate ≥ 0.95× | short P99 ≤ 1.05× | long P99 ≤ 1.05× | GPU return P50 ≤ 1 ms and ≥ 5× better "
        "| correctness | model |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        v, r = x["verdict"], x["verdict"]["ratios"]

        def ok(k):
            return "pass" if v["checks"][k] else "**FAIL**"

        L.append(
            f"| {m} | {pct(r['aggregate'])} {ok('aggregate_throughput')} | {pct(r['short_p99'])} {ok('short_p99')} "
            f"| {pct(r['long_p99'])} {ok('long_p99')} | ×{f(r['gpu_return_p50_gain'], 1)} {ok('gpu_completion_isolation')} "
            f"| {ok('correctness')} | {'PASS' if v['passed'] else '**FAIL**'} |"
        )
    L.append(f"\nOverall: {'PASS' if res['passed'] else '**FAIL**'}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    res = summarise()
    js = json.dumps(res, indent=1, sort_keys=True) + "\n"
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
