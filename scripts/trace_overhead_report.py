"""Summarise scripts/bench_trace_overhead.py windows: each condition against the base.

    uv run python scripts/trace_overhead_report.py benchmarks/tracing/overhead.jsonl
    uv run python scripts/trace_overhead_report.py --micro benchmarks/tracing/micro.jsonl

Per (label, condition): the median over all windows of each metric, and the window-to-window
spread (min-max of the per-round medians). The change against the base (--base, default
main/none) is the ratio of medians; the noise band is the spread of the base's own
per-round medians relative to its median.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = (
    ("req_s", "throughput (req/s)"),
    ("p50_ms", "e2e P50 (ms)"),
    ("p95_ms", "e2e P95 (ms)"),
    ("p99_ms", "e2e P99 (ms)"),
    ("cpu_s_per_s", "CPU (cores busy)"),
)


def summarise(lines):
    by = defaultdict(lambda: defaultdict(list))
    for x in lines:
        by[(x["label"], x["condition"])][x["round"]].append(x)
    out = {}
    for key, rounds in by.items():
        ws = [w for r in rounds.values() for w in r]
        s = {"windows": len(ws), "rounds": len(rounds), "requests": sum(w["requests"] for w in ws)}
        for m, _ in METRICS:
            per_round = [float(np.median([w[m] for w in r])) for r in rounds.values()]
            s[m] = float(np.median([w[m] for w in ws]))
            s[m + "_round_min"], s[m + "_round_max"] = min(per_round), max(per_round)
        out[key] = s
    return out


def micro(lines, base_key):
    """scripts/bench_trace_micro.py blocks: median µs per request, min-max over blocks."""
    by = defaultdict(list)
    for x in lines:
        by[(x["label"], x["condition"])].append(x["us_per_request"])
    base = float(np.median(by[base_key]))
    print("| Code | trace | blocks | requests | µs per request, median | min-max | vs base |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for k in sorted(by, key=lambda k: (k != base_key, k[0], ["none", "noop", "recorder"].index(k[1]))):
        v = by[k]
        med = float(np.median(v))
        n = sum(x["requests"] for x in lines if (x["label"], x["condition"]) == k)
        delta = "" if k == base_key else f"{med - base:+.2f} µs ({med / base - 1:+.2%})"
        print(f"| {k[0]} | {k[1]} | {len(v)} | {n} | {med:.2f} | {min(v):.2f}-{max(v):.2f} | {delta} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--base", default="main/none")
    ap.add_argument("--micro", action="store_true", help="the input is bench_trace_micro.py output")
    args = ap.parse_args()
    lines = [json.loads(x) for x in args.path.read_text().splitlines() if x.strip()]
    if args.micro:
        micro(lines, tuple(args.base.split("/")))
        return
    s = summarise(lines)
    base = s[tuple(args.base.split("/"))]
    order = sorted(s, key=lambda k: (k != tuple(args.base.split("/")), k[0], ["none", "noop", "recorder"].index(k[1])))
    print("| Code | trace | windows | requests | " + " | ".join(label for _, label in METRICS) + " |")
    print("|---|---|---:|---:|" + "---:|" * len(METRICS))
    for k in order:
        v = s[k]
        cells = []
        for m, _ in METRICS:
            delta = v[m] / base[m] - 1
            cells.append(f"{v[m]:.2f} ({delta:+.2%})" if k != tuple(args.base.split("/")) else f"{v[m]:.2f}")
        print(f"| {k[0]} | {k[1]} | {v['windows']} | {v['requests']} | " + " | ".join(cells) + " |")
    print()
    print("Noise band of the base (per-round medians, min to max, relative to its median):")
    for m, label in METRICS:
        lo, hi = base[m + "_round_min"] / base[m] - 1, base[m + "_round_max"] / base[m] - 1
        print(f"- {label}: {lo:+.2%} to {hi:+.2%}")


if __name__ == "__main__":
    main()
