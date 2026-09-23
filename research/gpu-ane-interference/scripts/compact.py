"""Compact interference.py raw runs for the repository: one record per request, nothing dropped.

    uv run python research/gpu-ane-interference/scripts/compact.py FULL.json.gz ... --out-dir raw/

The full runs keep every backend phase record separately (tens of MB per run). This keeps
every request of every window, joins its backend phase record into it (device_exec, host,
cpu_ms: what analyze.py derives from the phases), rounds times to 10 us (the harness itself
cannot resolve finer: a timestamp costs ~1 us and the dispatcher's wake-up ~10-50 us), and
drops only the phase list. analyze.py reads both forms and gives the same results within
that rounding.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from analyze import Phases, rows

UNIT = 0.01  # ms: every time below is an integer number of 10-us units


def encode(recs: list, phases: Phases) -> dict:
    """Integer columns: arrival as the gap from the previous arrival, every other time as a
    duration (pre, queue, service, post), plus the backend phase split joined from `phases`."""

    def q(x):
        return None if x is None else int(round(x / UNIT))

    cols = {k: [] for k in ("d_arrival", "pre", "queue", "service", "post", "forward", "device_exec", "host",
                            "cpu", "match", "in_window", "seed")}
    extra = {k: [] for k in ("device", "shape", "reason", "estimate", "backlog", "gpu_backlog", "ane_backlog")}
    prev = 0
    for r in recs:
        a = q(r["arrival"])
        cols["d_arrival"].append(a - prev)
        prev = a
        started = "device_start" in r
        cols["pre"].append(q(r["queue_enter"] - r["arrival"]) if started else None)
        cols["queue"].append(q(r["device_start"] - r["queue_enter"]) if started else None)
        cols["service"].append(q(r["device_end"] - r["device_start"]) if started else None)
        cols["post"].append(q(r["response"] - r["device_end"]) if started else q(r["response"] - r["arrival"]))
        cols["forward"].append(q(r.get("forward_ms")))
        p = phases.within(r["device"], r["device_start"] - 0.01, r["device_end"] + 0.01) if started else None
        ex = sum(y - x for x, y in p["dev"]) if p is not None else None
        cols["device_exec"].append(q(ex))
        cols["host"].append(q((p["t1"] - p["t0"]) - ex) if p is not None else None)
        cols["cpu"].append(q(p["cpu_ms"]) if p is not None else None)
        cols["match"].append(int(bool(r.get("match"))))
        cols["in_window"].append(int(r["in_window"]))
        cols["seed"].append(r.get("seed"))
        extra["device"].append(r.get("device"))
        extra["shape"].append(r.get("shape"))
        extra["reason"].append(r.get("reason"))
        extra["estimate"].append(q(r.get("estimate_ms")))
        extra["backlog"].append(q(r.get("backlog_at_enter_ms")))
        extra["gpu_backlog"].append(q(r.get("gpu_backlog_ms")))
        extra["ane_backlog"].append(q(r.get("ane_backlog_ms")))
    for k, v in extra.items():  # constant within a device stream: store once
        cols[k] = v[0] if len(set(v)) <= 1 and v else v
    return cols


def decode(enc: dict) -> list:
    """Inverse of `encode`: records with absolute times in ms (what analyze.py expects)."""
    n = len(enc["d_arrival"])
    out, t = [], 0
    for i in range(n):
        t += enc["d_arrival"][i]
        r = {"arrival": t * UNIT, "match": bool(enc["match"][i]), "in_window": bool(enc["in_window"][i]),
             "seed": enc["seed"][i]}
        for k in ("device", "shape", "reason", "estimate", "backlog", "gpu_backlog", "ane_backlog"):
            v = enc[k][i] if isinstance(enc[k], list) else enc[k]
            name = {"estimate": "estimate_ms", "backlog": "backlog_at_enter_ms", "gpu_backlog": "gpu_backlog_ms",
                    "ane_backlog": "ane_backlog_ms"}.get(k, k)
            if v is not None:
                r[name] = v * UNIT if k in ("estimate", "backlog", "gpu_backlog", "ane_backlog") else v
        if enc["service"][i] is not None:
            r["queue_enter"] = r["arrival"] + enc["pre"][i] * UNIT
            r["device_start"] = r["queue_enter"] + enc["queue"][i] * UNIT
            r["device_end"] = r["device_start"] + enc["service"][i] * UNIT
            r["response"] = r["device_end"] + enc["post"][i] * UNIT
        else:
            r["response"] = r["arrival"] + enc["post"][i] * UNIT
        for k, name in (("forward", "forward_ms"), ("device_exec", "device_exec"), ("host", "host"), ("cpu", "cpu_ms")):
            if enc[k][i] is not None:
                r[name] = enc[k][i] * UNIT
        out.append(r)
    return out


def compact(d: dict) -> dict:
    phases = Phases(d["phases"])
    for w in d["windows"]:
        for st in w["streams"].values():
            st["encoded"] = encode(rows(st["records"]), phases)
            del st["records"]
    d["phases"] = None
    d["compact"] = {"unit_ms": UNIT, "format": "compact.py encode/decode", "phases": "joined into each request"}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for p in args.paths:
        with gzip.open(p, "rt") as f:
            d = json.load(f)
        out = args.out_dir / p.name
        with gzip.open(out, "wt", compresslevel=9) as f:
            json.dump(compact(d), f, separators=(",", ":"))
        print(f"{p.name}: {p.stat().st_size / 1e6:.1f} MB -> {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
