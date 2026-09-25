"""Post-hoc: where the hetero-onset E-core residency sits (existing #99 raw data only, not a gate).

    uv run python research/coreml-dependency-qos/scripts/placement_posthoc.py [--check]

Reads raw/laya-{B,O,Q}-r1.json.gz (the recount series: cumulative PROC_PIDTHREADCOUNTS per
thread and perf level, sampled every 100 ms) and writes placement_posthoc.{md,json}. It does not
change #99's outcome.

Per hetero transition (t0 = the window's start_ns), every sampled thread's counter deltas are
assigned to the interval midpoint and summed per bucket: baseline [-2, 0), 0-0.5, 0.5-1, 1-2,
2-4, 4-8 and late [10, 20) s. Per thread and bucket: CPU time on P and E, the E share,
instructions, cycles, IPC and the relative effective cycle rate (cycles / CPU ns; relative only).

Groups (the recount sampler reads these parent threads only; other parent threads, such as the
loader or the sampler itself, are not in the data):
  chain         laya-ane-dispatch, client-short, coreml-callback
  parent-other  MainThread, client-long, laya-gpu-dispatch
  worker        every thread of the GPU worker process (all named gpu-worker)
A thread is active in a transition if it used >= ACTIVE_MS of CPU in [t0, t0 + 4 s); only active
threads enter a group aggregate, weighted by their CPU time, so idle or short-lived threads do not
dilute it. The worker's threads are not identifiable by name; they are listed by tid and CPU.

E->P switch time, per active thread: the start of the first 0.5 s bin at or after t0 from which
every later bin up to t0 + 20 s with >= BIN_MIN_MS of CPU has E share < 0.5 (none: "never").

Classification, fixed in this script before its output was read:
  per transition, over W = [t0 + 0.5, t0 + 4): a group is E-dominant if its active-weighted E
  share >= 0.5, and P-dominant if <= 0.25.
    P3  chain, parent-other and worker all E-dominant
    P2  chain and parent-other E-dominant, worker P-dominant
    P1  chain E-dominant, parent-other P-dominant
    P4  anything else (including chain not E-dominant)
  overall: the class shared by every transition whose chain is E-dominant; otherwise P4.

Also, per window of each run (all 8 conditions): the E share of all sampled CPU of the parent and
of the auto instance's GPU worker. Window starts are the hetero start_ns +/- 22.5 s steps (20 s
windows, 2.0 s idle, 0.5 s lead). gpu_only runs on the separate GPU-only instance, whose threads
are not sampled.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RAW = EXP / "raw"
RUNS = ("laya-B-r1", "laya-O-r1", "laya-Q-r1")
S = 1_000_000_000
BUCKETS = (("baseline -2-0", -2.0, 0.0), ("0-0.5", 0.0, 0.5), ("0.5-1", 0.5, 1.0), ("1-2", 1.0, 2.0),
           ("2-4", 2.0, 4.0), ("4-8", 4.0, 8.0), ("late 10-20", 10.0, 20.0))  # fmt: skip
W = (0.5, 4.0)
ACTIVE_MS = 20.0
BIN_S, BIN_MIN_MS = 0.5, 2.0
E_DOM, P_DOM = 0.5, 0.25
GROUPS = {
    "chain": ("laya-ane-dispatch", "client-short", "coreml-callback"),
    "parent-other": ("MainThread", "client-long", "laya-gpu-dispatch"),
}


def group_of(name: str, pid: int, parent: int) -> str:
    if pid != parent:
        return "worker"
    for g, names in GROUPS.items():
        if name in names:
            return g
    return "parent-unlisted"


def intervals(rc: dict, series: dict):
    """(mid_ns, P(instr, cyc, cpu), E(instr, cyc, cpu)) deltas between a thread's consecutive rows."""
    t = rc["t_ns"]
    rows = series["rows"]
    for a, b in zip(rows, rows[1:]):
        d = [y - x for x, y in zip(a[1:], b[1:])]
        yield (t[a[0]] + t[b[0]]) // 2, (d[0], d[1], d[2]), (d[4], d[5], d[6])


def acc() -> dict:
    return {"p_instr": 0, "p_cyc": 0, "p_cpu": 0, "e_instr": 0, "e_cyc": 0, "e_cpu": 0}


def add(x: dict, p, e) -> None:
    x["p_instr"] += p[0]
    x["p_cyc"] += p[1]
    x["p_cpu"] += p[2]
    x["e_instr"] += e[0]
    x["e_cyc"] += e[1]
    x["e_cpu"] += e[2]


def derive(x: dict) -> dict:
    cpu = x["p_cpu"] + x["e_cpu"]
    cyc = x["p_cyc"] + x["e_cyc"]
    return {
        "cpu_ms": cpu / 1e6,
        "e_share": x["e_cpu"] / cpu if cpu else None,
        "p_share": x["p_cpu"] / cpu if cpu else None,
        "instructions": x["p_instr"] + x["e_instr"],
        "cycles": cyc,
        "ipc": (x["p_instr"] + x["e_instr"]) / cyc if cyc else None,
        "rate_p": x["p_cyc"] / x["p_cpu"] if x["p_cpu"] else None,
        "rate_e": x["e_cyc"] / x["e_cpu"] if x["e_cpu"] else None,
    }


def switch_time(bins: list[dict]) -> float | None | str:
    """Start (s from t0) of the first bin from which every later bin with CPU is P-dominant."""
    live = [(lo, b) for lo, b in bins if b["cpu_ms"] >= BIN_MIN_MS]
    if not live:
        return None
    for i, (lo, _) in enumerate(live):
        if all(b["e_share"] < 0.5 for _, b in live[i:]):
            return lo
    return "never"


def transition(rc: dict, parent: int, t0: int) -> dict:
    threads = {}
    for key, s in rc["series"].items():
        g = group_of(s["name"], s["pid"], parent)
        b = {name: acc() for name, _, _ in BUCKETS}
        w = acc()
        onset = acc()
        nb = int(round((20.0 + 2.0) / BIN_S))
        bins = [acc() for _ in range(nb)]
        for mid, p, e in intervals(rc, s):
            x = (mid - t0) / S
            for name, lo, hi in BUCKETS:
                if lo <= x < hi:
                    add(b[name], p, e)
            if W[0] <= x < W[1]:
                add(w, p, e)
            if 0.0 <= x < 4.0:
                add(onset, p, e)
            k = int((x + 2.0) // BIN_S)
            if 0 <= k < nb:
                add(bins[k], p, e)
        on = derive(onset)
        threads[key] = {
            "name": s["name"],
            "pid": s["pid"],
            "tid": s["tid"],
            "group": g,
            "active": on["cpu_ms"] >= ACTIVE_MS,
            "onset": on,
            "W": derive(w),
            "_W": w,
            "buckets": {n: derive(v) for n, v in b.items()},
            "_buckets": b,
            "switch_s": switch_time(
                [(-2.0 + i * BIN_S, derive(v)) for i, v in enumerate(bins) if -2.0 + i * BIN_S >= 0]
            ),
        }
    groups = {}
    for g in ("chain", "parent-other", "worker"):
        members = [t for t in threads.values() if t["group"] == g and t["active"]]
        tot = acc()
        for t in members:
            add(
                tot,
                (t["_W"]["p_instr"], t["_W"]["p_cyc"], t["_W"]["p_cpu"]),
                (t["_W"]["e_instr"], t["_W"]["e_cyc"], t["_W"]["e_cpu"]),
            )
        bk = {}
        for n, _, _ in BUCKETS:
            x = acc()
            for t in members:
                y = t["_buckets"][n]
                add(x, (y["p_instr"], y["p_cyc"], y["p_cpu"]), (y["e_instr"], y["e_cyc"], y["e_cpu"]))
            bk[n] = derive(x)
        groups[g] = {"active_threads": [f"{t['name']}:{t['tid']}" for t in members], "W": derive(tot), "buckets": bk}
    for t in threads.values():
        del t["_W"], t["_buckets"]

    def dom(g):
        e = groups[g]["W"]["e_share"]
        if e is None:
            return None
        return "E" if e >= E_DOM else "P" if e <= P_DOM else "mixed"

    d = {g: dom(g) for g in groups}
    if d["chain"] != "E":
        cls = "P4"
    elif d["parent-other"] == "E" and d["worker"] == "E":
        cls = "P3"
    elif d["parent-other"] == "E" and d["worker"] == "P":
        cls = "P2"
    elif d["parent-other"] == "P":
        cls = "P1"
    else:
        cls = "P4"
    return {"threads": threads, "groups": groups, "dominance_W": d, "class": cls}


def window_shares(r: dict, parent: int) -> list[dict]:
    rc = r["recount"]
    t = rc["t_ns"]
    order = r["window_order"]
    hetero = [i for i, (_, n) in enumerate(order) if n == "hetero"]
    starts: dict[int, int] = {}
    for k, i in enumerate(hetero):
        for j in range(len(order)):
            starts.setdefault(j, r["windows"][k]["start_ns"] + int((j - i) * 22.5 * S))
    out = []
    for j, (cyc, name) in enumerate(order):
        lo, hi = starts[j], starts[j] + 20 * S
        agg = {"parent": [0, 0], "worker": [0, 0]}
        for s in rc["series"].values():
            g = "parent" if s["pid"] == parent else "worker"
            for a, b in zip(s["rows"], s["rows"][1:]):
                if lo <= (t[a[0]] + t[b[0]]) // 2 < hi:
                    agg[g][0] += b[3] - a[3]
                    agg[g][1] += b[7] - a[7]
        row = {"cycle": cyc, "condition": name}
        for g, (pc, ec) in agg.items():
            row[g] = {"cpu_s": (pc + ec) / S, "e_share": ec / (pc + ec) if pc + ec else None}
        out.append(row)
    return out


def analyse(raw: Path = RAW) -> dict:
    out = {"runs": {}, "transitions": []}
    for run in RUNS:
        r = json.load(gzip.open(raw / f"{run}.json.gz"))["research"]
        rc = r["recount"]
        parent = next(s["pid"] for s in rc["series"].values() if s["name"] == "laya-ane-dispatch")
        worker = r["workers"]["gpu"]["pid"]
        out["runs"][run] = {"parent_pid": parent, "worker_pid": worker, "windows": window_shares(r, parent)}
        for cyc, w in enumerate(r["windows"]):
            tr = transition(rc, parent, w["start_ns"])
            tr.update(run=run, cycle=cyc, after="solo_long" if cyc == 0 else "gpu_only")
            out["transitions"].append(tr)
    chained = [t["class"] for t in out["transitions"] if t["dominance_W"]["chain"] == "E"]
    out["overall"] = chained[0] if chained and all(c == chained[0] for c in chained) else "P4"
    out["classes"] = [(t["run"], t["cycle"], t["class"]) for t in out["transitions"]]
    return out


def f(x, n=2):
    return "–" if x is None else x if isinstance(x, str) else f"{x:.{n}f}"


def render(res: dict) -> str:
    L = [
        "# Post-hoc: where the hetero-onset E-core residency sits (#99 raw data only)\n",
        "Not a gate; #99's outcome is unchanged. Definitions and the classification rule are in "
        "`scripts/placement_posthoc.py`, fixed before its output was read.\n",
        f"Overall class: **{res['overall']}**. Per transition: "
        + ", ".join(f"{r.replace('laya-', '')} c{c} {k}" for r, c, k in res["classes"])
        + "\n",
        "## Group aggregates over W = [t0 + 0.5, t0 + 4) (active threads, CPU-weighted)\n",
        "| run | cycle | after | group | active threads | CPU ms | E share | IPC | rate P / E | dominance |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in res["transitions"]:
        for g, x in t["groups"].items():
            w = x["W"]
            L.append(
                f"| {t['run']} | {t['cycle']} | {t['after']} | {g} | {len(x['active_threads'])} | {f(w['cpu_ms'], 1)} "
                f"| {f(w['e_share'])} | {f(w['ipc'])} | {f(w['rate_p'])} / {f(w['rate_e'])} | {t['dominance_W'][g]} |"
            )
    L += [
        "\n## Group E share by bucket (active threads, CPU-weighted)\n",
        "| run | cycle | group | " + " | ".join(n for n, _, _ in BUCKETS) + " |",
        "|---|---|---|" + "---|" * len(BUCKETS),
    ]
    for t in res["transitions"]:
        for g, x in t["groups"].items():
            L.append(
                f"| {t['run']} | {t['cycle']} | {g} | "
                + " | ".join(f(x["buckets"][n]["e_share"]) for n, _, _ in BUCKETS)
                + " |"
            )
    L += [
        "\n## Every window: E share of all sampled CPU, parent and GPU worker\n",
        "gpu_only runs on the separate GPU-only instance, whose threads are not sampled.\n",
        "| run | cycle | condition | parent E share (CPU s) | worker E share (CPU s) |",
        "|---|---|---|---|---|",
    ]
    for run, x in res["runs"].items():
        for w in x["windows"]:
            L.append(
                f"| {run} | {w['cycle']} | {w['condition']} | {f(w['parent']['e_share'])} ({f(w['parent']['cpu_s'], 1)}) "
                f"| {f(w['worker']['e_share'])} ({f(w['worker']['cpu_s'], 1)}) |"
            )
    L += [
        "\n## Per active thread: E share over W, and the E->P switch time\n",
        "Switch: start (s from t0) of the first 0.5 s bin from which every later bin with CPU is P-dominant; "
        "`never` = still E-dominant somewhere up to t0 + 20 s.\n",
        "| run | cycle | group | thread | onset CPU ms | E share W | IPC W | rate P / E W | switch s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t in res["transitions"]:
        act = sorted((x for x in t["threads"].values() if x["active"]), key=lambda x: (x["group"], x["name"], x["tid"]))
        for x in act:
            w = x["W"]
            L.append(
                f"| {t['run']} | {t['cycle']} | {x['group']} | {x['name']}:{x['tid']} | {f(x['onset']['cpu_ms'], 1)} "
                f"| {f(w['e_share'])} | {f(w['ipc'])} | {f(w['rate_p'])} / {f(w['rate_e'])} | {f(x['switch_s'], 1)} |"
            )
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="verify the outputs are up to date")
    a = ap.parse_args()
    res = analyse()
    md, js = render(res), json.dumps(res, indent=1, sort_keys=True) + "\n"
    pm, pj = EXP / "placement_posthoc.md", EXP / "placement_posthoc.json"
    if a.check:
        stale = [p.name for p, c in ((pm, md), (pj, js)) if not p.exists() or p.read_text() != c]
        if stale:
            raise SystemExit(f"stale: {stale}")
        print("outputs are up to date")
        return
    pm.write_text(md)
    pj.write_text(js)
    print(f"overall: {res['overall']}")


if __name__ == "__main__":
    main()
