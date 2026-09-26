"""Phase 1, the recovery experiment: phase1_results.json and phase1_tables.md from raw/ (../phase1.md).

    uv run python research/coreml-adaptive-breaker/scripts/phase1_analyze.py [--check]
    uv run python research/coreml-adaptive-breaker/scripts/phase1_analyze.py runs     # "cell rep" lines, in order
    uv run python research/coreml-adaptive-breaker/scripts/phase1_analyze.py reruns   # machine re-runs ("cell rep-b")

Inputs:
- raw/laya-<cell>-r<rep>[-b].json.gz, written by run_config.py;
- raw/<run>.before.json and raw/<run>.after.json, written by machine_snapshot.py;
- raw/failed/<run>.<attempt>.log, from run_phase1.sh's crash rule.

Every threshold below is phase1.md's, and none changes once the first run has started.

The two signal families stay separate:
- **The detector (C3)** reads `prepare_ms` only.
- **The ground truth, onset and recovery** read user-visible latency (e2e = response − submit of
  the short requests) and completion cadence.
- **Prepare** enters recovery only as a necessary extra condition: at most 2 host-slow requests in
  a window. It can make a recovery later, never earlier. It is also reported as a diagnostic.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = HERE.parents[2]
RAW = ROOT / "raw"
OUT_JSON = ROOT / "phase1_results.json"
OUT_MD = ROOT / "phase1_tables.md"
S = 1_000_000_000

# ------------------------------------------------------------------ frozen by phase1.md
RUNS = (
    ("R", 1), ("A", 1), ("R", 2), ("B", 1),
    ("R", 3), ("B", 2), ("R", 4), ("A", 2),
    ("R", 5), ("A", 3), ("R", 6), ("B", 3),
)  # fmt: skip
HOST_SLOW_MS = 0.3  # C3's threshold, also the host-slow diagnostic
C3_K = 3
ARM_S = 1.0  # C3 counts from 1.0 s after the episode start (breaker.py)
A_SKIP_S = 1.0  # A: every A measure starts at t0 + 1 s
SPAN_S, SLOW_RATIO, STEP_S = 1.0, 1.2, 0.1  # Phase 0's ground truth (#103's H6)
BIN_S, BIN_SHARE = 0.5, 0.10  # #103's H5
W = 25  # recovery: requests per rolling window
HOLD_S = 1.0  # recovery: every window starting within HOLD_S of the recovery point passes
ENV_Q = 99.9  # recovery: the A envelope is this percentile of the Phase 1 A windows
HOST_SLOW_MAX = 2  # recovery: at most this many host-slow requests in a window
RECOVERY_MS = 1000.0  # primary gate: trip -> sustained A-like recovery
ONSET_TRIP_MS = 250.0  # C3's target: onset -> trip
THROUGHPUT_RATIO = 0.95
MIN_B_SUSTAINED = 3  # of 6 B episodes, else INCONCLUSIVE (phenotype not reproduced)
MIN_R_TRIPPED, MIN_R_CONFIRMED = 6, 4  # else INCONCLUSIVE (not enough natural slow state)
A_MAX_BAD = 1  # of 6 A episodes: Phase 0 sustained, or no A-like point within 1 s of t0 + 1 s
ROUTES = {
    "solo_short": {"short": "ane"},
    "solo_long": {"long": "gpu"},
    "hetero": {"short": "ane", "long": "gpu"},
    "gpu_only": {"short": "gpu", "long": "gpu"},
}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


qual = _load("staged_handoff_qual_analyze", REPO / "research" / "coreml-staged-handoff" / "scripts" / "qual_analyze.py")


def pinned_tree() -> str | None:
    """phase1.md's pinned laya_apple tree (a line `laya_apple tree: <sha>`)."""
    p = ROOT / "phase1.md"
    m = re.search(r"laya_apple`? tree: `?([0-9a-f]{40})", p.read_text()) if p.exists() else None
    return m.group(1) if m else None


def allowed(x: float) -> float:
    return max(1.05 * x, x + 1.0)


def run_name(cell: str, rep, suffix: str = "") -> str:
    return f"laya-{cell}-r{rep}{suffix}"


# ------------------------------------------------------------------ per-window data


def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


def requests(run: dict, lo: int, end: int) -> dict:
    tr = run["_trace"]
    m = (tr["target"] == "ane") & (tr["submit_ns"] >= lo) & (tr["submit_ns"] < end)
    o = np.argsort(tr["submit_ns"][m], kind="stable")
    sub = tr["submit_ns"][m][o]
    return {
        "submit": sub,
        "id": tr["request_id"][m][o],
        "prep": (tr["prepared_ns"][m][o] - sub) / 1e6,
        "e2e": (tr["response_ns"][m][o] - sub) / 1e6,
        "resp": tr["response_ns"][m][o],
        "start": tr["service_start_ns"][m][o],
        "end": tr["service_end_ns"][m][o],
    }


def slow_spans(rq: dict, lo: int, end: int, limit: float) -> list[bool]:
    out = []
    for s in range(lo, end - S + 1, S):
        m = (rq["submit"] >= s) & (rq["submit"] < s + S)
        out.append(bool(m.any() and np.median(rq["e2e"][m]) > limit))
    return out


def host_bins(rq: dict, lo: int, end: int) -> list[bool]:
    out, step = [], int(BIN_S * S)
    for s in range(lo, end - step + 1, step):  # full bins only
        m = (rq["submit"] >= s) & (rq["submit"] < s + step)
        out.append(bool(m.any() and (rq["prep"][m] > HOST_SLOW_MS).mean() >= BIN_SHARE))
    return out


def _run2(flags: list[bool]) -> bool:
    return any(a and b for a, b in zip(flags, flags[1:]))


def _longest(flags) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def phase0_truth(rq: dict, lo: int, end: int, m_a: float) -> dict:
    """Phase 0's ground truth, unchanged: sustained = 2 consecutive slow 1 s spans from lo."""
    limit = SLOW_RATIO * m_a
    flags = slow_spans(rq, lo, end, limit)
    out = {"slow_spans": int(sum(flags)), "longest": _longest(flags), "sustained": _run2(flags)}
    if out["sustained"]:
        i = next(k for k in range(len(flags) - 1) if flags[k] and flags[k + 1])
        onset, step = lo + i * S, int(STEP_S * S)

        def slow(a):
            m = (rq["submit"] >= a) & (rq["submit"] < a + S)
            return bool(m.any() and np.median(rq["e2e"][m]) > limit)

        while onset - step >= lo and slow(onset - step):
            onset -= step
        out["onset_ns"] = onset
    return out


def rolling(rq: dict, i0: int = 0) -> dict:
    e, s, hs = rq["e2e"][i0:], rq["submit"][i0:], (rq["prep"][i0:] > HOST_SLOW_MS).astype(int)
    n = len(e) - W + 1
    if n <= 0:
        return {
            "t": np.zeros(0, dtype=np.int64),
            "med": np.zeros(0),
            "p95": np.zeros(0),
            "span": np.zeros(0),
            "hs": np.zeros(0),
        }
    return {
        "t": s[:n],
        "med": np.array([np.median(e[k : k + W]) for k in range(n)]),
        "p95": np.array([np.percentile(e[k : k + W], 95) for k in range(n)]),
        "span": (s[W - 1 :] - s[:n]) / 1e6,
        "hs": np.convolve(hs, np.ones(W, dtype=int), "valid"),
    }


def good(r: dict, env: dict) -> np.ndarray:
    return (r["med"] <= env["med"]) & (r["p95"] <= env["p95"]) & (r["span"] <= env["span"]) & (r["hs"] <= HOST_SLOW_MAX)


def recovery_point(r: dict, env: dict) -> int | None:
    """The first window start t such that every window starting in [t, t + HOLD_S) passes, and the
    data reaches t + HOLD_S; None when there is no such point in the episode."""
    g, t = good(r, env), r["t"]
    if not len(t):
        return None
    hold = int(HOLD_S * S)
    for k in range(len(t)):
        if t[k] + hold > t[-1]:
            return None
        if not g[k]:
            continue
        j = np.searchsorted(t, t[k] + hold, side="left")
        if g[k:j].all():
            return int(t[k])
    return None


def aggregate(run: dict, lo: int, end: int) -> float | None:
    tr = run["_trace"]
    m = (tr["response_ns"] >= lo) & (tr["response_ns"] < end)
    return float(m.sum() / ((end - lo) / S)) if end > lo else None


def gpu_return(run: dict, lo: int, end: int) -> dict:
    g = run.get("gpu_return") or {}
    rec, ret = np.asarray(g.get("received_ns", [])), np.asarray(g.get("return_us", []), dtype=float)
    m = (rec >= lo) & (rec < end) if len(rec) else np.zeros(0, dtype=bool)
    x = ret[m] / 1000.0 if len(rec) else np.zeros(0)
    return {"n": int(len(x)), "p50_ms": _pct(x, 50), "p99_ms": _pct(x, 99)}


def decisions(run: dict) -> tuple[np.ndarray, np.ndarray, list]:
    rows = run["research"]["handoff"]["decisions"]
    return np.array([r[0] for r in rows], dtype=np.int64), np.array([r[1] for r in rows]), [r[2] for r in rows]


def request_of(rq: dict, t_ns: int):
    """The ANE request whose forward contains decision time t_ns (decide() runs inside the forward)."""
    m = (rq["start"] <= t_ns) & (rq["end"] >= t_ns)
    return int(rq["id"][m][0]) if m.sum() == 1 else None


def c3_offline(prep: np.ndarray) -> int | None:
    run = 0
    for i, x in enumerate(prep):
        run = run + 1 if x > HOST_SLOW_MS else 0
        if run >= C3_K:
            return i
    return None


# ------------------------------------------------------------------ runs


def load(path: Path):
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt") as fh:
                return json.load(fh)
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def prepare(run: dict) -> dict:
    tr = run["research"]["trace"]
    run["_trace"] = {k: np.asarray(v) for k, v in tr.items()}
    run["_windows"] = [(w["start_ns"], w["end_ns"]) for w in run["research"]["windows"] if w["instance"] == "auto"]
    # every decision -> the ANE request whose forward contains it (-1: none recorded, outside the
    # hetero windows' trace)
    t = run["_trace"]
    ane = t["target"] == "ane"
    o = np.argsort(t["service_start_ns"][ane], kind="stable")
    st, en, ids = t["service_start_ns"][ane][o], t["service_end_ns"][ane][o], t["request_id"][ane][o]
    rows = (run["research"].get("handoff") or {}).get("decisions") or []
    dt = np.array([r[0] for r in rows], dtype=np.int64)
    k = np.searchsorted(st, dt, side="right") - 1
    ok = (k >= 0) & (en[np.maximum(k, 0)] >= dt) if len(st) else np.zeros(len(dt), dtype=bool)
    run["_dec_req"] = np.where(ok, ids[np.maximum(k, 0)] if len(st) else -1, -1)
    return run


def correctness(run: dict) -> dict:
    mism, routing, loss = 0, [], []
    for w in run["part_a"]["windows"]:
        for s, st in w["streams"].items():
            mism += st["mismatches"]
            want = ROUTES[w["condition"]].get(s)
            if want is None or set(st["devices"]) != {want}:
                routing.append(f"c{w['cycle']} {w['condition']} {s}: {st['devices']}")
    het = [w for w in run["part_a"]["windows"] if w["condition"] == "hetero"]
    tr = run["_trace"]
    for w, (t0, end) in zip(het, run["_windows"]):
        for s, target in (("short", "ane"), ("long", "gpu")):
            n = int(((tr["target"] == target) & (tr["submit_ns"] >= t0) & (tr["submit_ns"] < end)).sum())
            if n != w["streams"][s]["n"]:
                loss.append(f"c{w['cycle']} {s}: client {w['streams'][s]['n']} vs trace {n}")
    if len(het) != len(run["_windows"]) or len(run["part_a"]["windows"]) != 8:
        loss.append("window count")
    return {"mismatches": mism, "routing_failures": routing, "request_loss": loss}


def a_reference(a_runs: list[dict]) -> dict:
    e2e, rolls = [], []
    for run in a_runs:
        for t0, end in run["_windows"]:
            rq = requests(run, t0 + int(A_SKIP_S * S), end)
            e2e.append(rq["e2e"])
            rolls.append(rolling(rq))
    if not e2e:
        return {}
    cat = lambda k: np.concatenate([r[k] for r in rolls])  # noqa: E731
    return {
        "m_a_ms": float(np.median(np.concatenate(e2e))),
        "envelope": {"med": _pct(cat("med"), ENV_Q), "p95": _pct(cat("p95"), ENV_Q), "span": _pct(cat("span"), ENV_Q)},
    }


def episode(run: dict, name: str, k: int, ref: dict) -> dict:
    cell = run["research"]["cell"]
    t0, end = run["_windows"][k]
    lo = t0 + int(A_SKIP_S * S) if cell == "A" else t0
    rq = requests(run, lo, end)
    m_a, env = ref["m_a_ms"], ref["envelope"]
    e = {
        "run": name,
        "window": k,
        "cell": cell,
        "n": int(len(rq["e2e"])),
        "median_ms": _pct(rq["e2e"], 50),
        "p95_ms": _pct(rq["e2e"], 95),
        "p99_ms": _pct(rq["e2e"], 99),
        "host_slow_share": float((rq["prep"] > HOST_SLOW_MS).mean()) if len(rq["prep"]) else None,
        "aggregate_req_s": aggregate(run, lo, end),
        "_count": int(((run["_trace"]["response_ns"] >= lo) & (run["_trace"]["response_ns"] < end)).sum()),
        "_seconds": (end - lo) / S,
        "gpu_return": gpu_return(run, lo + (0 if cell == "A" else S), end),
        "truth": phase0_truth(rq, lo, end, m_a),
    }
    tr = e["truth"]
    if "onset_ns" in tr:
        e["onset_s"] = (tr.pop("onset_ns") - t0) / 1e9
    if cell == "A":
        r = rolling(rq)
        p = recovery_point(r, env)
        e["a_like_from_s"] = None if p is None else (p - lo) / 1e9  # A's own check, from t0 + 1 s
        return e
    if cell == "B":  # C3 as R would apply it: from requests completing >= t0 + ARM_S
        j0 = int(np.searchsorted(rq["resp"], t0 + int(ARM_S * S))) if len(rq["resp"]) else 0
        i = c3_offline(rq["prep"][j0:])
        e["c3_offline"] = None
        if i is not None:
            i += j0
            e["c3_offline"] = {"request_id": int(rq["id"][i]), "t_s": (int(rq["resp"][i]) - t0) / 1e9}
        return e
    e.update(r_episode(run, rq, t0, end, m_a, env))
    return e


def r_episode(run, rq, t0, end, m_a, env) -> dict:
    """Cell R: structure, detector consistency, timing and recovery for one hetero window. Trips,
    observations and decisions belong to the window by their request (submitted in [t0, end)),
    not by their own timestamps."""
    b = run["research"]["breaker"]
    ids = set(rq["id"].tolist())
    rows = run["research"]["handoff"]["decisions"]
    dmask = np.isin(run["_dec_req"], list(ids))
    wd = np.array([r[0] for r, m in zip(rows, dmask) if m], dtype=np.int64)
    wp = np.array([r[1] for r, m in zip(rows, dmask) if m])
    ws = [r[2] for r, m in zip(rows, dmask) if m]
    wreq = run["_dec_req"][dmask]
    trips = [x for x in b["trips"] if x[2] in ids]
    obs = [o for o in b["observations"] if o[1] in ids]
    out: dict = {"trips": len(trips), "structure": [], "rearmed_in_window": 0}
    bad = out["structure"]
    if len(trips) > 1:
        bad.append(f"{len(trips)} trips in one episode")
    # every ANE request of the window observed exactly once, with the trace's own prepare value
    seen = [o[1] for o in obs]
    if sorted(seen) != sorted(ids):
        bad.append(f"observations cover {len(set(seen))} of {len(ids)} requests ({len(seen)} rows)")
    prep_of = dict(zip(rq["id"].tolist(), rq["prep"].tolist()))
    if any(abs(prep_of.get(o[1], -1.0) - o[2]) > 1e-3 for o in obs):
        bad.append("an observation's prepare differs from the trace")
    # the online count against its own rule: only armed HEALTHY observations move it
    count = 0
    for _t, rid, prep, before, after in obs:
        if before in ("idle", "arming"):
            count = 0
        elif before == "healthy":
            count = count + 1 if prep > HOST_SLOW_MS else 0
        if after != count:
            bad.append(f"C3 count {after} != {count} at request {rid} ({before})")
            break
    # episodes started inside the window (transitions into HEALTHY)
    starts = [k for k in range(len(ws)) if ws[k] == "healthy" and (k == 0 or ws[k - 1] != "healthy")]
    out["rearmed_in_window"] = max(0, len(starts) - 1)
    ep_start = int(wd[starts[0]]) if starts else None
    out["episode_start_s"] = None if ep_start is None else (ep_start - t0) / 1e9
    arm = None if ep_start is None else ep_start + int(ARM_S * S)
    # offline C3 on the trace, from the first armed HEALTHY observation
    healthy = [o[1] for o in obs if o[3] == "healthy"]
    off = None
    if healthy:
        j0 = int(np.where(rq["id"] == healthy[0])[0][0])
        i = c3_offline(rq["prep"][j0:])
        off = None if i is None else int(rq["id"][j0 + i])
    online = int(trips[0][2]) if trips else None
    if off != online:
        bad.append(f"online trip {online} != offline C3 {off}")
    if not trips:
        if (wp == 0).any():
            bad.append("sync decision without a trip")
        return out
    trip_ns = int(trips[0][0])
    trip_i = int(np.where(rq["id"] == online)[0][0])
    after = wreq > online  # requests after the tripping one choose their path after the trip
    if not after.any():
        out["no_request_after_trip"] = True  # the trip came on the window's last request
        return out
    if (wp[after] != 0).any() or any(s != "open" for s, a in zip(ws, after) if a):
        bad.append("async decision after the trip (retry)")
    if (wp[~after] != 1).any():
        bad.append("sync decision before the trip")
    first_a_ns = int(wd[after][0])
    if first_a_ns < trip_ns:
        bad.append("first sync decision precedes the trip")
    ida = int(wreq[after][0])
    idx_a = int(np.where(rq["id"] == ida)[0][0])
    last_async = int(wreq[~after][-1]) if (~after).any() else None
    sub_a = int(rq["submit"][idx_a])
    out.update(
        {
            "trip_s": (trip_ns - t0) / 1e9,
            "trip_request_id": online,
            "breaker_open_s": (trip_ns - t0) / 1e9,  # the same instant: the trip opens the breaker under one lock
            "last_async_request_id": last_async,
            "first_a_request_id": ida,
            "async_forwards_after_trip": int((wp[after] == 1).sum()),
            "trip_to_first_a_ms": (first_a_ns - trip_ns) / 1e6,
            "first_bad_prepare_s": (int(rq["submit"][trip_i - C3_K + 1]) - t0) / 1e9,
        }
    )
    # user-visible onset, searched from the arm time (the detector is blind before it by design):
    # the first rolling window starting at or after the arm time and at or before the trip whose
    # median or P95 is outside the A-like envelope; onset = the submit of that window's first request
    # above the envelope P95. Prepare is not read.
    limit = SLOW_RATIO * m_a
    r0 = rolling(rq)
    lat_bad = (r0["med"] > env["med"]) | (r0["p95"] > env["p95"])
    cand = np.where(lat_bad & (r0["t"] >= (arm or t0)) & (r0["t"] <= trip_ns))[0]
    onset = None
    if len(cand):
        k0 = int(np.searchsorted(rq["submit"], r0["t"][cand[0]]))
        above = np.where(rq["e2e"][k0 : k0 + W] > env["p95"])[0]
        onset = int(rq["submit"][k0 + (above[0] if len(above) else 0)])
    out["onset_s"] = None if onset is None else (onset - t0) / 1e9
    out["onset_to_trip_ms"] = None if onset is None else (trip_ns - onset) / 1e6
    # confirmed slow: the short requests of the second before the trip, up to the first A request
    m = (rq["submit"] >= max(t0, trip_ns - S)) & (rq["submit"] < sub_a)
    pre = rq["e2e"][m]
    out["pre_trip_median_ms"] = _pct(pre, 50)
    out["confirmed_slow_at_trip"] = bool(len(pre) and np.median(pre) > limit)
    out["suspected_false_trip"] = onset is None and not out["confirmed_slow_at_trip"]
    # recovery, from the first A request
    p = recovery_point(rolling(rq, idx_a), env)
    out["recovered_at_s"] = None if p is None else (p - t0) / 1e9
    out["first_a_to_recovery_ms"] = None if p is None else (p - sub_a) / 1e6
    out["trip_to_recovery_ms"] = None if p is None else (p - trip_ns) / 1e6
    # after the fallback settles: [first A + 1 s, end)
    post = sub_a + S
    spans = slow_spans(rq, post, end, limit)
    bins = host_bins(rq, post, end)
    tail = rq["submit"] >= post
    out["post"] = {
        "slow_spans": int(sum(spans)),
        "persistent_latency": _run2(spans),
        "persistent_host_slow": _run2(bins),
        "p99_ms": _pct(rq["e2e"][tail], 99),
        "median_ms": _pct(rq["e2e"][tail], 50),
        "aggregate_req_s": aggregate(run, post, end),
        "gpu_return": gpu_return(run, post, end),
        "_count": int(((run["_trace"]["response_ns"] >= post) & (run["_trace"]["response_ns"] < end)).sum()),
        "_seconds": (end - post) / S,
    }
    return out


def run_record(raw: Path, cell: str, rep, suffix: str, logs: list[str]) -> dict:
    name = run_name(cell, rep, suffix)
    rec = {"run": name, "cell": cell, "present": (raw / f"{name}.json.gz").exists()}
    rec["crashes"] = sum(bool(re.fullmatch(re.escape(name) + r"\.\d+\.log", x)) for x in logs)
    before, after = load(raw / f"{name}.before.json"), load(raw / f"{name}.after.json")
    rec["machine"] = qual.snapshot_checks(before, after) if rec["present"] else []
    return rec


def slots(raw: Path) -> list[dict]:
    logs = [p.name for p in (raw / "failed").glob("*.log")] if (raw / "failed").exists() else []
    out = []
    for cell, rep in RUNS:
        s = run_record(raw, cell, rep, "", logs)
        b = run_record(raw, cell, rep, "-b", logs)
        s["rerun"] = b
        s["machine_invalid"] = False
        if s["present"] and s["machine"]:
            s["used"] = b["run"] if b["present"] and not b["machine"] else None
            s["machine_invalid"] = b["present"] and bool(b["machine"])
        else:
            s["used"] = s["run"] if s["present"] else None
        # run_phase1.sh stops on any R crash and on a second crash of an A / B run (or its -b re-run)
        s["stops"] = bool((cell == "R" and (s["crashes"] or b["crashes"])) or s["crashes"] >= 2 or b["crashes"] >= 2)
        out.append(s)
    return out


def summarise(raw: Path = RAW) -> dict:
    tree = pinned_tree()
    sl = slots(raw)
    stopped = any(s["stops"] for s in sl)
    pending = [] if stopped else [s["run"] for s in sl if s["used"] is None and not s["machine_invalid"]]
    runs = {}
    for s in sl:
        if s["used"]:
            r = load(raw / f"{s['used']}.json.gz")
            runs[s["used"]] = prepare(r) if r else None
    hard, fails, invalid, inconclusive = [], [], [], []  # hard: correctness / structure, never excused
    per_run = {}
    for s in sl:
        cell = s["cell"]
        for n, c in ((s["run"], s["crashes"]), (s["rerun"]["run"], s["rerun"]["crashes"])):
            if c and cell == "R":
                hard.append(f"{n} crashed")
            elif c >= 2:
                invalid.append(f"{n} crashed twice")
        if s["machine_invalid"]:
            invalid.append(f"{s['run']}: machine failed twice")
        name = s["used"]
        if not name:
            continue
        run = runs[name]
        if run is None or run.get("research", {}).get("experiment") != "coreml-adaptive-breaker":
            invalid.append(f"{name}: unreadable or not a Phase 1 record")
            runs[name] = None
            continue
        rt = run.get("runtime") or {}
        if rt.get("laya_apple_tree") != tree or rt.get("dirty") is not False:
            invalid.append(f"{name}: runtime {rt.get('laya_apple_tree')} dirty={rt.get('dirty')}")
        if run["research"]["cell"] != cell or run["args"].get("cycles") != 2 or run["args"].get("seconds") != 20.0:
            invalid.append(f"{name}: protocol deviation")
        c = correctness(run)
        per_run[name] = {"cell": cell, **c}
        if c["mismatches"] or c["routing_failures"] or c["request_loss"]:
            (hard if cell == "R" else invalid).append(f"{name}: correctness {c}")
        if cell == "R":
            br = run["research"].get("breaker") or {}
            ds = [r[2] for r in run["research"]["handoff"]["decisions"]]
            n_dec = sum(1 for k in range(len(ds)) if ds[k] == "healthy" and (k == 0 or ds[k - 1] != "healthy"))
            if not br.get("installed") or br.get("episodes") != n_dec:
                hard.append(f"{name}: breaker episodes {br.get('episodes')} != HEALTHY runs in its decisions {n_dec}")
            if run["research"]["handoff"].get("errors"):
                hard.append(f"{name}: handoff errors {run['research']['handoff']['errors']}")
            wins = run["_windows"]
            tr = run["_trace"]
            win_ids = set(
                tr["request_id"][
                    (tr["target"] == "ane")
                    & np.any([(tr["submit_ns"] >= a) & (tr["submit_ns"] < b) for a, b in wins], axis=0)
                ].tolist()
            )
            if [t for t in br.get("trips", []) if t[2] not in win_ids]:
                hard.append(f"{name}: trip outside a hetero window")
    ok_runs = [(s, runs[s["used"]]) for s in sl if s["used"] and runs.get(s["used"])]
    ref = a_reference([r for s, r in ok_runs if s["cell"] == "A"])
    eps = [episode(r, s["used"], k, ref) for s, r in ok_runs for k in range(len(r["_windows"]))] if ref else []
    A = [e for e in eps if e["cell"] == "A"]
    B = [e for e in eps if e["cell"] == "B"]
    R = [e for e in eps if e["cell"] == "R"]
    T = [e for e in R if e["trips"]]
    for e in R:
        tag = f"{e['run']} w{e['window']}"
        hard.extend(f"{tag}: {x}" for x in e["structure"])
        if e["rearmed_in_window"]:
            fails.append(f"{tag}: the breaker re-armed inside the hetero window (an ANE forward gap > 1 s)")
        if not e["trips"] and e["truth"]["sustained"]:
            fails.append(f"{tag}: sustained slow state and no trip (missed)")
        if e["trips"] and "trip_to_recovery_ms" in e:
            if e["trip_to_recovery_ms"] is None or e["trip_to_recovery_ms"] > RECOVERY_MS:
                fails.append(f"{tag}: no sustained A-like recovery within {RECOVERY_MS:.0f} ms of the trip")
            if e["post"]["persistent_latency"] or e["post"]["persistent_host_slow"]:
                fails.append(f"{tag}: the slow state persists after the fallback")
            if e["onset_to_trip_ms"] is not None and e["onset_to_trip_ms"] > ONSET_TRIP_MS:
                fails.append(f"{tag}: onset -> trip {e['onset_to_trip_ms']:.0f} ms > {ONSET_TRIP_MS:.0f} ms")
    post_ok = None
    TP = [e for e in T if "post" in e]
    if TP and A:
        r_rate = sum(e["post"]["_count"] for e in TP) / sum(e["post"]["_seconds"] for e in TP)
        a_rate = sum(e["_count"] for e in A) / sum(e["_seconds"] for e in A)
        a_p99 = float(np.median([e["p99_ms"] for e in A]))
        r_p99 = float(np.median([e["post"]["p99_ms"] for e in TP]))
        post_ok = {
            "throughput_ratio": r_rate / a_rate,
            "r_post_req_s": r_rate,
            "a_req_s": a_rate,
            "a_p99_ms": a_p99,
            "r_post_p99_ms": r_p99,
            "allowed_ms": allowed(a_p99),
        }
        if r_rate / a_rate < THROUGHPUT_RATIO:
            fails.append(f"post-fallback throughput {r_rate / a_rate:.3f} x A < {THROUGHPUT_RATIO}")
        if r_p99 > allowed(a_p99):
            fails.append(f"post-fallback median episode P99 {r_p99:.2f} ms > allowed {allowed(a_p99):.2f} ms")
    a_bad = [
        f"{e['run']} w{e['window']}"
        for e in A
        if e["truth"]["sustained"] or e.get("a_like_from_s") is None or e["a_like_from_s"] > HOLD_S
    ]
    if len(a_bad) > A_MAX_BAD:
        invalid.append(f"A side: {len(a_bad)} of {len(A)} A episodes slow or not A-like ({', '.join(a_bad)})")
    b_sus = sum(e["truth"]["sustained"] for e in B)
    confirmed = sum(bool(e.get("confirmed_slow_at_trip")) for e in T)
    if B and b_sus < MIN_B_SUSTAINED:
        inconclusive.append(
            f"B phenotype not reproduced: {b_sus} of {len(B)} B episodes sustained (< {MIN_B_SUSTAINED})"
        )
    if R and (len(T) < MIN_R_TRIPPED or confirmed < MIN_R_CONFIRMED):
        inconclusive.append(
            f"not enough natural slow state: {len(T)} R trips (need {MIN_R_TRIPPED}), "
            f"{confirmed} confirmed slow at the trip (need {MIN_R_CONFIRMED})"
        )
    if pending and not hard:
        outcome = "PENDING"
    elif hard:
        outcome = "FAIL"
    elif invalid:
        outcome = "INCONCLUSIVE"
    elif fails:
        outcome = "FAIL"
    elif inconclusive:
        outcome = "INCONCLUSIVE"
    else:
        outcome = "PASS"

    def dist(xs):
        xs = [x for x in xs if x is not None]
        return {"n": len(xs), "median": _pct(xs, 50), "p95": _pct(xs, 95), "worst": max(xs) if xs else None}

    stats = {
        "A": {"episodes": len(A), "sustained": sum(e["truth"]["sustained"] for e in A)},
        "B": {
            "episodes": len(B),
            "sustained": b_sus,
            "c3_offline_trips": sum(1 for e in B if e.get("c3_offline")),
        },
        "R": {
            "episodes": len(R),
            "sustained_whole_window": sum(e["truth"]["sustained"] for e in R),
            "tripped": len(T),
            "confirmed_slow_at_trip": confirmed,
            "suspected_false_trips": sum(bool(e.get("suspected_false_trip")) for e in T),
            "recovered_within_1s": sum(
                1 for e in T if e.get("trip_to_recovery_ms") is not None and e["trip_to_recovery_ms"] <= RECOVERY_MS
            ),
            "onset_to_trip_ms": dist([e.get("onset_to_trip_ms") for e in T]),
            "trip_to_first_a_ms": dist([e.get("trip_to_first_a_ms") for e in T]),
            "first_a_to_recovery_ms": dist([e.get("first_a_to_recovery_ms") for e in T]),
            "trip_to_recovery_ms": dist([e.get("trip_to_recovery_ms") for e in T]),
        },
        "post_fallback": post_ok,
        "worst_recovery": None,
        "false_trip_cost": [],
    }
    if TP:
        w = max(TP, key=lambda e: np.inf if e["trip_to_recovery_ms"] is None else e["trip_to_recovery_ms"])
        stats["worst_recovery"] = {
            "episode": f"{w['run']} w{w['window']}",
            "trip_to_recovery_ms": w["trip_to_recovery_ms"],
        }
    for e in TP:
        if e.get("suspected_false_trip"):
            stats["false_trip_cost"].append(
                {
                    "episode": f"{e['run']} w{e['window']}",
                    "gpu_return_p50_ms": e["post"]["gpu_return"]["p50_ms"],
                    "gpu_return_p99_ms": e["post"]["gpu_return"]["p99_ms"],
                    "req_s": e["post"]["aggregate_req_s"],
                    "p99_ms": e["post"]["p99_ms"],
                }
            )
    ref_cost = {}
    for cell, xs in (("A", A), ("B", [e for e in B if not e["truth"]["sustained"]])):
        if xs:
            ref_cost[cell] = {
                "gpu_return_p50_ms": _pct(
                    [e["gpu_return"]["p50_ms"] for e in xs if e["gpu_return"]["p50_ms"] is not None], 50
                ),
                "req_s": _pct([e["aggregate_req_s"] for e in xs], 50),
                "p99_ms": _pct([e["p99_ms"] for e in xs], 50),
            }
    stats["false_trip_reference"] = ref_cost
    for e in eps:
        for k in [k for k in e if k.startswith("_")]:
            del e[k]
        if "post" in e:
            e["post"] = {k: v for k, v in e["post"].items() if not k.startswith("_")}
    return {
        "outcome": outcome,
        "pinned_tree": tree,
        "pending": pending,
        "hard_failures": hard,
        "failures": fails,
        "invalid": invalid,
        "inconclusive": inconclusive,
        "reference": ref,
        "stats": stats,
        "runs": per_run,
        "slots": sl,
        "episodes": eps,
    }


# ------------------------------------------------------------------ report


def f(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def _dist(d: dict, digits=0) -> str:
    return f"{f(d['median'], digits)} / {f(d['p95'], digits)} / {f(d['worst'], digits)} (n {d['n']})"


def tables(res: dict) -> str:
    st, ref = res["stats"], res["reference"] or {}
    L = [
        "# Phase 1: recovery experiment",
        "",
        f"Outcome: **{res['outcome']}**",
        "",
        "Generated by `scripts/phase1_analyze.py` from `raw/` under [`phase1.md`](phase1.md).",
        "",
    ]
    for title, key in (
        ("Pending runs", "pending"),
        ("Hard failures (R)", "hard_failures"),
        ("Failures", "failures"),
        ("Invalid", "invalid"),
        ("Inconclusive", "inconclusive"),
    ):
        if res[key]:
            L += [f"{title}:", ""] + [f"- {x}" for x in res[key]] + [""]
    if ref:
        env = ref["envelope"]
        L += [
            f"A reference: m_A {ref['m_a_ms']:.2f} ms (slow > {SLOW_RATIO * ref['m_a_ms']:.2f} ms); A-like envelope "
            f"(P{ENV_Q} of the Phase 1 A windows of {W} requests): median ≤ {env['med']:.2f} ms, P95 ≤ {env['p95']:.2f} ms, "
            f"span ≤ {env['span']:.1f} ms, host-slow ≤ {HOST_SLOW_MAX}.",
            "",
        ]
    if st:
        r = st["R"]
        L += [
            "## Summary",
            "",
            f"- A: {st['A']['sustained']} of {st['A']['episodes']} episodes sustained slow.",
            f"- B: {st['B']['sustained']} of {st['B']['episodes']} episodes sustained slow.",
            f"- R: {r['episodes']} episodes, {r['sustained_whole_window']} sustained over the whole window, {r['tripped']} tripped "
            f"({r['confirmed_slow_at_trip']} confirmed slow at the trip, {r['suspected_false_trips']} suspected false trips), "
            f"{r['recovered_within_1s']} recovered within 1 s.",
            "",
            "| R timing, ms | median / P95 / worst |",
            "|---|---|",
            f"| slow onset (from the arm time, episode start + {ARM_S:.0f} s) → detector trip | {_dist(r['onset_to_trip_ms'])} |",
            f"| detector trip → first A request | {_dist(r['trip_to_first_a_ms'], 2)} |",
            f"| first A request → A-like recovery | {_dist(r['first_a_to_recovery_ms'])} |",
            f"| **detector trip → sustained A-like recovery** | {_dist(r['trip_to_recovery_ms'])} |",
            "",
        ]
        if st["worst_recovery"]:
            w = st["worst_recovery"]
            L += [f"Worst recovery: {w['episode']}, trip → recovery {f(w['trip_to_recovery_ms'], 0)} ms.", ""]
        if st["false_trip_cost"]:
            ref_c = st["false_trip_reference"]
            L += [
                "Suspected false trips, after the fallback (reference medians: "
                + "; ".join(
                    f"{c} GPU-return P50 {f(x['gpu_return_p50_ms'], 3)} ms, {f(x['req_s'], 1)} req/s, P99 {f(x['p99_ms'])} ms"
                    for c, x in ref_c.items()
                )
                + "):",
                "",
            ]
            for x in st["false_trip_cost"]:
                L.append(
                    f"- {x['episode']}: GPU-return P50 / P99 {f(x['gpu_return_p50_ms'], 3)} / {f(x['gpu_return_p99_ms'], 3)} ms, "
                    f"{f(x['req_s'], 1)} req/s, P99 {f(x['p99_ms'])} ms"
                )
            L.append("")
        p = st["post_fallback"]
        if p:
            L += [
                f"After the fallback ([first A + 1 s, end), tripped R episodes pooled): {p['r_post_req_s']:.1f} req/s = "
                f"{p['throughput_ratio']:.3f} × A ({p['a_req_s']:.1f}); median episode P99 {p['r_post_p99_ms']:.2f} ms against "
                f"A {p['a_p99_ms']:.2f} ms (allowed {p['allowed_ms']:.2f}).",
                "",
            ]
    L += [
        "## Episodes",
        "",
        "| run | w | cell | n | median / P95 / P99 ms | host-slow | agg req/s | GPU ret P50 ms | sustained · onset s "
        "| trip s | onset→trip | trip→A | A→rec | trip→rec ms | post P99 / req/s / GPU P50 | notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in res["episodes"]:
        t = e["truth"]
        notes = []
        if e["cell"] == "A":
            notes.append(f"A-like from {f(e.get('a_like_from_s'))} s")
        if e.get("c3_offline") and e["cell"] == "B":
            notes.append(f"offline C3 (armed) {e['c3_offline']['t_s']:.2f} s")
        if e.get("rearmed_in_window"):
            notes.append("re-armed in window")
        if e.get("no_request_after_trip"):
            notes.append("trip on the last request")
        if e.get("confirmed_slow_at_trip"):
            notes.append("confirmed slow")
        if e.get("suspected_false_trip"):
            notes.append("suspected false trip")
        if e.get("structure"):
            notes += e["structure"]
        post = e.get("post") or {}
        L.append(
            f"| {e['run']} | {e['window']} | {e['cell']} | {e['n']} | {f(e['median_ms'])} / {f(e['p95_ms'])} / {f(e['p99_ms'])} "
            f"| {f(e['host_slow_share'], 3)} | {f(e['aggregate_req_s'], 1)} | {f(e['gpu_return']['p50_ms'], 3)} "
            f"| {'yes' if t['sustained'] else 'no'} · {f(e.get('onset_s'))} | {f(e.get('trip_s'))} "
            f"| {f(e.get('onset_to_trip_ms'), 0)} | {f(e.get('trip_to_first_a_ms'))} | {f(e.get('first_a_to_recovery_ms'), 0)} "
            f"| {f(e.get('trip_to_recovery_ms'), 0)} "
            f"| {f(post.get('p99_ms'))} / {f(post.get('aggregate_req_s'), 1)} / {f((post.get('gpu_return') or {}).get('p50_ms'), 3)} "
            f"| {'; '.join(notes)} |"
        )
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", nargs="?", choices=("runs", "reruns"))
    ap.add_argument("--check", action="store_true", help="fail if the outputs are stale")
    ap.add_argument("--raw", type=Path, default=RAW)
    a = ap.parse_args()
    if a.cmd == "runs":
        for cell, rep in RUNS:
            print(cell, rep)
        return 0
    if a.cmd == "reruns":
        for s in slots(a.raw):
            if s["present"] and s["machine"] and not s["rerun"]["present"]:
                print(s["cell"], f"{s['rerun']['run'].split('-r', 1)[1]}")
        return 0
    res = summarise(a.raw)
    js = json.dumps(res, indent=1) + "\n"
    md = tables(res)
    if a.check:
        stale = [p.name for p, t in ((OUT_JSON, js), (OUT_MD, md)) if not p.exists() or p.read_text() != t]
        if stale:
            print("stale:", ", ".join(stale))
            return 1
        return 0
    OUT_JSON.write_text(js)
    OUT_MD.write_text(md)
    print(md.split("\n## Episodes")[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
