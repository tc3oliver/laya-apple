"""1.5 release validation (Phases 2-6): val_results.json and val_tables.md from raw-val/ (../validation.md).

    uv run python research/coreml-adaptive-breaker/scripts/val_analyze.py [--check]
    uv run python research/coreml-adaptive-breaker/scripts/val_analyze.py runs <phase>     # run lines, in order
    uv run python research/coreml-adaptive-breaker/scripts/val_analyze.py reruns <phase>   # machine re-runs
    uv run python research/coreml-adaptive-breaker/scripts/val_analyze.py ready <phase>    # exit 1 unless it may run

**Inputs:**
- raw-val/<schedule>-<model>-<cell>-r<rep>[-b].json.gz, written by val_run.py;
- the machine snapshots <run>.before.json and <run>.after.json;
- the crash logs failed/<run>.<attempt>.log.

Every threshold is validation.md's.

**Signals.** The detector reads prepare only. Ground truth, recovery and safety read user-visible
latency and completion cadence, with the definitions of phase1.md. Prepare enters recovery only as
the extra necessary condition of the A-like window (at most 2 host-slow requests).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = HERE.parents[2]
RAW = ROOT / "raw-val"
OUT_JSON = ROOT / "val_results.json"
OUT_MD = ROOT / "val_tables.md"
S = 1_000_000_000


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p1 = _load("adaptive_phase1_for_val", HERE / "phase1_analyze.py")  # Phase 1's definitions
qual = p1.qual

# ------------------------------------------------------------------ frozen by validation.md
GUARD = 64
HETERO = ("hetero", "hetero_bursty")
_PAP = (("P", 1), ("A", 1), ("P", 2))
# phase -> [(schedule, model, short, long, seconds, extra flags, runs)]; evidence reuse (validation.md):
# small fresh production runs, one or two A references each, no A/B matrix
PHASES = {
    "2": [("product", "laya", 128, 512, 20.0, "", _PAP)],
    "3": [
        ("product", "laya-typed-decisions", 128, 1024, 20.0, "", _PAP),
        ("soak55", "laya-typed-decisions", 128, 1024, 20.0, "", (("P", 3),)),
    ],
    "4": [("mix", "laya-multilingual", 128, 512, 5.0, "--expect-rejected", (("P", 1),))],
    # addendum 1: phases 5 and 6 merged into one production product-mix soak; its hetero_bursty
    # episodes get a same-day A reference (a short bursty A run), its hetero episodes phase 2's A runs
    "5": [
        ("productsoak", "laya", 128, 512, 20.0, "", (("P", 1),)),
        ("bursty", "laya", 128, 512, 20.0, "", (("A", 1),)),
    ],
}
REFERENCE_PHASES = {"5": ("2",)}
EXTENSION: dict = {}  # addendum 1: no second soak
ORDER = ("2", "3", "4", "5")
NAMES = {"2": "laya", "3": "typed-decisions", "4": "multilingual smoke", "5": "product-mix soak"}
ROUTES = {
    "solo_short": {"short": "ane"},
    "solo_long": {"long": "gpu"},
    "hetero": {"short": "ane", "long": "gpu"},
    "hetero_bursty": {"short": "ane", "long": "gpu"},
    "gpu_only": {"short": "gpu", "long": "gpu"},
}
A_SKIP_S = 1.0
RECOVERY_MS = 1000.0
GPU_RETURN_P50_MS = 1.0  # per untripped P episode, and pooled
THROUGHPUT_RATIO = 1.00  # P / A, completions per second over [t0 + 1 s, end), pooled per phase
EPISODE_P99_MS = 0.5  # median P episode P99 <= median A episode P99 + this (short e2e, [t0 + 1 s, end))
BORDERLINE = {"throughput_ratio": 1.02, "episode_p99_ms": 0.3}  # phase 6 extension triggers
UNTRIPPED_SHARE = 0.80  # of P hetero episodes
ASYNC_RESIDENCY = 0.70  # async ANE forwards / all ANE forwards, P hetero windows
FALSE_TRIP_SHARE = 0.10  # suspected false trips / P hetero episodes
SOAK_MIN_EPISODES = 60
A_SIGNATURE_SHARE = 0.10  # A episodes with a sustained slow state, above which the phase is INVALID
TRACE = (
    "target",
    "submit_ns",
    "prepared_ns",
    "service_start_ns",
    "service_end_ns",
    "received_ns",
    "response_ns",
    "request_id",
)


def pinned() -> dict:
    """validation.md's pins: `laya_apple tree: <sha>`, `pyproject.toml blob: <sha>`, `uv.lock blob: <sha>`."""
    p = ROOT / "validation.md"
    text = p.read_text() if p.exists() else ""
    out = {}
    for key, pat in (
        ("tree", r"laya_apple`? tree: `?([0-9a-f]{40})"),
        ("pyproject", r"pyproject.toml`? blob: `?([0-9a-f]{40})"),
        ("uv_lock", r"uv.lock`? blob: `?([0-9a-f]{40})"),
    ):
        m = re.search(pat, text)
        out[key] = m.group(1) if m else None
    return out


def run_name(sched: str, model: str, cell: str, rep, suffix: str = "") -> str:
    return f"{sched}-{model}-{cell}-r{rep}{suffix}"


def run_lines(phase: str) -> list[str]:
    out = []
    for sched, model, short, long_, secs, extra, runs in PHASES[phase]:
        for cell, rep in runs:
            out.append(f"{sched} {model} {cell} {rep} {short} {long_} {secs} {extra}".rstrip())
    return out


def _pct(x, q):
    x = np.asarray(x, float)
    return float(np.percentile(x, q)) if len(x) else None


def _qs(x) -> dict:
    return {k: _pct(x, q) for k, q in (("median", 50), ("p95", 95), ("p99", 99), ("p999", 99.9))}


def load(p: Path):
    return p1.load(p) if p.exists() else None


# ------------------------------------------------------------------ one run


def prepare(run: dict) -> dict:
    rows = run["trace"]
    t = {c: np.asarray([r[i] for r in rows]) for i, c in enumerate(run["trace_columns"])}
    run["_t"] = t
    ane = t["target"] == "ane" if len(rows) else np.zeros(0, bool)
    o = np.argsort(t["service_start_ns"][ane], kind="stable") if len(rows) else np.zeros(0, int)
    st = t["service_start_ns"][ane][o] if len(rows) else np.zeros(0)
    en = t["service_end_ns"][ane][o] if len(rows) else np.zeros(0)
    ids = t["request_id"][ane][o] if len(rows) else np.zeros(0)
    dec = run.get("decisions") or []
    dt = np.array([d[0] for d in dec], dtype=np.int64)
    k = np.searchsorted(st, dt, side="right") - 1
    ok = (k >= 0) & (en[np.maximum(k, 0)] >= dt) if len(st) else np.zeros(len(dt), bool)
    run["_dec_req"] = np.where(ok, ids[np.maximum(k, 0)], -1) if len(st) else np.full(len(dt), -1)
    # a trip happens in observe(), inside the tripping request's completion: [received, response]
    rec, resp = (t["received_ns"][ane], t["response_ns"][ane]) if len(rows) else (np.zeros(0), np.zeros(0))
    aid = t["request_id"][ane] if len(rows) else np.zeros(0)
    run["_trip_req"] = []
    for tt in run.get("trips") or []:
        m = (rec <= tt) & (resp >= tt)
        run["_trip_req"].append(int(aid[m][0]) if m.sum() == 1 else -1)
    return run


def requests(run: dict, lo: int, end: int) -> dict:
    t = run["_t"]
    m = (t["target"] == "ane") & (t["submit_ns"] >= lo) & (t["submit_ns"] < end)
    o = np.argsort(t["submit_ns"][m], kind="stable")
    sub = t["submit_ns"][m][o]
    return {
        "submit": sub,
        "id": t["request_id"][m][o],
        "prep": (t["prepared_ns"][m][o] - sub) / 1e6,
        "e2e": (t["response_ns"][m][o] - sub) / 1e6,
        "resp": t["response_ns"][m][o],
        "start": t["service_start_ns"][m][o],
        "end": t["service_end_ns"][m][o],
    }


def gpu_returns(run: dict, lo: int, end: int) -> np.ndarray:
    t = run["_t"]
    m = (t["target"] == "gpu") & (t["received_ns"] >= lo) & (t["received_ns"] < end)
    return (t["received_ns"][m] - t["service_end_ns"][m]) / 1e6


def correctness(run: dict) -> dict:
    mism, routing, loss = 0, [], []
    t = run["_t"]
    for w in run["windows"]:
        for s, st in w["streams"].items():
            mism += st["mismatches"]
            want = ROUTES[w["condition"]].get(s)
            if want is None or set(st["devices"]) != {want}:
                routing.append(f"w{w['index']} {w['condition']} {s}: {st['devices']}")
        if w["instance"] == "auto":
            for s, st in w["streams"].items():
                target = ROUTES[w["condition"]][s]
                n = int(
                    ((t["target"] == target) & (t["submit_ns"] >= w["start_ns"]) & (t["submit_ns"] < w["end_ns"])).sum()
                )
                if n != len(st["latency_ms"]):
                    loss.append(f"w{w['index']} {s}: client {len(st['latency_ms'])} vs trace {n}")
    alive = run.get("workers_alive_at_end") or {}
    dead = [k for k, v in alive.items() if not v]
    return {"mismatches": mism, "routing_failures": routing, "request_loss": loss, "dead_workers": dead}


def structure(run: dict, w: dict, rq: dict) -> tuple[list[str], dict]:
    """P hetero window: leading armed (sync) decisions, exactly GUARD safe_sync (1..GUARD), then
    async_healthy (async), then optionally breaker_open (sync) to the end; at most one trip."""
    ids = set(rq["id"].tolist())
    dec = run.get("decisions") or []
    sel = [d for d, r in zip(dec, run["_dec_req"]) if r in ids]
    bad = []
    k = 0
    while k < len(sel) and sel[k][2] == "armed":
        if sel[k][1] != 0:
            bad.append("async decision while armed")
        k += 1
    for c in range(1, GUARD + 1):
        if k >= len(sel) or sel[k][2] not in ("safe_sync", "async_healthy") or sel[k][3] != c or sel[k][1] != 0:
            bad.append(f"guard decision {c} missing or wrong")
            break
        k += 1
    first_async = None
    while k < len(sel) and sel[k][2] == "async_healthy":
        if sel[k][1] != 1:
            bad.append("sync decision in async_healthy")
        if first_async is None:
            first_async = int(sel[k][0])
        k += 1
    first_open = int(sel[k][0]) if k < len(sel) else None
    while k < len(sel) and sel[k][2] == "breaker_open":
        if sel[k][1] != 0:
            bad.append("async decision after the breaker opened (retry)")
        k += 1
    if k < len(sel):
        retry = first_open is not None and any(d[1] == 1 for d in sel[k:])
        bad.append(
            "async decision after the breaker opened (retry)"
            if retry
            else f"unexpected decision {sel[k][2]} after the episode's states"
        )
    trips = [tt for tt, r in zip(run.get("trips") or [], run["_trip_req"]) if r in ids]
    if len(trips) > 1:
        bad.append(f"{len(trips)} trips in one episode")
    if bool(trips) != (first_open is not None):
        bad.append("trip and breaker_open disagree")
    if trips and first_open is not None and first_open < trips[0]:
        bad.append("breaker_open decided before the trip")
    n_async = sum(1 for d in sel if d[1] == 1)
    return bad, {
        "t_h": first_async,
        "trip": int(trips[0]) if trips else None,
        "first_a": first_open,
        "first_a_req": int(run["_dec_req"][[i for i, d in enumerate(dec) if d[0] == first_open][0]])
        if first_open
        else None,
        "forwards": len(sel),
        "async_forwards": n_async,
    }


def episode(run: dict, w: dict, cell: str, ref: dict) -> dict:
    t0, end = w["start_ns"], w["end_ns"]
    short = np.asarray(w["streams"]["short"]["latency_ms"], float)
    e = {
        "index": w["index"],
        "condition": w["condition"],
        "cell": cell,
        "client": _qs(short),
        "aggregate_req_s": float(sum(s["req_s"] for s in w["streams"].values())),
    }
    rq = requests(run, t0, end)
    env, m_a = ref["envelope"][w["condition"]], ref["m_a_ms"]
    lo = t0 + int(A_SKIP_S * S)  # steady span, both cells: past P's 64-forward guard (~0.75 s)
    tt = run["_t"]
    done = (tt["response_ns"] >= lo) & (tt["response_ns"] < end)
    e["steady"] = {
        "completions": int(done.sum()),
        "seconds": (end - lo) / S,
        "short_p99_ms": _pct(rq["e2e"][rq["submit"] >= lo], 99),
        "_e2e": rq["e2e"][rq["submit"] >= lo],
    }
    if cell == "A":
        lo = t0 + int(A_SKIP_S * S)
        sub = {k: v[rq["submit"] >= lo] for k, v in rq.items()}
        e["truth"] = p1.phase0_truth(sub, lo, end, m_a)
        e["truth"].pop("onset_ns", None)
        e["gpu_return_p50_ms"] = _pct(gpu_returns(run, lo, end), 50)
        return e
    bad, st = structure(run, w, rq)
    e["structure"] = bad
    e["forwards"], e["async_forwards"] = st["forwards"], st["async_forwards"]
    th = st["t_h"]
    e["t_h_s"] = None if th is None else (th - t0) / 1e9
    e["tripped"] = st["trip"] is not None
    if th is None:
        return e
    trip = st["trip"]
    healthy_end = trip if trip is not None else end
    gr = gpu_returns(run, th + S, healthy_end)
    e["gpu_return_async"] = {"n": int(len(gr)), "p50_ms": _pct(gr, 50), "p99_ms": _pct(gr, 99)}
    if trip is None:
        sub = {k: v[rq["submit"] >= th] for k, v in rq.items()}
        e["truth"] = p1.phase0_truth(sub, th, end, m_a)
        e["truth"].pop("onset_ns", None)
        return e
    e["trip_s"] = (trip - t0) / 1e9
    e["trip_to_first_a_ms"] = None if st["first_a"] is None else (st["first_a"] - trip) / 1e6
    limit = p1.SLOW_RATIO * m_a
    idx_a = None
    if st["first_a_req"] is not None and st["first_a_req"] in set(rq["id"].tolist()):
        idx_a = int(np.where(rq["id"] == st["first_a_req"])[0][0])
    if idx_a is None:
        e["no_request_after_trip"] = True
        return e
    sub_a = int(rq["submit"][idx_a])
    r0 = p1.rolling(rq)
    lat_bad = (r0["med"] > env["med"]) | (r0["p95"] > env["p95"])
    cand = np.where(lat_bad & (r0["t"] >= th) & (r0["t"] <= trip))[0]
    onset = None
    if len(cand):
        k0 = int(np.searchsorted(rq["submit"], r0["t"][cand[0]]))
        above = np.where(rq["e2e"][k0 : k0 + p1.W] > env["p95"])[0]
        onset = int(rq["submit"][k0 + (above[0] if len(above) else 0)])
    e["onset_to_trip_ms"] = None if onset is None else (trip - onset) / 1e6
    m = (rq["submit"] >= max(th, trip - S)) & (rq["submit"] < sub_a)
    pre = rq["e2e"][m]
    e["confirmed_slow"] = bool(len(pre) and np.median(pre) > limit)
    e["suspected_false_trip"] = onset is None and not e["confirmed_slow"]
    p = p1.recovery_point(p1.rolling(rq, idx_a), env)
    e["trip_to_recovery_ms"] = None if p is None else (p - trip) / 1e6
    post = sub_a + S
    e["post"] = {
        "persistent_latency": p1._run2(p1.slow_spans(rq, post, end, limit)),
        "persistent_host_slow": p1._run2(p1.host_bins(rq, post, end)),
        "gpu_return_p50_ms": _pct(gpu_returns(run, post, end), 50),
    }
    return e


# ------------------------------------------------------------------ slots, machine, crashes


def _logs(raw: Path) -> list[str]:
    return [p.name for p in (raw / "failed").glob("*.log")] if (raw / "failed").exists() else []


def _crashes(logs, name) -> int:
    return sum(bool(re.fullmatch(re.escape(name) + r"\.\d+\.log", x)) for x in logs)


def _slot(raw, logs, sched, model, short, long_, secs, extra, cell, rep) -> dict:
    base = run_name(sched, model, cell, rep)
    reb = base + "-b"
    s = {"schedule": sched, "model": model, "cell": cell, "rep": rep, "run": base, "rerun": reb}
    s["line"] = f"{sched} {model} {cell} {rep} {short} {long_} {secs} {extra}".rstrip()
    s["crashes"] = {base: _crashes(logs, base), reb: _crashes(logs, reb)}
    s["present"] = {n: (raw / f"{n}.json.gz").exists() for n in (base, reb)}
    s["machine"] = {
        n: qual.snapshot_checks(load(raw / f"{n}.before.json"), load(raw / f"{n}.after.json"))
        for n in (base, reb)
        if s["present"][n]
    }
    s["machine_invalid"] = False
    if s["present"][base] and s["machine"][base]:
        s["used"] = reb if s["present"][reb] and not s["machine"][reb] else None
        s["machine_invalid"] = s["present"][reb] and bool(s["machine"][reb])
    else:
        s["used"] = base if s["present"][base] else None
    c = s["crashes"][base] + s["crashes"][reb]
    s["stops"] = bool((cell == "P" and c) or s["crashes"][base] >= 2 or s["crashes"][reb] >= 2)
    return s


def slots(raw: Path, phase: str, extended: bool = False) -> list[dict]:
    logs = _logs(raw)
    out = []
    for sched, model, short, long_, secs, extra, runs in PHASES[phase]:
        for cell, rep in runs:
            out.append(_slot(raw, logs, sched, model, short, long_, secs, extra, cell, rep))
    if extended and phase in EXTENSION:
        sched, model, short, long_, secs, extra, (cell, rep) = EXTENSION[phase]
        out.append(_slot(raw, logs, sched, model, short, long_, secs, extra, cell, rep))
    return out


def _runs(raw: Path, sl: list[dict], pins: dict, fail: list, invalid: list) -> dict:
    runs = {}
    for s in sl:
        for n, c in s["crashes"].items():
            if c and s["cell"] == "P":
                fail.append(f"{n} crashed")
            elif c >= 2:
                invalid.append(f"{n} crashed twice")
        if s["machine_invalid"]:
            invalid.append(f"{s['run']}: machine failed twice")
        if not s["used"]:
            continue
        run = load(raw / f"{s['used']}.json.gz")
        if run is None:
            invalid.append(f"{s['used']}: unreadable")
            continue
        rt = run.get("runtime") or {}
        if (
            rt.get("laya_apple_tree") != pins["tree"]
            or rt.get("pyproject_blob") != pins["pyproject"]
            or rt.get("uv_lock_blob") != pins["uv_lock"]
            or rt.get("dirty") is not False
        ):
            invalid.append(f"{s['used']}: runtime {rt.get('laya_apple_tree')} dirty={rt.get('dirty')}")
        a = run.get("args") or {}
        if a.get("cell") != s["cell"] or a.get("model") != s["model"] or a.get("schedule") != s["schedule"]:
            invalid.append(f"{s['used']}: protocol deviation")
        runs[s["used"]] = prepare(run)
    return runs


def _reference(a_runs: list[dict]) -> dict:
    ref = {"envelope": {}}
    e2e = []
    for cond in HETERO:
        rolls = []
        for run in a_runs:
            for w in run["windows"]:
                if w["condition"] != cond:
                    continue
                rq = requests(run, w["start_ns"] + int(A_SKIP_S * S), w["end_ns"])
                e2e.append(rq["e2e"])
                rolls.append(p1.rolling(rq))
        if rolls:
            cat = lambda k: np.concatenate([r[k] for r in rolls])  # noqa: E731
            ref["envelope"][cond] = {k: _pct(cat(k), p1.ENV_Q) for k in ("med", "p95", "span")}
    if e2e:
        ref["m_a_ms"] = float(np.median(np.concatenate(e2e)))
    return ref


def _triggers(st: dict) -> list[str]:
    """Phase 6's extension triggers (validation.md)."""
    out = []
    if st["suspected_false_trips"]:
        out.append(f"{st['suspected_false_trips']} suspected false trip(s)")
    if st["throughput_ratio"] is not None and st["throughput_ratio"] < BORDERLINE["throughput_ratio"]:
        out.append(f"throughput ratio {st['throughput_ratio']:.3f} < {BORDERLINE['throughput_ratio']}")
    if st["episode_p99_delta_ms"] is not None and st["episode_p99_delta_ms"] > BORDERLINE["episode_p99_ms"]:
        out.append(f"episode P99 delta {st['episode_p99_delta_ms']:+.2f} ms > +{BORDERLINE['episode_p99_ms']}")
    return out


def judge(raw: Path, phase: str, extended: bool | None = None) -> dict:
    """One phase. Phase 6: judged on r1 first; if r1 triggers an extension, r2 is required and the
    phase is judged on both."""
    if extended is None and phase in EXTENSION:
        first = judge(raw, phase, extended=False)
        trig = first.get("triggers") or []
        if first["status"] in ("PASS",) and trig:
            out = judge(raw, phase, extended=True)
            out["triggers"] = trig
            return out
        return first
    pins = pinned()
    sl = slots(raw, phase, bool(extended))
    stopped = any(s["stops"] for s in sl)
    pending = [] if stopped else [s["run"] for s in sl if s["used"] is None and not s["machine_invalid"]]
    fail, invalid = [], []
    runs = _runs(raw, sl, pins, fail, invalid)
    out = {"phase": phase, "name": NAMES[phase], "pending": pending, "slots": sl, "extended": bool(extended)}
    if phase == "4":
        return judge_smoke(out, sl, runs, fail, invalid)
    for s in sl:
        run = runs.get(s["used"])
        if run is None:
            continue
        c = correctness(run)
        broken = c["mismatches"] or c["routing_failures"] or c["request_loss"] or c["dead_workers"]
        if s["cell"] == "P":
            snap = run.get("handoff_snapshot") or {}
            if not snap.get("enabled") or not snap.get("consistent"):
                fail.append(
                    f"{s['used']}: handoff snapshot {snap.get('state')} consistent={snap.get('consistent')} "
                    f"({snap.get('disabled_reason')})"
                )
            n_het = sum(1 for w in run["windows"] if w["condition"] in HETERO)
            if snap.get("episodes") != n_het:
                fail.append(f"{s['used']}: {snap.get('episodes')} episodes for {n_het} hetero windows")
            if (snap.get("breaker") or {}).get("trips") != len(run.get("trips") or []):
                fail.append(f"{s['used']}: snapshot trips != logged trips")
            if -1 in run["_trip_req"]:
                fail.append(f"{s['used']}: a trip not inside one request's completion")
            if "ane_handoff" not in (run.get("info_start") or {}):
                fail.append(f"{s['used']}: the production default did not use adaptive execution")
        if broken:
            (fail if s["cell"] == "P" else invalid).append(f"{s['used']}: correctness {c}")
    # the A reference: this phase's A runs, or the reference phases' (phase 6)
    a_runs = [runs[s["used"]] for s in sl if s["cell"] == "A" and s["used"] in runs]
    a_slots = [s for s in sl if s["cell"] == "A"]
    for q in REFERENCE_PHASES.get(phase, ()):
        qs = slots(raw, q)
        a_slots += [s for s in qs if s["cell"] == "A"]
        a_runs += [r for s, r in _runs(raw, [s for s in qs if s["cell"] == "A"], pins, [], invalid).items()]
    ref = _reference(a_runs)
    if pending or "m_a_ms" not in ref:
        out.update(
            status="PENDING" if pending else "INVALID",
            fail=fail,
            invalid=invalid + ([] if pending else ["no A reference"]),
        )
        return out
    out["reference"] = {"m_a_ms": ref["m_a_ms"], "envelope": ref["envelope"], "a_runs": [s["used"] for s in a_slots]}
    P, A = [], []
    for s in sl:
        run = runs.get(s["used"])
        if run is None or s["cell"] != "P":
            continue
        for w in run["windows"]:
            if w["condition"] in HETERO:
                e = episode(run, w, "P", ref)
                e["run"] = s["used"]
                P.append(e)
    for run in a_runs:
        for w in run["windows"]:
            if w["condition"] in HETERO and w["condition"] in ref["envelope"]:
                A.append(episode(run, w, "A", ref))
    T = [e for e in P if e.get("tripped")]
    U = [e for e in P if e.get("t_h_s") is not None and not e.get("tripped")]
    for e in P:
        tag = f"{e['run']} w{e['index']}"
        fail += [f"{tag}: {x}" for x in e.get("structure", [])]
        if e.get("t_h_s") is None:
            fail.append(f"{tag}: no handoff to async")
        if not e.get("tripped") and (e.get("truth") or {}).get("sustained"):
            fail.append(f"{tag}: sustained slow state with no trip (exposed)")
        if e.get("tripped") and not e.get("no_request_after_trip"):
            if e["trip_to_recovery_ms"] is None or e["trip_to_recovery_ms"] > RECOVERY_MS:
                fail.append(f"{tag}: no sustained A-like recovery within {RECOVERY_MS:.0f} ms of the trip")
            if e["post"]["persistent_latency"] or e["post"]["persistent_host_slow"]:
                fail.append(f"{tag}: the slow state persists after the fallback")
        gr = (e.get("gpu_return_async") or {}).get("p50_ms")
        if not e.get("tripped") and e.get("t_h_s") is not None and (gr is None or gr > GPU_RETURN_P50_MS):
            fail.append(f"{tag}: GPU-return P50 {gr} ms during healthy async > {GPU_RETURN_P50_MS} ms")
    a_sig = sum(1 for e in A if e["truth"]["sustained"])
    if A and a_sig > A_SIGNATURE_SHARE * len(A):
        invalid.append(f"A side: {a_sig} of {len(A)} A episodes sustained slow")
    n_p = len(P)
    fwd = sum(e.get("forwards", 0) for e in P)
    residency = sum(e.get("async_forwards", 0) for e in P) / fwd if fwd else None
    untripped = len(U) / n_p if n_p else None
    false_trips = sum(1 for e in T if e.get("suspected_false_trip"))
    if untripped is None or untripped < UNTRIPPED_SHARE:
        fail.append(f"product value: {len(U)} of {n_p} P episodes stayed async (< {UNTRIPPED_SHARE:.0%})")
    if residency is None or residency < ASYNC_RESIDENCY:
        fail.append(f"product value: async residency {residency} < {ASYNC_RESIDENCY:.0%}")
    if n_p and false_trips > FALSE_TRIP_SHARE * n_p:
        fail.append(f"false trips: {false_trips} of {n_p} P episodes (> {FALSE_TRIP_SHARE:.0%})")
    # throughput and latency against the A reference, over each episode's steady span, per condition
    # (addendum 1: a bursty episode is compared with bursty A episodes, a hetero one with hetero A)
    ratio = dp99 = None
    by_cond = {}
    for cond in HETERO:
        pc = [e for e in P if e["condition"] == cond]
        ac = [e for e in A if e["condition"] == cond]
        if not pc:
            continue
        if not ac:
            invalid.append(f"no A reference for {cond}")
            continue
        p_rate = sum(e["steady"]["completions"] for e in pc) / sum(e["steady"]["seconds"] for e in pc)
        a_rate = sum(e["steady"]["completions"] for e in ac) / sum(e["steady"]["seconds"] for e in ac)
        r = p_rate / a_rate
        p99_p = float(np.median([e["steady"]["short_p99_ms"] for e in pc]))
        p99_a = float(np.median([e["steady"]["short_p99_ms"] for e in ac]))
        by_cond[cond] = {"throughput_ratio": r, "episode_p99_delta_ms": p99_p - p99_a, "p": len(pc), "a": len(ac)}
        if r < THROUGHPUT_RATIO:
            fail.append(
                f"throughput {cond}: P {p_rate:.1f} = {r:.3f} x A {a_rate:.1f} completions/s (< {THROUGHPUT_RATIO})"
            )
        if p99_p - p99_a > EPISODE_P99_MS:
            fail.append(f"latency {cond}: median episode P99 {p99_p:.2f} ms > A {p99_a:.2f} + {EPISODE_P99_MS} ms")
        ratio = r if ratio is None else min(ratio, r)
        dp99 = p99_p - p99_a if dp99 is None else max(dp99, p99_p - p99_a)
    if phase == "5" and n_p < SOAK_MIN_EPISODES:
        fail.append(f"soak: {n_p} P hetero episodes < {SOAK_MIN_EPISODES}")
    rec = [e["trip_to_recovery_ms"] for e in T if e.get("trip_to_recovery_ms") is not None]
    ott = [e["onset_to_trip_ms"] for e in T if e.get("onset_to_trip_ms") is not None]
    gr_u = [e["gpu_return_async"]["p50_ms"] for e in U if (e.get("gpu_return_async") or {}).get("p50_ms") is not None]
    a_gr = [e["gpu_return_p50_ms"] for e in A if e.get("gpu_return_p50_ms") is not None]
    p_lat = np.concatenate([e["steady"]["_e2e"] for e in P]) if P else np.zeros(0)
    a_lat = np.concatenate([e["steady"]["_e2e"] for e in A]) if A else np.zeros(0)

    def dist(xs):
        return {"n": len(xs), "median": _pct(xs, 50), "p95": _pct(xs, 95), "worst": max(xs) if xs else None}

    out["stats"] = {
        "p_episodes": n_p,
        "a_episodes": len(A),
        "untripped": len(U),
        "untripped_share": untripped,
        "async_residency": residency,
        "trips": len(T),
        "confirmed_slow_trips": sum(1 for e in T if e.get("confirmed_slow")),
        "suspected_false_trips": false_trips,
        "trip_rate": len(T) / n_p if n_p else None,
        "false_trip_rate": false_trips / n_p if n_p else None,
        "onset_to_trip_ms": dist(ott),
        "trip_to_first_a_ms": dist([e["trip_to_first_a_ms"] for e in T if e.get("trip_to_first_a_ms") is not None]),
        "trip_to_recovery_ms": dist(rec),
        "a_sustained": a_sig,
        "gpu_return_p50_ms": {
            "P_healthy_median": _pct(gr_u, 50),
            "P_healthy_worst": max(gr_u) if gr_u else None,
            "A_median": _pct(a_gr, 50),
        },
        "throughput_ratio": ratio,  # the worst condition
        "episode_p99_delta_ms": dp99,
        "by_condition": by_cond,
        "latency_ms": {"P": _qs(p_lat), "A": _qs(a_lat)},
    }
    for e in P + A:
        e["steady"].pop("_e2e", None)
    out["episodes"] = P
    out["triggers"] = _triggers(out["stats"]) if phase in EXTENSION and not extended else []
    out.update(status=_status(pending, fail, invalid), fail=fail, invalid=invalid)
    return out


def _status(pending, fail, invalid) -> str:
    if pending:
        return "PENDING"
    if invalid:
        return "INVALID"
    return "FAIL" if fail else "PASS"


def judge_smoke(out, sl, runs, fail, invalid) -> dict:
    s = sl[0]
    run = runs.get(s["used"])
    if run is None:
        out.update(status=_status(out["pending"], fail, invalid), fail=fail, invalid=invalid)
        return out
    c = correctness(run)
    if c["mismatches"] or c["routing_failures"] or c["request_loss"] or c["dead_workers"]:
        fail.append(f"correctness {c}")
    for tag in ("info_start", "info_end"):
        info = run.get(tag) or {}
        if info.get("ane_placement") != "process":
            fail.append(f"{tag}: ane_placement {info.get('ane_placement')!r}, not 'process'")
        if "ane_handoff" in info:
            fail.append(f"{tag}: adaptive execution applied to laya-multilingual")
    if run.get("decisions") or run.get("trips"):
        fail.append("handoff decisions logged for laya-multilingual")
    rej = run.get("expect_rejected") or {}
    if rej.get("raised") != "ValueError":
        fail.append(f"ane_handoff=True did not raise ValueError: {rej}")
    out.update(status=_status(out["pending"], fail, invalid), fail=fail, invalid=invalid, correctness=c)
    return out


def ready(raw: Path, phase: str) -> tuple[bool, str]:
    for p in ORDER[: ORDER.index(phase)]:
        st = judge(raw, p)["status"]
        if st != "PASS":
            return False, f"phase {p} is {st}"
    j = judge(raw, phase)
    if j["status"] in ("FAIL", "INVALID") and not j["pending"]:
        return False, f"phase {phase} is already {j['status']}"
    if any(x.endswith("crashed") for x in j.get("fail", [])):
        return False, f"phase {phase} has a P crash"
    return True, "ok"


def reruns(raw: Path, phase: str) -> list[str]:
    """Machine re-runs, then (phase 6) the extension run once r1 triggered it."""
    out = [
        s["line"].replace(f" {s['rep']} ", f" {s['rep']}-b ", 1)
        for s in slots(raw, phase, extended=True)
        if s["present"][s["run"]] and s["machine"].get(s["run"]) and not s["present"][s["rerun"]]
    ]
    if phase in EXTENSION:
        first = judge(raw, phase, extended=False)
        ext = slots(raw, phase, extended=True)[-1]
        if (
            first["status"] == "PASS"
            and first.get("triggers")
            and ext["used"] is None
            and not ext["present"][ext["run"]]
        ):
            out.append(ext["line"])
    return out


# ------------------------------------------------------------------ report


def f(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def summarise(raw: Path = RAW) -> dict:
    phases = {}
    for p in ORDER:
        phases[p] = judge(raw, p)
        if phases[p]["status"] != "PASS":
            for q in ORDER[ORDER.index(p) + 1 :]:
                phases[q] = {"phase": q, "name": NAMES[q], "status": "NOT RUN"}
            break
    done = all(phases[p]["status"] == "PASS" for p in ORDER)
    stop = next((p for p in ORDER if phases[p]["status"] not in ("PASS",)), None)
    outcome = "RELEASE" if done else f"phase {stop} {phases[stop]['status']}"
    return {"outcome": outcome, "pins": pinned(), "phases": phases}


def tables(res: dict) -> str:
    L = [
        "# 1.5 release validation",
        "",
        f"Outcome: **{res['outcome']}**",
        "",
        "Generated by `scripts/val_analyze.py` under [`validation.md`](validation.md).",
        "",
    ]
    for p, j in res["phases"].items():
        L += [f"## Phase {p}: {j['name']} — {j['status']}", ""]
        for k in ("pending", "fail", "invalid", "triggers"):
            if j.get(k):
                L += [f"{k}:", ""] + [f"- {x}" for x in j[k]] + [""]
        st = j.get("stats")
        if not st:
            continue
        lat, gr = st["latency_ms"], st["gpu_return_p50_ms"]

        def d(x, digits=0):
            return f"{f(x['median'], digits)} / {f(x['p95'], digits)} / {f(x['worst'], digits)} (n {x['n']})"

        L += [
            f"- P hetero episodes {st['p_episodes']} (A reference {st['a_episodes']}); stayed async "
            f"{st['untripped']} ({f(100 * (st['untripped_share'] or 0), 1)}%); async residency "
            f"{f(100 * (st['async_residency'] or 0), 1)}% of ANE forwards.",
            f"- Breaker trips {st['trips']} (rate {f(st['trip_rate'], 3)}): confirmed slow {st['confirmed_slow_trips']}, "
            f"suspected false {st['suspected_false_trips']} (rate {f(st['false_trip_rate'], 3)}).",
            f"- onset → trip ms {d(st['onset_to_trip_ms'])}; trip → first A ms {d(st['trip_to_first_a_ms'], 2)}; "
            f"trip → sustained A-like ms {d(st['trip_to_recovery_ms'])}.",
            f"- GPU return P50 ms: P healthy async median {f(gr['P_healthy_median'], 3)} (worst episode "
            f"{f(gr['P_healthy_worst'], 3)}); A {f(gr['A_median'], 3)}.",
            f"- Throughput (steady span, all completions): P / A = {f(st['throughput_ratio'], 3)}.",
            f"- Short e2e ms, median / P95 / P99 / P99.9: P {f(lat['P']['median'])} / {f(lat['P']['p95'])} / "
            f"{f(lat['P']['p99'])} / {f(lat['P']['p999'])}; A {f(lat['A']['median'])} / {f(lat['A']['p95'])} / "
            f"{f(lat['A']['p99'])} / {f(lat['A']['p999'])}; median episode P99 P − A {f(st['episode_p99_delta_ms'])} ms.",
            f"- A reference episodes with a sustained slow state: {st['a_sustained']}.",
            *[
                f"- {c}: {x['p']} P / {x['a']} A episodes; throughput P / A {f(x['throughput_ratio'], 3)}; "
                f"median episode P99 P − A {f(x['episode_p99_delta_ms'])} ms."
                for c, x in (st.get("by_condition") or {}).items()
            ],
            "",
        ]
        trips = [e for e in j.get("episodes", []) if e.get("tripped")]
        if trips:
            L += [
                "| run | w | trip s | onset→trip ms | trip→A ms | trip→recovery ms | confirmed | suspected false | post GPU P50 |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            for e in trips:
                L.append(
                    f"| {e['run']} | {e['index']} | {f(e.get('trip_s'))} | {f(e.get('onset_to_trip_ms'), 0)} "
                    f"| {f(e.get('trip_to_first_a_ms'))} | {f(e.get('trip_to_recovery_ms'), 0)} | {e.get('confirmed_slow')} "
                    f"| {e.get('suspected_false_trip')} | {f((e.get('post') or {}).get('gpu_return_p50_ms'), 3)} |"
                )
            L.append("")
    return "\n".join(L)


def _json(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", nargs="?", choices=("runs", "reruns", "ready"))
    ap.add_argument("phase", nargs="?", choices=ORDER)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--raw", type=Path, default=RAW)
    a = ap.parse_args()
    if a.cmd == "runs":
        print("\n".join(run_lines(a.phase)))
        return 0
    if a.cmd == "reruns":
        for line in reruns(a.raw, a.phase):  # nothing at all when none is left: run_val.sh stops on empty
            print(line)
        return 0
    if a.cmd == "ready":
        ok, why = ready(a.raw, a.phase)
        if not ok:
            print(why, file=sys.stderr)
        return 0 if ok else 1
    res = summarise(a.raw)
    js = json.dumps(res, indent=1, default=_json) + "\n"
    md = tables(res)
    if a.check:
        stale = [p.name for p, t in ((OUT_JSON, js), (OUT_MD, md)) if not p.exists() or p.read_text() != t]
        if stale:
            print("stale:", ", ".join(stale))
            return 1
        return 0
    OUT_JSON.write_text(js)
    OUT_MD.write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
