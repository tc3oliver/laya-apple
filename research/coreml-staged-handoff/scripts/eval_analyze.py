"""The H64 production evaluation: eval_results.json and eval_tables.md from raw-eval/ (../evaluation.md).

    uv run python research/coreml-staged-handoff/scripts/eval_analyze.py [--check] [--raw DIR] [--out DIR]
    uv run python research/coreml-staged-handoff/scripts/eval_analyze.py runs <phase>
    uv run python research/coreml-staged-handoff/scripts/eval_analyze.py reruns <phase> [--raw DIR]
    uv run python research/coreml-staged-handoff/scripts/eval_analyze.py ready <phase> [--raw DIR]

Inputs: raw-eval/<schedule>-<model>-<cell>-r<rep>.json.gz (prod_run.py), their .before.json /
.after.json machine snapshots (machine_snapshot.py), the machine re-runs <run>-b.json.gz (same
snapshots) and raw-eval/failed/<run>.<attempt>.log (the crash rule).

Subcommands (for run_eval.sh):
- `runs <phase>` prints the phase's runs in their fixed order, one per line:
  "schedule model cell rep short long seconds [extra-flags]".
- `reruns <phase>` prints the same line, with rep "<N>-b", for every run whose snapshots fail
  qualification item 5 and that has no -b re-run yet; nothing while a main run of the phase is missing.
- `ready <phase>` exits 0 only if every earlier phase is PASS (a phase runs only if the previous one
  passed), the phase's own status is not already FAIL or INVALID, and no P run of the phase has a
  crash log.

Runs (evaluation.md's table; P rN is paired with A rN, adjacent in its block of four: two pairs
per block, so 6, 4 and 4 run pairs are the latency clusters of phases 1, 2 and 4; phase 5's 2 pairs
give 22 clusters of 6 consecutive episodes):
  phase 1  mix-laya                     P1 A1 A2 P2 | A3 P3 P4 A4 | P5 A5 A6 P6
  phase 2  mix-laya-typed-decisions     P1 A1 A2 P2 | A3 P3 P4 A4           (L128 / L1024)
  phase 3  mix-laya-multilingual        A1 (5 s windows, --expect-rejected)
  phase 4  product-laya                 P1 A1 A2 P2 | A3 P3 P4 A4
  phase 5  soak55-laya                  P1 A1 A2 P2

Measures are evaluation.md's. Per hetero window (an episode; t0 = start_ns, end = end_ns):
- short requests: the auto trace's ANE rows submitted in [t0, end), in submit order; prepare =
  prepared - submit; each one's latency is the short stream's client latency matched by order (one
  closed-loop short client) when the counts agree, else response - submit from the trace (reported);
- episode median / P95 / P99 / P99.9 over every client latency of the short stream;
- t_h (P) = the first async decision attributed to the window; GPU return = received - service_end
  of the trace's GPU rows received in [t_h + 1 s, end) (P) or [t0 + 1 s, end) (A);
- host-slow bins: 0.5 s bins from t_h (P) or t0 + 1 s (A) to end, the last one possibly shorter;
  a bin is host-slow when >= 0.10 of the short requests submitted in it have prepare > 0.3 ms;
- slow spans: 1 s spans over the same range; a span is slow when the median latency of the short
  requests submitted in it is > 1.2 x the phase's pooled A hetero short median.
- aggregate req/s: the sum of the window's streams' req/s.

Interpretation choices where evaluation.md leaves room (the stricter reading each time):
- Floating-point limits are compared exactly (no tolerance). A measure that cannot be computed
  (no t_h, no GPU row in range, no short request) fails its gate.
- Decisions are attributed to the window whose start precedes them, up to the next window's start
  (the last window: to the end of the run), so a decision made for a request in flight at a window's
  end still counts. H2 also fails for any decision attributed to a window without a short stream on
  the auto instance (solo_long, gpu_only, idle) or made before the first window. H2's hetero rule
  (evaluation.md): zero or more leading armed / sync / count 0 decisions (the short request reaching
  decide() before the long request's GPU submit), then exactly 64 guard decisions with counts 1..64
  in order (sync_guard 1..63, the 64th async_steady, all sync), then t_h = the first async decision,
  immediately after the 64th, and no sync decision after t_h. An armed decision after the guard
  started, 63 or 65 guard decisions all fail.
- Host-slow and slow spans use the short (ANE) requests only (#94's rule, as in the qualification):
  every L512 / L1024 prepare exceeds 0.3 ms by construction. An empty host-slow bin is not host-slow;
  an empty full 1 s span is slow (no short request was submitted for a whole second); an empty final
  partial span is not judged.
- H3's phase-pooled GPU-return P99 and the outcome's pooled P50 pool the GPU-return samples of every
  P episode of the phase. H4's pooled hetero aggregate is the mean of the run's episode aggregates
  (every hetero window of a schedule has the same length, so this is the time-weighted pool).
- H1 also requires both workers ("ane", "gpu") recorded alive; a missing handoff snapshot fails H1.
  A crash of a P run (any failed/ log of it, including a -b re-run) is an H1 failure even when a
  later attempt produced the file. The runner stops at a P crash.
- Machine re-runs (-b) replace the run for pooling, pairing, H3 pooled, H4, H7 and the latency
  test. The superseded P run is still judged on every per-run and per-episode hard gate (H1, H2, H3
  per episode, H5, H6): a P hard failure is never excused by the machine.
- INVALID covers every present run of the phase, used or superseded: runtime (laya_apple tree,
  pyproject.toml and uv.lock blobs at 8d9e798, not dirty),
  protocol (args, window sequence and lengths, --expect-rejected only in phase 3, P's handoff guard
  64 / gap 1.0 s) and the A side (A has no handoff snapshot, no ane_handoff key in info() and no
  logged decision). A mismatches / routing failures in any A file, and A's H5 / H6 signature share
  over the used A episodes (> 10% is INVALID).
- Precedence inside a phase: a P hard failure that no other run can change (H1, H2, H3 per episode,
  H5, on a run with the right runtime and protocol) is FAIL at once, even with runs pending or an
  INVALID condition present (reported alongside). Otherwise INVALID, then pending, then the gates
  that use A or the whole phase (H3 pooled, H4, H6, H7, latency, pooled P50, soak episode count).
- Phase 3 (multilingual smoke, A only): its mismatches, routing failures, a crash (any failed/ log
  of the run, also when the in-place re-run succeeded), workers not alive, info() placement / key,
  a decision or handoff snapshot, and anything other than a ValueError mentioning ane_handoff from
  the rejected load all fail the smoke (FAIL -> CLOSE); runtime, protocol and machine failures make
  it INVALID. "Before loading" is not observable in the record and is not checked further.
- P episodes expected, from prod_run.schedule(): phase 1 12, phase 2 8, phase 4 24 (6 hetero
  windows per product run, 4 P runs), phase 5 132 (the soak gate needs >= 100).
- Tail events: the threshold is the median of the used A episodes' P99 in the phase + 2.0 ms
  (strictly exceeded). A tail event with no measurable pair is "tail with other symptoms". The
  isolated label needs P95 and median deltas <= +0.3 ms, req/s >= 0.95 x the pair's, GPU-return
  P50 <= 1.0 ms, zero host-slow bins and zero slow spans.
- Order: 1, 2, 3, 4, 5. A phase after a non-PASS phase is "not reached" (its would-be status and
  data are still reported).
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
EXP = HERE.parent
RAW = EXP / "raw-eval"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prod_run = _load("eval_prod_run", HERE / "prod_run.py")  # schedule(), STREAMS, TRACE_COLUMNS
qual = _load("eval_qual_analyze", HERE / "qual_analyze.py")  # snapshot_checks (qualification item 5)

S = 1_000_000_000
TREE = "d541fdc820f59fcb7b27a5f20110f828892be16f"
PYPROJECT_BLOB = "2d2d3d684b086c0c651cd1ea96e09550ce2e5e85"
UV_LOCK_BLOB = "10f6ba84dd28791684cc102d651c1d5c38cc0178"
GUARD, GAP_S = 64, 1.0
PHASES = {
    "1": {"schedule": "mix", "model": "laya", "short": 128, "long": 512, "seconds": 20.0, "expect_rejected": False},
    "2": {
        "schedule": "mix",
        "model": "laya-typed-decisions",
        "short": 128,
        "long": 1024,
        "seconds": 20.0,
        "expect_rejected": False,
    },
    "3": {
        "schedule": "mix",
        "model": "laya-multilingual",
        "short": 128,
        "long": 512,
        "seconds": 5.0,
        "expect_rejected": True,
    },
    "4": {"schedule": "product", "model": "laya", "short": 128, "long": 512, "seconds": 20.0, "expect_rejected": False},
    "5": {"schedule": "soak55", "model": "laya", "short": 128, "long": 512, "seconds": 20.0, "expect_rejected": False},
}
_B1 = (("P", 1), ("A", 1), ("A", 2), ("P", 2))
_B2 = (("A", 3), ("P", 3), ("P", 4), ("A", 4))
_B3 = (("P", 5), ("A", 5), ("A", 6), ("P", 6))
RUNS = {"1": _B1 + _B2 + _B3, "2": _B1 + _B2, "3": (("A", 1),), "4": _B1 + _B2, "5": _B1}
ORDER = ("1", "2", "3", "4", "5")
LATENCY_PHASES = ("1", "2", "4", "5")
ROUTES = {
    "solo_short": {"short": {"ane"}},
    "solo_long": {"long": {"gpu"}},
    "hetero": {"short": {"ane"}, "long": {"gpu"}},
    "gpu_only": {"short": {"gpu"}, "long": {"gpu"}},
    "idle": {},
}
HOST_SLOW_MS, BIN_S, BIN_SHARE = 0.3, 0.5, 0.10
SPAN_S, SLOW_RATIO = 1.0, 1.2
A_FROM_S, GPU_FROM_S = 1.0, 1.0
GPU_P50_MS, GPU_P99_MS = 1.0, 1.0
THROUGHPUT_RATIO = 0.95
CONSECUTIVE = 2
TAIL_MS, TAIL_MIN, TAIL_FACTOR = 2.0, 2, 2
MEDIAN_MS, UPPER_MS = 0.5, 1.0
B, SEED, UPPER_Q = 20_000, 20260926, 95
SOAK_GROUP, SOAK_MIN_P = 6, 100
A_SIGNATURE_SHARE = 0.10
ISOLATED_MS = 0.3
RUN_NAME = re.compile(
    r"^(?P<schedule>mix|product|soak55)-(?P<model>laya-typed-decisions|laya-multilingual|laya)"
    r"-(?P<cell>A|P)-r(?P<rep>\d+)(?P<b>-b)?\.json\.gz$"
)
CANDIDATE = "PRODUCT-CANDIDATE: H64 is the leading 1.5 production candidate."
QS = (("median", 50), ("p95", 95), ("p99", 99), ("p999", 99.9))


def run_name(phase: str, cell: str, rep) -> str:
    c = PHASES[phase]
    return f"{c['schedule']}-{c['model']}-{cell}-r{rep}"


def phase_of(schedule: str, model: str) -> str | None:
    return next((p for p, c in PHASES.items() if c["schedule"] == schedule and c["model"] == model), None)


def run_line(phase: str, cell: str, rep) -> str:
    c = PHASES[phase]
    extra = " --expect-rejected" if c["expect_rejected"] else ""
    return f"{c['schedule']} {c['model']} {cell} {rep} {c['short']} {c['long']} {c['seconds']:g}{extra}"


def _pct(x, q):
    x = np.asarray(x, float)
    return float(np.percentile(x, q)) if len(x) else None


def _qs(x) -> dict:
    return {k: _pct(x, q) for k, q in QS}


def _longest(flags) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def _gate(value, limit, op: str) -> dict:
    """Exact comparison; an unmeasurable value fails."""
    if value is None:
        ok = False
    elif op == "<=":
        ok = value <= limit
    elif op == ">=":
        ok = value >= limit
    elif op == "<":
        ok = value < limit
    else:
        ok = value == limit
    return {"value": value, "limit": limit, "op": op, "pass": bool(ok)}


# ------------------------------------------------------------------ one run


def trace(run: dict) -> dict:
    cols = run.get("trace_columns") or prod_run.TRACE_COLUMNS
    rows = run.get("trace") or []
    out = {"target": np.asarray([str(r[cols.index("target")]) for r in rows], dtype=object)}
    for c in prod_run.TRACE_COLUMNS[1:]:
        out[c] = np.asarray([r[cols.index(c)] for r in rows], np.int64)
    order = np.argsort(out["submit_ns"], kind="stable")
    return {k: v[order] for k, v in out.items()}


def attribute(windows: list[dict], decisions: list) -> tuple[list[list], list]:
    """Decisions per window: [start_i, start_{i+1}); the last window to the end. Also the strays."""
    per = [[] for _ in windows]
    stray = []
    starts = [w["start_ns"] for w in windows]
    for d in sorted(decisions, key=lambda r: r[0]):
        i = int(np.searchsorted(starts, d[0], side="right")) - 1
        (stray if i < 0 else per[i]).append(list(d))
    return per, stray


def structure(dec: list) -> dict:
    """H2 on one P hetero window's decisions (in time order): zero or more leading armed / sync
    decisions (the short request arriving before the GPU job starts), then exactly 64 guard
    decisions with counts 1..64 (sync_guard 1..63, the 64th reported as async_steady, all sync),
    then async only."""
    lead = 0
    while lead < len(dec) and [dec[lead][1], dec[lead][2], dec[lead][3]] == [0, "armed", 0]:
        lead += 1
    h = next((i for i, d in enumerate(dec) if d[1] == 1), None)
    guard = dec[lead : lead + GUARD]
    after = [] if h is None else dec[h + 1 :]
    want = [[0, "async_steady" if k == GUARD else "sync_guard", k] for k in range(1, GUARD + 1)]
    first = guard[0] if guard else None
    checks = {
        "guard_starts_sync_guard_1": first is not None and [first[1], first[2], first[3]] == [0, "sync_guard", 1],
        "t_h_exists": h is not None,
        "guard_sequence_1_to_64": [[d[1], d[2], d[3]] for d in guard] == want,
        "t_h_follows_the_64th": h is not None and h == lead + GUARD,
        "no_sync_after_t_h": h is not None and not any(d[1] == 0 for d in after),
    }
    return {
        "t_h_ns": None if h is None else int(dec[h][0]),
        "leading_armed": lead,
        "first_decision": None if not dec else {"path": dec[0][1], "state": dec[0][2], "count": dec[0][3]},
        "sync_before_t_h": None if h is None else sum(d[1] == 0 for d in dec[lead:h]),
        "sync_after_t_h": sum(d[1] == 0 for d in after),
        "decisions": len(dec),
        "checks": checks,
        "failed": [k for k, v in checks.items() if not v],
    }


def _bins(sub, flag_fn, lo, end, width_s) -> list:
    """Per bin of `width_s` over [lo, end): None (empty) or flag_fn(mask)."""
    if lo is None or end <= lo:
        return []
    w = int(width_s * S)
    out = []
    for k in range(int(np.ceil((end - lo) / w))):
        a, b = lo + k * w, min(lo + (k + 1) * w, end)
        m = (sub >= a) & (sub < b)
        out.append({"full": b - a == w, "flag": flag_fn(m) if m.any() else None})
    return out


def episode(tr: dict, w: dict, dec: list, cell: str, prev: dict | None) -> dict:
    t0, end = w["start_ns"], w["end_ns"]
    sub_all = tr["submit_ns"]
    ane = (tr["target"] == "ane") & (sub_all >= t0) & (sub_all < end)
    sub = sub_all[ane]
    prep = (tr["prepared_ns"][ane] - sub) / 1e6
    short = w["streams"].get("short") or {}
    client = np.asarray(short.get("latency_ms") or [], float)
    if len(client) == len(sub):
        lat, source = client, "client"
    else:
        lat, source = (tr["response_ns"][ane] - sub) / 1e6, "trace"
    st = structure(dec) if cell == "P" else None
    th = st["t_h_ns"] if st else None
    lo = th if cell == "P" else t0 + int(A_FROM_S * S)
    g_lo = None if lo is None else (th + int(GPU_FROM_S * S) if cell == "P" else lo)
    if g_lo is None:
        ret = np.zeros(0)
    else:
        g = (tr["target"] == "gpu") & (tr["received_ns"] >= g_lo) & (tr["received_ns"] < end)
        ret = (tr["received_ns"][g] - tr["service_end_ns"][g]) / 1e6
    host = _bins(sub, lambda m: bool((prep[m] > HOST_SLOW_MS).mean() >= BIN_SHARE), lo, end, BIN_S)
    host_flags = [bool(b["flag"]) for b in host]
    return {
        "index": w["index"],
        "cycle": w.get("cycle"),
        "preceded_by": None if prev is None else prev["condition"],
        "t_h_s": None if th is None else (th - t0) / S,
        "structure": st,
        "n_short": int(len(client)),
        "n_short_trace": int(len(sub)),
        "latency_source": source,
        "short_ms": _qs(client),
        "req_s": {s: x.get("req_s") for s, x in w["streams"].items()},
        "aggregate_req_s": float(sum(x.get("req_s") or 0.0 for x in w["streams"].values())),
        "gpu_return_ms": {
            "from_s": None if g_lo is None else (g_lo - t0) / S,
            "n": int(len(ret)),
            **{f"p{q}": _pct(ret, q) for q in (50, 95, 99)},
        },  # fmt: skip
        "host_slow": {
            "from_s": None if lo is None else (lo - t0) / S,
            "bins": len(host),
            "slow_bins": int(sum(host_flags)),
            "longest": _longest(host_flags) if lo is not None else None,
        },
        "_sub": sub,
        "_lat": lat,
        "_client": client,
        "_lo": lo,
        "_end": end,
        "_ret": ret,
    }


def slow_spans(e: dict, a_median: float | None) -> dict:
    if a_median is None or e["_lo"] is None:
        return {"spans": None, "slow": None, "longest": None}
    lat = e["_lat"]
    bins = _bins(e["_sub"], lambda m: bool(np.median(lat[m]) > SLOW_RATIO * a_median), e["_lo"], e["_end"], SPAN_S)
    flags = [b["flag"] if b["flag"] is not None else b["full"] for b in bins]
    return {"spans": len(bins), "slow": int(sum(flags)), "longest": _longest(flags)}


def _snapshot_checks(snap) -> list[str]:
    if not isinstance(snap, dict):
        return ["no handoff snapshot"]
    bad = []
    if snap.get("disabled") or snap.get("enabled") is not True or snap.get("disabled_reason") is not None:
        bad.append(f"handoff disabled at the end ({snap.get('disabled_reason')})")
    if snap.get("consistent") is not True:
        bad.append("handoff snapshot not consistent")
    return bad


def run_stats(run: dict, name: str, phase: str, cell: str, logs: list[str]) -> dict:
    cfg = PHASES[phase]
    windows = run.get("windows") or []
    args = run.get("args") or {}
    sched = prod_run.schedule(cfg["schedule"], cfg["seconds"])
    snap = run.get("handoff_snapshot")
    infos = [x for x in (run.get("info_start"), run.get("info_end")) if isinstance(x, dict)]
    rt = run.get("runtime") or {}
    runtime = {
        "tree": rt.get("laya_apple_tree") == TREE,
        "pyproject": rt.get("pyproject_blob") == PYPROJECT_BLOB,
        "uv_lock": rt.get("uv_lock_blob") == UV_LOCK_BLOB,
        "clean": rt.get("dirty") is False,
    }
    protocol = {
        "cell": args.get("cell") == cell,
        "model": args.get("model") == cfg["model"],
        "lengths": args.get("short") == cfg["short"] and args.get("long") == cfg["long"],
        "schedule": args.get("schedule") == cfg["schedule"],
        "seconds": cfg["schedule"] != "mix" or args.get("seconds") == cfg["seconds"],
        "expect_rejected": bool(args.get("expect_rejected")) == cfg["expect_rejected"],
        "windows": [w["condition"] for w in windows] == [x["condition"] for x in sched],
        "window_lengths": len(windows) == len(sched)
        and all(abs(w["end_ns"] - w["start_ns"] - x["seconds"] * S) <= 1e6 for w, x in zip(windows, sched)),
    }
    if cell == "P":
        protocol["guard"] = isinstance(snap, dict) and snap.get("guard") == GUARD and snap.get("gap_s") == GAP_S
    else:  # the A side: 1.4's default path, no handoff at all
        protocol["a_default_path"] = (
            snap is None and not run.get("decisions") and len(infos) == 2 and all("ane_handoff" not in i for i in infos)
        )
    mismatches = sum(int(s.get("mismatches", 0)) for w in windows for s in w["streams"].values())
    routing = []
    for w in windows:
        want = ROUTES.get(w["condition"], {})
        for s in sorted(set(want) | set(w["streams"])):
            got = set((w["streams"].get(s) or {}).get("devices") or {})
            if s not in want or got != want[s]:
                routing.append(f"w{w['index']}:{w['condition']}:{s}:{sorted(got)}")
    alive = run.get("workers_alive_at_end")
    workers_alive = isinstance(alive, dict) and all(bool(alive.get(k)) for k in ("ane", "gpu"))
    crashes = [x for x in logs if x.startswith(name + ".")]
    tr = trace(run)
    per, stray = attribute(windows, run.get("decisions") or [])
    h1 = []
    if mismatches:
        h1.append(f"{mismatches} mismatches")
    if routing:
        h1.append("routing: " + ", ".join(routing))
    if crashes:
        h1.append("crash: " + ", ".join(crashes))
    if not workers_alive:
        h1.append(f"workers not alive at the end: {alive}")
    h2 = []
    eps = []
    for i, w in enumerate(windows):
        if w["condition"] == "hetero":
            e = episode(tr, w, per[i], cell, windows[i - 1] if i else None)
            eps.append(e)
            if cell == "P" and e["structure"]["failed"]:
                h2.append(f"w{w['index']}: " + ", ".join(e["structure"]["failed"]))
        elif cell == "P" and w["condition"] == "solo_short":
            bad = [d for d in per[i] if d[1] != 0 or d[2] != "armed"]
            if bad:
                h2.append(f"w{w['index']} solo_short: {len(bad)} decisions not armed / sync")
        elif cell == "P" and per[i]:
            h2.append(f"w{w['index']} {w['condition']}: {len(per[i])} decisions without a short stream")
    if cell == "P":
        h1 += _snapshot_checks(snap)
        if stray:
            h2.append(f"{len(stray)} decisions before the first window")
        n_het = sum(w["condition"] == "hetero" for w in windows)
        if not isinstance(snap, dict) or snap.get("episodes") != n_het:
            h2.append(f"snapshot episodes {None if not isinstance(snap, dict) else snap.get('episodes')} != {n_het}")
    return {
        "run": name,
        "phase": phase,
        "cell": cell,
        "runtime": rt,
        "runtime_ok": all(runtime.values()),
        "protocol": protocol,
        "protocol_ok": all(protocol.values()),
        "mismatches": mismatches,
        "routing_failures": routing,
        "crash_logs": crashes,
        "workers_alive_at_end": alive,
        "workers_alive": workers_alive,
        "handoff_snapshot": snap,
        "info": {"start": run.get("info_start"), "end": run.get("info_end")},
        "expect_rejected": run.get("expect_rejected"),
        "decisions": len(run.get("decisions") or []),
        "H1": h1,
        "H2": h2,
        "episodes": eps,
        "run_wall_s": run.get("run_wall_s"),
    }


# ------------------------------------------------------------------ files


def load(p: Path):
    if not p.exists():
        return None
    if p.suffix == ".gz":
        with gzip.open(p, "rt") as fh:
            return json.load(fh)
    return json.loads(p.read_text())


def failed_logs(raw: Path) -> list[str]:
    d = raw / "failed"
    return sorted(p.name for p in d.glob("*.log")) if d.exists() else []


def present(raw: Path) -> tuple[set[str], list[str]]:
    names, other = set(), []
    for p in sorted(raw.glob("*.json.gz")) if raw.exists() else []:
        x = RUN_NAME.match(p.name)
        ph = x and phase_of(x["schedule"], x["model"])
        base = p.name[: -len(".json.gz")]
        if ph and (x["cell"], int(x["rep"])) in RUNS[ph]:
            names.add(base)
        else:
            other.append(p.name)
    return names, other


def machine(raw: Path, name: str) -> list[str]:
    return qual.snapshot_checks(load(raw / f"{name}.before.json"), load(raw / f"{name}.after.json"))


def _count_logs(logs: list[str], name: str) -> int:
    return sum(bool(re.fullmatch(re.escape(name) + r"\.\d+\.log", x)) for x in logs)


def slots(raw: Path, phase: str, names: set[str], logs: list[str]) -> list[dict]:
    """Per planned run: which file is used, machine findings, crash logs, and the run's state."""
    out = []
    for cell, rep in RUNS[phase]:
        base = run_name(phase, cell, rep)
        reb = base + "-b"
        s = {"cell": cell, "rep": rep, "run": base, "rerun": reb, "used": None, "superseded": None}
        s["crashes"] = {base: _count_logs(logs, base), reb: _count_logs(logs, reb)}
        s["machine"] = {n: machine(raw, n) for n in (base, reb) if n in names}
        if base not in names:
            s["state"] = "missing"
        elif not s["machine"][base]:
            s["used"], s["state"] = base, "ok"
        elif reb not in names:
            s["state"] = "machine rerun pending"
        else:
            s["used"], s["superseded"] = reb, base
            s["state"] = "ok" if not s["machine"][reb] else "machine failed twice"
        out.append(s)
    return out


def runs_lines(phase: str) -> list[str]:
    return [run_line(phase, c, r) for c, r in RUNS[phase]]


def reruns(raw: Path, phase: str) -> list[str]:
    names, _ = present(raw)
    sl = slots(raw, phase, names, failed_logs(raw))
    if any(s["state"] == "missing" for s in sl):
        return []
    return [run_line(phase, s["cell"], f"{s['rep']}-b") for s in sl if s["state"] == "machine rerun pending"]


# ------------------------------------------------------------------ phase judgement


def bootstrap(clusters) -> dict:
    """Median of all deltas; one-sided 95% upper bounds: cluster (resampled with replacement, each
    drawn cluster contributing all its deltas) and transition-level (report only)."""
    d = np.asarray(clusters, float)
    n = len(d)
    flat = d.ravel()
    idx = np.random.default_rng(SEED).integers(0, n, size=(B, n))
    cl = np.median(d[idx].reshape(B, -1), axis=1)
    idx2 = np.random.default_rng(SEED).integers(0, len(flat), size=(B, len(flat)))
    tr = np.median(flat[idx2], axis=1)
    return {
        "n_clusters": int(n),
        "cluster_size": int(d.shape[1]),
        "median_ms": float(np.median(flat)),
        "cluster_upper_ms": float(np.percentile(cl, UPPER_Q)),
        "transition_upper_ms": float(np.percentile(tr, UPPER_Q)),
    }


def _d(a, b):
    return None if a is None or b is None else a - b


def pair_episodes(p: dict, a: dict) -> list[dict]:
    ae = {e["index"]: e for e in a["episodes"]}
    out = []
    for e in p["episodes"]:
        x = ae.get(e["index"])
        out.append(
            {
                "index": e["index"],
                "preceded_by": e["preceded_by"],
                "p99": _d(e["short_ms"]["p99"], x and x["short_ms"]["p99"]),
                "median": _d(e["short_ms"]["median"], x and x["short_ms"]["median"]),
                "p95": _d(e["short_ms"]["p95"], x and x["short_ms"]["p95"]),
                "p999": _d(e["short_ms"]["p999"], x and x["short_ms"]["p999"]),
                "req_s": _d(e["aggregate_req_s"], x and x["aggregate_req_s"]),
                "req_s_ratio": None
                if not x or not x["aggregate_req_s"]
                else e["aggregate_req_s"] / x["aggregate_req_s"],
            }
        )
    return out


def latency(phase: str, pairs: list[dict]) -> dict:
    deltas = [[x["p99"] for x in p["deltas"]] for p in pairs]
    flat = [v for c in deltas for v in c]
    out = {"n": len(flat), "clusters": None, "pass": False}
    if not flat or any(v is None for v in flat) or len({len(c) for c in deltas}) != 1:
        out["reason"] = "unmeasurable"
        return out
    if phase == "5":
        clusters = [c[k : k + SOAK_GROUP] for c in deltas for k in range(0, len(c), SOAK_GROUP)]
        if any(len(c) != SOAK_GROUP for c in clusters):
            out["reason"] = "soak episodes are not whole 6-episode groups"
            return out
    else:
        clusters = deltas
    b = bootstrap(clusters)
    g = {
        "median": _gate(b["median_ms"], MEDIAN_MS, "<="),
        "cluster_upper": _gate(b["cluster_upper_ms"], UPPER_MS, "<="),
    }
    rep = {}
    for k in ("median", "p95", "p999", "req_s"):
        v = [x[k] for p in pairs for x in p["deltas"] if x[k] is not None]
        rep[k] = _pct(v, 50)
    out.update(clusters=b["n_clusters"], **b, gates=g, reported_median_deltas=rep)
    out["pass"] = all(x["pass"] for x in g.values())
    return out


def _strip(e: dict) -> dict:
    return {k: v for k, v in e.items() if not k.startswith("_")}


def judge_smoke(sl: list[dict], stats: dict) -> dict:
    s = sl[0]
    r = stats.get(s["used"]) if s["used"] else None
    if r is None:
        return {"pass": None, "checks": None}
    infos = [r["info"]["start"], r["info"]["end"]]
    rej = r["expect_rejected"] or {}
    checks = {
        "no_mismatch": r["mismatches"] == 0,
        "no_routing_failure": not r["routing_failures"],
        "no_crash": sum(s["crashes"].values()) == 0,
        "workers_alive": r["workers_alive"],
        "process_placement": all(isinstance(i, dict) and i.get("ane_placement") == "process" for i in infos),
        "no_ane_handoff_key": all(isinstance(i, dict) and "ane_handoff" not in i for i in infos),
        "no_handoff": r["handoff_snapshot"] is None and r["decisions"] == 0,
        "rejected_with_value_error": rej.get("raised") == "ValueError" and "ane_handoff" in (rej.get("message") or ""),
    }
    return {"pass": all(checks.values()), "checks": checks, "failed": [k for k, v in checks.items() if not v]}


def judge_phase(raw: Path, phase: str, names: set[str], logs: list[str]) -> dict:
    sl = slots(raw, phase, names, logs)
    stats = {}
    for s in sl:
        for n in (s["run"], s["rerun"]):
            if n in names:
                stats[n] = run_stats(load(raw / f"{n}.json.gz"), n, phase, s["cell"], logs)
    out = {"slots": sl, "runs": stats}
    invalid, pending, fail = [], [], []
    for n, r in stats.items():
        if not r["runtime_ok"]:
            invalid.append(f"{n}: runtime differs ({r['runtime']})")
        if not r["protocol_ok"]:
            invalid.append(f"{n}: protocol deviation ({', '.join(k for k, v in r['protocol'].items() if not v)})")
    for s in sl:
        if s["state"] == "machine failed twice":
            invalid.append(f"{s['run']}: machine snapshot failed twice ({'; '.join(s['machine'][s['rerun']])})")
        elif s["state"] == "machine rerun pending":
            pending.append(f"{s['rerun']} (machine: {'; '.join(s['machine'][s['run']])})")
        if s["cell"] == "A" and phase != "3":
            for n, k in s["crashes"].items():
                if k >= 2:
                    invalid.append(f"{n}: A crashed twice")
        if s["cell"] == "P" and sum(s["crashes"].values()):
            fail.append(f"H1 {s['run']}: crash ({sum(s['crashes'].values())} failed logs)")
        if s["state"] == "missing" and not (s["cell"] == "P" and s["crashes"][s["run"]]):
            pending.append(s["run"])
    if phase == "3":
        sm = judge_smoke(sl, stats)
        out["smoke"] = sm
        crashed = sum(sl[0]["crashes"].values())
        if sm["pass"] is False or crashed:
            fail.append("multilingual smoke: " + ", ".join(sm.get("failed") or ["no_crash"]))
        return _status(out, fail, invalid, pending, [])
    # self-contained P hard failures, on every present P run with the right runtime and protocol
    judged_p = [r for r in stats.values() if r["cell"] == "P" and r["runtime_ok"] and r["protocol_ok"]]
    for r in judged_p:
        fail += [f"H1 {r['run']}: {x}" for x in r["H1"] if not x.startswith("crash")]
        fail += [f"H2 {r['run']}: {x}" for x in r["H2"]]
        for e in r["episodes"]:
            p50 = e["gpu_return_ms"]["p50"]
            if not _gate(p50, GPU_P50_MS, "<=")["pass"]:
                fail.append(f"H3 {r['run']} w{e['index']}: GPU-return P50 {_fmt(p50, 3)} ms > {GPU_P50_MS}")
            hl = e["host_slow"]["longest"]
            if hl is None or hl >= CONSECUTIVE:
                fail.append(f"H5 {r['run']} w{e['index']}: {hl} consecutive host-slow bins")
    a_all = [r for r in stats.values() if r["cell"] == "A"]
    for r in a_all:
        if r["mismatches"] or r["routing_failures"]:
            invalid.append(f"{r['run']}: A has {r['mismatches']} mismatches, routing failures {r['routing_failures']}")
    used = {(s["cell"], s["rep"]): stats[s["used"]] for s in sl if s["used"] in stats}
    complete = len(used) == len(sl) and not pending
    late = full_phase(phase, out, used, judged_p) if complete else None
    return _status(out, fail, invalid, pending, late)


def full_phase(phase: str, out: dict, used: dict, judged_p: list[dict]) -> tuple[list[str], bool]:
    """Everything that needs the whole phase: A reference, H3 pooled, H4, H6, H7, latency."""
    reps = sorted({rep for _, rep in used})
    p_used = [used[("P", r)] for r in reps]
    a_used = [used[("A", r)] for r in reps]
    a_client = np.concatenate([np.zeros(0)] + [e["_client"] for r in a_used for e in r["episodes"]])
    a_median = _pct(a_client, 50)
    for r in {id(x): x for x in p_used + a_used + judged_p}.values():
        for e in r["episodes"]:
            e["slow_spans"] = slow_spans(e, a_median)
    fail = []
    # H6 on every judged P run (used and superseded)
    for r in judged_p:
        for e in r["episodes"]:
            ln = e["slow_spans"]["longest"]
            if ln is None or ln >= CONSECUTIVE:
                fail.append(f"H6 {r['run']} w{e['index']}: {ln} consecutive slow spans")
    # A signatures (validity)
    a_eps = [e for r in a_used for e in r["episodes"]]
    sig = [
        e
        for e in a_eps
        if (e["host_slow"]["longest"] or 0) >= CONSECUTIVE or (e["slow_spans"]["longest"] or 0) >= CONSECUTIVE
    ]
    share = len(sig) / len(a_eps) if a_eps else None
    out["A_signatures"] = {"episodes": len(a_eps), "with_H5_or_H6": len(sig), "share": share}
    # H3 pooled
    p_ret = np.concatenate([np.zeros(0)] + [e["_ret"] for r in p_used for e in r["episodes"]])
    a_ret = np.concatenate([np.zeros(0)] + [e["_ret"] for r in a_used for e in r["episodes"]])
    gpu = {c: {f"p{q}": _pct(x, q) for q in (50, 95, 99)} | {"n": int(len(x))} for c, x in (("P", p_ret), ("A", a_ret))}
    out["gpu_return_ms"] = gpu
    h3p = _gate(gpu["P"]["p99"], GPU_P99_MS, "<=")
    if not h3p["pass"]:
        fail.append(f"H3 pooled P GPU-return P99 {_fmt(gpu['P']['p99'], 3)} ms > {GPU_P99_MS}")
    p50 = _gate(gpu["P"]["p50"], GPU_P50_MS, "<=")
    if not p50["pass"]:
        fail.append(f"pooled P GPU-return P50 {_fmt(gpu['P']['p50'], 3)} ms > {GPU_P50_MS}")
    # pooled latency per cell
    out["pooled_short_ms"] = {
        "P": _qs(np.concatenate([np.zeros(0)] + [e["_client"] for r in p_used for e in r["episodes"]])),
        "A": _qs(a_client),
    }
    out["A_pooled_short_median_ms"] = a_median
    # H4 and the pairs
    pairs = []
    for rep in reps:
        p, a = used[("P", rep)], used[("A", rep)]
        pa = float(np.mean([e["aggregate_req_s"] for e in p["episodes"]])) if p["episodes"] else None
        aa = float(np.mean([e["aggregate_req_s"] for e in a["episodes"]])) if a["episodes"] else None
        g = _gate(pa, None if aa is None else THROUGHPUT_RATIO * aa, ">=")
        pairs.append(
            {
                "P": p["run"],
                "A": a["run"],
                "P_aggregate_req_s": pa,
                "A_aggregate_req_s": aa,
                "ratio": None if not aa or pa is None else pa / aa,
                "H4": g,
                "deltas": pair_episodes(p, a),
            }
        )
        if not g["pass"]:
            fail.append(f"H4 {p['run']} / {a['run']}: pooled aggregate {_fmt(pa, 1)} < 0.95 x {_fmt(aa, 1)}")
    out["pairs"] = pairs
    # H7 and the tail events
    a_p99 = [e["short_ms"]["p99"] for e in a_eps]
    ref = _pct([v for v in a_p99 if v is not None], 50) if a_p99 and None not in a_p99 else None
    pair_of = {}
    for rep in reps:
        p, a = used[("P", rep)], used[("A", rep)]
        ae = {e["index"]: e for e in a["episodes"]}
        pe = {e["index"]: e for e in p["episodes"]}
        for i, e in pe.items():
            pair_of[id(e)] = ae.get(i)
        for i, e in ae.items():
            pair_of[id(e)] = pe.get(i)
    tails = []
    for cell, runs in (("P", p_used), ("A", a_used)):
        for r in runs:
            for e in r["episodes"]:
                v = e["short_ms"]["p99"]
                if ref is None or (v is not None and v - ref <= TAIL_MS):
                    continue
                tails.append(
                    {"run": r["run"], "cell": cell, "index": e["index"], "p99_ms": v, **_label(e, pair_of.get(id(e)))}
                )
    n_p = sum(t["cell"] == "P" for t in tails)
    n_a = sum(t["cell"] == "A" for t in tails)
    lim = max(TAIL_MIN, TAIL_FACTOR * n_a)
    out["tails"] = {"threshold_ms": None if ref is None else ref + TAIL_MS, "A_median_episode_p99_ms": ref,
                    "events": tails, "P": n_p, "A": n_a, "limit": lim}  # fmt: skip
    if ref is None:
        fail.append("H7: the A episode P99 reference cannot be computed")
    elif n_p > lim:
        fail.append(f"H7: {n_p} P tail events > max(2, 2 x {n_a})")
    # latency non-inferiority
    lat = latency(phase, pairs)
    out["latency"] = lat
    if not lat["pass"]:
        fail.append(
            "latency non-inferiority: "
            + (
                lat.get("reason")
                or f"median {lat['median_ms']:+.3f} ms, cluster upper {lat['cluster_upper_ms']:+.3f} ms"
            )
        )
    n_p_eps = sum(len(r["episodes"]) for r in p_used)
    out["P_episodes"] = n_p_eps
    out["P_episodes_expected"] = len(p_used) * sum(
        x["condition"] == "hetero" for x in prod_run.schedule(PHASES[phase]["schedule"], PHASES[phase]["seconds"])
    )
    if phase == "5" and n_p_eps < SOAK_MIN_P:
        fail.append(f"soak: {n_p_eps} P episodes < {SOAK_MIN_P}")
    out["hard_gates"] = {
        "H3_episode_max_p50_ms": _max([e["gpu_return_ms"]["p50"] for r in judged_p for e in r["episodes"]]),
        "H3_pooled_p99": h3p,
        "H4_min_ratio": _min([p["ratio"] for p in pairs]),
        "H5_longest_bins": _max([e["host_slow"]["longest"] for r in judged_p for e in r["episodes"]]),
        "H6_longest_spans": _max([e["slow_spans"]["longest"] for r in judged_p for e in r["episodes"]]),
        "H7": {"P": n_p, "A": n_a, "limit": lim},
        "pooled_P_gpu_p50": p50,
    }
    return fail, (share is not None and share > A_SIGNATURE_SHARE)


def _max(v):
    v = [x for x in v if x is not None]
    return max(v) if v else None


def _min(v):
    v = [x for x in v if x is not None]
    return min(v) if v else None


def _label(e: dict, x: dict | None) -> dict:
    if x is None:
        return {"label": "tail with other symptoms", "why": ["no pair"]}
    why = []
    for k in ("p95", "median"):
        d = _d(e["short_ms"][k], x["short_ms"][k])
        if d is None or d > ISOLATED_MS:
            why.append(f"{k} {_fmt(d, 3, sign=True)} ms vs pair")
    if not (x["aggregate_req_s"] and e["aggregate_req_s"] >= THROUGHPUT_RATIO * x["aggregate_req_s"]):
        why.append(f"req/s {_fmt(e['aggregate_req_s'], 1)} < 0.95 x {_fmt(x['aggregate_req_s'], 1)}")
    p50 = e["gpu_return_ms"]["p50"]
    if p50 is None or p50 > GPU_P50_MS:
        why.append(f"GPU-return P50 {_fmt(p50, 3)} ms")
    if e["host_slow"]["slow_bins"] or e["host_slow"]["longest"] is None:
        why.append(f"{e['host_slow']['slow_bins']} host-slow bins")
    sp = e.get("slow_spans") or {}
    if sp.get("slow") is None or sp["slow"]:
        why.append(f"{sp.get('slow')} slow spans")
    return {"label": "isolated tail" if not why else "tail with other symptoms", "why": why}


def _status(out: dict, fail: list, invalid: list, pending: list, late) -> dict:
    late_fail, a_sig = late if late else ([], False)
    if a_sig:
        invalid.append(f"A has H5 / H6 signatures in {out['A_signatures']['share']:.0%} of its episodes (> 10%)")
    out.update(fail=fail + late_fail, invalid=invalid, pending=pending)
    if fail:
        out.update(status="FAIL", reasons=fail)
    elif invalid:
        out.update(status="INVALID", reasons=invalid)
    elif pending:
        out.update(status="pending", reasons=pending)
    elif late_fail:
        out.update(status="FAIL", reasons=late_fail)
    else:
        out.update(status="PASS", reasons=[])
    return out


# ------------------------------------------------------------------ summary


def summarise(raw: Path) -> dict:
    names, other = present(raw)
    logs = failed_logs(raw)
    phases = {}
    blocker = None
    outcome = None
    for ph in ORDER:
        j = judge_phase(raw, ph, names, logs)
        if blocker:
            j.update(judged_status=j["status"], status="not reached")
        phases[ph] = j
        if blocker or j["status"] == "PASS":
            continue
        blocker = ph
        why = "; ".join(j["reasons"])
        if j["status"] == "FAIL":
            outcome = f"CLOSE: phase {ph} failed: {why}. H64 is closed and 1.4 stays."
        elif j["status"] == "INVALID":
            outcome = f"INVALID: phase {ph}: {why}"
        else:
            outcome = f"phase {ph} pending: {why}"
    for j in phases.values():
        for r in j["runs"].values():
            r["episodes"] = [_strip(e) for e in r["episodes"]]
    return {
        "design": {
            "runtime_tree": TREE,
            "guard": GUARD,
            "gap_s": GAP_S,
            "phases": PHASES,
            "runs": {ph: [f"{c}{r}" for c, r in v] for ph, v in RUNS.items()},
            "host_slow": {"prepare_ms": HOST_SLOW_MS, "bin_s": BIN_S, "share": BIN_SHARE},
            "slow_span": {"span_s": SPAN_S, "ratio": SLOW_RATIO},
            "gpu_return": {
                "from_t_h_s": GPU_FROM_S,
                "A_from_t0_s": A_FROM_S,
                "p50_ms": GPU_P50_MS,
                "p99_ms": GPU_P99_MS,
            },
            "throughput_ratio": THROUGHPUT_RATIO,
            "tail": {"ms": TAIL_MS, "min": TAIL_MIN, "factor": TAIL_FACTOR, "isolated_ms": ISOLATED_MS},
            "latency": {"median_ms": MEDIAN_MS, "upper_ms": UPPER_MS, "B": B, "seed": SEED, "q": UPPER_Q},
            "soak": {"group": SOAK_GROUP, "min_P_episodes": SOAK_MIN_P},
            "A_signature_share": A_SIGNATURE_SHARE,
        },
        "failed_logs": logs,
        "unexpected_files": other,
        "phases": phases,
        "outcome": outcome or CANDIDATE,
    }


# ------------------------------------------------------------------ tables


def _fmt(x, d=2, sign=False):
    if x is None:
        return "–"
    if isinstance(x, bool):
        return str(x)
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def tables(res: dict) -> str:
    L = ["# H64 production evaluation (evaluation.md)", "", f"Outcome: {res['outcome']}", ""]
    L.append(
        f"Runtime under test: laya_apple tree {TREE}, pyproject {PYPROJECT_BLOB}, uv.lock {UV_LOCK_BLOB}, clean. Failed logs: {', '.join(res['failed_logs']) or 'none'}"
    )
    if res["unexpected_files"]:
        L.append(f"Files outside the plan (not judged): {', '.join(res['unexpected_files'])}")
    L += ["", "| phase | status | reasons |", "|---|---|---|"]
    for ph, j in res["phases"].items():
        st = j["status"] + (f" (would be {j['judged_status']})" if "judged_status" in j else "")
        L.append(f"| {ph} | {st} | {'; '.join(j['reasons']) or '–'} |")
    for ph, j in res["phases"].items():
        c = PHASES[ph]
        L += ["", f"## Phase {ph}: {c['model']}, L{c['short']} / L{c['long']}, {c['schedule']}", ""]
        L.append(f"Status: **{j['status']}**")
        for k in ("fail", "invalid", "pending"):
            if j[k]:
                L.append(f"- {k}: " + "; ".join(j[k]))
        L += ["", "### Runs and validity", ""]
        L += ["| run | used | runtime ok | protocol ok | machine | crash logs | mismatches | routing | workers alive "
              "| H1 | H2 |", "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for s in j["slots"]:
            for n in (s["run"], s["rerun"]):
                r = j["runs"].get(n)
                if r is None:
                    if n == s["run"]:
                        L.append(f"| {n} | {s['state']} | – | – | – | {s['crashes'][n]} | – | – | – | – | – |")
                    continue
                mach = "; ".join(s["machine"].get(n) or []) or "ok"
                bad_p = ", ".join(k for k, v in r["protocol"].items() if not v)
                L.append(
                    f"| {n} | {'yes' if s['used'] == n else ('superseded' if s['superseded'] == n else s['state'])} "
                    f"| {r['runtime_ok']} | {r['protocol_ok']}{' (' + bad_p + ')' if bad_p else ''} | {mach} "
                    f"| {len(r['crash_logs'])} | {r['mismatches']} | {len(r['routing_failures'])} | {r['workers_alive']} "
                    f"| {'; '.join(r['H1']) or '–'} | {'; '.join(r['H2']) or '–'} |"
                )
        if "smoke" in j:
            sm = j["smoke"]
            L += ["", "### Multilingual smoke", ""]
            if sm["checks"] is None:
                L.append("pending")
            else:
                L += ["| check | pass |", "|---|---|"] + [f"| {k} | {v} |" for k, v in sm["checks"].items()]
                L.append(f"\nSmoke passes: **{sm['pass']}**")
            continue
        if "hard_gates" in j:
            hg = j["hard_gates"]
            L += ["", "### Hard gates (P)", "", "| gate | value | limit | pass |", "|---|---|---|---|"]
            L.append(f"| H1 correctness | {sum(1 for x in j['fail'] if x.startswith('H1'))} findings | 0 | "
                     f"{not any(x.startswith('H1') for x in j['fail'])} |")  # fmt: skip
            L.append(f"| H2 handoff state | {sum(1 for x in j['fail'] if x.startswith('H2'))} findings | 0 | "
                     f"{not any(x.startswith('H2') for x in j['fail'])} |")  # fmt: skip
            L.append(f"| H3 episode GPU-return P50 (max) | {_fmt(hg['H3_episode_max_p50_ms'], 3)} | <= {GPU_P50_MS} "
                     f"| {not any(x.startswith('H3') and ' w' in x for x in j['fail'])} |")  # fmt: skip
            g = hg["H3_pooled_p99"]
            L.append(f"| H3 pooled GPU-return P99 | {_fmt(g['value'], 3)} | <= {GPU_P99_MS} | {g['pass']} |")
            L.append(f"| H4 min pooled aggregate ratio | {_fmt(hg['H4_min_ratio'], 3)} | >= {THROUGHPUT_RATIO} "
                     f"| {not any(x.startswith('H4') for x in j['fail'])} |")  # fmt: skip
            L.append(f"| H5 longest host-slow bins | {hg['H5_longest_bins']} | < {CONSECUTIVE} "
                     f"| {not any(x.startswith('H5') for x in j['fail'])} |")  # fmt: skip
            L.append(f"| H6 longest slow spans | {hg['H6_longest_spans']} | < {CONSECUTIVE} "
                     f"| {not any(x.startswith('H6') for x in j['fail'])} |")  # fmt: skip
            h7 = hg["H7"]
            L.append(f"| H7 P tail events | {h7['P']} (A {h7['A']}) | <= {h7['limit']} | {h7['P'] <= h7['limit']} |")
            g = hg["pooled_P_gpu_p50"]
            L.append(f"| pooled P GPU-return P50 (outcome) | {_fmt(g['value'], 3)} | <= {GPU_P50_MS} | {g['pass']} |")
            L.append(f"\nP episodes: {j['P_episodes']} of {j['P_episodes_expected']} expected. A signatures "
                     f"(H5 / H6): {j['A_signatures']['with_H5_or_H6']} of {j['A_signatures']['episodes']} A episodes.")  # fmt: skip
        if "latency" in j:
            lat = j["latency"]
            L += ["", "### Latency non-inferiority (delta P99 = P episode - paired A episode)", ""]
            if lat.get("gates"):
                L.append(
                    f"- median delta: {lat['median_ms']:+.3f} ms (<= +{MEDIAN_MS}): {lat['gates']['median']['pass']}"
                )
                L.append(f"- cluster bootstrap one-sided 95% upper bound ({lat['clusters']} clusters of "
                         f"{lat['cluster_size']}, B = {B}, seed {SEED}): {lat['cluster_upper_ms']:+.3f} ms "
                         f"(<= +{UPPER_MS}): {lat['gates']['cluster_upper']['pass']}")  # fmt: skip
                L.append(f"- transition-level upper bound (report only): {lat['transition_upper_ms']:+.3f} ms")
                rp = lat["reported_median_deltas"]
                L.append(f"- median deltas (report only): episode median {_fmt(rp['median'], 3, True)}, P95 "
                         f"{_fmt(rp['p95'], 3, True)}, P99.9 {_fmt(rp['p999'], 3, True)} ms, "
                         f"req/s {_fmt(rp['req_s'], 2, True)}")  # fmt: skip
            else:
                L.append(f"- not computable: {lat.get('reason')}")
            L.append(f"- pass: **{lat['pass']}**")
        if "pooled_short_ms" in j:
            L += ["", "### Pooled per cell", "", "| cell | median | P95 | P99 | P99.9 | GPU return P50 / P95 / P99 (n) |",
                  "|---|---|---|---|---|---|"]  # fmt: skip
            for cell in ("P", "A"):
                s, g = j["pooled_short_ms"][cell], j["gpu_return_ms"][cell]
                L.append(f"| {cell} | {_fmt(s['median'])} | {_fmt(s['p95'])} | {_fmt(s['p99'])} | {_fmt(s['p999'])} "
                         f"| {_fmt(g['p50'], 3)} / {_fmt(g['p95'], 3)} / {_fmt(g['p99'], 3)} ({g['n']}) |")  # fmt: skip
            L.append(
                f"\nA pooled hetero short median (slow-span reference): {_fmt(j['A_pooled_short_median_ms'], 3)} ms"
            )
        if "pairs" in j:
            L += ["", "### Throughput per run pair", "", "| P | A | P agg req/s | A agg req/s | ratio | episode ratios |",
                  "|---|---|---|---|---|---|"]  # fmt: skip
            for p in j["pairs"]:
                er = ", ".join(_fmt(x["req_s_ratio"], 3) for x in p["deltas"])
                L.append(f"| {p['P']} | {p['A']} | {_fmt(p['P_aggregate_req_s'], 1)} | {_fmt(p['A_aggregate_req_s'], 1)} "
                         f"| {_fmt(p['ratio'], 3)} | {er} |")  # fmt: skip
        if "tails" in j:
            t = j["tails"]
            L += ["", f"### Tail events (episode P99 > A median episode P99 {_fmt(t['A_median_episode_p99_ms'])} + "
                  f"{TAIL_MS} ms)", ""]  # fmt: skip
            if not t["events"]:
                L.append("none")
            for x in t["events"]:
                L.append(f"- {x['run']} w{x['index']} ({x['cell']}): P99 {_fmt(x['p99_ms'])} ms, **{x['label']}**"
                         + (f" ({'; '.join(x['why'])})" if x["why"] else ""))  # fmt: skip
        eps = [(r["run"], e) for r in j["runs"].values() for e in r["episodes"]]
        if eps:
            L += ["", "### Episodes", "",
                  "| run | window | after | t_h s | median | P95 | P99 | P99.9 | agg req/s | GPU return P50 / P95 / P99 "
                  "| host-slow bins (longest) | slow spans (longest) | latency source |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
            for n, e in eps:
                s, g, sp = e["short_ms"], e["gpu_return_ms"], e.get("slow_spans") or {}
                L.append(
                    f"| {n} | {e['index']} | {e['preceded_by']} | {_fmt(e['t_h_s'], 3)} | {_fmt(s['median'])} "
                    f"| {_fmt(s['p95'])} | {_fmt(s['p99'])} | {_fmt(s['p999'])} | {_fmt(e['aggregate_req_s'], 1)} "
                    f"| {_fmt(g['p50'], 3)} / {_fmt(g['p95'], 3)} / {_fmt(g['p99'], 3)} "
                    f"| {e['host_slow']['slow_bins']} ({e['host_slow']['longest']}) "
                    f"| {sp.get('slow', '–')} ({sp.get('longest', '–')}) | {e['latency_source']} |"
                )
    return "\n".join(L) + "\n"


def _json(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", nargs="?", choices=("runs", "reruns", "ready"))
    ap.add_argument("phase", nargs="?", choices=ORDER)
    ap.add_argument("--check", action="store_true", help="fail if eval_results.json / eval_tables.md are stale")
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=EXP)
    a = ap.parse_args(argv)
    if a.cmd and not a.phase:
        ap.error(f"{a.cmd} needs a phase")
    if a.cmd == "runs":
        print("\n".join(runs_lines(a.phase)))
        return 0
    if a.cmd == "reruns":
        for x in reruns(a.raw, a.phase):
            print(x)
        return 0
    res = summarise(a.raw)
    if a.cmd == "ready":
        before = [ph for ph in ORDER[: ORDER.index(a.phase)] if res["phases"][ph]["status"] != "PASS"]
        if before:
            print(
                f"phase {a.phase} is not reached: "
                + ", ".join(f"phase {p} {res['phases'][p]['status']}" for p in before)
            )
            return 1
        own = res["phases"][a.phase]
        own = own.get("judged_status", own["status"])
        crashes = [x for x in res["failed_logs"] for rep in ("", "-b") for _, r in RUNS[a.phase]
                   if re.fullmatch(re.escape(run_name(a.phase, "P", f"{r}{rep}")) + r"\.\d+\.log", x)]  # fmt: skip
        if own in ("FAIL", "INVALID") or crashes:
            print(
                f"phase {a.phase} is already {own}"
                + (f"; P crash logs: {', '.join(sorted(set(crashes)))}" if crashes else "")
            )
            return 1
        return 0
    js = json.dumps(res, indent=1, sort_keys=True, default=_json) + "\n"
    md = tables(res)
    targets = ((a.out / "eval_results.json", js), (a.out / "eval_tables.md", md))
    if a.check:
        stale = [p.name for p, s in targets if not p.exists() or p.read_text() != s]
        if stale:
            print(f"stale: {stale}")
            return 1
        print("outputs are up to date")
        return 0
    for p, s in targets:
        p.write_text(s)
    print(res["outcome"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
