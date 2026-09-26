"""Phase 0: replay candidate slow-state detectors on the recorded PB-ASYNC episodes.

    uv run python research/coreml-adaptive-breaker/scripts/replay.py [--check]

No new measurement. Inputs are the raw runs of #92, #96, #99, #102 and #103, read in place:
- research/coreml-async-predict/raw (#92), research/coreml-async-transient/raw (#96),
  research/coreml-dependency-qos/raw (#99): #94's harness format (research.trace);
- research/coreml-staged-handoff/raw and raw-conf (#102): the same format, with the handoff's
  decisions in research.handoff;
- research/coreml-staged-handoff/raw-eval and raw-qual (#103): prod_run.py's format.

Episodes: every hetero window of the auto instance. The async span starts at t_a:
- PB-ASYNC, B, O and Q: t_a = t0, the window start (async from the first request);
- H32, H64 and P: the first async decision in [t0, end);
- A (and PB-SYNC): no async span; A is replayed from t0 + 1 s as a reference only.

Signals, all taken from RequestTrace fields the runtime already records (laya_apple/trace.py),
over the ANE-routed requests submitted in [t_a, end), in submit order:
- prepare_ms = prepared - submit (Laya.prepare: tokenising and layout on the calling thread). A
  request is host-slow when prepare_ms > 0.3 ms (#94's rule). The value is known before routing,
  so a detector sees it at prepared_ns, before the request is dispatched;
- e2e_ms = response - submit, used only for the ground truth below, never by a detector.

Ground truth (user-facing latency; it never reads prepare_ms):
- m_A: the pooled median e2e of the ANE requests in the same study's A hetero windows (each A
  window from t0 + 1 s). #99 had no A cell, so it uses #96's A (the same harness, one day apart);
- slow span: a 1 s span from t_a whose median e2e is > 1.2 x m_A (#103's H6 rule);
- sustained episode: 2 or more consecutive slow spans (#103's H6);
- onset: the first span of the first such run, extended back in 0.1 s steps while the 1 s window
  starting 0.1 s earlier is still slow (and starts at or after t_a): the earliest second of traffic
  whose median is already slow;
- healthy episode: an async episode that is not sustained.

Detectors (each trips at most once per episode, at the prepared_ns of the request that trips it):
- C2 / C3: 2 / 3 consecutive host-slow requests;
- R3of5 / R4of8: 3 of the last 5 / 4 of the last 8 requests host-slow;
- EWMA: prepare EWMA (alpha 0.25, starting at the first value) > 0.3 ms;
- MED5: the median of the last 5 prepare values > 0.3 ms.

Per detector: recall over sustained episodes, detection delay (trip - onset, ms; negative means it
tripped before the first slow second began) and in requests, false trips over healthy episodes,
and the trip time in the #102 confirmation's H64 r8 cycle 1 (the isolated P99 tail).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESEARCH = ROOT.parent
OUT_JSON = ROOT / "replay.json"
OUT_MD = ROOT / "replay.md"

HOST_SLOW_MS = 0.3
SPAN_S = 1.0
SLOW_RATIO = 1.2
STEP_S = 0.1
A_SKIP_S = 1.0

# study -> (issue, [raw dirs], A-reference study)
STUDIES = {
    "async-predict": ("#92", ["coreml-async-predict/raw"], "async-predict"),
    "async-transient": ("#96", ["coreml-async-transient/raw"], "async-transient"),
    "dependency-qos": ("#99", ["coreml-dependency-qos/raw"], "async-transient"),
    "handoff-screen": ("#102", ["coreml-staged-handoff/raw"], "handoff-screen"),
    "handoff-confirm": ("#102", ["coreml-staged-handoff/raw-conf"], "handoff-confirm"),
    "production-eval": (
        "#103",
        ["coreml-staged-handoff/raw-eval", "coreml-staged-handoff/raw-qual"],
        "production-eval",
    ),
}
ASYNC_FROM_T0 = {"PB-ASYNC", "B", "O", "Q"}
HANDOFF = {"H32", "H64", "P"}
SKIP = {"PB-SYNC"}


# ------------------------------------------------------------------------------ detectors


def _consecutive(k):
    def rule(slow, prep):
        run = 0
        for i, s in enumerate(slow):
            run = run + 1 if s else 0
            if run >= k:
                return i
        return None

    return rule


def _rolling(k, n):
    def rule(slow, prep):
        c = np.cumsum(np.concatenate([[0], slow.astype(int)]))
        for i in range(len(slow)):
            lo = max(0, i + 1 - n)
            if i + 1 >= min(n, k) and c[i + 1] - c[lo] >= k:
                return i
        return None

    return rule


def _ewma(alpha):
    def rule(slow, prep):
        e = None
        for i, x in enumerate(prep):
            e = x if e is None else alpha * x + (1 - alpha) * e
            if e > HOST_SLOW_MS:
                return i
        return None

    return rule


def _median(n):
    def rule(slow, prep):
        for i in range(n - 1, len(prep)):
            if np.median(prep[i + 1 - n : i + 1]) > HOST_SLOW_MS:
                return i
        return None

    return rule


DETECTORS = {
    "C2": _consecutive(2),
    "C3": _consecutive(3),
    "R3of5": _rolling(3, 5),
    "R4of8": _rolling(4, 8),
    "EWMA": _ewma(0.25),
    "MED5": _median(5),
}


# ------------------------------------------------------------------------------ loading


def _cols(rows, columns):
    return {c: np.array([r[i] for r in rows]) for i, c in enumerate(columns)}


def load_run(path: Path) -> dict:
    d = json.load(gzip.open(path))
    if "research" in d:  # #94's harness
        r = d["research"]
        cell = r["cell"]
        tr = {k: np.asarray(v) for k, v in r["trace"].items()}
        wins = [(w["start_ns"], w["end_ns"]) for w in r["windows"] if w["instance"] == "auto"]
        dec = None
        if r.get("handoff") and r["handoff"].get("decisions"):
            rows = r["handoff"]["decisions"]
            dec = (np.array([x[0] for x in rows]), np.array([x[1] for x in rows]))
    else:  # prod_run.py
        cell = d["args"]["cell"]
        tr = _cols(d["trace"], d["trace_columns"])
        wins = [(w["start_ns"], w["end_ns"]) for w in d["windows"] if w["condition"] == "hetero"]
        dec = None
        if d.get("decisions"):
            rows = d["decisions"]
            dec = (np.array([x[0] for x in rows]), np.array([x[1] for x in rows]))
    return {"cell": cell, "trace": tr, "windows": wins, "decisions": dec}


def async_start(run: dict, t0: int, end: int) -> int | None:
    cell = run["cell"]
    if cell in ASYNC_FROM_T0:
        return t0
    if cell in HANDOFF:
        t, path = run["decisions"]
        m = (t >= t0) & (t < end) & (path == 1)
        return int(t[m].min()) if m.any() else None
    return None


def requests(run: dict, lo: int, end: int) -> dict:
    tr = run["trace"]
    m = (tr["target"] == "ane") & (tr["submit_ns"] >= lo) & (tr["submit_ns"] < end)
    order = np.argsort(tr["submit_ns"][m], kind="stable")
    sub = tr["submit_ns"][m][order]
    return {
        "submit": sub,
        "prepared": tr["prepared_ns"][m][order],
        "prep": (tr["prepared_ns"][m][order] - sub) / 1e6,
        "e2e": (tr["response_ns"][m][order] - sub) / 1e6,
    }


# ------------------------------------------------------------------------------ ground truth


def _slow(rq, a, b, limit):
    m = (rq["submit"] >= a) & (rq["submit"] < b)
    return bool(m.any() and np.median(rq["e2e"][m]) > limit)


def ground_truth(rq: dict, t_a: int, end: int, m_a: float) -> dict:
    limit = SLOW_RATIO * m_a
    span = int(SPAN_S * 1e9)
    starts = list(range(t_a, end - span + 1, span))
    flags = [_slow(rq, s, s + span, limit) for s in starts]
    first = None
    for i in range(len(flags) - 1):
        if flags[i] and flags[i + 1]:
            first = i
            break
    longest = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        longest = max(longest, cur)
    out = {"slow_spans": int(sum(flags)), "longest_run": longest, "spans": len(flags), "sustained": first is not None}
    if first is not None:
        onset = starts[first]
        step = int(STEP_S * 1e9)
        while onset - step >= t_a and _slow(rq, onset - step, onset - step + span, limit):
            onset -= step
        out["onset_ns"] = onset
        out["onset_s"] = (onset - t_a) / 1e9
    return out


# ------------------------------------------------------------------------------ replay


def episodes() -> tuple[list[dict], dict]:
    eps, m_a = [], {}
    runs = {}
    for study, (issue, dirs, _) in STUDIES.items():
        for d in dirs:
            for p in sorted((RESEARCH / d).glob("*.json.gz")):
                runs[(study, p)] = load_run(p)
    for study in STUDIES:
        vals = []
        for (s, p), run in runs.items():
            if s != study or run["cell"] != "A":
                continue
            for t0, end in run["windows"]:
                vals.append(requests(run, t0 + int(A_SKIP_S * 1e9), end)["e2e"])
        if vals:
            m_a[study] = float(np.median(np.concatenate(vals)))
    for (study, p), run in runs.items():
        issue, _, ref = STUDIES[study]
        if run["cell"] in SKIP:
            continue
        for k, (t0, end) in enumerate(run["windows"]):
            t_a = async_start(run, t0, end)
            is_a = run["cell"] == "A"
            lo = t0 + int(A_SKIP_S * 1e9) if is_a else t_a
            if lo is None:
                eps.append(
                    {"study": study, "issue": issue, "run": p.name, "window": k, "cell": run["cell"], "error": "no t_a"}
                )
                continue
            rq = requests(run, lo, end)
            slow = rq["prep"] > HOST_SLOW_MS
            gt = ground_truth(rq, lo, end, m_a[ref])
            e = {
                "study": study,
                "issue": issue,
                "run": p.name,
                "window": k,
                "cell": run["cell"],
                "reference": "A" if is_a else "async",
                "t_a_s": (lo - t0) / 1e9,
                "n": int(len(slow)),
                "req_rate": len(slow) / ((end - lo) / 1e9),
                "host_slow_share": float(slow.mean()) if len(slow) else None,
                "median_e2e_ms": float(np.median(rq["e2e"])) if len(slow) else None,
                "m_a_ms": m_a[ref],
                "truth": gt,
                "detectors": {},
            }
            for name, rule in DETECTORS.items():
                i = rule(slow, rq["prep"])
                if i is None:
                    e["detectors"][name] = None
                    continue
                trip = int(rq["prepared"][i])
                r = {"index": int(i), "trip_s": (trip - lo) / 1e9}
                if gt["sustained"]:
                    onset = gt["onset_ns"]
                    r["delay_ms"] = (trip - onset) / 1e6
                    r["delay_requests"] = int(i - np.searchsorted(rq["submit"], onset))
                e["detectors"][name] = r
            eps.append(e)
    return eps, m_a


def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


def summarise(eps: list[dict]) -> dict:
    asy = [e for e in eps if e.get("reference") == "async"]
    pos = [e for e in asy if e["truth"]["sustained"]]
    neg = [e for e in asy if not e["truth"]["sustained"]]
    a = [e for e in eps if e.get("reference") == "A"]
    out = {
        "async_episodes": len(asy),
        "sustained": len(pos),
        "healthy": len(neg),
        "a_episodes": len(a),
        "detectors": {},
    }
    for name in DETECTORS:
        hit = [e for e in pos if e["detectors"][name] is not None]
        d = np.array([e["detectors"][name]["delay_ms"] for e in hit])
        dr = np.array([e["detectors"][name]["delay_requests"] for e in hit])
        ft = [e for e in neg if e["detectors"][name] is not None]
        at = [e for e in a if e["detectors"][name] is not None]
        out["detectors"][name] = {
            "recall": f"{len(hit)}/{len(pos)}",
            "missed": [f"{e['run']} w{e['window']}" for e in pos if e["detectors"][name] is None],
            "delay_ms": {"median": _pct(d, 50), "p95": _pct(d, 95), "worst": float(d.max()) if len(d) else None},
            "delay_requests": {
                "median": _pct(dr, 50),
                "p95": _pct(dr, 95),
                "worst": int(dr.max()) if len(dr) else None,
            },
            "false_trips": f"{len(ft)}/{len(neg)}",
            "false_trip_episodes": [
                f"{e['study']} {e['run']} w{e['window']} @{e['detectors'][name]['trip_s']:.2f}s" for e in ft
            ],
            "a_trips": f"{len(at)}/{len(a)}",
        }
    return out


def f(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def tables(res: dict) -> str:
    s = res["summary"]
    L = [
        "# Phase 0: detector replay on recorded episodes",
        "",
        "Generated by `scripts/replay.py` from #92, #96, #99, #102 and #103's raw runs. No new measurement.",
        "",
        f"Async episodes: {s['async_episodes']} ({s['sustained']} sustained, {s['healthy']} healthy by the latency "
        f"ground truth). A episodes replayed as a reference: {s['a_episodes']}.",
        "",
        "A-reference medians (m_A, ms): " + ", ".join(f"{k} {v:.2f}" for k, v in res["m_a_ms"].items()) + ".",
        "",
        "## Detectors",
        "",
        "| detector | recall | delay ms median / P95 / worst | delay requests median / P95 / worst | false trips (healthy) | A trips |",
        "|---|---|---|---|---|---|",
    ]
    for name, x in s["detectors"].items():
        dm, dr = x["delay_ms"], x["delay_requests"]
        L.append(
            f"| {name} | {x['recall']} | {f(dm['median'], 0)} / {f(dm['p95'], 0)} / {f(dm['worst'], 0)} "
            f"| {f(dr['median'], 0)} / {f(dr['p95'], 0)} / {dr['worst'] if dr['worst'] is not None else '–'} "
            f"| {x['false_trips']} | {x['a_trips']} |"
        )
    L += ["", "False-trip episodes:", ""]
    for name, x in s["detectors"].items():
        L.append(f"- {name}: " + ("; ".join(x["false_trip_episodes"]) or "none"))
    L += [
        "",
        "## Episodes",
        "",
        "| study | run | w | cell | t_a s | n | host-slow | median e2e | slow spans / longest | sustained onset s | "
        + " | ".join(DETECTORS)
        + " |",
        "|---|---|---|---|---|---|---|---|---|---|" + "---|" * len(DETECTORS),
    ]
    for e in res["episodes"]:
        if "error" in e:
            L.append(f"| {e['study']} | {e['run']} | {e['window']} | {e['cell']} | {e['error']} |")
            continue
        t = e["truth"]
        cells = []
        for name in DETECTORS:
            r = e["detectors"][name]
            if r is None:
                cells.append("–")
            elif "delay_ms" in r:
                cells.append(f"{r['trip_s']:.2f} ({r['delay_ms']:+.0f})")
            else:
                cells.append(f"{r['trip_s']:.2f}")
        L.append(
            f"| {e['study']} | {e['run'].replace('.json.gz', '')} | {e['window']} | {e['cell']}{' (A ref)' if e['reference'] == 'A' else ''} "
            f"| {f(e['t_a_s'])} | {e['n']} | {f(e['host_slow_share'], 3)} | {f(e['median_e2e_ms'])} "
            f"| {t['slow_spans']}/{t['spans']} · {t['longest_run']} | {f(t.get('onset_s'))} | "
            + " | ".join(cells)
            + " |"
        )
    L += [
        "",
        "Detector cells: trip time in s from t_a (A: from t0 + 1 s); in a sustained episode, the delay from onset in ms "
        "follows in brackets.",
        "",
    ]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if replay.json / replay.md are stale")
    args = ap.parse_args()
    eps, m_a = episodes()
    res = {"m_a_ms": m_a, "summary": summarise(eps), "episodes": eps}
    js = json.dumps(res, indent=1, sort_keys=False) + "\n"
    md = tables(res)
    if args.check:
        stale = [p.name for p, t in ((OUT_JSON, js), (OUT_MD, md)) if not p.exists() or p.read_text() != t]
        if stale:
            print("stale:", ", ".join(stale))
            return 1
        return 0
    OUT_JSON.write_text(js)
    OUT_MD.write_text(md)
    print(md[: md.index("## Episodes")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
