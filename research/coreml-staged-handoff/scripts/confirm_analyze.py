"""The H64 confirmation: confirm_results.json and confirm_tables.md from raw-conf/ (../confirmation.md).

    uv run python research/coreml-staged-handoff/scripts/confirm_analyze.py [--check] [--raw DIR] [--out DIR]
    uv run python research/coreml-staged-handoff/scripts/confirm_analyze.py reruns [--raw DIR]

Inputs: raw-conf/laya-{A,H64}-r{1..8}.json.gz (run_config.py, run_conf.sh's fixed order), the A
validity re-runs raw-conf/laya-A-r<N>-b.json.gz, and raw-conf/failed/<run>.<attempt>.log (the crash
rule). `reruns` prints "<N>-b" for each A rN that exists, fails the A validity guard and has no
rN-b yet, one per line; it prints nothing while any of the 16 main runs is missing.

The screen's analyze.py (loaded by path, unchanged) supplies every per-run and per-transition
measure (run_stats / transitions: t_h, structure, window and onset P99, native and predict means,
transient from t_h, steady host-slow, GPU return P50 / P95 / P99, aggregate req/s, mismatches,
routing failures, protocol), its gate helper (_gate: "<=" / ">=" inclusive with #94's EPS, "<"
strict, an unmeasurable value fails, a missing reference is pending) and the A validity guard
(a_validity). New here: the post-handoff E residency (below), the pairing, the hard gates, the
population latency gate with its bootstrap, the outlier guard and the outcome.

Post-handoff E residency (placement_posthoc's recount series: cumulative per-thread P / E counters,
each counter interval assigned to its midpoint, as analyze.mechanism does): groups are the screen's
mechanism groups, parent-active (chain + parent-other threads of the process that runs
laya-ane-dispatch) and the auto GPU worker (every thread of another process); a thread enters its
group only if it used >= 20 ms of CPU in [t0, t0 + 4 s); each group is CPU-weighted. Bins are
[t_h + 0.5 k, t_h + 0.5 (k + 1)) up to the window end. A bin is E-resident if either group has
>= 2 ms of CPU in it and an E share >= 0.5. Long E residency: >= 2 consecutive E-resident bins.

Pairs: H64 rN <-> A rN (A rN-b when A rN failed the validity guard), cycle c <-> cycle c. Per H64
transition, every hard gate (confirmation.md 1-6) with value, limit and pass; per pair,
delta_p99 = H64 window P99 - A window P99 and, reported only, the same delta for onset P99, P95,
median (part_a's short p95_ms / p50_ms) and aggregate req/s. Population gate: median of the 16
deltas <= +0.5 ms and the one-sided 95% cluster bootstrap upper bound <= +1.0 ms (the 8 run pairs
resampled with replacement, each contributing both deltas, B = 20000,
numpy.random.default_rng(20260926), the 95th percentile of the resampled medians). The
transition-level bound (16 deltas resampled, a fresh generator with the same seed) is reported
only. Outlier guard: any delta_p99 > +2.0 ms fails.

Interpretation choices where confirmation.md leaves room (the stricter reading each time):
- A validity is analyze.a_validity unchanged: mismatches and routing failures in every window of
  the A run (not only its hetero windows), and the A references (onset P99, predict mean) must be
  measurable.
- Protocol (INVALID if wrong, for any run present, including an A run later replaced by its re-run):
  analyze's cycles 2, seconds 20, guard (64 for H64, none for A), handoff installed without errors;
  plus model laya, L128 / L512, gap 1.0 s and the recorded cell.
- Crash: any failed/ log of an H64 run is a correctness failure of that run (never excused), even
  when the re-run's file is missing. Two failed/ logs of the same A run (A rN or A rN-b) make the
  study INVALID.
- An A rN-b that exists while A rN passed validity is not used (reported). An A rN-b that fails
  validity makes the study INVALID. Any other file is reported and not judged.
- A hard gate that cannot be measured fails (analyze's rule); the E residency cannot be measured
  without t_h or without the recount series, so it then fails. The last bin may be shorter than
  0.5 s (it ends at the window end) and is still judged. Consecutive E-resident bins count as long
  residency even when a different group makes each bin resident. The structure gate also checks
  sync_before_t_h == 64 explicitly.
- The Core ML gates (native ratio / plus) are hard gates; a failure is not listed among the named
  FAILED reasons of confirmation.md, but it prevents CONFIRMED and INVALID does not apply, so it is
  FAILED ("Core ML").
- Precedence: a failure of H64's own measures that no A run can change (crash, correctness,
  structure, E residency, host slow, GPU isolation, on a run with the right protocol) is FAILED at
  once, even with runs pending or an INVALID condition present (reported alongside). Otherwise
  INVALID, then pending, then the gates that use the paired A (throughput, Core ML, latency).
- delta_p99 > 2.0 fails with #94's EPS (a delta of 2.0 + 1e-9 passes as float rounding), as every
  inclusive gate of the screen.
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
RAW = EXP / "raw-conf"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


an = _load("staged_handoff_analyze_for_confirmation", HERE / "analyze.py")  # the screen's, unchanged
pp = an.pp
a94 = an.a94

S = an.S
EPS = an.EPS
CELL = "H64"
GUARD = 64
GAP_S = 1.0
REPS = tuple(range(1, 9))
ORDER = (
    ("A", 1), ("H64", 1), ("H64", 2), ("A", 2),
    ("H64", 3), ("A", 3), ("A", 4), ("H64", 4),
    ("A", 5), ("H64", 5), ("H64", 6), ("A", 6),
    ("H64", 7), ("A", 7), ("A", 8), ("H64", 8),
)  # fmt: skip
E_BIN_S = 0.5
E_BIN_MIN_MS = 2.0
E_SHARE = 0.5
LONG_E_BINS = 2
GPU_P50_MS = 1.0
GPU_P99_MS = 1.0
MEDIAN_MS = 0.5
UPPER_MS = 1.0
OUTLIER_MS = 2.0
B = 20_000
SEED = 20260926
UPPER_Q = 95
HARD = (
    "mismatches",
    "routing",
    "crash",
    "structure",
    "sync_before_t_h",
    "e_residency",
    "transient_from_th",
    "steady_host_slow",
    "gpu_return_p50",
    "gpu_return_p99",
    "throughput",
    "native_ratio",
    "native_plus",
)
A_DEPENDENT = ("throughput", "native_ratio", "native_plus")
CATEGORY = {
    "mismatches": "correctness",
    "routing": "correctness",
    "crash": "correctness",
    "structure": "correctness",
    "sync_before_t_h": "correctness",
    "e_residency": "#96 / #99-type E residency",
    "transient_from_th": "persistent host-slow state",
    "steady_host_slow": "persistent host-slow state",
    "gpu_return_p50": "GPU isolation",
    "gpu_return_p99": "GPU isolation",
    "throughput": "throughput",
    "native_ratio": "Core ML",
    "native_plus": "Core ML",
}
CONFIRMED = "CONFIRMED: H64 confirmed as the leading production candidate."
CLOSES = "the H64 route closes, production stays A, and no other N is tried"
NAME = re.compile(r"^laya-(?P<cell>A|H64)-r(?P<rep>[1-8])(?P<b>-b)?\.json\.gz$")
LOG = re.compile(r"^(?P<run>laya-(?:A|H64)-r[1-8](?:-b)?)\.(?P<attempt>\d+)\.log$")


# ------------------------------------------------------------------ per transition additions


def e_residency(run: dict, t0: int, end: int, th: int | None) -> dict | None:
    """The 0.5 s bins from t_h (parent-active / worker, CPU-weighted); None if unmeasurable."""
    rc = run["research"].get("recount") or {}
    series = rc.get("series") or {}
    parent = next((s["pid"] for s in series.values() if s["name"] == "laya-ane-dispatch"), None)
    if th is None or parent is None or not rc.get("t_ns") or end <= th:
        return None
    width = int(E_BIN_S * S)
    nb = -(-(end - th) // width)
    agg = {g: [pp.acc() for _ in range(nb)] for g in ("parent", "worker")}
    active = {"parent": [], "worker": []}
    for s in series.values():
        g = {"chain": "parent", "parent-other": "parent", "worker": "worker"}.get(
            pp.group_of(s["name"], s["pid"], parent)
        )
        if g is None:
            continue
        onset = pp.acc()
        bins = [pp.acc() for _ in range(nb)]
        for mid, p, e in pp.intervals(rc, s):
            if t0 + int(an.ONSET[0] * S) <= mid < t0 + int(an.ONSET[1] * S):
                pp.add(onset, p, e)
            if th <= mid < end:
                pp.add(bins[(mid - th) // width], p, e)
        if pp.derive(onset)["cpu_ms"] < an.MECH_ACTIVE_MS - EPS:
            continue
        active[g].append(f"{s['name']}:{s['tid']}")
        for a, b in zip(agg[g], bins):
            an._merge(a, b)
    out, run_len, longest = [], 0, 0
    for k in range(nb):
        row = {"start_s": k * E_BIN_S, "end_s": min((k + 1) * width, end - th) / S}
        hit = []
        for g in agg:
            d = pp.derive(agg[g][k])
            row[g] = {"cpu_ms": d["cpu_ms"], "e_share": d["e_share"]}
            if d["cpu_ms"] >= E_BIN_MIN_MS - EPS and d["e_share"] is not None and d["e_share"] >= E_SHARE - EPS:
                hit.append(g)
        row["resident"] = hit
        run_len = run_len + 1 if hit else 0
        longest = max(longest, run_len)
        out.append(row)
    resident = [r["start_s"] for r in out if r["resident"]]
    return {
        "active_threads": active,
        "bins": out,
        "resident_bins_s": resident,
        "longest_consecutive": longest,
        "long": longest >= LONG_E_BINS,
    }


# ------------------------------------------------------------------ loading


def load(path: Path) -> dict:
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def present(raw: Path) -> tuple[dict[str, Path], list[str]]:
    """{run name: path} of the files this study reads, and every other laya-* file."""
    runs, other = {}, []
    for p in sorted(raw.glob("laya-*.json.gz")):
        m = NAME.match(p.name)
        if m and not (m["b"] and m["cell"] != "A"):
            runs[p.name[: -len(".json.gz")]] = p
        else:
            other.append(p.name)
    return runs, other


def crash_logs(raw: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p in sorted((raw / "failed").glob("*.log")):
        if m := LOG.match(p.name):
            out.setdefault(m["run"], []).append(p.name)
    return out


def stats(path: Path, logs: dict) -> dict:
    run = load(path)
    res = run["research"]
    name = path.name[: -len(".json.gz")]
    cell = NAME.match(path.name)["cell"]
    assert res["experiment"] == "coreml-staged-handoff" and res["cell"] == cell, path
    assert res["base"]["cell"] == "PB-ASYNC", path
    st = an.run_stats(run, name, cell, crashed=cell == CELL and bool(logs.get(name)))
    h = res.get("handoff") or {}
    args = run["args"]
    st["protocol"].update(
        model=args.get("model") == an.design.MODEL,
        short=args.get("short") == an.design.SHORT,
        long=args.get("long") == an.design.LONG,
        gap=h.get("gap_s") == GAP_S,
        cell=h.get("cell") == cell,
    )
    st["protocol_ok"] = all(st["protocol"].values())
    st["run_wall_s"] = res.get("run_wall_total_s", res.get("run_wall_s"))
    st["seconds"] = args.get("seconds")
    pws = {k: pw for k, pw, _ in a94.hetero(run)}
    for t in st["transitions"]:
        short = pws[t["cycle"]]["streams"]["short"]
        t["short_p95_ms"] = short.get("p95_ms")
        t["short_p50_ms"] = short.get("p50_ms")
        th = t["structure"]["t_h_ns"]
        t["e_residency"] = None if cell == "A" else e_residency(run, t["t0_ns"], t["end_ns"], th)
    return st


def reruns(raw: Path) -> list[str]:
    runs, _ = present(raw)
    if any(f"laya-{c}-r{r}" not in runs for c, r in ORDER):
        return []
    logs = crash_logs(raw)
    out = []
    for r in REPS:
        if f"laya-A-r{r}-b" in runs:
            continue
        if not an.a_validity([stats(runs[f"laya-A-r{r}"], logs)])["valid"]:
            out.append(f"{r}-b")
    return out


# ------------------------------------------------------------------ gates


def hard_gates(t: dict, run: dict, a: dict | None) -> dict:
    """Every hard safety gate of one H64 transition against its paired A transition."""
    st, g, er = t["structure"], t["gpu_return_ms"], t["e_residency"]
    tr = t["transient_from_th"]
    agg = None if a is None else a["aggregate_req_s"]
    pm = None if a is None else a["predict_mean_ms"]
    x = {
        "mismatches": an._gate(run["mismatches"], 0, "=="),
        "routing": an._gate(len(run["routing_failures"]), 0, "=="),
        "crash": an._gate(run["crashed"], False, "=="),
        "structure": an._gate(st["valid"], True, "=="),
        "sync_before_t_h": an._gate(st.get("sync_before_t_h"), GUARD, "=="),
        "e_residency": an._gate(None if er is None else er["longest_consecutive"], LONG_E_BINS - 1, "<="),
        "transient_from_th": an._gate(None if tr is None else tr["duration_s"], an.TRANSIENT_S, "<="),
        "steady_host_slow": an._gate(t["steady_host_slow_share"], an.STEADY_HOST_SLOW, "<"),
        "gpu_return_p50": an._gate(g["p50"], GPU_P50_MS, "<="),
        "gpu_return_p99": an._gate(g["p99"], GPU_P99_MS, "<="),
        "throughput": an._gate(t["aggregate_req_s"], None if agg is None else an.THROUGHPUT_RATIO * agg, ">="),
        "native_ratio": an._gate(t["native_mean_ms"], None if pm is None else an.NATIVE["ratio"] * pm, "<="),
        "native_plus": an._gate(t["native_mean_ms"], None if pm is None else pm + an.NATIVE["plus_ms"], "<="),
    }
    flags = [v["pass"] for v in x.values()]
    return {
        "gates": x,
        "failed": [k for k, v in x.items() if v["pass"] is False],
        "pending": [k for k, v in x.items() if v["pass"] is None],
        "pass": None if None in flags else all(flags),
    }


def _delta(h, a):
    return None if h is None or a is None else h - a


def bootstrap(deltas: np.ndarray) -> dict:
    """Median, cluster (run-pair) and transition-level one-sided 95% upper bounds; deltas (8, 2)."""
    d = np.asarray(deltas, float)
    n = len(d)
    flat = d.ravel()
    idx = np.random.default_rng(SEED).integers(0, n, size=(B, n))
    cluster = np.median(d[idx].reshape(B, -1), axis=1)
    idx2 = np.random.default_rng(SEED).integers(0, len(flat), size=(B, len(flat)))
    trans = np.median(flat[idx2], axis=1)
    return {
        "median_ms": float(np.median(flat)),
        "cluster_upper_ms": float(np.percentile(cluster, UPPER_Q)),
        "transition_upper_ms": float(np.percentile(trans, UPPER_Q)),
        "worst_ms": float(flat.max()),
    }


def latency(pairs: list[dict]) -> dict:
    rows = [(p, x) for p in pairs for x in p["transitions"]]
    d = [x["deltas"]["window_p99_ms"] for _, x in rows]
    out = {"n": len(d), "deltas_ms": d}
    if len(pairs) != len(REPS) or len(d) != 2 * len(REPS) or any(v is None for v in d):
        out.update(complete=False, **{"pass": None})
        return out
    b = bootstrap(np.asarray(d).reshape(len(REPS), 2))
    g = {
        "median": an._gate(b["median_ms"], MEDIAN_MS, "<="),
        "cluster_upper": an._gate(b["cluster_upper_ms"], UPPER_MS, "<="),
        "outlier": an._gate(b["worst_ms"], OUTLIER_MS, "<="),
    }
    out.update(
        complete=True,
        **b,
        gates=g,
        outliers=[f"{p['h64']} c{x['cycle']}" for p, x in rows if x["deltas"]["window_p99_ms"] > OUTLIER_MS + EPS],
        population_pass=g["median"]["pass"] and g["cluster_upper"]["pass"],
        outlier_pass=g["outlier"]["pass"],
    )
    out["pass"] = out["population_pass"] and out["outlier_pass"]
    return out


# ------------------------------------------------------------------ summary


def summarise(raw: Path) -> dict:
    runs, other = present(raw)
    logs = crash_logs(raw)
    st = {n: stats(p, logs) for n, p in runs.items()}
    missing = [f"{c} r{r}" for c, r in ORDER if f"laya-{c}-r{r}" not in runs]
    invalid, pending, own_fail, notes = [], [], [], []

    for name, ls in sorted(logs.items()):
        if name.startswith("laya-A-") and len(ls) >= 2:
            invalid.append(f"{name} failed twice ({', '.join(ls)})")
    for name, s in sorted(st.items()):
        if not s["protocol_ok"]:
            bad = [k for k, v in s["protocol"].items() if not v]
            invalid.append(f"{name} protocol deviation ({', '.join(bad)})")

    validity = {n: an.a_validity([s]) for n, s in st.items() if s["cell"] == "A"}
    pairs = []
    for r in REPS:
        a_name, b_name, h_name = f"laya-A-r{r}", f"laya-A-r{r}-b", f"laya-H64-r{r}"
        used = None
        if a_name in st:
            if validity[a_name]["valid"]:
                used = a_name
                if b_name in st:
                    notes.append(f"{b_name} not used: {a_name} passed the A validity guard")
            elif b_name in st:
                if validity[b_name]["valid"]:
                    used = b_name
                else:
                    invalid.append(f"{b_name} (the re-run of {a_name}) failed the A validity guard")
            else:
                pending.append(f"A r{r}-b (A r{r} failed the A validity guard)")
        if h_name not in st or used is None:
            continue
        h, a = st[h_name], st[used]
        a_tr = {t["cycle"]: t for t in a["transitions"]}
        a_order = {t["cycle"]: t["preceded_by"] for t in a["transitions"]}
        if a_order != {0: "solo_long", 1: "gpu_only"}:  # an A-side harness defect, not an H64 failure
            invalid.append(f"{used} does not have exactly the cycle-0 / cycle-1 hetero transitions ({a_order})")
        trs = []
        for t in h["transitions"]:
            at = a_tr.get(t["cycle"])
            if at is not None and at["preceded_by"] != t["preceded_by"]:
                at = None
            g = hard_gates(t, h, at)
            deltas = {
                k: _delta(t[k], None if at is None else at[k])
                for k in ("window_p99_ms", "onset_p99_ms", "short_p95_ms", "short_p50_ms", "aggregate_req_s")
            }
            trs.append(
                {
                    "cycle": t["cycle"],
                    "preceded_by": t["preceded_by"],
                    "a_cycle": None if at is None else at["cycle"],
                    **g,
                    "deltas": deltas,
                }
            )
        pairs.append({"rep": r, "h64": h_name, "a": used, "transitions": trs})

    # H64's own failures (no A run can change them): crash logs and A-independent gates
    for name, ls in sorted(logs.items()):
        if name.startswith("laya-H64-"):
            own_fail.append(("correctness", f"{name} crashed ({', '.join(ls)})"))
    for name, s in sorted(st.items()):
        if s["cell"] != CELL or not s["protocol_ok"]:
            continue
        for t in s["transitions"]:
            g = hard_gates(t, s, None)
            bad = [k for k in g["failed"] if k not in A_DEPENDENT and k != "crash"]
            for k in bad:
                own_fail.append((CATEGORY[k], f"{name} c{t['cycle']}: {k}"))

    lat = latency(pairs)
    judged = [x for p in pairs for x in p["transitions"]]
    if own_fail:
        outcome = "FAILED: " + _reasons(own_fail) + f"; {CLOSES}"
        extra = invalid + ([f"pending: {', '.join(missing + pending)}"] if missing or pending else [])
        if extra:
            outcome += " (also: " + "; ".join(extra) + ")"
    elif invalid:
        outcome = "INVALID: " + "; ".join(invalid)
    elif missing or pending:
        outcome = "pending: " + ", ".join(missing + pending)
    else:
        fails = [
            (CATEGORY[k], f"{p['h64']} c{x['cycle']}: {k}")
            for p in pairs
            for x in p["transitions"]
            for k in x["failed"]
        ]
        if len(judged) != 2 * len(REPS):
            fails.append(("correctness", f"{len(judged)} of {2 * len(REPS)} H64 transitions paired"))
        if any(x["pending"] for x in judged):
            fails.append(("correctness", "a paired A transition is missing"))
        if lat["complete"]:
            if not lat["population_pass"]:
                fails.append(
                    (
                        "latency confirmation",
                        f"population gate: median {lat['median_ms']:+.3f} ms (<= +{MEDIAN_MS}), "
                        f"cluster bootstrap upper bound {lat['cluster_upper_ms']:+.3f} ms (<= +{UPPER_MS})",
                    )
                )
            if not lat["outlier_pass"]:
                fails.append(("latency confirmation", "outlier guard: " + ", ".join(lat["outliers"])))
        else:
            fails.append(("latency confirmation", "delta_p99 not measurable in every transition pair"))
        outcome = CONFIRMED if not fails else "FAILED: " + _reasons(fails) + f"; {CLOSES}"

    def public(s):
        return {
            **{k: v for k, v in s.items() if k != "transitions"},
            "transitions": [{k: v for k, v in t.items() if not k.startswith("_")} for t in s["transitions"]],
        }

    return {
        "design": {
            "order": [list(x) for x in ORDER],
            "guard": GUARD,
            "gap_s": GAP_S,
            "e_residency": {"bin_s": E_BIN_S, "bin_min_ms": E_BIN_MIN_MS, "e_share": E_SHARE, "long_bins": LONG_E_BINS},
            "gpu_return_ms": {"p50": GPU_P50_MS, "p99": GPU_P99_MS, "from_s": an.GPU_RETURN_FROM_S},
            "transient_s": an.TRANSIENT_S,
            "steady_host_slow": an.STEADY_HOST_SLOW,
            "throughput_ratio": an.THROUGHPUT_RATIO,
            "native": an.NATIVE,
            "latency": {"median_ms": MEDIAN_MS, "upper_ms": UPPER_MS, "outlier_ms": OUTLIER_MS, "B": B, "seed": SEED},
        },
        "failed_logs": logs,
        "unexpected_files": other,
        "notes": notes,
        "missing": missing,
        "pending": pending,
        "invalid": invalid,
        "runs": {n: public(s) for n, s in st.items()},
        "a_validity": validity,
        "pairs": pairs,
        "latency": lat,
        "outcome": outcome,
    }


def _reasons(items: list[tuple[str, str]]) -> str:
    by: dict[str, list[str]] = {}
    for c, s in items:
        by.setdefault(c, []).append(s)
    return "; ".join(f"{c} ({', '.join(v)})" for c, v in by.items())


# ------------------------------------------------------------------ tables

f = an.f


def _range(xs, d=3):
    xs = [x for x in xs if x is not None]
    return f"{f(min(xs), d)}–{f(max(xs), d)}" if xs else "–"


def tables(res: dict) -> str:
    L = ["# H64 confirmation: A vs H64 at the hetero transition (laya, L128 / L512)", ""]
    L.append(f"Outcome: {res['outcome']}")
    L.append("")
    L.append(
        f"Crash logs (`raw-conf/failed/`): {', '.join(x for v in res['failed_logs'].values() for x in v) or 'none'}"
    )
    if res["unexpected_files"]:
        L.append(f"\nFiles outside the design (not judged): {', '.join(res['unexpected_files'])}")
    for n in res["notes"]:
        L.append(f"\n- {n}")
    L += [
        "\n## Runs\n",
        "| run | protocol ok | mismatches | routing failures | crashed | episodes | forwards sync / async | wall s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in res["runs"].values():
        h = r["handoff"]
        L.append(
            f"| {r['run']} | {r['protocol_ok']} | {r['mismatches']} | {', '.join(r['routing_failures']) or '–'} "
            f"| {r['crashed']} | {h['episodes']} | {h['forwards_sync']} / {h['forwards_async']} | {f(r['run_wall_s'], 0)} |"
        )
    L += ["\n## A validity\n", "| run | valid | failed checks |", "|---|---|---|"]
    for n, v in res["a_validity"].items():
        bad = sorted({f"c{x['cycle']}: {k}" for x in v["per_transition"] for k in x["failed"]})
        L.append(f"| {n} | {v['valid']} | {', '.join(bad) or '–'} |")
    L += [
        "\n## Hard safety gates (every H64 transition)\n",
        "| H64 transition | A | after | " + " | ".join(HARD) + " | failed |",
        "|---|---|---|" + "---|" * (len(HARD) + 1),
    ]
    for p in res["pairs"]:
        for x in p["transitions"]:
            L.append(
                f"| {p['h64']} c{x['cycle']} | {p['a']} c{f(x['a_cycle'])} | {x['preceded_by']} | "
                + " | ".join(an._cell(x["gates"][g]) for g in HARD)
                + f" | {', '.join(x['failed']) or ('pending: ' + ', '.join(x['pending']) if x['pending'] else '–')} |"
            )
    L += [
        "\n## Post-handoff E residency\n",
        "0.5 s bins from t_h; a bin is E-resident if parent-active or the worker has >= 2 ms CPU and E share >= 0.5.\n",
        "| H64 transition | bins | E-resident bins (s from t_h) | longest consecutive | long |",
        "|---|---|---|---|---|",
    ]
    for r in res["runs"].values():
        if r["cell"] != CELL:
            continue
        for t in r["transitions"]:
            e = t["e_residency"]
            if e is None:
                L.append(f"| {r['run']} c{t['cycle']} | – | not measurable | – | – |")
                continue
            res_s = ", ".join(f"{x:.1f}" for x in e["resident_bins_s"]) or "none"
            L.append(
                f"| {r['run']} c{t['cycle']} | {len(e['bins'])} | {res_s} | {e['longest_consecutive']} | {e['long']} |"
            )
    L += [
        "\n## User-visible latency (H64 − paired A)\n",
        "delta_p99 is the gate; the other deltas are reported only.\n",
        "| pair | cycle | H64 window P99 | A window P99 | delta P99 | delta onset P99 | delta P95 | delta median | delta req/s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for p in res["pairs"]:
        h = {t["cycle"]: t for t in res["runs"][p["h64"]]["transitions"]}
        a = {t["cycle"]: t for t in res["runs"][p["a"]]["transitions"]}
        for x in p["transitions"]:
            d = x["deltas"]
            at = a.get(x["a_cycle"]) or {}
            L.append(
                f"| {p['h64']} ↔ {p['a']} | {x['cycle']} | {f(h[x['cycle']]['window_p99_ms'])} | {f(at.get('window_p99_ms'))} "
                f"| {f(d['window_p99_ms'], 3)} | {f(d['onset_p99_ms'], 3)} | {f(d['short_p95_ms'], 3)} "
                f"| {f(d['short_p50_ms'], 3)} | {f(d['aggregate_req_s'], 1)} |"
            )
    lat = res["latency"]
    if lat["complete"]:
        g = lat["gates"]
        L += [
            "",
            f"- Median of the {lat['n']} deltas: {an._cell(g['median'])}",
            f"- Cluster (run-pair) bootstrap one-sided 95% upper bound (B = {B}, seed {SEED}): {an._cell(g['cluster_upper'])}",
            f"- Transition-level bootstrap upper bound (report only): {f(lat['transition_upper_ms'], 3)}",
            f"- Worst delta (outlier guard): {an._cell(g['outlier'])}",
            f"- Population gate: **{lat['population_pass']}**; outlier guard: **{lat['outlier_pass']}**",
        ]
    else:
        L.append(f"\nPopulation gate: pending ({lat['n']} of {2 * len(REPS)} deltas)")
    hs = [t for r in res["runs"].values() if r["cell"] == CELL for t in r["transitions"]]
    As = [t for p in res["pairs"] for t in res["runs"][p["a"]]["transitions"]]
    L += [
        "\n## GPU return, throughput, correctness\n",
        "| cell | GPU return P50 | P95 | P99 | aggregate req/s |",
        "|---|---|---|---|---|",
    ]
    for c, ts in (("H64 (from t_h + 1 s)", hs), ("A (whole window, paired)", As)):
        L.append(
            f"| {c} | "
            + " | ".join(_range([t["gpu_return_ms"][q] for t in ts]) for q in ("p50", "p95", "p99"))
            + f" | {_range([t['aggregate_req_s'] for t in ts], 1)} |"
        )
    hr = [r for r in res["runs"].values() if r["cell"] == CELL]
    L += [
        "",
        f"Correctness (H64): mismatches {sum(r['mismatches'] for r in hr)}, routing failures "
        f"{sum(len(r['routing_failures']) for r in hr)}, crashed runs {sum(bool(r['crashed']) for r in hr)}, "
        f"structurally valid transitions {sum(bool(t['structure']['valid']) for t in hs)} of {len(hs)}",
    ]
    if res["invalid"]:
        L.append(f"\nStudy validity issues: {'; '.join(res['invalid'])}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", nargs="?", choices=("reruns",), help="print the A validity re-runs still needed")
    ap.add_argument("--check", action="store_true", help="fail if confirm_results.json / confirm_tables.md are stale")
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=EXP)
    a = ap.parse_args()
    if a.cmd == "reruns":
        for x in reruns(a.raw):
            print(x)
        return
    res = summarise(a.raw)
    js = json.dumps(res, indent=1, sort_keys=True, default=a94._json) + "\n"
    md = tables(res)
    targets = ((a.out / "confirm_results.json", js), (a.out / "confirm_tables.md", md))
    if a.check:
        stale = [p.name for p, s in targets if not p.exists() or p.read_text() != s]
        if stale:
            sys.exit(f"stale: {stale}")
        print("outputs are up to date")
        return
    for p, s in targets:
        p.write_text(s)
    print(md)


if __name__ == "__main__":
    main()
