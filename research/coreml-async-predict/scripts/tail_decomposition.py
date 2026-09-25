"""Post-hoc, existing-data-only: where the PB-ASYNC short-stream tail occurs (#92's raw data).

    uv run python research/coreml-async-predict/scripts/tail_decomposition.py [--check]

Not a gate, and it does not change #92's preregistered outcome. For every short (ANE) request of
a hetero window, the end-to-end time is split into stages from fields the run already recorded:

  RequestTrace (laya_apple/model.py, executor.py), per request:
    submit -> prepared -> routed -> queue_enter -> dispatch -> service_start -> service_end
    -> received -> response
  the binding's stamps (run_config.py), per forward, joined on service_start <= entry < service_end:
    A        entry, exit                       (coremltools predict, GIL held)
    PB-SYNC  entry, python_before, native_before, native_after, python_after, exit
    PB-ASYNC entry, submit_before, submit_after, callback_entry, wake, exit
  the client's own latency (bench_concurrency.closed_loop, time.monotonic around laya.predict),
  matched to the trace by order within the window; client - (response - submit) is the time
  outside laya.predict's traced span (call entry before submit, and the client thread's wake
  after the result is set).

Groups: every request of A's windows; PB-ASYNC's normal window (short P99 < 13 ms) and its tail
windows (>= 13 ms); the slowest 1%, 5% and 10% of PB-ASYNC short requests by client latency; and
PB-SYNC as a reference for a different mechanism. Per group, the mean of each stage and its P50,
so a whole-distribution shift and a few-request tail can be told apart. Writes
tail_decomposition.json and tail_decomposition.md next to criteria.md.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
RAW = EXP / "raw"
SLOW_MS = 13.0

TRACE_STAGES = [  # (name, from, to) on RequestTrace
    ("prepare", "submit_ns", "prepared_ns"),
    ("route", "prepared_ns", "routed_ns"),
    ("check + enqueue", "routed_ns", "queue_enter_ns"),
    ("queue / dispatcher wake", "queue_enter_ns", "dispatch_ns"),
    ("dispatch → service_start", "dispatch_ns", "service_start_ns"),
]
POST_STAGES = [
    ("service_end → finish (future resolved)", "service_end_ns", "received_ns"),
    ("format answers", "received_ns", "response_ns"),
]
BINDING = {  # the predict's own stamps, columns 1..4, named per cell
    "A": [],
    "PB-SYNC": [
        ("pre (inputs, pool)", 0, 1),
        ("to native", 1, 2),
        ("native predict", 2, 3),
        ("GIL re-acquire", 3, 4),
        ("post (pool, outputs)", 4, 5),
    ],
    "PB-ASYNC": [
        ("pre (inputs, pool)", 0, 1),
        ("async send", 1, 2),
        ("Core ML + callback GIL", 2, 3),
        ("callback → waiter wake", 3, 4),
        ("post (pool, outputs)", 4, 5),
    ],
}


def load(path: Path) -> dict:
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def hetero_windows(run: dict) -> list[dict]:
    return [w for w in run["part_a"]["windows"] if w["condition"] == "hetero"]


def hetero_bounds(run: dict) -> list[tuple[int, int]]:
    seen: dict = {}
    for w in run["windows_t"]:
        if w["instance"] == "auto":
            seen.setdefault((w["start_ns"], w["end_ns"]), set()).add(w["stream"])
    return sorted(k for k, s in seen.items() if s == {"short", "long"})


def requests(run: dict, cell: str) -> list[dict]:
    """One dict per short request in a hetero window: stage durations (ms) and context."""
    t = run["research"]["trace"]
    n = len(t["request_id"])
    rows = [{k: t[k][i] for k in t} for i in range(n)]
    gpu_recv = np.sort(np.asarray([r["received_ns"] for r in rows if r["target"] == "gpu"], dtype=np.int64))
    short = sorted((r for r in rows if r["target"] == "ane"), key=lambda r: r["submit_ns"])
    pr = np.asarray(run["research"]["predicts"], dtype=np.int64).reshape(-1, 6)
    pr = pr[np.argsort(pr[:, 0])]
    cb = run["research"].get("callbacks") or {}
    cb_rows = np.asarray(cb.get("hetero") or [], dtype=np.int64).reshape(-1, 3)
    bounds = hetero_bounds(run)
    wins = hetero_windows(run)
    out = []
    for k, ((lo, hi), w) in enumerate(zip(bounds, wins)):
        mine = [r for r in short if lo <= r["submit_ns"] < hi]
        client = w["streams"]["short"]["latency_ms"]
        m = min(len(mine), len(client))
        for i in range(m):
            r = mine[i]
            d = {"cycle": w["cycle"], "window_p99": w["streams"]["short"]["p99_ms"], "client": client[i]}
            d["traced"] = (r["response_ns"] - r["submit_ns"]) / 1e6
            d["outside traced span"] = d["client"] - d["traced"]
            for name, a, b in TRACE_STAGES + POST_STAGES:
                d[name] = (r[b] - r[a]) / 1e6
            j = np.searchsorted(pr[:, 0], r["service_start_ns"])
            if j < len(pr) and pr[j, 0] < r["service_end_ns"]:
                p = pr[j]
                d["features"] = (p[0] - r["service_start_ns"]) / 1e6
                if BINDING[cell]:
                    for name, a, b in BINDING[cell]:
                        d[name] = (p[b] - p[a]) / 1e6
                else:
                    d["predict (whole)"] = (p[5] - p[0]) / 1e6
                d["action head (exit → service_end)"] = (r["service_end_ns"] - p[5]) / 1e6
                if cell == "PB-ASYNC" and len(cb_rows):
                    c = np.searchsorted(cb_rows[:, 0], p[3])
                    d["callback thread"] = int(cb_rows[c, 1]) if c < len(cb_rows) and cb_rows[c, 0] == p[3] else None
            # GPU completions (parent-side received_ns) inside this request's service span
            s, e = r["service_start_ns"], r["service_end_ns"]
            d["GPU completions during service"] = int(
                np.searchsorted(gpu_recv, e, "right") - np.searchsorted(gpu_recv, s, "left")
            )
            d["service"] = (e - s) / 1e6
            d["pre-service (submit → service_start)"] = (s - r["submit_ns"]) / 1e6
            d["post-service (service_end → response)"] = (r["response_ns"] - e) / 1e6
            out.append(d)
    return out


ORDER = (
    ["client", "traced", "outside traced span", "pre-service (submit → service_start)"]
    + [n for n, _, _ in TRACE_STAGES]
    + ["service", "features", "predict (whole)"]
    + [n for n, _, _ in BINDING["PB-SYNC"]]
    + [n for n, _, _ in BINDING["PB-ASYNC"] if n not in {x for x, _, _ in BINDING["PB-SYNC"]}]
    + ["action head (exit → service_end)", "post-service (service_end → response)"]
    + [n for n, _, _ in POST_STAGES]
    + ["GPU completions during service"]
)


def summary(reqs: list[dict]) -> dict:
    out = {"n": len(reqs)}
    for k in ORDER:
        v = [r[k] for r in reqs if r.get(k) is not None]
        if v:
            out[k] = {"mean": float(np.mean(v)), "p50": float(np.median(v)), "p99": float(np.percentile(v, 99))}
    return out


HOST_SLOW_PREPARE_MS = 0.3  # client-thread prepare; about 0.12 ms normally, 5x+ in the host-slow state


def onset_profile(run: dict, reqs: list[dict]) -> list[dict]:
    """Per hetero window: the share of short requests in the host-slow state (client-thread prepare
    above HOST_SLOW_PREPARE_MS) per 2 s of the window, and client P99 inside and outside it."""
    t = run["research"]["trace"]
    short = sorted(
        ({k: t[k][i] for k in t} for i in range(len(t["target"])) if t["target"][i] == "ane"),
        key=lambda r: r["submit_ns"],
    )[: len(reqs)]
    out = []
    for lo, hi in hetero_bounds(run):
        idx = [i for i, r in enumerate(short) if lo <= r["submit_ns"] < hi]
        ts = np.asarray([(short[i]["submit_ns"] - lo) / 1e9 for i in idx])
        slow = np.asarray([reqs[i]["prepare"] > HOST_SLOW_PREPARE_MS for i in idx])
        cl = np.asarray([reqs[i]["client"] for i in idx])
        out.append(
            {
                "window_p99": reqs[idx[0]]["window_p99"],
                "host_slow_share": float(slow.mean()),
                "share_per_2s": [float(slow[(ts >= a) & (ts < a + 2)].mean()) for a in range(0, 20, 2)],
                "client_p99_host_slow": float(np.percentile(cl[slow], 99)) if slow.sum() > 5 else None,
                "client_p99_otherwise": float(np.percentile(cl[~slow], 99)),
            }
        )
    return out


def cpu_vs_wall(run: dict, reqs: list[dict]) -> dict:
    """ANE dispatcher-thread CPU per forward against the wall time of its CPU-active stages, for
    the fastest 90% and slowest 10% / 1% of requests by client latency: CPU rising with wall means
    the work itself ran slower; wall rising alone means waiting."""
    fw = {int(a): int(c) for a, _, c in run["research"]["forwards"]["ane"]}
    t = run["research"]["trace"]
    short = sorted(
        ({k: t[k][i] for k in t} for i in range(len(t["target"])) if t["target"][i] == "ane"),
        key=lambda r: r["submit_ns"],
    )[: len(reqs)]
    active = [
        "features",
        "pre (inputs, pool)",
        "async send",
        "to native",
        "post (pool, outputs)",
        "action head (exit → service_end)",
    ]
    order = np.argsort([-r["client"] for r in reqs])
    n = len(reqs)
    groups = {"fastest 90%": order[n // 10 :], "slowest 10%": order[: n // 10], "slowest 1%": order[: max(1, n // 100)]}
    out = {}
    for g, idx in groups.items():
        cpu = [fw[short[i]["service_start_ns"]] / 1e6 for i in idx if short[i]["service_start_ns"] in fw]
        out[g] = {
            "ane_thread_cpu_ms": float(np.mean(cpu)),
            "active_stage_wall_ms": float(np.mean([sum(reqs[i].get(k) or 0 for k in active) for i in idx])),
            "prepare_ms": float(np.mean([reqs[i]["prepare"] for i in idx])),
        }
    slow = np.zeros(n, bool)
    slow[groups["slowest 10%"]] = True
    out["p_next_slow_given_slow"] = float(slow[1:][slow[:-1]].mean())
    return out


def analyse(raw: Path) -> dict:
    runs = {c: load(raw / f"laya-{c}-r1.json.gz") for c in ("A", "PB-SYNC", "PB-ASYNC")}
    cells = {c: requests(r, c) for c, r in runs.items()}
    pa = cells["PB-ASYNC"]
    groups = {
        "A (all windows)": cells["A"],
        "PB-ASYNC normal window": [r for r in pa if r["window_p99"] < SLOW_MS],
        "PB-ASYNC tail windows": [r for r in pa if r["window_p99"] >= SLOW_MS],
    }
    by_client = sorted(pa, key=lambda r: r["client"], reverse=True)
    for q in (1, 5, 10):
        groups[f"PB-ASYNC slowest {q}%"] = by_client[: max(1, round(len(pa) * q / 100))]
    groups["PB-ASYNC fastest 90%"] = by_client[max(1, round(len(pa) * 10 / 100)) :]
    groups["PB-SYNC (all windows, reference)"] = cells["PB-SYNC"]
    a_slow = sorted(cells["A"], key=lambda r: r["client"], reverse=True)
    groups["A slowest 1%"] = a_slow[: max(1, round(len(a_slow) / 100))]
    res = {"slow_ms": SLOW_MS, "groups": {g: summary(r) for g, r in groups.items()}}
    # whole-distribution shift vs a few-request tail: client-latency percentiles per group
    res["client_percentiles"] = {
        g: {str(q): float(np.percentile([r["client"] for r in groups[g]], q)) for q in (50, 90, 95, 99, 99.9)}
        for g in ("A (all windows)", "PB-ASYNC normal window", "PB-ASYNC tail windows")
    }
    # which stage carries the slowest requests' excess over the fastest 90% (mean, ms)
    base, slow = res["groups"]["PB-ASYNC fastest 90%"], res["groups"]["PB-ASYNC slowest 1%"]
    res["excess_slowest_1pct_over_fastest_90pct"] = {
        k: slow[k]["mean"] - base[k]["mean"] for k in ORDER if k in slow and k in base
    }
    res["onset"] = {c: onset_profile(runs[c], cells[c]) for c in runs}
    res["cpu_vs_wall"] = {c: cpu_vs_wall(runs[c], cells[c]) for c in runs}
    res["callback_threads"] = sorted({r["callback thread"] for r in pa if r.get("callback thread") is not None})
    return res


def table(res: dict) -> str:
    g = res["groups"]
    cols = [
        "A (all windows)",
        "PB-ASYNC normal window",
        "PB-ASYNC tail windows",
        "PB-ASYNC slowest 10%",
        "PB-ASYNC slowest 5%",
        "PB-ASYNC slowest 1%",
        "PB-SYNC (all windows, reference)",
    ]
    L = [
        "# PB-ASYNC short-tail decomposition (post-hoc, #92 raw data, not a gate)\n",
        "Mean ms per short request in hetero windows (P50 in parentheses). Stages are consecutive; "
        "`client` is the latency bench_concurrency measured around `laya.predict`.\n",
        "| stage | " + " | ".join(cols) + " |",
        "|---|" + "---:|" * len(cols),
        "| requests | " + " | ".join(str(g[c]["n"]) for c in cols) + " |",
    ]
    for k in ORDER:
        cells = []
        for c in cols:
            v = g[c].get(k)
            cells.append("–" if v is None else f"{v['mean']:.3f} ({v['p50']:.3f})")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L += [
        "\n## Client latency percentiles (ms)\n",
        "| group | P50 | P90 | P95 | P99 | P99.9 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for grp, p in res["client_percentiles"].items():
        L.append(f"| {grp} | " + " | ".join(f"{p[q]:.2f}" for q in ("50", "90", "95", "99", "99.9")) + " |")
    L += [
        "\n## Excess of PB-ASYNC's slowest 1% over its fastest 90% (mean ms per stage)\n",
        "| stage | excess ms |",
        "|---|---:|",
    ]
    for k, v in res["excess_slowest_1pct_over_fastest_90pct"].items():
        L.append(f"| {k} | {v:+.3f} |")
    L += [
        "\n## Host-slow share over each hetero window (client-thread prepare > 0.3 ms)\n",
        "| cell | window | window short P99 | host-slow share | per 2 s of the window | client P99 host-slow / otherwise |",
        "|---|---|---:|---:|---|---|",
    ]
    for c, ws in res["onset"].items():
        for k, w in enumerate(ws):
            inside = "–" if w["client_p99_host_slow"] is None else f"{w['client_p99_host_slow']:.2f}"
            L.append(
                f"| {c} | {k} | {w['window_p99']:.2f} | {w['host_slow_share']:.1%} | "
                + " ".join(f"{x:.0%}" for x in w["share_per_2s"])
                + f" | {inside} / {w['client_p99_otherwise']:.2f} |"
            )
    L += [
        "\n## CPU time against wall time (ANE dispatcher thread), by client-latency group\n",
        "| cell | group | ANE thread CPU / forward | wall of CPU-active stages | client-thread prepare |",
        "|---|---|---:|---:|---:|",
    ]
    for c, v in res["cpu_vs_wall"].items():
        for g in ("fastest 90%", "slowest 10%", "slowest 1%"):
            x = v[g]
            L.append(
                f"| {c} | {g} | {x['ane_thread_cpu_ms']:.3f} | {x['active_stage_wall_ms']:.3f} | {x['prepare_ms']:.3f} |"
            )
        L.append(f"| {c} | P(next of slowest 10% \\| slowest 10%) | {v['p_next_slow_given_slow']:.2f} | | |")
    L.append(f"\nCallback threads seen (native ids): {len(res['callback_threads'])}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    res = analyse(RAW)
    js = json.dumps(res, indent=1, sort_keys=True) + "\n"
    md = table(res)
    targets = ((EXP / "tail_decomposition.json", js), (EXP / "tail_decomposition.md", md))
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
