"""Analyse open-loop S5 replicates (decision rules of PREREGISTRATION.md, applied literally).

    analyze_openloop.py [--include-invalid] <replicate dir> [<replicate dir> ...]

--include-invalid pools invalid cycles too (smoke / diagnosis only; the outcome rules never use it).

Per replicate and cell: validity flags, LLM tokens/s and TTFT P95 per arm (generations lying fully
inside a window, as llm_analyze.py; valid cycles only), B/A ratios, load, queue, lag, ANE/GPU service
and response times, answer mismatches. Then the three-way outcome over the valid replicates.
"""

import json
import sys
from pathlib import Path

import numpy as np

LAG_P99_MS = 14.0  # Amendment 1; the original 5 ms is reported alongside, not used


def pct(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def load_rep(d):
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text())
    ix = [json.loads(x) for x in (d / "index.jsonl").read_text().splitlines()] if (d / "index.jsonl").exists() else []
    wins = [json.loads((d / i["file"]).read_text()) for i in ix if (d / i["file"]).exists()]
    gens = [json.loads(x) for x in (d / "llm.jsonl").read_text().splitlines()] if (d / "llm.jsonl").exists() else []
    return meta, wins, gens


def window_stats(w):
    iss, sch = np.array(w["issued_ms"]), np.array(w["sched_ms"])
    sta, end = np.array(w["started_ms"]), np.array(w["ended_ms"])
    ok = iss >= 0
    lag = (iss - sch)[ok]
    errs = sum(1 for e in w["error"] if e >= 0 and w["error_tab"][e] != "dropped")
    devs = np.array([w["device_tab"][d] if d >= 0 else "" for d in w["device"]])
    done = end >= 0
    s = {
        "lag": lag, "q": np.array(w["queue"]), "drops": w["dropped"], "errors": errs, "drained": w["drained"],
        "n": w["n"], "done": int(done.sum()), "devs": {}, "drain_s": w["drain_s"],
    }
    for dv in sorted({d for d in devs if d}):
        m = done & (devs == dv)
        s["devs"][dv] = ((end - sta)[m], (end - sch)[m])
    s["lag5"] = pct(lag, 99) <= 5.0
    s["valid"] = s["drops"] == 0 and errs == 0 and pct(lag, 99) <= LAG_P99_MS and w["drained"]
    return s


def llm(gens, wins):
    g = [x for x in gens if any(x["t0"] >= w["t0_wall"] and x["t1"] <= w["t1_wall"] for w in wins)]
    if not g:
        return float("nan"), float("nan"), 0
    return (float(np.mean([x["tokens"] / (x["t1"] - x["t0"]) for x in g])),
            float(np.percentile([x["ttft_s"] for x in g], 95) * 1e3), len(g))


def analyse(d, ncyc_min=0.75, include_invalid=False):
    meta, wins, gens = load_rep(d)
    win_s = meta["args"]["window_s"]
    by = {}
    for w in wins:
        by.setdefault((w["cell"], w["cycle"]), []).append(w)
    cells = sorted(meta["args"]["cells"])
    cycles = list(range(meta["args"]["cycles"]))
    st = {id(w): window_stats(w) for w in wins}
    # a missing or half-present (cell, cycle) is an invalid cycle, never an error
    valid_cyc = {(c, y): len(by.get((c, y), [])) == 2 and all(st[id(w)]["valid"] for w in by[(c, y)]) for c in cells for y in cycles}
    complete = len(wins) == meta["n_windows"] and (Path(d) / "meta_end.json").exists()
    # schedule identity across arms of a cycle
    sched_ok = all(len({w["sched_hash"] for w in v}) == 1 for v in by.values())
    need = int(np.ceil(ncyc_min * len(cycles)))
    n_valid = {c: sum(valid_cyc[(c, y)] for y in cycles) for c in cells}
    rep_valid = complete and all(n >= need for n in n_valid.values()) and sched_ok
    res = {"dir": str(d), "meta": meta, "wins": [(w["cell"], w["cycle"], w["arm"], st[id(w)]) for w in wins], "cells": {}, "valid": rep_valid, "complete": complete, "n_valid": n_valid, "sched_ok": sched_ok, "need": need}
    for c in cells:
        arms = sorted({w["arm"] for (cc, _), v in by.items() if cc == c for w in v})
        row = {"arms": {}}
        for arm in arms:
            ws = [w for (cc, y), v in by.items() if cc == c and (include_invalid or valid_cyc[(cc, y)]) for w in v if w["arm"] == arm]
            sts = [st[id(w)] for w in ws]
            tps, ttft, ng = llm(gens, ws)
            lag = np.concatenate([s["lag"] for s in sts]) if sts else np.array([])
            q = np.concatenate([s["q"] for s in sts]) if sts else np.array([])
            r = {"tps": tps, "ttft95": ttft, "ngen": ng, "offered": np.mean([s["n"] for s in sts]) / win_s if sts else float("nan"),
                 "done": np.mean([s["done"] for s in sts]) / win_s if sts else float("nan"),
                 "drops": sum(s["drops"] for s in sts), "q_mean": float(q.mean()) if len(q) else float("nan"),
                 "q_p99": pct(q, 99), "q_max": float(q.max()) if len(q) else float("nan"),
                 "lag_p50": pct(lag, 50), "lag_p99": pct(lag, 99), "lag_max": float(lag.max()) if len(lag) else float("nan"),
                 "dev": {}}
            for dv in sorted({k for s in sts for k in s["devs"]}):
                sv = np.concatenate([s["devs"][dv][0] for s in sts if dv in s["devs"]])
                rp = np.concatenate([s["devs"][dv][1] for s in sts if dv in s["devs"]])
                r["dev"][dv] = {"svc50": pct(sv, 50), "svc99": pct(sv, 99), "rsp50": pct(rp, 50), "rsp99": pct(rp, 99), "n": len(sv)}
            row["arms"][arm] = r
        # answer mismatches between the two arms of the same cycle (identical request by position)
        mm = tot = 0
        for (cc, y), v in by.items():
            if cc != c or not (include_invalid or valid_cyc[(cc, y)]) or len(v) != 2:
                continue
            ans = [[w["answer_tab"][i] if i >= 0 else None for i in w["answer"]] for w in v]
            mm += sum(x != y_ for x, y_ in zip(*ans))
            tot += len(ans[0])
        row["mismatch"] = (mm, tot)
        res["cells"][c] = row
    return res


def ratio(r, c, num, den, k):
    a = r["cells"].get(c, {}).get("arms", {})
    return a[num][k] / a[den][k] if num in a and den in a else float("nan")


def md_rep(r):
    print(f"\n### {r['dir']}  valid={r['valid']}  valid cycles/cell {r['n_valid']} (need {r['need']}), schedule identity {r['sched_ok']}, complete {r['complete']}\n")
    print("| cell | cycle | arm | n | done | drops | errors | lag P99 ms | drained (s) | valid | lag<=5ms | ANE svc P50 ms |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c, y, arm, s in r["wins"]:
        ane = [v for k, v in s["devs"].items() if "ane" in k.lower()]
        print(f"| {c} | {y} | {arm} | {s['n']} | {s['done']} | {s['drops']} | {s['errors']} | {pct(s['lag'], 99):.2f} | {s['drained']} ({s['drain_s']}) | {s['valid']} | {s['lag5']} | "
              f"{pct(ane[0][0], 50) if ane else float('nan'):.2f} |")
    print()
    print("| cell | arm | LLM tok/s | TTFT P95 ms | gens | offered/s | done/s | drops | queue mean/P99/max | lag P50/P99/max ms |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for c, row in r["cells"].items():
        for arm, x in row["arms"].items():
            print(f"| {c} | {arm} | {x['tps']:.1f} | {x['ttft95']:.1f} | {x['ngen']} | {x['offered']:.1f} | {x['done']:.1f} | {x['drops']} | "
                  f"{x['q_mean']:.1f}/{x['q_p99']:.0f}/{x['q_max']:.0f} | {x['lag_p50']:.2f}/{x['lag_p99']:.2f}/{x['lag_max']:.2f} |")
    print("\n| cell | device | svc P50 B/A (A2/A) | svc P99 | resp P50 | resp P99 | answer mismatches |")
    print("|---|---|---|---|---|---|---|")
    for c, row in r["cells"].items():
        arms = row["arms"]
        if "A" not in arms:
            continue
        num = "B" if "B" in arms else "A2"
        if num not in arms:
            continue
        for dv in arms["A"]["dev"]:
            if dv in arms[num]["dev"]:
                v = [arms[num]["dev"][dv][k] / arms["A"]["dev"][dv][k] for k in ("svc50", "svc99", "rsp50", "rsp99")]
                print(f"| {c} | {dv} | " + " | ".join(f"{t:.3f}" for t in v) + f" | {row['mismatch'][0]}/{row['mismatch'][1]} |")
    print("\n| ratio | tok/s | TTFT P95 |\n|---|---|---|")
    for lab, c, n, d in (("eq B/A", "eq", "B", "A"), ("hi B/A", "hi", "B", "A"), ("null A2/A", "null", "A2", "A")):
        print(f"| {lab} | {ratio(r, c, n, d, 'tps'):.3f} | {ratio(r, c, n, d, 'ttft95'):.3f} |")
    for arm in ("A", "B"):
        e, h = r["cells"].get("eq", {}).get("arms", {}).get(arm), r["cells"].get("hi", {}).get("arms", {}).get(arm)
        if e and h:
            print(f"| dose {arm}(hi)/{arm}(eq) | {h['tps'] / e['tps']:.3f} | {h['ttft95'] / e['ttft95']:.3f} |")


def outcome(reps):
    ok = [r for r in reps if r["valid"]]
    if len(ok) != 3:
        return f"NO OUTCOME: {len(ok)} valid replicate(s) among {[r['dir'] for r in reps]}; valid: {[r['dir'] for r in ok]}; exactly 3 required"
    tps = [ratio(r, "eq", "B", "A", "tps") for r in ok]
    tt = [ratio(r, "eq", "B", "A", "ttft95") for r in ok]
    if all(x >= 0.97 for x in tps) and all(x <= 1.05 for x in tt):
        o = "No inherent W8 interference detected"
    elif sum(x < 0.97 for x in tps) >= 2 and np.mean(tps) < 0.97:
        o = "W8 interference confirmed (at this load)"
    else:
        o = "Inconclusive"
    noise = [(ratio(r, "null", "A2", "A", "tps"), ratio(r, "null", "A2", "A", "ttft95")) for r in ok]
    nl = any(abs(a - 1) > 0.015 or abs(b - 1) > 0.015 for a, b in noise)
    return (f"{o}\n\neq B/A tok/s {np.round(tps, 3).tolist()} (pooled mean = mean of the per-replicate ratios: {np.mean(tps):.3f}), TTFT P95 {np.round(tt, 3).tolist()}\n"
            f"null A2/A (tok/s, TTFT) {[(round(a, 3), round(b, 3)) for a, b in noise]}  noise-limited: {nl}")


if __name__ == "__main__":
    inv = "--include-invalid" in sys.argv
    reps = [analyse(d, include_invalid=inv) for d in sys.argv[1:] if not d.startswith("--")]
    for r in reps:
        md_rep(r)
    print("\n## Outcome\n")
    print(outcome(reps))
