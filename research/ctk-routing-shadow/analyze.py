"""Score the arms and simulate cascades. Thresholds are chosen on dev only, then applied to eval.

Usage: analyze.py [LAYA_VARIANT]. Writes results/metrics.json and results/metrics.md.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from rules import RULE_SURE, classify

HERE = Path(__file__).parent
R = HERE / "results"
LABELS = ["lint-fix", "small-bugfix", "refactor", "docs", "needs-human", "needs-more-info"]
AUTO = LABELS[:4]  # classes a cheap agent may act on without a human
ABST = "needs-more-info"
TOL = 0.01  # dev: a cascade may lose at most 1 point of accuracy against Claude-only


def load(path):
    return {r["id"]: r for r in map(json.loads, path.open())} if path.exists() else {}


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, math.ceil(q / 100 * len(xs)) - 1)] if xs else None


def ece(pairs, bins=10):
    if not pairs:
        return None
    b = defaultdict(list)
    for conf, ok in pairs:
        b[min(int(conf * bins), bins - 1)].append((conf, ok))
    return sum(len(v) / len(pairs) * abs(sum(c for c, _ in v) / len(v) - sum(o for _, o in v) / len(v)) for v in b.values())


def arm_rows(split, variant):
    gold = {r["id"]: r for r in map(json.loads, (HERE / "data" / f"{split}.jsonl").open())}
    rule, laya, claude = {}, load(R / f"laya_{variant}_{split}.jsonl"), load(R / f"claude_{split}.jsonl")
    for i, g in gold.items():
        p, c = classify(g["text"])
        rule[i] = {"pred": p, "conf": c, "latency_ms": 0.0}
    claude = {i: r for i, r in claude.items() if "error" not in r}
    return gold, {"rules": rule, "laya": laya, "claude": claude}


def score(gold, preds):
    ids = [i for i in gold if i in preds]
    ok = [preds[i]["pred"] == gold[i]["label"] for i in ids]
    cls = {}
    for c in LABELS:
        tp = sum(preds[i]["pred"] == c and gold[i]["label"] == c for i in ids)
        pp = sum(preds[i]["pred"] == c for i in ids)
        ap = sum(gold[i]["label"] == c for i in ids)
        cls[c] = {"precision": tp / pp if pp else None, "recall": tp / ap if ap else None, "support": ap}
    sl = defaultdict(list)
    for i, o in zip(ids, ok):
        sl[gold[i]["slice"]].append(o)
    lat = [preds[i]["latency_ms"] for i in ids]
    conf = [(preds[i]["conf"], o) for i, o in zip(ids, ok) if preds[i].get("conf") is not None]
    nh = [i for i in ids if gold[i]["label"] == "needs-human"]
    return {
        "n": len(ids), "accuracy": sum(ok) / len(ok) if ok else None, "per_class": cls,
        "abstain_rate": sum(preds[i]["pred"] == ABST for i in ids) / len(ids) if ids else None,
        "ece": ece(conf) if any(c for c, _ in conf) else None,
        "slice_accuracy": {k: sum(v) / len(v) for k, v in sl.items()},
        "unsafe_auto_rate": sum(preds[i]["pred"] in AUTO for i in nh) / len(nh) if nh else None,
        "latency_ms": {"p50": pct(lat, 50), "p95": pct(lat, 95), "p99": pct(lat, 99)},
        "cost_usd": sum(preds[i].get("cost_usd") or 0 for i in ids),
    }


def cascade(gold, first, claude, accept):
    """first-arm answer is final when accept(first_row) is true; else Claude's (a missing one counts wrong)."""
    ids = list(gold)
    esc, ok, lat, cost = 0, 0, [], 0.0
    for i in ids:
        f = first[i]
        if accept(f):
            pred, ms = f["pred"], f["latency_ms"]
        else:
            esc += 1
            c = claude.get(i)
            pred, ms = (c["pred"] if c else None), f["latency_ms"] + (c["latency_ms"] if c else 0)
            cost += (c or {}).get("cost_usd") or 0
        ok += pred == gold[i]["label"]
        lat.append(ms)
    return {"claude_call_fraction": esc / len(ids), "accuracy": ok / len(ids), "cost_usd": cost,
            "mean_latency_ms": sum(lat) / len(lat), "p95_latency_ms": pct(lat, 95), "n": len(ids)}


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "described"
    out = {"laya_variant": variant, "laya_blocked": (R / "laya_BLOCKED.txt").exists()}
    res = {}
    for split in ("dev", "eval"):
        gold, arms = arm_rows(split, variant)
        res[split] = (gold, arms)
        out[split] = {a: score(gold, p) for a, p in arms.items() if p}
    gd, ad = res["dev"]
    ge, ae = res["eval"]
    out["cascades"] = {}
    if ad["laya"] and ae["laya"] and ad["claude"]:
        # Laya-first threshold: the largest call reduction on dev within TOL of Claude-only accuracy.
        base = cascade(gd, ad["claude"], ad["claude"], lambda f: False)["accuracy"]
        best = None
        for tau in [x / 100 for x in range(20, 100, 2)] + [1.01]:
            r = cascade(gd, ad["laya"], ad["claude"], lambda f, t=tau: f["conf"] >= t and f["pred"] != ABST)
            if r["accuracy"] >= base - TOL and (best is None or r["claude_call_fraction"] < best[1]["claude_call_fraction"]):
                best = (tau, r)
        tau = best[0]
        sweep = {f"{t}": cascade(ge, ae["laya"], ae["claude"], lambda f, t=t: f["conf"] >= t and f["pred"] != ABST)
                 for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)}
        out["cascades"]["laya_tau_chosen_on_dev"] = tau
        out["cascades"]["laya_first"] = cascade(ge, ae["laya"], ae["claude"], lambda f: f["conf"] >= tau and f["pred"] != ABST)
        out["cascades"]["laya_first_eval_sweep_expost"] = sweep
    if ae["claude"]:
        out["cascades"]["claude_only"] = cascade(ge, ae["claude"], ae["claude"], lambda f: False)
        out["cascades"]["rules_first"] = cascade(ge, ae["rules"], ae["claude"], lambda f: f["conf"] >= RULE_SURE and f["pred"] != ABST)
        out["cascades"]["rules_dev_check"] = cascade(gd, ad["rules"], ad["claude"], lambda f: f["conf"] >= RULE_SURE and f["pred"] != ABST)
    (R / "metrics.json").write_text(json.dumps(out, indent=1))
    lines = ["# Metrics (eval split unless noted)", ""]
    def f(x):
        return "n/a" if x is None else f"{x:.3f}"

    lines += ["| arm | n | acc | abstain | ECE | unsafe-auto | p50 ms | p95 | p99 | cost USD |", "|---|---|---|---|---|---|---|---|---|---|"]
    for a, s in out["eval"].items():
        L = s["latency_ms"]
        lines.append(f"| {a} | {s['n']} | {f(s['accuracy'])} | {f(s['abstain_rate'])} | {f(s['ece'])} | {f(s['unsafe_auto_rate'])} | "
                     f"{f(L['p50'])} | {f(L['p95'])} | {f(L['p99'])} | {s['cost_usd']:.4f} |")
    lines += ["", "## Accuracy by slice", "", "| arm | " + " | ".join(sorted({k for s in out["eval"].values() for k in s["slice_accuracy"]})) + " |"]
    keys = sorted({k for s in out["eval"].values() for k in s["slice_accuracy"]})
    lines.append("|---|" + "---|" * len(keys))
    for a, s in out["eval"].items():
        lines.append(f"| {a} | " + " | ".join(f(s["slice_accuracy"].get(k)) for k in keys) + " |")
    for a, s in out["eval"].items():
        lines += ["", f"## Per-class: {a}", "", "| class | precision | recall | support |", "|---|---|---|---|"]
        lines += [f"| {c} | {f(v['precision'])} | {f(v['recall'])} | {v['support']} |" for c, v in s["per_class"].items()]
    lines += ["", "## Cascades (eval)", "", "| strategy | Claude call fraction | accuracy | mean ms | p95 ms | Claude cost USD |", "|---|---|---|---|---|---|"]
    for k in ("claude_only", "rules_first", "laya_first"):
        c = out["cascades"].get(k)
        if c:
            lines.append(f"| {k} | {c['claude_call_fraction']:.3f} | {c['accuracy']:.3f} | {c['mean_latency_ms']:.0f} | {c['p95_latency_ms']:.0f} | {c['cost_usd']:.4f} |")
    if "laya_first_eval_sweep_expost" in out["cascades"]:
        lines += ["", f"Laya tau chosen on dev: {out['cascades']['laya_tau_chosen_on_dev']}. Ex-post sweep on eval (not used for any choice):", "",
                  "| tau | call fraction | accuracy |", "|---|---|---|"]
        lines += [f"| {t} | {c['claude_call_fraction']:.3f} | {c['accuracy']:.3f} |" for t, c in out["cascades"]["laya_first_eval_sweep_expost"].items()]
    (R / "metrics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
