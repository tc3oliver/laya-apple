"""B/A end-to-end ratios (P50, P99, throughput) per replicate and pooled, from raw/phase3/s1/rep*.json."""
import glob
import json

import numpy as np

T = {2: 4.3027, 9: 2.2622}
gm = {"p50": [], "p99": [], "thr": []}
for f in sorted(glob.glob("research/ane-w8-l64-gate/raw/phase3/s1/rep*.json")):
    d = json.load(open(f))
    by = {"A": [], "B": []}
    for w in d["windows"]:
        by[w["arm"]].append(w)
    errs = {k: sum(w["errors"] for w in v) for k, v in by.items()}
    ch = {k: sorted({c for w in v for c in w["choices"]}) for k, v in by.items()}
    r50 = [np.percentile(b["e2e_ms"], 50) / np.percentile(a["e2e_ms"], 50) for a, b in zip(by["A"], by["B"])]
    thr = [(a["wall_s"] / b["wall_s"]) for a, b in zip(by["A"], by["B"])]
    pa = np.percentile([x for w in by["A"] for x in w["e2e_ms"]], [50, 95, 99])
    pb = np.percentile([x for w in by["B"] for x in w["e2e_ms"]], [50, 95, 99])
    g = lambda v: float(np.exp(np.mean(np.log(v))))
    gm["p50"].append(np.log(g(r50)))
    gm["p99"].append(np.log(pb[2] / pa[2]))
    gm["thr"].append(np.log(g(thr)))
    print(f.split("/")[-1], f"P50 B/A {g(r50):.4f}", f"P99 B/A {pb[2]/pa[2]:.4f}", f"thr B/A {g(thr):.4f}",
          "A p50/95/99", np.round(pa, 2), "B", np.round(pb, 2), "errors", errs, "choices", ch)
for k, v in gm.items():
    v = np.array(v)
    h = T[len(v) - 1] * v.std(ddof=1) / len(v) ** 0.5
    print(k, f"pooled B/A {np.exp(v.mean()):.4f} [{np.exp(v.mean()-h):.4f},{np.exp(v.mean()+h):.4f}]")
