"""B/A end-to-end P50 / P99 / throughput / errors / answer mismatches per replicate and pooled, from raw/phase3/mix/rep*.json."""
import glob
import json
import sys

import numpy as np

T = {2: 4.3027}
d = sys.argv[1] if len(sys.argv) > 1 else "mix"
files = sorted(glob.glob(f"research/ane-w8-l64-gate/raw/phase3/{d}/rep*.json"))
recs = [json.load(open(f)) for f in files]
pool = {}
for f, r in zip(files, recs):
    for cl, cell in r["cells"].items():
        by = {"A": [], "B": []}
        for w in cell:
            by[w["arm"]].append(w)
        lat = {k: np.array([x for w in v for x in w["lat_ms"]]) for k, v in by.items()}
        ane = {k: np.array([x for w in v for x, dv in zip(w["lat_ms"], w["device"]) if dv == "ane"]) for k, v in by.items()}
        thr = np.exp(np.mean([np.log(a["wall_s"] / b["wall_s"]) for a, b in zip(by["A"], by["B"])]))
        err = {k: sum(len(w["errors"]) for w in v) for k, v in by.items()}
        mism = sum(x != y for a, b in zip(by["A"], by["B"]) for x, y in zip(a["answer"], b["answer"]))
        ratios = (np.percentile(lat["B"], 50) / np.percentile(lat["A"], 50),
                  np.percentile(lat["B"], 99) / np.percentile(lat["A"], 99), thr,
                  np.percentile(ane["B"], 50) / np.percentile(ane["A"], 50))
        pool.setdefault(cl, []).append(ratios)
        print(f.split("/")[-1], f"c={cl}", "all P50 %.3f P99 %.3f thr %.3f | ane-req P50 %.3f" % ratios,
              "A p50/p99 %.1f/%.1f" % tuple(np.percentile(lat["A"], [50, 99])),
              "B %.1f/%.1f" % tuple(np.percentile(lat["B"], [50, 99])),
              "ane share %.2f" % np.mean([x == "ane" for w in by["A"] for x in w["device"]]), "errors", err, "answer mismatches", mism)
for cl, rs in pool.items():
    g = np.log(np.array(rs))
    h = T[len(rs) - 1] * g.std(axis=0, ddof=1) / len(rs) ** 0.5
    names = ["P50", "P99", "thr", "aneP50"]
    print(f"c={cl} pooled B/A", "  ".join(f"{n} {np.exp(m):.3f} [{np.exp(m-x):.3f},{np.exp(m+x):.3f}]" for n, m, x in zip(names, g.mean(0), h)))
