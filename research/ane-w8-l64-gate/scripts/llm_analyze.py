"""S5: LLM tokens/s and TTFT per ANE arm (generations lying fully inside an A or B window), per
concurrency and replicate, vs the LLM-only baseline. Reads raw/phase3/s3/rep*.json + rep*.llm.jsonl
and raw/phase3/llm_only/rep*.jsonl. Gate: B tokens/s >= 0.97 A, B TTFT P95 <= 1.05 A."""
import glob
import json

import numpy as np

D = "research/ane-w8-l64-gate/raw/phase3"


def load(f):
    return [json.loads(x) for x in open(f)]


agg = {}
for f in sorted(glob.glob(f"{D}/llm_only/rep*.jsonl")):
    g = load(f)[3:]
    print("llm-only", f.split("/")[-1], "n", len(g), "tps P50 %.1f" % np.median([x["decode_tps"] for x in g]),
          "TTFT P50/P95 ms %.1f/%.1f" % tuple(1e3 * np.percentile([x["ttft_s"] for x in g], [50, 95])))
for f in sorted(glob.glob(f"{D}/s3/rep*.json")):
    r, gens = json.load(open(f)), load(f.replace(".json", ".llm.jsonl"))
    for cl, cell in r["cells"].items():
        st = {"A": [], "B": []}
        for w in cell:
            st[w["arm"]] += [g for g in gens if g["t0"] >= w["t0_wall"] and g["t1"] <= w["t1_wall"]]
        row = {}
        for k, v in st.items():
            row[k] = (np.mean([g["tokens"] / (g["t1"] - g["t0"]) for g in v]), np.percentile([g["ttft_s"] for g in v], 95) * 1e3, len(v))
        rt, rf = row["B"][0] / row["A"][0], row["B"][1] / row["A"][1]
        agg.setdefault(cl, []).append((rt, rf))
        print(f.split("/")[-1], f"c={cl}", "gen tok/s A %.1f B %.1f (n %d/%d)" % (row["A"][0], row["B"][0], row["A"][2], row["B"][2]),
              "TTFT P95 ms A %.1f B %.1f" % (row["A"][1], row["B"][1]), "B/A tok/s %.3f TTFT %.3f" % (rt, rf))
for cl, v in agg.items():
    v = np.array(v)
    print(f"c={cl} B/A over replicates: tok/s {v[:,0].round(3)} (gate >=0.97), TTFT P95 {v[:,1].round(3)} (gate <=1.05)")
