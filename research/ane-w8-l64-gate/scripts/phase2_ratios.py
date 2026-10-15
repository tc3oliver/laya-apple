"""B/A predict and forward P50 ratios per replicate and pooled (GM, 95% t-interval over replicates)."""
import glob
import json

import numpy as np

P = "research/ane-w8-l64-gate/raw/phase2"
for k in ("predict", "forward"):
    gms = []
    for i in (1, 2, 3):
        d = json.load(open(glob.glob(f"{P}/rep{i}/**/latency.json", recursive=True)[0]))
        r = []
        for c in range(10):
            w = {x["arm"]: x for x in d["windows"] if x["cycle"] == c}
            r.append(w["candidate"][k + "_p50_ms"] / w["baseline"][k + "_p50_ms"])
        l = np.log(r)
        h = 2.2622 * l.std(ddof=1) / 10**0.5
        gms.append(l.mean())
        print(k, i, d["status"], d["baseline"]["probe_gate"], d["candidate"]["probe_gate"],
              d["baseline"]["artifact_sha256"][:12], f"GM {np.exp(l.mean()):.4f} [{np.exp(l.mean()-h):.4f},{np.exp(l.mean()+h):.4f}]",
              f"pairs {min(r):.4f}..{max(r):.4f}", f"maxdlogit {d['candidate_vs_baseline_max_abs_logit']:.4f}")
    g = np.array(gms); h = 4.3027 * g.std(ddof=1) / 3**0.5
    print(k, "pooled", f"{np.exp(g.mean()):.4f} [{np.exp(g.mean()-h):.4f},{np.exp(g.mean()+h):.4f}]")
