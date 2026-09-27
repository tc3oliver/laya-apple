"""Addendum 2: per-run paired statistics and the replication verdict for typed L64 w8-pt.

    uv run python research/ane-w8/scripts/replicate_analyze.py [--raw research/ane-w8/raw/replication-l64]

Reads <raw>/run-<n>/laya-typed-decisions/L64-w8-pt/latency.json. Per run: analyze.latency_stats
(unchanged paired gate), each arm's P99 over all its predict samples, and the load average.
Pooled: geometric mean of every run's pair ratios. Verdict per addendum-2.md. Writes
<raw>/replication.json and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import latency_stats  # noqa: E402
from common import RAW, write_json  # noqa: E402

ORIGINAL = 0.649
BAND = 0.1
REPEATS = 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW / "replication-l64"))
    args = ap.parse_args()
    raw = Path(args.raw).resolve()
    runs = []
    for d in sorted(raw.glob("run-*")):
        p = d / "laya-typed-decisions" / "L64-w8-pt" / "latency.json"
        if not p.exists():
            continue
        lat = json.loads(p.read_text())
        s = latency_stats(lat)
        arm = {a: [x for w in lat["windows"] if w["arm"] == a for x in w["predict_ms"]] for a in ("baseline", "candidate")}
        runs.append(
            {
                "run": d.name,
                "ratio": s["predict_t"]["geomean"],
                "ci": [s["predict_t"]["lo"], s["predict_t"]["hi"]],
                "bootstrap": [s["predict_bootstrap"]["lo"], s["predict_bootstrap"]["hi"]],
                "forward_ratio": s["forward_t"]["geomean"],
                "fp16_p50_ms": s["baseline_predict_p50_ms"],
                "w8_p50_ms": s["candidate_predict_p50_ms"],
                "fp16_p99_ms": float(np.percentile(arm["baseline"], 99)),
                "w8_p99_ms": float(np.percentile(arm["candidate"], 99)),
                "loadavg_before": lat["environment_before"]["loadavg"],
                "loadavg_after": lat["environment_after"]["loadavg"],
                "probe_ratio_w8": lat["candidate"].get("probe", {}).get("ratio"),
                "probe_gate_w8": lat["candidate"].get("probe_gate"),
                "probe_ratio_fp16": lat["baseline"].get("probe", {}).get("ratio"),
                "verdict_1_05": s["verdict"],
                "predict_ratios": s["predict_ratios"],
            }
        )
    out = {"original": ORIGINAL, "runs": runs}
    if runs:
        all_ratios = [r for run in runs for r in run["predict_ratios"]]
        pooled = math.exp(float(np.mean(np.log(all_ratios))))
        per_run = [r["ratio"] for r in runs]
        out["pooled_geomean"] = pooled
        out["run_ratio_sd"] = float(np.std(per_run, ddof=1)) if len(per_run) > 1 else None
        out["run_ratio_range"] = [min(per_run), max(per_run)]
        complete = len(runs) == REPEATS
        every_below_1 = all(r["ci"][1] < 1.0 for r in runs)
        within = abs(pooled - ORIGINAL) <= BAND
        out["rule"] = {"complete": complete, "every_upper_below_1": every_below_1, "pooled_within_0_1": within}
        out["reproduced"] = bool(complete and every_below_1 and within) if complete else None
    write_json(raw / "replication.json", out)
    for r in runs:
        print(
            f"{r['run']}: ratio {r['ratio']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  "
            f"FP16 P50/P99 {r['fp16_p50_ms']:.2f}/{r['fp16_p99_ms']:.2f}  W8 P50/P99 {r['w8_p50_ms']:.2f}/{r['w8_p99_ms']:.2f}  "
            f"probe {r['probe_ratio_w8']}  load {r['loadavg_before'][0]:.2f}->{r['loadavg_after'][0]:.2f}"
        )
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    main()
