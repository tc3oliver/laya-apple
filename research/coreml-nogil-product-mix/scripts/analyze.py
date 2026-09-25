"""The GIL-release product-mix experiment: results.json and tables.md from raw/.

    uv run python research/coreml-nogil-product-mix/scripts/analyze.py [--check]

Inputs: raw/<model>-<config>-r{1,2}.json.gz (run_all.sh). The criteria, their definitions and
the stop rules are in ../criteria.md and were committed before the campaign ran; this file
implements them. The per-configuration summary is #57's own (`placement_summary` in
benchmarks/ane-process-isolation/analyze.py, imported, not copied), so every gated number is
computed exactly as the #57 gate computed it.

--raw / --out / --rounds / --models exist for checking the harness on smoke runs kept outside
the repository. The campaign uses the defaults.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent


def _gate57():
    """benchmarks/ane-process-isolation/analyze.py (#57), loaded by path: it shares this file's name."""
    path = EXP.parents[1] / "benchmarks" / "ane-process-isolation" / "analyze.py"
    spec = importlib.util.spec_from_file_location("gate57_analyze", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate57 = _gate57()

# model: (short, long, production configuration). The shapes are the v1.0 commands.
MODELS = {
    "laya": (128, 512, "A"),
    "laya-typed-decisions": (128, 1024, "A"),
    "laya-multilingual": (96, 1024, "B"),
}
CANDIDATES = ("C", "D")
ROUNDS = (1, 2)
LIMITS = gate57.LIMITS  # the #57 thresholds, unchanged


def load(path: Path) -> dict:
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def verdict(prod: dict, cand: dict, gain_applies: bool) -> dict:
    """#57's criteria: 'process' is the candidate, 'thread' the production configuration."""
    agg = cand["aggregate_req_s"] / prod["aggregate_req_s"]
    short = cand["streams"]["short"]["p99_ms_median"] / prod["streams"]["short"]["p99_ms_median"]
    long_ = cand["streams"]["long"]["p99_ms_median"] / prod["streams"]["long"]["p99_ms_median"]
    rc, rp = cand["gpu_return_ms"]["p50"], prod["gpu_return_ms"]["p50"]
    gain = None if rc is None or rp is None else (math.inf if rc == 0 else rp / rc)
    isolation = rc is not None and rc <= LIMITS["return_p50_max_ms"]
    if gain_applies:
        isolation = isolation and gain is not None and gain >= LIMITS["return_min_gain"]
    checks = {
        "correctness": prod["mismatches_all_windows"] == 0 and cand["mismatches_all_windows"] == 0,
        "aggregate_throughput": agg >= LIMITS["aggregate_min_ratio"],
        "short_p99": short <= LIMITS["p99_max_ratio"],
        "long_p99": long_ <= LIMITS["p99_max_ratio"],
        "gpu_completion_isolation": isolation,
    }
    return {
        # JSON has no infinity: a candidate P50 of 0 µs (the legs are whole µs) is stored as None
        "ratios": {
            "aggregate": agg,
            "short_p99": short,
            "long_p99": long_,
            "gpu_return_p50_gain": None if gain == math.inf else gain,
        },
        "gain_applies": gain_applies,
        "checks": checks,
        "passed": all(checks.values()),
    }


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def extras(runs: list[dict]) -> dict:
    """Non-gating: the ANE thread's GIL re-acquire wait, and executing-thread CPU per forward.
    Both are recorded by run_config.py for hetero windows only."""
    out: dict = {"forwards": {}}
    for dev in ("ane", "gpu"):
        rows = [r["research"]["forwards"].get(dev) for r in runs]
        if any(x is None for x in rows):
            out["forwards"][dev] = None
            continue
        a = np.array([f for x in rows for f in x], dtype=np.int64).reshape(-1, 3)
        span, cpu = (a[:, 1] - a[:, 0]) / 1e6, a[:, 2] / 1e6
        out["forwards"][dev] = {
            "placement": runs[0]["research"]["workers"][dev]["placement"],
            "n": len(a),
            "service_ms_p50": pct(span, 50),
            "service_ms_mean": float(span.mean()) if len(a) else None,
            "thread_cpu_ms_mean": float(cpu.mean()) if len(a) else None,
            "thread_cpu_ms_p50": pct(cpu, 50),
        }
    stamps = [r["research"]["gil_wait"] for r in runs]
    if any(s is None for s in stamps):
        out["gil_wait"] = None
    else:
        s = np.array([x for st in stamps for x in st], dtype=np.int64).reshape(-1, 4)
        wait, release, predict = (s[:, 3] - s[:, 2]) / 1e6, (s[:, 1] - s[:, 0]) / 1e6, (s[:, 2] - s[:, 1]) / 1e6
        out["gil_wait"] = {
            "n": len(s),
            "reacquire_ms": {
                "p50": pct(wait, 50),
                "p95": pct(wait, 95),
                "p99": pct(wait, 99),
                "mean": float(wait.mean()) if len(s) else None,
            },
            "release_ms_p50": pct(release, 50),
            "predict_native_ms_p50": pct(predict, 50),
        }
    return out


def outcome(passed: dict) -> str:
    """The stop rules in criteria.md, applied to the overall candidate verdicts."""
    if passed["D"]:
        return "D passes: D is the candidate production design"
    if passed["C"]:
        return "only C passes: C is the candidate production design; the GPU worker stays a process"
    return "C and D both fail: FAIL recorded; the Swift-worker fallback research is preregistered next"


def summarise(raw: Path, rounds, models) -> dict:
    res = {"limits": LIMITS, "rounds": list(rounds), "models": {}}
    for m in models:
        short, long_, prod = MODELS[m]
        configs = (prod, *CANDIDATES)
        runs = {c: [load(raw / f"{m}-{c}-r{i}.json.gz") for i in rounds] for c in configs}
        for c, rs in runs.items():
            for r in rs:
                assert r["research"]["config"] == c and r["args"]["short"] == short and r["args"]["long"] == long_
        s = {c: gate57.placement_summary(rs) for c, rs in runs.items()}
        # criteria.md, "Valid runs": every hetero request on its intended device
        routed = {
            c: all(
                set(d) == {dev} for st, dev in (("short", "ane"), ("long", "gpu")) for d in x["streams"][st]["devices"]
            )
            for c, x in s.items()
        }
        res["models"][m] = {
            "hetero_streams_on_intended_devices": routed,
            "production": prod,
            "args": {k: runs[prod][0]["args"][k] for k in ("short", "long", "seconds", "cycles")},
            "versions": {k: runs["C"][0]["research"][k] for k in ("laya_apple", "pyobjc", "coremltools", "mlx")},
            "configs": s,
            "extras": {c: extras(rs) for c, rs in runs.items()},
            "verdicts": {c: verdict(s[prod], s[c], gain_applies=prod == "A") for c in CANDIDATES},
        }
    res["candidates"] = {c: all(x["verdicts"][c]["passed"] for x in res["models"].values()) for c in CANDIDATES}
    res["outcome"] = outcome(res["candidates"])
    return res


def f(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def rel(r):
    return f"{(r - 1) * 100:+.1f}%"


def tables(res: dict) -> str:
    L = ["# Core ML predict without the GIL on the product mix (v1.0 closed-loop mix, hetero windows)\n"]
    L += [
        "| model | config | aggregate req/s | short req/s | short P99 | long req/s | long P99 "
        "| GPU return P50 / P99 | mismatches |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        for c, s in x["configs"].items():
            sh, lo, g = s["streams"]["short"], s["streams"]["long"], s["gpu_return_ms"]
            label = f"{c} (production)" if c == x["production"] else c
            L.append(
                f"| {m} | {label} | {f(s['aggregate_req_s'], 1)} | {f(sh['req_s_median'], 1)} "
                f"| {f(sh['p99_ms_median'])} | {f(lo['req_s_median'], 1)} | {f(lo['p99_ms_median'])} "
                f"| {f(g['p50'], 3)} / {f(g['p99'], 3)} | {s['mismatches_all_windows']} |"
            )
    L += [
        "\n## Criteria (candidate vs the model's production configuration)\n",
        "| model | candidate | aggregate ≥ 0.95× | short P99 ≤ 1.05× | long P99 ≤ 1.05× "
        "| GPU return P50 ≤ 1 ms and ≥ 5× better | correctness | model |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        for c, v in x["verdicts"].items():
            r = v["ratios"]

            def ok(k, v=v):
                return "pass" if v["checks"][k] else "**FAIL**"

            gain = f"×{f(r['gpu_return_p50_gain'], 1)}" if v["gain_applies"] else "≤ 1 ms only (vs B)"
            L.append(
                f"| {m} | {c} vs {x['production']} | {rel(r['aggregate'])} {ok('aggregate_throughput')} "
                f"| {rel(r['short_p99'])} {ok('short_p99')} | {rel(r['long_p99'])} {ok('long_p99')} "
                f"| {gain} {ok('gpu_completion_isolation')} | {ok('correctness')} "
                f"| {'PASS' if v['passed'] else '**FAIL**'} |"
            )
    L.append("")
    models = ", ".join(res["models"])
    for c, p in res["candidates"].items():
        L.append(f"- {c}: {'PASS' if p else '**FAIL**'} ({models})")
    for m, x in res["models"].items():
        off = [c for c, ok in x["hetero_streams_on_intended_devices"].items() if not ok]
        if off:
            L.append(f"- **{m}: a hetero stream was not served entirely by its device in {', '.join(off)}**")
    L.append(f"\nStop rule: {res['outcome']}\n")
    L += [
        "## Not gating: GIL re-acquire wait and executing-thread CPU\n",
        "| model | config | ANE GIL re-acquire P50 / P95 / P99 / mean ms | ANE predict (native) P50 "
        "| ANE thread CPU per forward | ANE placement | GPU thread CPU per forward | GPU service P50 | GPU placement |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        for c, e in x["extras"].items():
            g, a, gp = e["gil_wait"], e["forwards"]["ane"], e["forwards"]["gpu"]
            wait = "–" if g is None else " / ".join(f(g["reacquire_ms"][k], 3) for k in ("p50", "p95", "p99", "mean"))
            L.append(
                f"| {m} | {c} | {wait} | {f(None if g is None else g['predict_native_ms_p50'])} "
                f"| {f(None if a is None else a['thread_cpu_ms_mean'], 3)} | {'–' if a is None else a['placement']} "
                f"| {f(None if gp is None else gp['thread_cpu_ms_mean'], 3)} "
                f"| {f(None if gp is None else gp['service_ms_p50'])} | {'–' if gp is None else gp['placement']} |"
            )
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if results.json / tables.md are stale")
    ap.add_argument("--raw", type=Path, default=EXP / "raw")
    ap.add_argument("--out", type=Path, default=EXP)
    ap.add_argument("--rounds", type=int, nargs="+", default=list(ROUNDS))
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    a = ap.parse_args()
    res = summarise(a.raw, a.rounds, a.models)
    js = json.dumps(res, indent=1, sort_keys=True) + "\n"
    md = tables(res)
    targets = ((a.out / "results.json", js), (a.out / "tables.md", md))
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
