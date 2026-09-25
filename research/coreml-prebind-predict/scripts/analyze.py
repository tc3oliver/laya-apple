"""The prebound-predict experiment: results.json and tables.md from raw/.

    uv run python research/coreml-prebind-predict/scripts/analyze.py [--check]

Inputs: raw/<model>-<config>-r{1,2}.json.gz (run_all.sh) and raw/check.json (check_prebind.py).
The criteria, their definitions, the interpretation and the non-gating records are in
../criteria.md, committed before any data; this file implements them.
  - Gate: #77's `verdict` (research/coreml-nogil-product-mix/scripts/analyze.py, loaded by
    path), which applies #57's criteria with #57's LIMITS to summaries computed by #57's own
    `placement_summary` (benchmarks/ane-process-isolation/analyze.py). Nothing is re-derived.
  - Non-gating: scripts/derive.py (unit-tested), including the fixed-exposure-window collision
    definition of criteria.md's "Prior evidence".

--raw / --out / --rounds / --models exist for checking the harness on smoke runs kept outside
the repository. The campaign uses the defaults.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))

import derive  # noqa: E402
import gate  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mix77 = _load("mix77_analyze", EXP.parent / "coreml-nogil-product-mix" / "scripts" / "analyze.py")
gate57 = mix77.gate57

# model: (short, long, production configuration). The shapes are the v1.0 commands.
MODELS = {
    "laya": (128, 512, "A"),
    "laya-typed-decisions": (128, 1024, "A"),
    "laya-multilingual": (96, 1024, "B"),
}
CANDIDATES = ("C", "PB")
ROUNDS = (1, 2)
LIMITS = gate57.LIMITS  # the #57 thresholds, unchanged
IN_PROCESS = ("A", "C", "PB")  # configurations whose ANE forward runs in the benchmark process


def load(path: Path) -> dict:
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def dist(values_ns) -> dict:
    x = np.asarray(values_ns, dtype=np.float64) / 1e6
    return {
        "n": int(len(x)),
        "p50_ms": pct(x, 50),
        "p95_ms": pct(x, 95),
        "p99_ms": pct(x, 99),
        "mean_ms": float(x.mean()) if len(x) else None,
    }


def trace_rows(run: dict, target: str) -> dict:
    t = run["research"]["trace"]
    keep = [i for i, x in enumerate(t["target"]) if x == target]
    return {k: np.asarray([v[i] for i in keep], dtype=np.int64) for k, v in t.items() if k != "target"}


def ane_tail_threshold(prod_runs: list[dict]) -> int:
    """criteria.md: P99 of response_ns - submit_ns over P's hetero ANE requests, pooled."""
    lat = np.concatenate([(lambda a: a["response_ns"] - a["submit_ns"])(trace_rows(r, "ane")) for r in prod_runs])
    return int(np.percentile(lat, 99))


def collisions(runs: list[dict], threshold_ns: int) -> dict:
    """Fixed exposure window (criteria.md, "Prior evidence") and overlap counts, pooled over runs."""
    lat, col, over = [], [], []
    for r in runs:
        ane, gpu = trace_rows(r, "ane"), trace_rows(r, "gpu")
        lat.append(ane["response_ns"] - ane["submit_ns"])
        col.append(derive.collide_fixed(ane["service_start_ns"], gpu["received_ns"]))
        spans = np.stack([ane["service_start_ns"], ane["service_end_ns"]], 1)
        completions = np.stack([gpu["received_ns"], gpu["response_ns"]], 1)
        over.append(derive.overlap_counts(spans, completions))
    lat, col, over = np.concatenate(lat), np.concatenate(col), np.concatenate(over)
    out = derive.tail_association(lat, col, threshold_ns)
    out["threshold_ms"] = threshold_ns / 1e6
    out["forward_overlap_mean"] = float(over.mean()) if len(over) else None
    out["forward_overlap_any_fraction"] = float((over > 0).mean()) if len(over) else None
    return out


def in_process_records(runs: list[dict]) -> dict:
    """Stage timings, GIL re-acquire wait and pre-stage overlap: in-process ANE, hetero windows."""
    st = {k: [] for k in (*derive.STAGES, "predict")}
    pre_over = []
    for r in runs:
        res = r["research"]
        bounds = mix77.gate57.hetero_bounds(r)
        fw = np.asarray(res["forwards"]["ane"], dtype=np.int64).reshape(-1, 3)
        fw = fw[derive.in_windows(fw[:, 0], bounds)]
        pr = np.asarray(res["predicts"], dtype=np.int64).reshape(-1, 6)
        idx = derive.join_predicts(fw[:, :2], pr)
        ok = idx >= 0
        for f, i in zip(fw[ok], idx[ok]):
            for k, v in derive.stages(f[:2], pr[i]).items():
                if v is not None:
                    st[k].append(v)
        if ok.any() and pr[idx[ok], 2].all():  # native stamps present (C, PB)
            gpu = trace_rows(r, "gpu")
            pre = np.stack([fw[ok, 0], pr[idx[ok], 2]], 1)
            pre_over.append(derive.overlap_counts(pre, np.stack([gpu["received_ns"], gpu["response_ns"]], 1)))
    out = {k: dist(v) for k, v in st.items() if v}
    if pre_over:
        p = np.concatenate(pre_over)
        out["pre_overlap_mean"] = float(p.mean())
        out["pre_overlap_any_fraction"] = float((p > 0).mean())
    return out


def forward_cpu(runs: list[dict]) -> dict:
    out = {}
    for dev in ("ane", "gpu"):
        rows = [r["research"]["forwards"].get(dev) for r in runs]
        if any(x is None for x in rows):
            out[dev] = None
            continue
        cpu, span = [], []
        for r, x in zip(runs, rows):
            a = np.asarray(x, dtype=np.int64).reshape(-1, 3)
            a = a[derive.in_windows(a[:, 0], mix77.gate57.hetero_bounds(r))]
            cpu.append(a[:, 2])
            span.append(a[:, 1] - a[:, 0])
        cpu, span = np.concatenate(cpu) / 1e6, np.concatenate(span) / 1e6
        out[dev] = {
            "placement": runs[0]["research"]["workers"][dev]["placement"],
            "n": int(len(cpu)),
            "thread_cpu_ms_mean": float(cpu.mean()) if len(cpu) else None,
            "service_ms_p50": pct(span, 50),
        }
    return out


def _client_short_per_request(w: dict) -> float | None:
    d = derive.thread_cpu_delta(w["before"], w["after"])
    n = w["streams"].get("short", 0)
    return d.get("client-short", 0) / n if n else None


def _ane_cpu(run: dict, w: dict) -> np.ndarray:
    fw = run["research"]["forwards"]["ane"]
    fw = np.asarray(fw if fw is not None else [], dtype=np.int64).reshape(-1, 3)
    return fw[derive.in_windows(fw[:, 0], [(w["start_ns"], w["end_ns"])]), 2]


def windows_records(runs: list[dict], prod_runs: list[dict] | None) -> dict:
    """Per-window GIL probe, per-thread CPU and the slow-CPU flag; medians over hetero windows.

    Slow-CPU flag (criteria.md): a candidate window's client-short CPU per request over that of
    P's matched window (same round, same cycle), flagged above derive.SLOW_CPU_RATIO. The client
    thread runs the same code in every configuration. None for P itself."""
    probe_p99, probe_over, probe_frac, flags, ratios, ane_ratios = [], [], [], [], [], []
    threads: dict[str, list] = {}
    for i, r in enumerate(runs):
        ws = sorted(r["research"]["windows"], key=lambda w: w["start_ns"])
        pws = None if prod_runs is None else sorted(prod_runs[i]["research"]["windows"], key=lambda w: w["start_ns"])
        for k, w in enumerate(ws):
            g = w["gil_probe"]
            probe_p99.append(g["p99_ms"])
            probe_over.append(g["over_1ms_fraction"])
            d = derive.thread_cpu_delta(w["before"], w["after"])
            span = w["after"]["t_ns"] - w["before"]["t_ns"]
            probe_frac.append(d.get("gil-probe", 0) / span)
            for name, v in d.items():
                threads.setdefault(name, []).append(v / 1e6)
            if pws is None:
                continue
            c, p = _client_short_per_request(w), _client_short_per_request(pws[k])
            s = derive.slow_cpu([c], [p]) if c is not None and p is not None else {"ratio": None, "flag": None}
            flags.append(s["flag"])
            ratios.append(s["ratio"])
            ane_ratios.append(derive.slow_cpu(_ane_cpu(r, w), _ane_cpu(prod_runs[i], pws[k]))["ratio"])

    def med(x):
        x = [v for v in x if v is not None]
        return float(np.median(x)) if x else None

    return {
        "gil_probe_p99_ms_median": med(probe_p99),
        "gil_probe_over_1ms_fraction_median": med(probe_over),
        "gil_probe_cpu_fraction_of_core_median": med(probe_frac),
        "thread_cpu_ms_per_window_median": {k: med(v) for k, v in sorted(threads.items())},
        "slow_cpu": None
        if prod_runs is None
        else {
            "limit": derive.SLOW_CPU_RATIO,
            "client_short_ratios": ratios,
            "flags": flags,
            "flagged_windows": sum(1 for f in flags if f),
            "ane_thread_cpu_ratios": ane_ratios,
        },
    }


def gil_wait(runs: list[dict]) -> dict | None:
    rows = []
    for r in runs:
        pr = r["research"]["predicts"]
        if pr is None:
            return None
        pr = np.asarray(pr, dtype=np.int64).reshape(-1, 6)
        pr = pr[derive.in_windows(pr[:, 0], mix77.gate57.hetero_bounds(r)) & (pr[:, 3] > 0)]
        rows.append(pr[:, 4] - pr[:, 3])
    w = np.concatenate(rows) if rows else np.zeros(0)
    return dist(w) if len(w) else None


def load_time(runs: list[dict]) -> dict:
    res = [r["research"] for r in runs]
    return {"crossings": res[0]["crossings"], "backings": res[0]["backings"]}


def interpretation(model: str, verdict: str) -> str:
    """criteria.md, "Preregistered interpretation", per model."""
    if verdict == gate.PASS:
        return (
            f"PB PASS on {model}: supports 'the many PyObjC/GIL handoffs are a significant execution-layer "
            "cost' for this model (not attributed to GPU-completion collisions alone)"
        )
    if verdict == gate.FAIL:
        return f"PB FAIL on {model}: FAIL recorded"
    return (
        f"PB INCONCLUSIVE on {model}: reported as inconclusive; no production change follows; a larger "
        "preregistered replication (for example 9 cycles, n = 18 pairs) decides"
    )


def outcome(pb: dict) -> str:
    """pb: {model: PASS | FAIL | INCONCLUSIVE}."""
    if all(v == gate.PASS for v in pb.values()):
        return "PB PASS on all three models: a production binding is considered next, in its own PR"
    parts = []
    failed = [m for m, v in pb.items() if v == gate.FAIL]
    undecided = [m for m, v in pb.items() if v == gate.INCONCLUSIVE]
    if failed:
        parts.append(
            "PB FAIL on " + ", ".join(failed) + ": the paused equal-load experiment (research/coreml-nogil-equal-load) "
            "runs next; if it passes, workload/SoC contention is considered next; if it fails, the CPU slow state, "
            "host scheduling and Python runtime contention are prioritised"
        )
    if undecided:
        parts.append(
            "PB INCONCLUSIVE on " + ", ".join(undecided) + ": no production change; a larger preregistered "
            "replication decides"
        )
    return "; ".join(parts)


def summarise(raw: Path, rounds, models) -> dict:
    res = {"limits": LIMITS, "rounds": list(rounds), "models": {}}
    check = raw / "check.json"
    res["check"] = json.loads(check.read_text()) if check.exists() else None
    for m in models:
        short, long_, prod = MODELS[m]
        configs = (prod, *CANDIDATES)
        runs = {c: [load(raw / f"{m}-{c}-r{i}.json.gz") for i in rounds] for c in configs}
        for c, rs in runs.items():
            for r in rs:
                assert r["research"]["config"] == c and r["args"]["short"] == short and r["args"]["long"] == long_
        s = {c: gate57.placement_summary(rs) for c, rs in runs.items()}
        routed = {
            c: all(
                set(d) == {dev} for st, dev in (("short", "ane"), ("long", "gpu")) for d in x["streams"][st]["devices"]
            )
            for c, x in s.items()
        }
        threshold = ane_tail_threshold(runs[prod])
        # correctness and GPU completion isolation exactly as #77 (pooled, #57's definitions); the
        # other three criteria by the paired gate (gate.py). The pooled ratios are kept as context.
        point = {c: mix77.verdict(s[prod], s[c], gain_applies=prod == "A") for c in CANDIDATES}
        verdicts = {
            c: {
                **gate.paired_verdict(
                    runs[prod],
                    runs[c],
                    LIMITS,
                    correctness=point[c]["checks"]["correctness"],
                    isolation=point[c]["checks"]["gpu_completion_isolation"],
                ),
                "gain_applies": point[c]["gain_applies"],
                "gpu_return_p50_gain": point[c]["ratios"]["gpu_return_p50_gain"],
                "pooled_point_estimate_not_the_verdict": point[c]["ratios"],
            }
            for c in CANDIDATES
        }
        res["models"][m] = {
            "hetero_streams_on_intended_devices": routed,
            "production": prod,
            "args": {k: runs[prod][0]["args"][k] for k in ("short", "long", "seconds", "cycles")},
            "versions": {k: runs["PB"][0]["research"][k] for k in ("laya_apple", "pyobjc", "coremltools", "mlx")},
            "configs": s,
            "verdicts": verdicts,
            "interpretation": interpretation(m, verdicts["PB"]["verdict"]),
            "extras": {
                c: {
                    "gil_wait": gil_wait(rs) if c in IN_PROCESS else None,
                    "stages": in_process_records(rs) if c in IN_PROCESS else None,
                    "forwards": forward_cpu(rs),
                    "collisions": collisions(rs, threshold),
                    "windows": windows_records(rs, None if c == prod else runs[prod]),
                    "load": load_time(rs),
                }
                for c, rs in runs.items()
            },
        }
    res["per_model"] = {c: {m: x["verdicts"][c]["verdict"] for m, x in res["models"].items()} for c in CANDIDATES}
    res["outcome"] = outcome(res["per_model"]["PB"])
    return res


def f(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def crossing_text(load: dict, bucket: int) -> str:
    c = (load.get("crossings") or {}).get(str(bucket))
    if not c:
        return "–"
    return (
        f"{c['total']} (predict {c['ctypes']}, sends {c['objc_send']}, pool {c['pool_push'] + c['pool_pop']}); "
        f"re-entries {c.get('reentries', '–')}"
    )


def short_bucket(load: dict, short: int) -> int | None:
    bs = sorted(int(b) for b in (load.get("crossings") or {}))
    return next((b for b in bs if b >= short), None)


def tables(res: dict) -> str:
    L = ["# One prediction crossing on the ANE request path (hetero-only v1.0 closed-loop mix)\n"]
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
        "\n## Criteria: paired gate (candidate vs the model's production configuration, per model)\n",
        "Geometric mean of the matched-pair ratios (same round, same cycle) with its 95% t-interval "
        "(df = n − 1); PASS / FAIL / INCONCLUSIVE as in criteria.md.\n",
        "| model | candidate | pairs | aggregate ≥ 0.95× | short P99 ≤ 1.05× | long P99 ≤ 1.05× "
        "| GPU return P50 ≤ 1 ms (and ≥ 5× vs A) | correctness | model |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def ci(v, k, crit, kind="t"):
        t = v["stats"][k][kind]
        verdict = v["checks"][crit] if kind == "t" else v["bootstrap_sensitivity"][crit]
        head = f"{t['geomean']:.3f} " if kind == "t" else ""
        return f"{head}[{t['lo']:.3f}, {t['hi']:.3f}] {verdict}"

    for m, x in res["models"].items():
        for c, v in x["verdicts"].items():
            gain = f"×{f(v['gpu_return_p50_gain'], 1)}" if v["gain_applies"] else "≤ 1 ms only (vs B)"
            L.append(
                f"| {m} | {c} vs {x['production']} | {v['pairs']} | {ci(v, 'aggregate', 'aggregate_throughput')} "
                f"| {ci(v, 'short_p99', 'short_p99')} | {ci(v, 'long_p99', 'long_p99')} "
                f"| {gain} {v['checks']['gpu_completion_isolation']} | {v['checks']['correctness']} "
                f"| **{v['verdict']}** |"
            )
    L += [
        "\nSensitivity only (never the verdict): seeded percentile bootstrap over pairs, 10,000 resamples.\n",
        "| model | candidate | aggregate | short P99 | long P99 |",
        "|---|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        for c, v in x["verdicts"].items():
            L.append(
                f"| {m} | {c} vs {x['production']} | {ci(v, 'aggregate', 'aggregate_throughput', 'bootstrap')} "
                f"| {ci(v, 'short_p99', 'short_p99', 'bootstrap')} | {ci(v, 'long_p99', 'long_p99', 'bootstrap')} |"
            )
    L.append("")
    for m, x in res["models"].items():
        L.append(f"- {x['interpretation']}")
        off = [c for c, ok in x["hetero_streams_on_intended_devices"].items() if not ok]
        if off:
            L.append(f"- **{m}: a hetero stream was not served entirely by its device in {', '.join(off)}**")
    L.append(f"\nOutcome: {res['outcome']}\n")
    L.append(
        "C's verdicts are a same-campaign reference for PB. The paired gate applies from this preregistration "
        "on; it is not applied to #77, whose FAIL stands.\n"
    )

    L += [
        "## Not gating: crossings, GIL re-acquire wait, ANE stages\n",
        "Crossings: one forward on the short stream's bucket, counted at load (`crossings.py`; a lower bound "
        "for C). Stages: medians over in-process ANE forwards in hetero windows (A has no native stamps).\n",
        "| model | config | crossings per forward | re-acquire P50 / P99 / mean ms | features | pre | native "
        "| re-acquire | post | tail | predict (whole) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        for c, e in x["extras"].items():
            st, g = e["stages"] or {}, e["gil_wait"]
            b = short_bucket(e["load"], x["args"]["short"])

            def p50(k, st=st):
                return f(st[k]["p50_ms"], 3) if k in st else "–"

            wait = "–" if g is None else " / ".join(f(g[k], 3) for k in ("p50_ms", "p99_ms", "mean_ms"))
            L.append(
                f"| {m} | {c} | {crossing_text(e['load'], b) if b else '–'} | {wait} | {p50('features')} "
                f"| {p50('pre')} | {p50('native')} | {p50('reacquire')} | {p50('post')} | {p50('tail')} "
                f"| {p50('predict')} |"
            )
    L += [
        "\n## Not gating: GPU completion collisions and the ANE tail\n",
        "Fixed exposure window: a GPU `received_ns` in [ANE service_start − 1 ms, service_start + 0.3 ms]. "
        "Tail: ANE request `response_ns − submit_ns` above the production configuration's P99 "
        "(hetero windows). Odds ratio with 0.5 added to each cell.\n",
        "| model | config | tail threshold ms | in tail | colliding | tail requests colliding | odds ratio "
        "| GPU completions overlapping a forward (mean / any) | … overlapping its pre stage (mean / any) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        for c, e in x["extras"].items():
            k, st = e["collisions"], e["stages"] or {}
            pre = (
                f"{f(st['pre_overlap_mean'], 3)} / {f(st['pre_overlap_any_fraction'], 3)}"
                if "pre_overlap_mean" in st
                else "–"
            )
            L.append(
                f"| {m} | {c} | {f(k['threshold_ms'])} | {f(k['tail_fraction'], 3)} | {f(k['collide_fraction'], 3)} "
                f"| {f(k['tail_colliding_fraction'], 3)} | {f(k['odds_ratio'], 1)} "
                f"| {f(k['forward_overlap_mean'], 3)} / {f(k['forward_overlap_any_fraction'], 3)} | {pre} |"
            )
    L += [
        "\n## Not gating: GIL probe, thread CPU, slow-CPU flag\n",
        "GIL probe: lateness of a 1 ms sleep grid in the benchmark process (median over hetero windows of the "
        "per-window P99). Thread CPU: ms per hetero window (median). Slow-CPU flag: candidate windows whose "
        "client-short CPU per request exceeds 1.25x that of P's matched window.\n",
        "| model | config | probe P99 ms | probe > 1 ms | probe CPU (core) | client-short | client-long "
        "| ane-dispatch | gpu-dispatch | ANE thread CPU/forward | GPU thread CPU/forward | slow-CPU windows |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        for c, e in x["extras"].items():
            w, fc = e["windows"], e["forwards"]
            t = w["thread_cpu_ms_per_window_median"]
            sc = w["slow_cpu"]
            slow = "– (reference)" if sc is None else f"{sc['flagged_windows']} / {len(sc['flags'])}"

            def th(name, t=t):
                return f(t.get(name), 0)

            L.append(
                f"| {m} | {c} | {f(w['gil_probe_p99_ms_median'], 3)} "
                f"| {f(w['gil_probe_over_1ms_fraction_median'], 4)} "
                f"| {f(w['gil_probe_cpu_fraction_of_core_median'], 4)} | {th('client-short')} | {th('client-long')} "
                f"| {th('laya-ane-dispatch')} | {th('laya-gpu-dispatch')} "
                f"| {f(None if fc['ane'] is None else fc['ane']['thread_cpu_ms_mean'], 3)} "
                f"| {f(None if fc['gpu'] is None else fc['gpu']['thread_cpu_ms_mean'], 3)} "
                f"| {slow} |"
            )
    chk = res.get("check")
    if chk:
        L += [
            "\n## Pre-campaign check (`raw/check.json`)\n",
            f"- PB bit-identical to coremltools on every bucket of every model: {chk['PB_bit_identical_everywhere']}",
            f"- C bit-identical to coremltools on every bucket of every model: {chk['C_bit_identical_everywhere']}",
        ]
        modes = sorted(
            {
                mode
                for mm in chk["models"].values()
                for bb in mm["buckets"].values()
                for mode in bb["PB_backings"]["modes"].values()
            }
        )
        L.append(f"- PB output modes seen: {', '.join(modes)}")
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
