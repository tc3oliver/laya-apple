"""R1, prebound predict under #77's full #57 protocol: results.json and tables.md from raw/.

    uv run python research/coreml-prebind-full-protocol/scripts/analyze.py [--check]

Inputs: raw/<model>-<config>-r<round>.json.gz (run_all.sh, design.py's run order) and
raw/check.json (#83's check_prebind.py). The criteria and the sequential rule are in
../criteria.md, committed before any data; this file implements them.
  - Looks: per model at the end of stages 1, 2, 3 (18, 36, 54 matched hetero pairs per
    candidate). A look exists only when every run of its rounds exists for P, C and PB.
  - Gate at each look: correctness and GPU completion isolation by #77's `verdict` on #57's
    `placement_summary` (pooled, #57's LIMITS, the >= 5x gain against A applies); aggregate,
    short P99 and long P99 by #83's paired gate (gate.py, 95% t primary, bootstrap sensitivity).
    A Bonferroni-over-three-looks interval is reported next to it as sensitivity only.
  - The deciding verdict is PB's; the first look with PASS or FAIL is final, INCONCLUSIVE at
    stage 3 is final at the cap. Runs beyond a model's final look are listed, never used. C is
    judged at the same looks for reference only.
  - Non-gating extras reuse #83's analyze.py (loaded by path), with an R1 variant of its
    per-window records (R1 runs no GIL probe), plus the window-history diagnostic.

--raw / --out / --models exist for checking the harness on smoke runs kept outside the
repository. The campaign uses the defaults.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RAW = EXP / "raw"
sys.path.insert(0, str(HERE))

import design  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pb83 = _load("pb83_analyze", EXP.parent / "coreml-prebind-predict" / "scripts" / "analyze.py")
gate = pb83.gate  # #83's paired gate, unchanged
derive = pb83.derive
mix77 = pb83.mix77
gate57 = mix77.gate57

MODELS = design.MODELS
CANDIDATES = ("C", "PB")
LIMITS = gate57.LIMITS  # the #57 thresholds, unchanged
LOOKS = design.MAX_STAGE
BONFERRONI_CONFIDENCE = 1 - 0.05 / LOOKS  # sensitivity only, never the verdict
CRITERIA = (  # (pair-ratio key, criterion, verdict function, limit key)
    ("aggregate", "aggregate_throughput", gate.lower_verdict, "aggregate_min_ratio"),
    ("short_p99", "short_p99", gate.upper_verdict, "p99_max_ratio"),
    ("long_p99", "long_p99", gate.upper_verdict, "p99_max_ratio"),
)
CAP = f"INCONCLUSIVE at cap ({design.pairs_at(design.MAX_STAGE)} pairs)"


def load(path: Path) -> dict:
    run = pb83.load(path)
    for w in run["part_a"]["windows"]:  # per-request latencies: not used here, and the bulk of a run
        for s in w["streams"].values():
            s.pop("latency_ms", None)
    return run


class Runs:
    """One model's run files, loaded on first use."""

    def __init__(self, raw: Path, model: str):
        self.raw, self.model, self._cache = raw, model, {}
        self.short, self.long, _ = MODELS[model]

    def path(self, config: str, rnd: int) -> Path:
        return self.raw / design.run_file(self.model, design.config_name(self.model, config), rnd)

    def exists(self, config: str, rnd: int) -> bool:
        return self.path(config, rnd).exists()

    def complete(self, stage: int) -> bool:
        return all(self.exists(c, r) for c in design.CONFIGS for r in design.rounds_through(stage))

    def get(self, config: str, rounds) -> list[dict]:
        out = []
        for r in rounds:
            if (config, r) not in self._cache:
                run = load(self.path(config, r))
                res = run["research"]
                assert res["config"] == design.config_name(self.model, config), self.path(config, r)
                assert run["args"]["short"] == self.short and run["args"]["long"] == self.long
                assert run["args"]["seconds"] == design.SECONDS and run["args"]["cycles"] == design.CYCLES, self.path(
                    config, r
                )
                assert len(gate.hetero_by_cycle(run)) == design.CYCLES, self.path(config, r)
                self._cache[(config, r)] = run
            out.append(self._cache[(config, r)])
        return out

    def started(self, stage: int) -> bool:
        """At least one run file of the rounds this stage adds exists."""
        new = set(design.rounds_through(stage)) - (set(design.rounds_through(stage - 1)) if stage > 1 else set())
        return any(self.exists(c, r) for c in design.CONFIGS for r in new)

    def present(self) -> list[tuple[str, int]]:
        return sorted(
            (c, r)
            for c in design.CONFIGS
            for r in range(1, max(design.rounds_through(design.MAX_STAGE)) + 1)
            if self.exists(c, r)
        )


# ------------------------------------------------------------------ one look


def judge(prod_runs: list[dict], cand_runs: list[dict], s_prod: dict, s_cand: dict) -> dict:
    point = mix77.verdict(s_prod, s_cand, gain_applies=True)
    v = gate.paired_verdict(
        prod_runs,
        cand_runs,
        LIMITS,
        correctness=point["checks"]["correctness"],
        isolation=point["checks"]["gpu_completion_isolation"],
    )
    bonf = {}
    for key, crit, fn, lim in CRITERIA:
        ci = gate.t_interval(v["stats"][key]["ratios"], confidence=BONFERRONI_CONFIDENCE)
        bonf[crit] = {**ci, "confidence": BONFERRONI_CONFIDENCE, "would_be": fn(ci, LIMITS[lim])}
    return {
        **v,
        "gain_applies": True,
        "gpu_return_p50_gain": point["ratios"]["gpu_return_p50_gain"],
        "pooled_point_estimate_not_the_verdict": point["ratios"],
        "bonferroni_sensitivity_not_the_verdict": bonf,
    }


def look(runs: Runs, stage: int) -> dict:
    rounds = design.rounds_through(stage)
    rs = {c: runs.get(c, rounds) for c in design.CONFIGS}
    s = {c: gate57.placement_summary(x) for c, x in rs.items()}
    return {
        "stage": stage,
        "rounds": list(rounds),
        "pairs": design.pairs_at(stage),
        "configs": s,
        "verdicts": {c: judge(rs["P"], rs[c], s["P"], s[c]) for c in CANDIDATES},
    }


# ------------------------------------------------------------------ non-gating records


def windows_records(runs: list[dict], prod_runs: list[dict] | None) -> dict:
    """#83's windows_records without the GIL-probe fields (R1 runs no probe): per-thread CPU per
    hetero window (medians) and the slow-CPU flag against P's matched window (same round, same
    cycle). None for P itself."""
    flags, ratios, ane_ratios = [], [], []
    threads: dict[str, list] = {}
    for i, r in enumerate(runs):
        ws = sorted(r["research"]["windows"], key=lambda w: w["start_ns"])
        pws = None if prod_runs is None else sorted(prod_runs[i]["research"]["windows"], key=lambda w: w["start_ns"])
        for k, w in enumerate(ws):
            for name, v in derive.thread_cpu_delta(w["before"], w["after"]).items():
                threads.setdefault(name, []).append(v / 1e6)
            if pws is None:
                continue
            c, p = pb83._client_short_per_request(w), pb83._client_short_per_request(pws[k])
            s = derive.slow_cpu([c], [p]) if c is not None and p is not None else {"ratio": None, "flag": None}
            flags.append(s["flag"])
            ratios.append(s["ratio"])
            ane_ratios.append(derive.slow_cpu(pb83._ane_cpu(r, w), pb83._ane_cpu(prod_runs[i], pws[k]))["ratio"])

    def med(x):
        x = [v for v in x if v is not None]
        return float(np.median(x)) if x else None

    return {
        "thread_cpu_ms_per_window_median": {k: med(v) for k, v in sorted(threads.items())},
        "slow_cpu": None
        if prod_runs is None
        else {
            "limit": derive.SLOW_CPU_RATIO,
            "client_short_ratios": ratios,
            "flags": flags,
            "flagged_windows": sum(1 for f in flags if f),
            "computable_windows": sum(1 for f in flags if f is not None),
            "ane_thread_cpu_ratios": ane_ratios,
        },
    }


def expected_before(cycle: int) -> str:
    """bench_concurrency part_a: solo_short, solo_long, hetero, gpu_only; odd cycles reversed."""
    return "solo_long" if cycle % 2 == 0 else "gpu_only"


def preceding(run: dict) -> dict[int, str | None]:
    """{cycle: condition of the window run just before that cycle's hetero window}. part_a's
    windows are recorded in execution order."""
    ws = run["part_a"]["windows"]
    return {w["cycle"]: (ws[i - 1]["condition"] if i else None) for i, w in enumerate(ws) if w["condition"] == "hetero"}


def window_history(rs: dict[str, list[dict]]) -> dict:
    """Non-gating, for the protocol-history question: hetero windows split by the condition that
    ran just before them; per config the short P99 median, per candidate the geometric mean of
    the matched-pair ratios to P."""
    as_expected = True
    before = {}
    for c, runs in rs.items():
        groups: dict[str, list] = {}
        for r in runs:
            prev = preceding(r)
            as_expected &= all(p == expected_before(k) for k, p in prev.items())
            for k, w in gate.hetero_by_cycle(r).items():
                groups.setdefault(str(prev[k]), []).append(w["streams"]["short"]["p99_ms"])
        before[c] = {
            g: {"windows": len(v), "short_p99_median_ms": float(np.median(v))} for g, v in sorted(groups.items())
        }
    ratios = {}
    prev_p = [preceding(r) for r in rs["P"]]
    for c in CANDIDATES:
        groups = {}
        for rnd, k, p, w in gate.pair_windows(rs["P"], rs[c]):
            groups.setdefault(str(prev_p[rnd - 1][k]), []).append(
                (
                    w["streams"]["short"]["p99_ms"] / p["streams"]["short"]["p99_ms"],
                    gate.aggregate_req_s(w) / gate.aggregate_req_s(p),
                )
            )
        ratios[c] = {
            g: {
                "pairs": len(v),
                "short_p99_geomean_ratio": float(np.exp(np.mean(np.log([x[0] for x in v])))),
                "aggregate_geomean_ratio": float(np.exp(np.mean(np.log([x[1] for x in v])))),
            }
            for g, v in sorted(groups.items())
        }
    return {"order_as_expected": bool(as_expected), "by_config": before, "vs_P": ratios}


def report(runs: Runs, lk: dict) -> dict:
    rounds = lk["rounds"]
    rs = {c: runs.get(c, rounds) for c in design.CONFIGS}
    threshold = pb83.ane_tail_threshold(rs["P"])
    routing = [  # every hetero stream window not served entirely by its intended device
        {"config": c, "round": rnd, "cycle": k, "stream": st, "devices": w["streams"][st]["devices"]}
        for c, x in rs.items()
        for rnd, r in zip(rounds, x)
        for k, w in sorted(gate.hetero_by_cycle(r).items())
        for st, dev in (("short", "ane"), ("long", "gpu"))
        if set(w["streams"][st]["devices"]) != {dev}
    ]
    return {
        "look": lk["stage"],
        "routing_failures": routing,
        "args": {k: rs["P"][0]["args"][k] for k in ("short", "long", "seconds", "cycles")},
        "versions": {k: rs["PB"][0]["research"][k] for k in ("laya_apple", "pyobjc", "coremltools", "mlx")},
        "window_history": window_history(rs),
        "extras": {
            c: {
                "gil_wait": pb83.gil_wait(x),
                "stages": pb83.in_process_records(x),
                "forwards": pb83.forward_cpu(x),
                "collisions": pb83.collisions(x, threshold),
                "windows": windows_records(x, None if c == "P" else rs["P"]),
                "load": pb83.load_time(x),
            }
            for c, x in rs.items()
        },
    }


# ------------------------------------------------------------------ the sequential rule


def interpretation(c: str, pb: str) -> str:
    """criteria.md: C against PB at the final look, non-gating."""
    if pb == gate.PASS and c == gate.FAIL:
        return "#77's C regression reproduced under the full protocol and PB avoids it: supports handoff reduction"
    if pb == gate.PASS and c == gate.PASS:
        return (
            "#77's C regression not reproduced in this campaign: PB's pass is not attributed to handoff "
            "reduction; the #77/#83 divergence stays open"
        )
    return f"C {c}, PB {pb}: no statement on handoff reduction follows from this pair"


def model_result(runs: Runs) -> dict:
    out: dict = {
        "looks": {},
        "final": None,
        "needs_stage": None,
        "needs_stage_started": None,
        "unused_runs": [],
        "report": None,
        "runs_present": len(runs.present()),
    }
    for s in range(1, design.MAX_STAGE + 1):
        if not runs.complete(s):
            out["needs_stage"] = s
            out["needs_stage_started"] = runs.started(s)
            out["status"] = "stage 1 incomplete" if s == 1 else f"extension required (stage {s})"
            break
        lk = look(runs, s)
        out["looks"][str(s)] = lk
        v = lk["verdicts"]["PB"]["verdict"]
        if v in (gate.PASS, gate.FAIL) or s == design.MAX_STAGE:
            out["final"] = {
                "stage": s,
                "pairs": lk["pairs"],
                "verdict": v,
                "text": f"PB {v} at stage {s} ({lk['pairs']} pairs)" if v != gate.INCONCLUSIVE else CAP,
                "C_reference": lk["verdicts"]["C"]["verdict"],
                "interpretation": interpretation(lk["verdicts"]["C"]["verdict"], v),
            }
            out["status"] = "final"
            last = max(lk["rounds"])
            out["unused_runs"] = [
                design.run_file(runs.model, design.config_name(runs.model, c), r) for c, r in runs.present() if r > last
            ]
            break
    if out["looks"]:
        out["report"] = report(runs, out["looks"][str(max(int(k) for k in out["looks"]))])
    return out


def decide(models: dict) -> tuple[str, dict | None]:
    """The overall outcome and the next step (criteria.md), from each model's result."""
    final = {m: x["final"]["verdict"] for m, x in models.items() if x["final"]}
    failed = [m for m, v in final.items() if v == gate.FAIL]
    if failed:
        text = (
            f"PB FAIL on {', '.join(failed)}: R1 FAIL; a protocol-history causal experiment (fixed warm-up / "
            "solo_short / solo_long / gpu_only → hetero; P, C, PB matched) is preregistered next; no Swift/native "
            "worker, no GPU pacing"
        )
        # a FAIL stops only the start of new stages: stage 1, and any stage already started, completes
        started = {
            m: x["needs_stage"]
            for m, x in models.items()
            if x["needs_stage"] is not None and (x["needs_stage"] == 1 or x["needs_stage_started"])
        }
        if not started:
            return text, None
        stage = min(started.values())
        need = [m for m, s in started.items() if s == stage]
        return f"{text} (completing started stage {stage} for {', '.join(need)})", {"stage": stage, "models": need}
    if len(final) == len(models) and all(v == gate.PASS for v in final.values()):
        return f"PB PASS on {' and '.join(models)}: proceed to R2 (production binding feasibility)", None
    pending = {m: x["needs_stage"] for m, x in models.items() if x["needs_stage"] is not None}
    if pending:
        stage = min(pending.values())
        need = [m for m, s in pending.items() if s == stage]
        step = {"stage": stage, "models": need}
        if stage == 1:
            return f"stage 1 incomplete ({', '.join(need)})", step
        return f"extension required: stage {stage} ({design.pairs_at(stage)} pairs) for {', '.join(need)}", step
    capped = [m for m, v in final.items() if v == gate.INCONCLUSIVE]
    return f"PB INCONCLUSIVE at the cap on {', '.join(capped)}: 1.5.0 blocked; next step preregistered separately", None


def summarise(raw: Path, models=None) -> dict:
    models = list(MODELS) if models is None else list(models)
    check = raw / "check.json"
    res = {
        "limits": LIMITS,
        "design": {
            "stages": {
                str(s): {"rounds": list(design.rounds_through(s)), "pairs": design.pairs_at(s)} for s in (1, 2, 3)
            },
            "bonferroni_confidence": BONFERRONI_CONFIDENCE,
        },
        "check": json.loads(check.read_text()) if check.exists() else None,
        # run_all.sh keeps the console output of every crashed run there (criteria.md, crash rule)
        "failed_runs": sorted(p.name for p in (raw / "failed").glob("*.log")),
        "models": {m: model_result(Runs(raw, m)) for m in models},
    }
    res["outcome"], res["next"] = decide(res["models"])
    if set(models) != set(MODELS):
        res["outcome"] = f"partial model set ({', '.join(models)}): no preregistered outcome ({res['outcome']})"
    return res


# ------------------------------------------------------------------ tables


f = pb83.f


def ci_text(v: dict, key: str, crit: str, kind: str = "t") -> str:
    if kind == "bonferroni":
        b = v["bonferroni_sensitivity_not_the_verdict"][crit]
        return f"[{b['lo']:.3f}, {b['hi']:.3f}] {b['would_be']}"
    t = v["stats"][key][kind]
    verdict = v["checks"][crit] if kind == "t" else v["bootstrap_sensitivity"][crit]
    head = f"{t['geomean']:.3f} " if kind == "t" else ""
    return f"{head}[{t['lo']:.3f}, {t['hi']:.3f}] {verdict}"


def tables(res: dict) -> str:
    L = ["# R1: prebound predict under the full #57 product-mix protocol\n"]
    chk = res.get("check")
    L.append(
        "Pre-campaign check (`raw/check.json`): "
        + (
            "not run yet"
            if not chk
            else f"PB bit-identical everywhere {chk['PB_bit_identical_everywhere']}, "
            f"C bit-identical everywhere {chk['C_bit_identical_everywhere']}"
        )
        + "\n"
    )
    L += ["| model | status | final look | PB verdict | C (reference) | runs present | not used for the verdict |"]
    L.append("|---|---|---|---|---|---|---|")
    for m, x in res["models"].items():
        fin = x["final"]
        where = "–" if not fin else f"stage {fin['stage']} ({fin['pairs']} pairs)"
        L.append(
            f"| {m} | {x['status']} | {where} | {'–' if not fin else '**' + fin['text'] + '**'} "
            f"| {'–' if not fin else fin['C_reference']} | {x['runs_present']} "
            f"| {', '.join(x['unused_runs']) or '–'} |"
        )
    L.append(f"\nOutcome: {res['outcome']}\n")
    L.append(f"Crashed runs and re-runs (`raw/failed/`): {', '.join(res['failed_runs']) or 'none'}\n")
    nxt = res["next"]
    step = (
        "nothing (R1 has reached its outcome)"
        if nxt is None
        else f"stage {nxt['stage']} for {', '.join(nxt['models'])}"
    )
    L.append(f"Next: {step}\n")
    for m, x in res["models"].items():
        if x["final"]:
            L.append(f"- {m}: {x['final']['interpretation']}")
    L += [
        "\n## Criteria at each look: paired gate against P (A), 95% t (primary)\n",
        "Geometric mean of the matched-pair ratios (same round, same cycle) with its 95% t-interval "
        "(df = n − 1). PB decides; C is a reference judged at the same looks.\n",
        "| model | look | candidate | pairs | aggregate ≥ 0.95× | short P99 ≤ 1.05× | long P99 ≤ 1.05× "
        "| GPU return P50 ≤ 1 ms and ≥ 5× vs A | correctness | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        for s, lk in x["looks"].items():
            for c, v in lk["verdicts"].items():
                L.append(
                    f"| {m} | {s} | {c} vs A | {v['pairs']} | {ci_text(v, 'aggregate', 'aggregate_throughput')} "
                    f"| {ci_text(v, 'short_p99', 'short_p99')} | {ci_text(v, 'long_p99', 'long_p99')} "
                    f"| ×{f(v['gpu_return_p50_gain'], 1)} {v['checks']['gpu_completion_isolation']} "
                    f"| {v['checks']['correctness']} | **{v['verdict']}** |"
                )
    L += [
        "\nSensitivity only (never the verdict): seeded percentile bootstrap over pairs (10,000 resamples), "
        f"and the Bonferroni-over-three-looks t-interval (confidence {BONFERRONI_CONFIDENCE:.4f}) with the "
        "verdict it would give.\n",
        "| model | look | candidate | bootstrap aggregate | bootstrap short P99 | bootstrap long P99 "
        "| Bonferroni aggregate | Bonferroni short P99 | Bonferroni long P99 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        for s, lk in x["looks"].items():
            for c, v in lk["verdicts"].items():
                L.append(
                    f"| {m} | {s} | {c} vs A "
                    + "".join(f"| {ci_text(v, k, crit, 'bootstrap')} " for k, crit, _, _ in CRITERIA)
                    + "".join(f"| {ci_text(v, k, crit, 'bonferroni')} " for k, crit, _, _ in CRITERIA)
                    + "|"
                )
    L += [
        "\n## Per configuration at the reported look (pooled; not the verdict)\n",
        "| model | look | config | aggregate req/s | short req/s | short P99 | long req/s | long P99 "
        "| GPU return P50 / P99 | mismatches | pooled ratios vs A (aggregate / short P99 / long P99) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        if not x["report"]:
            continue
        lk = x["looks"][str(x["report"]["look"])]
        for c, s in lk["configs"].items():
            sh, lo, g = s["streams"]["short"], s["streams"]["long"], s["gpu_return_ms"]
            p = lk["verdicts"][c]["pooled_point_estimate_not_the_verdict"] if c in CANDIDATES else None
            pooled = "–" if p is None else f"{f(p['aggregate'], 3)} / {f(p['short_p99'], 3)} / {f(p['long_p99'], 3)}"
            L.append(
                f"| {m} | {lk['stage']} | {'A (P)' if c == 'P' else c} | {f(s['aggregate_req_s'], 1)} "
                f"| {f(sh['req_s_median'], 1)} | {f(sh['p99_ms_median'])} | {f(lo['req_s_median'], 1)} "
                f"| {f(lo['p99_ms_median'])} | {f(g['p50'], 3)} / {f(g['p99'], 3)} | {s['mismatches_all_windows']} "
                f"| {pooled} |"
            )
        for e in x["report"]["routing_failures"]:
            L.append(
                f"\n**{m}: hetero {e['stream']} stream not served entirely by its device: {e['config']} round "
                f"{e['round']} cycle {e['cycle']}, devices {e['devices']}**\n"
            )
    L += [
        "\n## Not gating: window history (the condition run just before each hetero window)\n",
        "bench_concurrency alternates the order per cycle, so a hetero window follows solo_long in even cycles "
        "and gpu_only in odd cycles. Candidate columns: geometric mean of the matched-pair ratios to A.\n",
        "| model | look | preceded by | A short P99 median | C short P99 median | PB short P99 median "
        "| C/A short P99 (pairs) | PB/A short P99 (pairs) | C/A aggregate | PB/A aggregate |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        if not x["report"]:
            continue
        h = x["report"]["window_history"]
        for g in sorted(h["by_config"]["P"]):
            b = {c: h["by_config"][c].get(g) for c in design.CONFIGS}
            r = {c: h["vs_P"][c].get(g) for c in CANDIDATES}

            def med(c, b=b):
                return "–" if b[c] is None else f(b[c]["short_p99_median_ms"])

            def rat(c, k, r=r):
                return "–" if r[c] is None else f(r[c][k], 3)

            L.append(
                f"| {m} | {x['report']['look']} | {g} | {med('P')} | {med('C')} | {med('PB')} "
                f"| {rat('C', 'short_p99_geomean_ratio')} ({'–' if r['C'] is None else r['C']['pairs']}) "
                f"| {rat('PB', 'short_p99_geomean_ratio')} ({'–' if r['PB'] is None else r['PB']['pairs']}) "
                f"| {rat('C', 'aggregate_geomean_ratio')} | {rat('PB', 'aggregate_geomean_ratio')} |"
            )
        if not h["order_as_expected"]:
            L.append(f"\n**{m}: the window order differs from bench_concurrency's alternation**\n")
    L += [
        "\n## Not gating: crossings, GIL re-acquire wait, ANE stages (hetero windows)\n",
        "| model | config | crossings per forward | re-acquire P50 / P99 / mean ms | features | pre | native "
        "| re-acquire | post | tail | predict (whole) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        if not x["report"]:
            continue
        for c, e in x["report"]["extras"].items():
            st, g = e["stages"] or {}, e["gil_wait"]
            b = pb83.short_bucket(e["load"], x["report"]["args"]["short"])

            def p50(k, st=st):
                return f(st[k]["p50_ms"], 3) if k in st else "–"

            wait = "–" if g is None else " / ".join(f(g[k], 3) for k in ("p50_ms", "p99_ms", "mean_ms"))
            L.append(
                f"| {m} | {c} | {pb83.crossing_text(e['load'], b) if b else '–'} | {wait} | {p50('features')} "
                f"| {p50('pre')} | {p50('native')} | {p50('reacquire')} | {p50('post')} | {p50('tail')} "
                f"| {p50('predict')} |"
            )
    L += [
        "\n## Not gating: GPU completion collisions and the ANE tail\n",
        "Fixed exposure window: a GPU `received_ns` in [ANE service_start − 1 ms, service_start + 0.3 ms]. "
        "Tail: ANE request latency above P's pooled hetero P99. Odds ratio with 0.5 added to each cell.\n",
        "| model | config | tail threshold ms | in tail | colliding | tail requests colliding | odds ratio "
        "| GPU completions overlapping a forward (mean / any) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        if not x["report"]:
            continue
        for c, e in x["report"]["extras"].items():
            k = e["collisions"]
            L.append(
                f"| {m} | {c} | {f(k['threshold_ms'])} | {f(k['tail_fraction'], 3)} | {f(k['collide_fraction'], 3)} "
                f"| {f(k['tail_colliding_fraction'], 3)} | {f(k['odds_ratio'], 1)} "
                f"| {f(k['forward_overlap_mean'], 3)} / {f(k['forward_overlap_any_fraction'], 3)} |"
            )
    L += [
        "\n## Not gating: thread CPU and the slow-CPU flag\n",
        "Thread CPU: ms per hetero window (median). Slow-CPU flag: candidate windows whose client-short CPU per "
        "request exceeds 1.25× that of A's matched window.\n",
        "| model | config | client-short | client-long | ane-dispatch | gpu-dispatch | ANE thread CPU/forward "
        "| GPU thread CPU/forward | slow-CPU windows |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        if not x["report"]:
            continue
        for c, e in x["report"]["extras"].items():
            w, fc = e["windows"], e["forwards"]
            t, sc = w["thread_cpu_ms_per_window_median"], w["slow_cpu"]
            if sc is None:
                slow = "– (reference)"
            elif not sc["computable_windows"]:
                slow = f"not computable (0 of {len(sc['flags'])})"
            else:
                slow = f"{sc['flagged_windows']} of {sc['computable_windows']} computable ({len(sc['flags'])} pairs)"
            L.append(
                f"| {m} | {c} | {f(t.get('client-short'), 0)} | {f(t.get('client-long'), 0)} "
                f"| {f(t.get('laya-ane-dispatch'), 0)} | {f(t.get('laya-gpu-dispatch'), 0)} "
                f"| {f(None if fc['ane'] is None else fc['ane']['thread_cpu_ms_mean'], 3)} "
                f"| {f(None if fc['gpu'] is None else fc['gpu']['thread_cpu_ms_mean'], 3)} | {slow} |"
            )
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if results.json / tables.md are stale")
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=EXP)
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    a = ap.parse_args()
    res = summarise(a.raw, a.models)
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
