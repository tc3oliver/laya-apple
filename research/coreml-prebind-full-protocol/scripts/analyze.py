"""R1, prebound predict under #77's full #57 protocol: results.json and tables.md from raw/.

    uv run python research/coreml-prebind-full-protocol/scripts/analyze.py [--check]

Inputs: raw/<model>-<config>-r<round>.json.gz (run_all.sh, design.py's blocks), raw/check.json
(#83's check_prebind.py) and raw/failed/*.log (run_all.sh's crash rule). The criteria and the
"fast fail, slow pass" addendum are in ../criteria.md, committed before any data was read; this
file implements them.
  - Interim looks, PB against P, futility only (FUTILITY STOP = FAIL, or CONTINUE; never PASS):
    n=6 on rounds 1-2 once both models' block 1 exists, n=12 on rounds 1-4 right after a model's
    block 2. Stop on: any mismatch in a P or PB run; GPU completion isolation failing (#77's
    `verdict`, >= 5x against A); a 99% t-interval entirely beyond a budget (short or long P99
    lower bound > 1.05x, aggregate upper bound < 0.95x); a PB run's second crash. A stop ends R1
    for both models.
  - Final look, n=18 on rounds 1-6: #83's paired gate unchanged (95% t, bootstrap sensitivity
    only, correctness, isolation). PASS needs both models; any FAIL is R1 FAIL; otherwise
    INCONCLUSIVE and the campaign stops, with the pairs a replication would need per criterion.
  - C: judged once, n=6 on rounds 1-2 against P's rounds 1-2 (#83's gate, reference only).
  - A second crash of a P or C run stops the campaign as invalid, not as a verdict.
  - Non-gating extras reuse #83's analyze.py (loaded by path), with an R1 variant of its
    per-window records (R1 runs no GIL probe), plus the window-history diagnostic.

--raw / --out / --models exist for checking the harness on runs kept outside the repository.
The campaign uses the defaults.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
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
LIMITS = gate57.LIMITS  # the #57 thresholds, unchanged
FUTILITY_CONFIDENCE = 0.99
PAIRS_SEARCH_MAX = 10_000
CRITERIA = (  # (pair-ratio key, criterion, verdict function, limit key)
    ("aggregate", "aggregate_throughput", gate.lower_verdict, "aggregate_min_ratio"),
    ("short_p99", "short_p99", gate.upper_verdict, "p99_max_ratio"),
    ("long_p99", "long_p99", gate.upper_verdict, "p99_max_ratio"),
)
RUN_NAME = re.compile(r"^(?P<model>.+)-(?P<config>A|C|PB)-r(?P<round>\d+)\.json\.gz$")
FAILED_NAME = re.compile(r"^(?P<run>.+-(?:A|C|PB)-r\d+)\.(?P<attempt>\d+)\.log$")
NEXT_STEP = "protocol-history experiment next"


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

    def block_complete(self, block: int) -> bool:
        return all((self.raw / design.run_file(m, c, r)).exists() for m, c, r in design.block_runs(self.model, block))

    def get(self, config: str, rounds) -> list[dict]:
        out = []
        for r in rounds:
            if (config, r) not in self._cache:
                p = self.path(config, r)
                run = load(p)
                assert run["research"]["config"] == design.config_name(self.model, config), p
                assert run["args"]["short"] == self.short and run["args"]["long"] == self.long, p
                assert run["args"]["seconds"] == design.SECONDS and run["args"]["cycles"] == design.CYCLES, p
                assert len(gate.hetero_by_cycle(run)) == design.CYCLES, p
                self._cache[(config, r)] = run
            out.append(self._cache[(config, r)])
        return out

    def present(self) -> list[str]:
        """This model's run files on disk (exact model name: laya does not match laya-typed-...)."""
        return sorted(
            p.name
            for p in self.raw.glob(f"{self.model}-*-r*.json.gz")
            if (x := RUN_NAME.match(p.name)) and x["model"] == self.model
        )


def crashes(raw: Path) -> dict[str, int]:
    """{run name: failure logs} from raw/failed (run_all.sh's crash rule)."""
    out: dict[str, int] = {}
    for p in sorted((raw / "failed").glob("*.log")):
        if x := FAILED_NAME.match(p.name):
            out[x["run"]] = out.get(x["run"], 0) + 1
    return out


# ------------------------------------------------------------------ looks


def futility_look(runs: Runs, block: int) -> dict:
    """Interim look after `block` (1 or 2): PB against P, 99% t-intervals; FUTILITY STOP or CONTINUE."""
    rounds = design.rounds_through(block)
    p, pb = runs.get("P", rounds), runs.get("PB", rounds)
    sp, spb = gate57.placement_summary(p), gate57.placement_summary(pb)
    point = mix77.verdict(sp, spb, gain_applies=True)
    ratios = gate.pair_ratios(gate.pair_windows(p, pb))
    intervals, triggered = {}, []
    if not point["checks"]["correctness"]:
        triggered.append("correctness")
    if not point["checks"]["gpu_completion_isolation"]:
        triggered.append("gpu_completion_isolation")
    for key, crit, fn, lim in CRITERIA:
        ci = gate.t_interval(ratios[key], confidence=FUTILITY_CONFIDENCE)
        v = fn(ci, LIMITS[lim])
        intervals[crit] = {**ci, "confidence": FUTILITY_CONFIDENCE, "verdict_99": v}
        if v == gate.FAIL:  # PASS and INCONCLUSIVE at 99% both mean continue
            triggered.append(crit)
    return {
        "kind": "futility",
        "block": block,
        "rounds": list(rounds),
        "pairs": len(ratios["short_p99"]),
        "decision": "FUTILITY STOP" if triggered else "CONTINUE",
        "triggered": triggered,
        "intervals_99": intervals,
        "mismatches": {"P": sp["mismatches_all_windows"], "PB": spb["mismatches_all_windows"]},
        "gpu_return_p50_ms": {"P": sp["gpu_return_ms"]["p50"], "PB": spb["gpu_return_ms"]["p50"]},
        "gpu_return_p50_gain": point["ratios"]["gpu_return_p50_gain"],
        "configs": {"P": sp, "PB": spb},
    }


def paired(prod_runs, cand_runs) -> dict:
    """#83's paired gate, unchanged, with #77's correctness and isolation (>= 5x against A)."""
    sp, sc = gate57.placement_summary(prod_runs), gate57.placement_summary(cand_runs)
    point = mix77.verdict(sp, sc, gain_applies=True)
    v = gate.paired_verdict(
        prod_runs,
        cand_runs,
        LIMITS,
        correctness=point["checks"]["correctness"],
        isolation=point["checks"]["gpu_completion_isolation"],
    )
    return {
        **v,
        "gain_applies": True,
        "gpu_return_p50_gain": point["ratios"]["gpu_return_p50_gain"],
        "pooled_point_estimate_not_the_verdict": point["ratios"],
        "configs": {"P": sp, "candidate": sc},
    }


def pairs_needed(sd: float, geomean: float, limit: float, confidence: float = gate.CONFIDENCE) -> int | None:
    """Smallest n with t(confidence, n - 1) * sd / sqrt(n) < |log(limit) - log(geomean)|, n <= 10000;
    None if the margin is 0 or not reached. t >= z, so the search starts at (z * sd / margin)^2."""
    margin = abs(math.log(limit) - math.log(geomean))
    if margin <= 0:
        return None
    if sd == 0:
        return 2
    z = gate.t_critical(PAIRS_SEARCH_MAX, confidence)  # lower bound of every t below 10000 df
    n = max(2, math.ceil((z * sd / margin) ** 2))
    while n <= PAIRS_SEARCH_MAX:
        if gate.t_critical(n - 1, confidence) * sd / math.sqrt(n) < margin:
            return n
        n += 1
    return None


def final_look(runs: Runs) -> dict:
    rounds = design.rounds_through(3)
    v = paired(runs.get("P", rounds), runs.get("PB", rounds))
    inconclusive = {}
    for key, crit, _, lim in CRITERIA:
        if v["checks"][crit] != gate.INCONCLUSIVE:
            continue
        t = v["stats"][key]["t"]
        sd = float(np.std(np.log(v["stats"][key]["ratios"]), ddof=1))
        n = pairs_needed(sd, t["geomean"], LIMITS[lim])
        inconclusive[crit] = {
            "geomean": t["geomean"],
            "ci95": [t["lo"], t["hi"]],
            "pair_log_sd": sd,
            "pairs_needed": n,
            "pairs_needed_text": str(n) if n is not None else "not resolvable at this effect size",
        }
    return {"kind": "final", "block": 3, "rounds": list(rounds), **v, "inconclusive": inconclusive}


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
    the matched-pair ratios to P (P's rounds 1..k for a candidate with k rounds)."""
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
    for c in (c for c in ("C", "PB") if rs.get(c)):
        prod = rs["P"][: len(rs[c])]
        prev_p = [preceding(r) for r in prod]
        groups = {}
        for rnd, k, p, w in gate.pair_windows(prod, rs[c]):
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


def report(runs: Runs, rounds, c_rounds) -> dict:
    """Extras on the rounds of the latest look (P, PB) and on rounds 1-2 (C)."""
    rs = {"P": runs.get("P", rounds), "PB": runs.get("PB", rounds), "C": runs.get("C", c_rounds) if c_rounds else []}
    prod_for = {"P": None, "PB": rs["P"], "C": rs["P"][: len(rs["C"])]}
    threshold = pb83.ane_tail_threshold(rs["P"])
    cfg_rounds = {"P": list(rounds), "PB": list(rounds), "C": list(c_rounds)}
    routing = [  # every hetero stream window not served entirely by its intended device
        {"config": c, "round": rnd, "cycle": k, "stream": st, "devices": w["streams"][st]["devices"]}
        for c, x in rs.items()
        for rnd, r in zip(cfg_rounds[c], x)
        for k, w in sorted(gate.hetero_by_cycle(r).items())
        for st, dev in (("short", "ane"), ("long", "gpu"))
        if set(w["streams"][st]["devices"]) != {dev}
    ]
    return {
        "rounds": {c: cfg_rounds[c] for c in rs if rs[c]},
        "routing_failures": routing,
        "args": {k: rs["P"][0]["args"][k] for k in ("short", "long", "seconds", "cycles")},
        "versions": {k: rs["PB"][0]["research"][k] for k in ("laya_apple", "pyobjc", "coremltools", "mlx")},
        "window_history": window_history({c: x for c, x in rs.items() if x}),
        "extras": {
            c: {
                "gil_wait": pb83.gil_wait(x),
                "stages": pb83.in_process_records(x),
                "forwards": pb83.forward_cpu(x),
                "collisions": pb83.collisions(x, threshold),
                "windows": windows_records(x, prod_for[c]),
                "load": pb83.load_time(x),
            }
            for c, x in rs.items()
            if x
        },
    }


# ------------------------------------------------------------------ the rule


def interpretation(c: str, pb: str) -> str:
    """criteria.md: C's n=6 reference verdict against PB's outcome, non-gating."""
    if pb == gate.PASS and c == gate.FAIL:
        return "#77's C regression reproduced under the full protocol and PB avoids it: supports handoff reduction"
    if pb == gate.PASS and c == gate.PASS:
        return (
            "#77's C regression not reproduced in this campaign: PB's pass is not attributed to handoff "
            "reduction; the #77/#83 divergence stays open"
        )
    return f"C {c} (n=6), PB {pb}: no statement on handoff reduction follows from this pair"


def summarise(raw: Path, models=None) -> dict:
    models = list(MODELS) if models is None else list(models)
    check = raw / "check.json"
    runs = {m: Runs(raw, m) for m in models}
    crash = crashes(raw)
    per = {m: {"looks": {}, "C_reference": None, "status": "running"} for m in models}
    used: dict[str, set] = {m: set() for m in models}
    outcome, step = None, None

    order = design.blocks(models)
    for m, b in order:
        if not runs[m].block_complete(b):
            step = {"model": m, "block": b}
            break
        # the looks this block makes due: n=12 right after the model's own block 2; n=6 and n=18 once
        # the last model's block 1 / block 3 exists (the loop reaches it only if the others exist)
        due = [m] if b == 2 else (models if m == models[-1] else [])
        for x in due:
            used[x] |= {design.run_file(*r) for bb in range(1, b + 1) for r in design.block_runs(x, bb)}
            if b == 3:
                per[x]["looks"]["18"] = final_look(runs[x])
            else:
                per[x]["looks"][str(design.pairs_at(b))] = futility_look(runs[x], b)
            if b == 1:
                per[x]["C_reference"] = paired(runs[x].get("P", (1, 2)), runs[x].get("C", (1, 2)))
        n = str(design.pairs_at(b))
        stopped = [x for x in due if per[x]["looks"][n].get("decision") == "FUTILITY STOP"]
        if stopped:
            parts = "; ".join(f"{x}: {', '.join(per[x]['looks'][n]['triggered'])}" for x in stopped)
            outcome = f"FUTILITY STOP at n={n} on {parts}; R1 FAIL; {NEXT_STEP}"
            break

    # second crashes (run_all.sh also stops on them): PB is futility, P or C invalidates the campaign
    second = sorted(r for r, k in crash.items() if k >= 2 and RUN_NAME.match(r + ".json.gz")["model"] in models)
    pb_second = [r for r in second if RUN_NAME.match(r + ".json.gz")["config"] == "PB"]
    other_second = [r for r in second if r not in pb_second]
    if outcome is None and pb_second:
        for r in pb_second:
            m = RUN_NAME.match(r + ".json.gz")["model"]
            n = max((int(k) for k in per[m]["looks"]), default=0)
            per[m]["status"] = f"FUTILITY STOP at n={n}"
        n = max((int(k) for x in per.values() for k in x["looks"]), default=0)
        outcome = f"FUTILITY STOP at n={n} on {', '.join(pb_second)}: PB second crash; R1 FAIL; {NEXT_STEP}"
    if outcome is None and other_second:
        outcome = f"invalid: second crash of {', '.join(other_second)}; campaign stopped, no verdict"

    finals = {m: per[m]["looks"]["18"]["verdict"] for m in models if "18" in per[m]["looks"]}
    if outcome is None and len(finals) == len(models):
        if all(v == gate.PASS for v in finals.values()):
            outcome = f"PB PASS on {' and '.join(models)} at n=18: proceed to R2"
        elif failed := [m for m, v in finals.items() if v == gate.FAIL]:
            outcome = f"PB FAIL at n=18 on {', '.join(failed)}: R1 FAIL; {NEXT_STEP}"
        else:
            undecided = [m for m, v in finals.items() if v == gate.INCONCLUSIVE]
            outcome = (
                f"INCONCLUSIVE at n=18 on {', '.join(undecided)}: campaign stopped; a targeted replication needs "
                "a new preregistration"
            )
    if outcome is None:
        outcome = f"running: next block {step['block']} of {step['model']}"
    else:
        step = None

    designed = design.designed_files(models)
    for m in models:
        x = per[m]
        looks = x["looks"]
        if looks:
            k = max(looks, key=int)
            if looks[k]["kind"] == "final":
                x["status"] = f"final: PB {looks[k]['verdict']} at n=18"
            elif looks[k]["decision"] == "FUTILITY STOP":
                x["status"] = f"FUTILITY STOP at n={k}"
            elif step is None and not x["status"].startswith("FUTILITY"):
                x["status"] = "stopped (R1 outcome reached)"
            x["report"] = report(runs[m], looks[k]["rounds"], (1, 2))
        else:
            x["report"] = None
        pb_final = None
        if looks.get("18"):
            pb_final = looks["18"]["verdict"]
        elif x["status"].startswith("FUTILITY"):
            pb_final = gate.FAIL
        x["interpretation"] = (
            interpretation(x["C_reference"]["verdict"], pb_final) if x["C_reference"] and pb_final else None
        )
        present = runs[m].present()
        keep = used[m] if step is None else designed
        x["runs_present"] = len(present)
        x["unused_runs"] = [p for p in present if p not in keep]
        x["crashes"] = {r: k for r, k in crash.items() if RUN_NAME.match(r + ".json.gz")["model"] == m}
    res = {
        "limits": LIMITS,
        "design": {
            "blocks": {str(b): list(o) for b, o in design.BLOCK_ORDERS.items()},
            "looks": {str(n): kind for kind, n in design.LOOKS.values()},
            "futility_confidence": FUTILITY_CONFIDENCE,
        },
        "check": json.loads(check.read_text()) if check.exists() else None,
        "failed_runs": sorted(p.name for p in (raw / "failed").glob("*.log")),
        "models": per,
        "outcome": outcome,
        "next": step,
    }
    if set(models) != set(MODELS):
        res["outcome"] = f"partial model set ({', '.join(models)}): no preregistered outcome ({outcome})"
    return res


# ------------------------------------------------------------------ tables


f = pb83.f


def ci_text(ci: dict, verdict: str, head: bool = True) -> str:
    lead = f"{ci['geomean']:.3f} " if head else ""
    return f"{lead}[{ci['lo']:.3f}, {ci['hi']:.3f}] {verdict}"


def tables(res: dict) -> str:
    L = ["# R1: prebound predict under the full #57 product-mix protocol (fast fail, slow pass)\n"]
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
    L += [
        "| model | status | C at n=6 (reference) | runs present | not used for the verdict |",
        "|---|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        c = x["C_reference"]
        L.append(
            f"| {m} | {x['status']} | {'–' if not c else c['verdict']} | {x['runs_present']} "
            f"| {', '.join(x['unused_runs']) or '–'} |"
        )
    L.append(f"\nOutcome: {res['outcome']}\n")
    L.append(f"Crashed runs and re-runs (`raw/failed/`): {', '.join(res['failed_runs']) or 'none'}\n")
    for m, x in res["models"].items():
        if x["interpretation"]:
            L.append(f"- {m}: {x['interpretation']}")
    L += [
        "\n## Interim looks: PB against A, futility only (99% t-intervals)\n",
        "FUTILITY STOP if any mismatch, isolation fails, or a 99% interval lies entirely beyond a budget "
        "(short or long P99 lower bound > 1.05×, aggregate upper bound < 0.95×). Never a PASS.\n",
        "| model | n | aggregate | short P99 | long P99 | GPU return P50 A / PB (gain) | mismatches A / PB "
        "| decision |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        for n, lk in x["looks"].items():
            if lk["kind"] != "futility":
                continue
            iv = lk["intervals_99"]
            g = lk["gpu_return_p50_ms"]
            L.append(
                f"| {m} | {n} | {ci_text(iv['aggregate_throughput'], iv['aggregate_throughput']['verdict_99'])} "
                f"| {ci_text(iv['short_p99'], iv['short_p99']['verdict_99'])} "
                f"| {ci_text(iv['long_p99'], iv['long_p99']['verdict_99'])} "
                f"| {f(g['P'], 3)} / {f(g['PB'], 3)} (×{f(lk['gpu_return_p50_gain'], 1)}) "
                f"| {lk['mismatches']['P']} / {lk['mismatches']['PB']} "
                f"| **{lk['decision']}**{' (' + ', '.join(lk['triggered']) + ')' if lk['triggered'] else ''} |"
            )
    L += [
        "\n## Paired gate (95% t): PB at n=18 (the verdict), C at n=6 (reference only)\n",
        "| model | candidate | pairs | aggregate ≥ 0.95× | short P99 ≤ 1.05× | long P99 ≤ 1.05× "
        "| GPU return P50 ≤ 1 ms and ≥ 5× vs A | correctness | verdict | bootstrap (aggregate / short / long) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        rows = [("C (n=6)", x["C_reference"])] + [("PB (n=18)", x["looks"].get("18"))]
        for name, v in rows:
            if not v:
                continue
            cells = [ci_text(v["stats"][k]["t"], v["checks"][crit]) for k, crit, _, _ in CRITERIA]
            boot = " / ".join(
                ci_text(v["stats"][k]["bootstrap"], v["bootstrap_sensitivity"][crit], head=False)
                for k, crit, _, _ in CRITERIA
            )
            L.append(
                f"| {m} | {name} | {v['pairs']} | {' | '.join(cells)} "
                f"| ×{f(v['gpu_return_p50_gain'], 1)} {v['checks']['gpu_completion_isolation']} "
                f"| {v['checks']['correctness']} | **{v['verdict']}** | {boot} |"
            )
    inc = [
        (m, c, d)
        for m, x in res["models"].items()
        if "18" in x["looks"]
        for c, d in x["looks"]["18"]["inconclusive"].items()
    ]
    if inc:
        L += [
            "\nInconclusive criteria at n=18: pairs a replication would need at this effect size and spread "
            "(smallest n with t(0.975, n − 1)·SD/√n below the distance of the geometric mean from the budget).\n",
            "| model | criterion | geomean | 95% CI | pair log-SD | pairs needed |",
            "|---|---|---|---|---|---|",
        ]
        for m, c, d in inc:
            L.append(
                f"| {m} | {c} | {d['geomean']:.3f} | [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}] "
                f"| {d['pair_log_sd']:.4f} | {d['pairs_needed_text']} |"
            )
    for m, x in res["models"].items():
        for e in (x["report"] or {}).get("routing_failures", []):
            L.append(
                f"\n**{m}: hetero {e['stream']} stream not served entirely by its device: {e['config']} round "
                f"{e['round']} cycle {e['cycle']}, devices {e['devices']}**\n"
            )
    L += [
        "\n## Not gating: window history (the condition run just before each hetero window)\n",
        "bench_concurrency alternates the order per cycle, so a hetero window follows solo_long in even cycles "
        "and gpu_only in odd cycles. Ratios: geometric mean of the matched-pair ratios to A (C on rounds 1–2).\n",
        "| model | preceded by | A short P99 median | C short P99 median | PB short P99 median "
        "| C/A short P99 (pairs) | PB/A short P99 (pairs) | C/A aggregate | PB/A aggregate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m, x in res["models"].items():
        if not x["report"]:
            continue
        h = x["report"]["window_history"]
        for g in sorted(h["by_config"]["P"]):
            b = {c: h["by_config"].get(c, {}).get(g) for c in design.CONFIGS}
            r = {c: h["vs_P"].get(c, {}).get(g) for c in ("C", "PB")}

            def med(c, b=b):
                return "–" if b[c] is None else f(b[c]["short_p99_median_ms"])

            def rat(c, k, r=r):
                return "–" if r[c] is None else f(r[c][k], 3)

            def npairs(c, r=r):
                return "–" if r[c] is None else r[c]["pairs"]

            L.append(
                f"| {m} | {g} | {med('P')} | {med('C')} | {med('PB')} "
                f"| {rat('C', 'short_p99_geomean_ratio')} ({npairs('C')}) "
                f"| {rat('PB', 'short_p99_geomean_ratio')} ({npairs('PB')}) "
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
        "Tail: ANE request latency above A's pooled hetero P99. Odds ratio with 0.5 added to each cell.\n",
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
