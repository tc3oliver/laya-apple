"""Product-mix proxies (G4): does the split make the single-question class's P99 worse?

    LAYA_APPLE_CACHE=<cache> HF_HUB_OFFLINE=1 uv run --extra ane --extra convert python \\
        research/intra-request-split/scripts/mix.py --mix switchyard --run-id switchyard
    ... --mix serve --run-id serve

Timing-sensitive: exclusive machine, as for latency.py. One process, one product instance
(common.open_laya, no extra bucket), rounds in the ABBA order of criteria.json
("protocol.mix.blocks"): "base" submits every request with Laya.submit (laya-apple 1.5 as
shipped); "split" submits multi-question requests through split.SplitSubmitter and every
single-question request with Laya.submit. Both replay the identical seeded open-loop timetable.

- switchyard: the switchyard-v1 workload (laya_apple.demos.switchyard.world.schedule, the frozen
  60 s timetable, its payloads, laya-typed-decisions) without the game, the gpu_only round or
  its result schema. Measured class: trains (1 question, <= 128 tokens); metric: completion -
  scheduled arrival; late = over 100 ms; answers checked against the oracle platform.
- serve: benchmarks/serve's decision timetable (scripts/bench_serve.client_schedule: 8 clients,
  Poisson 1 req/s each, 80% short_1q at 96 tokens, 20% mixed_3q at 128 tokens, seed 11, its
  build_requests), in process, without HTTP and without the LLM. Measured class: short_1q.

Completion is the moment the request's Future resolves (a done-callback timestamp), for both
configs. Before the first round, every distinct multi-question payload is answered once, idle,
with Laya.submit: that is the correctness reference for the multi-question answers of both
configs (FP16 gate).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import design  # noqa: E402
import split  # noqa: E402

SLEEP_S = 0.0005


def _wait_until(target_ns: int) -> None:
    while time.monotonic_ns() < target_ns:
        time.sleep(SLEEP_S)


def switchyard_workload(laya):
    from laya_apple.demos.switchyard import driver, world

    def build(items):
        return [
            {
                "t": a.offset_s,
                "cls": a.cls,
                "payload": pl,
                "oracle": a.train.oracle if a.train else None,
            }
            for a, pl in zip(items, driver.payloads(items, laya.tokenizer, laya.config))
        ]

    return {
        "measured_class": world.TRAIN,
        "question_id": world.QUESTION_ID,
        "deadline_ms": world.DEADLINE_MS,
        "warmup": build(world.warmup_schedule()),
        "round": build(world.schedule()),
        "schedule_sha256": world.schedule_sha256(world.schedule()),
    }


def serve_workload(laya, cfg: dict):
    bs = common.bench_serve()
    reqs = bs.build_requests("laya", cfg["short_len"], cfg["mixed_len"], cfg["short_variants"], cfg["mixed_variants"])
    mix = {"short_1q": cfg["short_share"], "mixed_3q": 1 - cfg["short_share"]}
    variants = {"short_1q": cfg["short_variants"], "mixed_3q": cfg["mixed_variants"]}

    def build(seconds, seed):
        sched = bs.client_schedule(cfg["clients"], cfg["rate_per_client"], seconds, seed, mix, variants)
        out = []
        for x in sched:
            r = reqs[x["cls"]][x["variant"]]
            out.append({"t": x["t"], "cls": x["cls"], "payload": (r["state"], r["questions"]), "oracle": None})
        return out

    return {
        "measured_class": "short_1q",
        "question_id": None,
        "deadline_ms": None,
        "warmup": build(cfg["warmup_s"], cfg["seed"] + 1),
        "round": build(cfg["window_s"], cfg["seed"]),
        "schedule_sha256": None,
    }


def run_round(laya, sub, items, config: str):
    """Open loop on the schedule; returns per-request records (after every request finished)."""
    lead_ns = int(0.5 * 1e9)
    start_ns = time.monotonic_ns() + lead_ns
    done_ns: dict = {}
    futs = []
    for i, it in enumerate(items):
        sched = start_ns + round(it["t"] * 1e9)
        _wait_until(sched)
        state, questions = it["payload"]
        sent = time.monotonic_ns()
        if config == "split" and len(questions) > 1:
            f = sub.submit(state, questions, mode="split")
        else:
            f = laya.submit(context=state, questions=questions)
        f.add_done_callback(lambda _f, i=i: done_ns.__setitem__(i, time.monotonic_ns()))
        futs.append((i, it, sched, sent, f))
    recs = []
    for i, it, sched, sent, f in futs:
        res = f.result(timeout=600)
        while i not in done_ns:  # the callback may still be running on the resolving thread
            time.sleep(SLEEP_S)
        rec = {
            "i": i,
            "cls": it["cls"],
            "latency_ms": (done_ns[i] - sched) / 1e6,
            "lag_ms": (sent - sched) / 1e6,
        }
        if isinstance(res, split.Outcome):
            rec["devices"] = sorted(res.devices)
            rec["k"] = res.plan.k
            rec["answers"] = res.answers
        else:
            rec["devices"] = [res.runtime.device]
            rec["reason"] = res.runtime.routing_reason
            rec["answers"] = res.answers
        recs.append(rec)
    return recs


def pair_ratios(rounds: list[dict], blocks: list[list[str]], measured: str) -> list[dict]:
    """split/base measured-class P99 per adjacent pair (design.adjacent_pairs), complete blocks only."""
    size = len(blocks[0])
    out = []
    for b, block in enumerate(blocks, 1):
        got = [r for r in rounds if r["block"] == b]
        if len(got) < size:
            break
        by_pos = {r["position"]: r for r in got}
        for ps, pb in design.adjacent_pairs(block, "split", "base"):
            s, base = by_pos[ps]["summary"][measured]["p99_ms"], by_pos[pb]["summary"][measured]["p99_ms"]
            out.append({"block": b, "split_position": ps, "base_position": pb, "ratio": s / base})
    return out


def futility_after_block(rounds: list[dict], mcfg: dict, measured: str) -> str | None:
    """criteria.json fast fail: after block 1 only, every pair's ratio above the futility limit."""
    if len(rounds) != len(mcfg["blocks"][0]):
        return None
    ratios = [p["ratio"] for p in pair_ratios(rounds, mcfg["blocks"][:1], measured)]
    if design.mix_futility(ratios, mcfg["futility_ratio"]):
        return f"{measured} P99 ratios {[round(r, 3) for r in ratios]} all above {mcfg['futility_ratio']}"
    return None


def summarize(recs, measured: str, qid, deadline_ms) -> dict:
    out = {}
    for cls in sorted({r["cls"] for r in recs}) + ["all"]:
        lat = np.asarray([r["latency_ms"] for r in recs if cls == "all" or r["cls"] == cls], np.float64)
        out[cls] = {
            "n": len(lat),
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
            "p99_ms": float(np.percentile(lat, 99)),
        }
    m = [r for r in recs if r["cls"] == measured]
    out["measured"] = measured
    out["measured_on_ane"] = sum("ane" in r["devices"] for r in m)
    out["lag_p99_ms"] = float(np.percentile([r["lag_ms"] for r in recs], 99))
    if deadline_ms is not None:
        out["late"] = sum(r["latency_ms"] > deadline_ms for r in m)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mix", required=True, choices=["switchyard", "serve"])
    p.add_argument("--run-id", required=True)
    a = p.parse_args(argv)
    crit = common.criteria()
    mcfg = crit["protocol"]["mix"]
    out_path = common.RAW / "mix" / f"{a.run_id}.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; raw data is never overwritten")
    model = mcfg[a.mix]["model"]
    env_start = common.environment()
    laya = common.open_laya(model)
    try:
        sub = split.SplitSubmitter(laya)
        wl = switchyard_workload(laya) if a.mix == "switchyard" else serve_workload(laya, mcfg["serve"])
        # correctness references: every distinct multi-question payload, idle, through Laya.submit
        refs = {}
        for it in wl["round"] + wl["warmup"]:
            state, questions = it["payload"]
            key = (json.dumps(state), json.dumps(questions, sort_keys=True))
            if len(questions) > 1 and key not in refs:
                refs[key] = laya.submit(context=state, questions=questions).result().answers
        rounds, stop = [], None
        order = [c for block in mcfg["blocks"] for c in block]
        for idx, config in enumerate(order):
            time.sleep(mcfg["idle_before_round_s"])
            run_round(laya, sub, wl["warmup"], config)  # unmeasured, its own seed
            recs = run_round(laya, sub, wl["round"], config)
            wrong, multi = 0, []
            for r, it in zip(recs, wl["round"]):
                ans = r.pop("answers")
                state, questions = it["payload"]
                if it["oracle"] is not None and ans[wl["question_id"]]["choice"] != it["oracle"]:
                    wrong += 1
                if len(questions) > 1:
                    c = common.compare_answers(refs[(json.dumps(state), json.dumps(questions, sort_keys=True))], ans)
                    multi.append({"i": r["i"], **c})
            s = summarize(recs, wl["measured_class"], wl["question_id"], wl["deadline_ms"])
            rounds.append(
                {
                    "index": idx,
                    "block": idx // len(mcfg["blocks"][0]) + 1,
                    "position": idx % len(mcfg["blocks"][0]) + 1,
                    "config": config,
                    "summary": s,
                    "oracle_wrong": wrong if wl["deadline_ms"] is not None else None,
                    "multi_question_checks": multi,
                    "requests": recs,
                    "handoff": laya.info().get("ane_handoff"),
                }
            )
            print(
                f"{a.run_id} round {idx + 1} {config:5s} {wl['measured_class']} P99 "
                f"{s[wl['measured_class']]['p99_ms']:8.2f} ms (all {s['all']['p99_ms']:.2f})",
                flush=True,
            )
            stop = futility_after_block(rounds, mcfg, wl["measured_class"])
            if stop:
                print(f"{a.run_id}: futility stop after block 1 ({stop})", flush=True)
                break
        info = laya.info()
    finally:
        laya.close()
    common.write_json(
        out_path,
        {
            "run_id": a.run_id,
            "mix": a.mix,
            "model": model,
            "protocol": mcfg,
            "policy": crit["policy"],
            "schedule_sha256": wl["schedule_sha256"],
            "requests_per_round": len(wl["round"]),
            "environment_start": env_start,
            "environment_end": common.environment(),
            "laya_info": info,
            "stopped": stop,
            "rounds": rounds,
        },
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
