"""Tracing observes only: the same requests with tracing off and on route and answer identically.

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run python \
        research/gpu-ane-interference/scripts/equivalence.py --model laya-typed-decisions \
        --ane-placement thread --out research/gpu-ane-interference/ledger/equivalence-laya-typed-decisions-thread.json

Every shape of the interference workload (interference.py's S, M, A, B, L, Md, S4) with
--seeds seeds each, sent one at a time (idle queues, so routing is a deterministic function of
the request) through `Laya(device="auto", execution="workers")`:

  off   trace=None, no backend hooks (the product as shipped)
  on    trace=callback, and jobtrace.py's backend hooks in this process and in the workers

For every request: target device, routing reason, ANE buckets and the answers must be
identical. The "on" run must also trace every request once, and every trace must join one
backend record. Under load, routing depends on timing, so it is compared as distributions in
the ledger runs instead (ledger/README.md).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import jobtrace  # noqa: E402


def run(model, placement, requests, trace):
    from laya_apple import Laya

    laya = Laya.from_pretrained(
        model, device="auto", execution="workers", ane_placement=placement, local_files_only=True, trace=trace
    )
    try:
        if trace is not None:
            for w in laya._workers.values():
                if w.placement == "thread":
                    jobtrace.tag_dispatcher(w)
        out = []
        for name, seed, state, questions in requests:
            r = laya.predict(context=state, questions=questions)
            rt = r.runtime
            out.append(
                {
                    "shape": name,
                    "seed": seed,
                    "request_id": rt.request_id,
                    "device": rt.device,
                    "reason": rt.routing_reason,
                    "buckets": list(rt.buckets),
                    "answers": r.answers,
                }
            )
        return out
    finally:
        laya.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="laya-typed-decisions")
    ap.add_argument("--ane-placement", choices=["thread", "process"], default="thread")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--part-a-short", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from laya_apple import Laya
    from laya_apple.workload import make_request

    probe = Laya.from_pretrained(args.model, device="gpu", local_files_only=True)
    spec, tok, cfg = probe.spec, probe.tokenizer, probe.config
    gpu_long = int(cfg.get("max_len", 512))
    shapes = {
        "S": (spec.ane_buckets[0], 1),
        "M": (128, 1),
        "A": (args.part_a_short, 1),
        "B": (max(spec.ane_buckets), 1),
        "L": (gpu_long, 1),
        "Md": (512 if gpu_long > 512 else 256, 1),
        "S4": (args.part_a_short, 4),
    }
    requests = []
    for name, (length, q) in shapes.items():
        for seed in range(args.seeds):
            state, questions = make_request(tok, cfg, length, n_questions=q, seed=100 + seed)
            requests.append((name, seed, state, questions))
    probe.close()

    off = run(args.model, args.ane_placement, requests, None)

    trace_dir = Path(tempfile.mkdtemp(prefix="laya-trace-"))
    os.environ["LAYA_TRACE_DIR"] = str(trace_dir)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(HERE / "hooks")] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
    jobtrace.install_phase_hooks()
    jobtrace.clear_phases()
    traces: list = []
    on = run(args.model, args.ane_placement, requests, traces.append)
    time.sleep(0.5)
    phases = list(jobtrace.phases())
    for f in sorted(trace_dir.glob("phases-*.json")):
        phases += json.loads(f.read_text())
    shutil.rmtree(trace_dir, ignore_errors=True)

    keys = ("device", "reason", "buckets", "answers")
    diffs = [
        {"shape": a["shape"], "seed": a["seed"], **{k: [a[k], b[k]] for k in keys if a[k] != b[k]}}
        for a, b in zip(off, on)
        if any(a[k] != b[k] for k in keys)
    ]
    ids = [r["request_id"] for r in on]
    traced = sorted(t.request_id for t in traces)
    by_id = {}
    for p in phases:
        if p["request_id"] is not None:
            by_id.setdefault(p["request_id"], []).append(p)
    joined = sum(1 for i in ids if len(by_id.get(i, [])) == 1)
    report = {
        "model": args.model,
        "ane_placement": args.ane_placement,
        "requests": len(requests),
        "shapes": {k: {"length": v[0], "questions": v[1]} for k, v in shapes.items()},
        "devices_off": {d: sum(r["device"] == d for r in off) for d in ("gpu", "ane")},
        "reasons_off": sorted({r["reason"] for r in off}),
        "identical_decisions_and_answers": not diffs,
        "differences": diffs,
        "traced_once": traced == sorted(ids),
        "backend_joined_1to1": joined,
        "backend_events_without_request_id": sum(1 for p in phases if p["request_id"] is None),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "differences"}, indent=2))
    if diffs or not report["traced_once"] or joined != len(ids):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
