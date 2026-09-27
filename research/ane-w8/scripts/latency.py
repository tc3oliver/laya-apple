"""Paired short-L ANE latency and the runtime placement probe, W8 vs the shipped FP16 artifact.

TIMING-SENSITIVE: run only in an exclusive hardware slot (oMLX stopped, nothing else running).

    uv run --extra ane python research/ane-w8/scripts/latency.py --config w8-pt \
        [--model laya] [--length 64] [--root <research-root>]

Per cell (criteria.md, "Latency" and "Placement probe"), in one process:

  - baseline: the shipped FP16 artifact, loaded through `load_verified` from LAYA_APPLE_CACHE
    (read only); candidate: the cell's W8 model.mlmodelc from the research root, CPU_AND_NE;
  - the runtime placement probe, `probe_placement` unchanged (ratio to CPU_ONLY <= 0.8), on both;
  - input: one exact-length request, `make_request(..., length, n_questions=1, seed=0)`, as
    `laya-apple benchmark` builds it; features built once and reused;
  - WARMUP forwards of each model, then CYCLES cycles of one WINDOW-forward window per model,
    order A B in even cycles and B A in odd cycles;
  - per forward: `predict_ms` (model.predict only) and `forward_ms` (ane_features + predict +
    host tail, the backend boundary). Every sample is kept.

Writes raw/<model>/<cell>/latency.json. The verdict is computed by analyze.py, not here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CELLS, RAW, cell_name, environment, log, now, research_root, write_json  # noqa: E402

WARMUP = 20
CYCLES = 10
WINDOW = 100


def run_cell(spec, length: int, config: str, root: Path) -> dict:
    import coremltools as ct

    from laya_apple.artifacts import COMPILED, artifact_dir, load_verified
    from laya_apple.backends.coreml_ane import HostWeights, ane_features, probe_placement
    from laya_apple.hub import checkpoint_path
    from laya_apple.prompt import Tokenizer, prepare
    from laya_apple.registry import ANE_COMPUTE_UNITS
    from laya_apple.workload import make_request

    cand_path = root / spec.name / spec.revision[:12] / f"bc1s-masked-{config}-L{length}-B1" / COMPILED
    if not cand_path.exists():
        return {"model": spec.name, "length": length, "config": config, "status": "skipped: no candidate artifact"}
    ckpt = checkpoint_path(spec, local_files_only=True)
    cfg = json.loads((ckpt / "rl_agent_config.json").read_text())
    enc = json.loads((ckpt / "encoder/config.json").read_text())
    tok = Tokenizer(ckpt / "tokenizer")
    host = HostWeights(ckpt, int(enc["local_attention"]))
    rec = {
        "experiment": "ane-w8/latency",
        "model": spec.name,
        "length": length,
        "config": config,
        "protocol": {"warmup": WARMUP, "cycles": CYCLES, "window": WINDOW, "order": "AB even / BA odd", "seed": 0},
        "started_at": now(),
        "environment_before": environment(),
    }

    t = time.perf_counter()
    base, manifest = load_verified(spec, length)
    rec["baseline"] = {"artifact_sha256": manifest.artifact_sha256, "load_s": time.perf_counter() - t}
    t = time.perf_counter()
    cand = ct.models.CompiledMLModel(str(cand_path), compute_units=getattr(ct.ComputeUnit, ANE_COMPUTE_UNITS))
    rec["candidate"] = {"load_s": time.perf_counter() - t}

    pad = {"ids": [tok.pad_token_id] * length, "markers": [1, 2], "qtype": 0}
    probe_feats = ane_features(
        [pad], length, 1, host.embedding, host.type_embedding, host.window(length), tok.pad_token_id
    )
    for tag, model, path in (("baseline", base, artifact_dir(spec, length) / COMPILED), ("candidate", cand, cand_path)):
        try:
            rec[tag]["probe"] = probe_placement(spec, length, model, path, probe_feats)
            rec[tag]["probe_gate"] = "PASS"
        except Exception as e:  # ComputeUnitMismatchError
            rec[tag]["probe_gate"] = "FAIL"
            rec[tag]["probe_error"] = str(e)

    state, questions = make_request(tok, cfg, length, n_questions=1, seed=0)
    item = prepare(tok, cfg, state, questions).items[0]
    if len(item["ids"]) != length:
        raise RuntimeError(f"workload row has {len(item['ids'])} tokens, not {length}")

    def feats_for():
        return ane_features(
            [item], length, 1, host.embedding, host.type_embedding, host.window(length), tok.pad_token_id
        )

    feats = feats_for()
    out_b, out_c = host.tail(base.predict(feats), [item]), host.tail(cand.predict(feats), [item])
    k = len(item["markers"])
    rec["workload_row"] = {"tokens": len(item["ids"]), "options": k}
    rec["candidate_vs_baseline_max_abs_logit"] = float(np.abs(out_b[0][0, :k] - out_c[0][0, :k]).max())

    def window(model):
        pred, fwd = [], []
        for _ in range(WINDOW):
            t0 = time.perf_counter()
            f = feats_for()
            t1 = time.perf_counter()
            o = model.predict(f)
            t2 = time.perf_counter()
            host.tail(o, [item])
            t3 = time.perf_counter()
            pred.append((t2 - t1) * 1e3)
            fwd.append((t3 - t0) * 1e3)
        return pred, fwd

    for model in (base, cand):
        for _ in range(WARMUP):
            model.predict(feats)
    windows = []
    for c in range(CYCLES):
        order = (("A", base), ("B", cand)) if c % 2 == 0 else (("B", cand), ("A", base))
        for tag, model in order:
            t = time.perf_counter()
            pred, fwd = window(model)
            windows.append(
                {
                    "cycle": c,
                    "arm": "baseline" if tag == "A" else "candidate",
                    "started_s": t,
                    "predict_ms": pred,
                    "forward_ms": fwd,
                    "predict_p50_ms": float(np.percentile(pred, 50)),
                    "forward_p50_ms": float(np.percentile(fwd, 50)),
                }
            )
    rec["windows"] = windows
    rec["environment_after"] = {"loadavg": os.getloadavg()}
    rec["finished_at"] = now()
    rec["status"] = "ok"
    return rec


def main() -> None:
    from laya_apple.registry import resolve

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", choices=sorted(CELLS), action="append")
    ap.add_argument("--length", type=int, action="append")
    ap.add_argument("--root")
    args = ap.parse_args()
    root = research_root(args.root)
    for model in args.model or list(CELLS):
        spec = resolve(model)
        for length in CELLS[model]:
            if args.length and length not in args.length:
                continue
            build = RAW / model / cell_name(length, args.config) / "build.json"
            if not build.exists() or json.loads(build.read_text()).get("step") != "done":
                log(f"{model} L{length} {args.config}: build not done; latency skipped")
                continue
            log(f"{model} L{length} {args.config}: latency")
            rec = run_cell(spec, length, args.config, root)
            write_json(RAW / model / cell_name(length, args.config) / "latency.json", rec)


if __name__ == "__main__":
    main()
