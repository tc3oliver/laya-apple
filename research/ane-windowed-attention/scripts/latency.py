"""Paired ANE latency and the runtime placement probe: windowed vs the masked BC1S graph.

TIMING-SENSITIVE: run only in an exclusive hardware slot (oMLX stopped, nothing else running).

    uv run --extra ane python research/ane-windowed-attention/scripts/latency.py \
        [--model laya] [--length 512] [--root <research-root>]

Per cell (criteria.md, "Latency"), in one process:

  - A (baseline): at 64/96/128 the shipped artifact through `load_verified` (LAYA_APPLE_CACHE,
    read only); at 256/512 the research-built masked artifact (build_windowed.py --variant masked);
  - B (candidate): the research-built windowed artifact; both CPU_AND_NE;
  - M (descriptive, 256/512 only): the MLX FP16 backend (`MLXBackend.forward`) on the same row;
  - the runtime placement probe, `probe_placement` unchanged, on A and B;
  - input: one exact-length request, `make_request(..., length, n_questions=1, seed=0)`;
  - WARMUP calls per arm, then CYCLES cycles: one WINDOW-forward window of A and of B, ordered
    A B in even cycles and B A in odd cycles, then (256/512) one window of M;
  - per ANE forward: `predict_ms` (model.predict only) and `forward_ms` (ane_features + predict +
    host tail, the backend boundary); per MLX forward: `forward_ms` (MLXBackend.forward). All
    samples are kept.

Writes raw/<model>/L<L>-windowed/latency.json. Verdicts are computed by analyze.py.
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
from common import (  # noqa: E402
    CELLS,
    LONG,
    RAW,
    artifact_path,
    cell_name,
    environment,
    log,
    now,
    research_root,
    write_json,
)

WARMUP = 20
CYCLES = 10
WINDOW = 100


def _done(model: str, length: int, variant: str) -> bool:
    p = RAW / model / cell_name(length, variant) / "build.json"
    return p.exists() and json.loads(p.read_text()).get("step") == "done"


def run_cell(spec, length: int, root: Path) -> dict:
    import coremltools as ct

    from laya_apple.artifacts import COMPILED, artifact_dir, load_verified
    from laya_apple.backends.coreml_ane import HostWeights, ane_features, probe_placement
    from laya_apple.hub import checkpoint_path
    from laya_apple.prompt import Tokenizer, prepare
    from laya_apple.registry import ANE_COMPUTE_UNITS
    from laya_apple.workload import make_request

    units = getattr(ct.ComputeUnit, ANE_COMPUTE_UNITS)
    ckpt = checkpoint_path(spec, local_files_only=True)
    cfg = json.loads((ckpt / "rl_agent_config.json").read_text())
    enc = json.loads((ckpt / "encoder/config.json").read_text())
    tok = Tokenizer(ckpt / "tokenizer")
    host = HostWeights(ckpt, int(enc["local_attention"]))
    long = length in LONG
    rec = {
        "experiment": "ane-windowed-attention/latency",
        "model": spec.name,
        "length": length,
        "protocol": {
            "warmup": WARMUP,
            "cycles": CYCLES,
            "window": WINDOW,
            "order": "A B even / B A odd" + (", then M" if long else ""),
            "seed": 0,
        },
        "started_at": now(),
        "environment_before": environment(),
    }

    t = time.perf_counter()
    if long:
        base_path = artifact_path(root, spec, "masked", length) / COMPILED
        base = ct.models.CompiledMLModel(str(base_path), compute_units=units)
        rec["baseline"] = {"source": "research masked build", "load_s": time.perf_counter() - t}
    else:
        base_path = artifact_dir(spec, length) / COMPILED
        base, manifest = load_verified(spec, length)
        rec["baseline"] = {
            "source": "shipped artifact",
            "artifact_sha256": manifest.artifact_sha256,
            "load_s": time.perf_counter() - t,
        }
    cand_path = artifact_path(root, spec, "windowed", length) / COMPILED
    t = time.perf_counter()
    cand = ct.models.CompiledMLModel(str(cand_path), compute_units=units)
    rec["candidate"] = {"load_s": time.perf_counter() - t}

    pad = {"ids": [tok.pad_token_id] * length, "markers": [1, 2], "qtype": 0}
    probe_feats = ane_features(
        [pad], length, 1, host.embedding, host.type_embedding, host.window(length), tok.pad_token_id
    )
    for tag, model, path in (("baseline", base, base_path), ("candidate", cand, cand_path)):
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
    k = len(item["markers"])
    lb, lc = host.tail(base.predict(feats), [item])[0], host.tail(cand.predict(feats), [item])[0]
    rec["workload_row"] = {"tokens": len(item["ids"]), "options": k}
    rec["candidate_vs_baseline_max_abs_logit"] = float(np.abs(lb[0, :k] - lc[0, :k]).max())

    mlx = None
    if long:
        from laya_apple.backends.mlx import MLXBackend

        t = time.perf_counter()
        mlx = MLXBackend(spec, ckpt, tok.pad_token_id, dtype="float16")
        rec["mlx"] = {"load_s": time.perf_counter() - t, "dtype": "float16"}
        lm = mlx.forward([item])[0]
        rec["mlx_vs_baseline_max_abs_logit"] = float(np.abs(lb[0, :k] - lm[0, :k]).max())

    def ane_window(model):
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
        return {"predict_ms": pred, "forward_ms": fwd}

    def mlx_window():
        fwd = []
        for _ in range(WINDOW):
            t0 = time.perf_counter()
            mlx.forward([item])
            fwd.append((time.perf_counter() - t0) * 1e3)
        return {"forward_ms": fwd}

    for model in (base, cand):
        for _ in range(WARMUP):
            model.predict(feats)
    if mlx is not None:
        for _ in range(WARMUP):
            mlx.forward([item])

    windows = []
    for c in range(CYCLES):
        order = [("baseline", base), ("candidate", cand)]
        if c % 2:
            order.reverse()
        if mlx is not None:
            order.append(("mlx", None))
        for arm, model in order:
            t = time.perf_counter()
            w = mlx_window() if arm == "mlx" else ane_window(model)
            w.update(cycle=c, arm=arm, started_s=t)
            for key in ("predict_ms", "forward_ms"):
                if key in w:
                    w[key.replace("_ms", "_p50_ms")] = float(np.percentile(w[key], 50))
            windows.append(w)
    rec["windows"] = windows
    rec["environment_after"] = {"loadavg": os.getloadavg()}
    rec["finished_at"] = now()
    rec["status"] = "ok"
    return rec


def main() -> None:
    from laya_apple.registry import resolve

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
            if not _done(model, length, "windowed") or (length in LONG and not _done(model, length, "masked")):
                log(f"{model} L{length}: a build is not done; latency skipped")
                continue
            log(f"{model} L{length}: latency")
            write_json(RAW / model / cell_name(length, "windowed") / "latency.json", run_cell(spec, length, root))


if __name__ == "__main__":
    main()
