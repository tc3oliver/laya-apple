"""Phase 2 screen for one arm in a fresh process: load, first call, runtime probe, repeat
consistency, warm percentiles. Timing-sensitive: run alone.

    screen_arm.py A|B --root <w8 root> --out <json> [--n 500]
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("arm", choices=["A", "B"])
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=500)
a = ap.parse_args()
out = Path(a.out)
if out.exists():
    sys.exit(f"{out} exists; refusing to overwrite")

import coremltools as ct  # noqa: E402

from laya_apple.artifacts import COMPILED, artifact_dir, load_verified  # noqa: E402
from laya_apple.backends.coreml_ane import HostWeights, ane_features, probe_placement  # noqa: E402
from laya_apple.hub import checkpoint_path  # noqa: E402
from laya_apple.prompt import Tokenizer, prepare  # noqa: E402
from laya_apple.registry import ANE_COMPUTE_UNITS, resolve  # noqa: E402
from laya_apple.workload import make_request  # noqa: E402

L = 64
spec = resolve("laya-typed-decisions")
ckpt = checkpoint_path(spec, local_files_only=True)
cfg = json.loads((ckpt / "rl_agent_config.json").read_text())
enc = json.loads((ckpt / "encoder/config.json").read_text())
tok = Tokenizer(ckpt / "tokenizer")
host = HostWeights(ckpt, int(enc["local_attention"]))
rec = {"arm": a.arm, "pid": os.getpid(), "loadavg_before": os.getloadavg(), "python": platform.python_version()}

t = time.perf_counter()
if a.arm == "A":
    model, manifest = load_verified(spec, L)
    path = artifact_dir(spec, L) / COMPILED
    rec["artifact_sha256"] = manifest.artifact_sha256
else:
    path = Path(a.root).expanduser() / spec.name / spec.revision[:12] / f"bc1s-masked-w8-pt-L{L}-B1" / COMPILED
    model = ct.models.CompiledMLModel(str(path), compute_units=getattr(ct.ComputeUnit, ANE_COMPUTE_UNITS))
rec["load_s"] = time.perf_counter() - t

state, questions = make_request(tok, cfg, L, n_questions=1, seed=0)
item = prepare(tok, cfg, state, questions).items[0]
assert len(item["ids"]) == L
feats = ane_features([item], L, 1, host.embedding, host.type_embedding, host.window(L), tok.pad_token_id)

t = time.perf_counter()
first = model.predict(feats)
rec["first_call_ms"] = (time.perf_counter() - t) * 1e3

pad = {"ids": [tok.pad_token_id] * L, "markers": [1, 2], "qtype": 0}
pf = ane_features([pad], L, 1, host.embedding, host.type_embedding, host.window(L), tok.pad_token_id)
try:
    rec["probe"] = probe_placement(spec, L, model, path, pf)
    rec["probe_gate"] = "PASS"
except Exception as e:
    rec["probe_gate"] = "FAIL"
    rec["probe_error"] = str(e)

# repeat-inference consistency: 20 predicts on identical input must be bit-identical
ref = {k: np.array(v) for k, v in first.items()}
rec["repeat_identical"] = all(
    all(np.array_equal(ref[k], np.array(v)) for k, v in model.predict(feats).items()) for _ in range(20)
)

for _ in range(20):
    model.predict(feats)
s = []
for _ in range(a.n):
    t = time.perf_counter()
    model.predict(feats)
    s.append((time.perf_counter() - t) * 1e3)
rec["warm_n"] = a.n
rec["warm_ms"] = {f"p{p}": float(np.percentile(s, p)) for p in (50, 95, 99)} | {"max": max(s)}
rec["samples_ms"] = s
import resource  # noqa: E402

rec["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
rec["loadavg_after"] = os.getloadavg()
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rec))
print(a.arm, {k: v for k, v in rec.items() if k != "samples_ms"})
