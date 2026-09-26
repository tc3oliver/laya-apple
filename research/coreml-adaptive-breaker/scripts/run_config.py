"""One run of Phase 1, the recovery experiment, in cell A, B or R (../phase1.md).

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run --with pyobjc-framework-CoreML==12.2.2 python \
        research/coreml-adaptive-breaker/scripts/run_config.py --cell R \
        --output research/coreml-adaptive-breaker/raw/laya-R-r1.json.gz

Every cell is #102's staged-handoff run: research/coreml-staged-handoff/scripts/run_config.py,
loaded by path and unchanged. It runs on #94's harness with laya at L128 / L512, 2 cycles of 20 s
windows and 2 hetero windows per run. Both Core ML paths are loaded and warm, and the cells differ
only in the path each ANE forward of the auto instance takes:

- **A:** always sync, the production coremltools path. This is #102's A.
- **B:** always async, PB-ASYNC. This is #102's B.
- **R:** B plus the frozen breaker of breaker.py.
  - C3 on the RequestTrace of every completed ANE request.
  - After a trip, sync for the rest of the hetero episode, then re-armed.

What this script adds:
- **Cell R:** #102's `make_policy` is replaced for R only. A and B keep #102's policies.
- **The trace hook:** `Laya.from_pretrained` is wrapped before #94's own wrapper, so the auto
  instance's trace callback also feeds `Breaker.observe`. The same trace is recorded by #94.
- **After the run:** #102 writes to a temporary file, and the record is re-labelled `experiment = "coreml-adaptive-breaker"`. It gains
  `research.breaker`: the trips, the observations and the episodes, empty for A and B. It also
  gains `runtime`: the laya-apple HEAD, the `laya_apple` tree, and whether any harness path
  (HARNESS) is dirty.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HANDOFF102 = ROOT / "research" / "coreml-staged-handoff" / "scripts"
CELLS = ("A", "B", "R")
HARNESS = (  # the code a run executes: the runtime and every research script the harness loads
    "laya_apple",
    "research/coreml-adaptive-breaker/scripts",
    "research/coreml-staged-handoff/scripts",
    "research/coreml-async-transient/scripts",
    "research/coreml-async-predict/scripts",
    "research/coreml-nogil-product-mix/scripts",
    "research/coreml-prebind-predict/scripts",
    "benchmarks/ane-process-isolation",
    "scripts",
)
MODEL, SHORT, LONG, CYCLES, SECONDS = "laya", 128, 512, 2, 20.0


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def runtime_revision() -> dict:
    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()

    return {
        "head": git("rev-parse", "HEAD"),
        "laya_apple_tree": git("rev-parse", "HEAD:laya_apple"),
        "dirty": bool(git("status", "--porcelain", "--", *HARNESS)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", choices=CELLS, required=True)
    ap.add_argument("--output", required=True, help=".json.gz")
    ap.add_argument("--seconds", type=float, default=SECONDS, help="window length (20 in Phase 1)")
    ap.add_argument("--cycles", type=int, default=CYCLES, help="cycles per run (2 in Phase 1)")
    a = ap.parse_args()

    breaker_mod = _load("adaptive_breaker_breaker", HERE / "breaker.py")
    rc102 = _load("staged_handoff_run_config", HANDOFF102 / "run_config.py")  # #102's, unchanged
    from laya_apple import Laya

    made: list = []
    base_policy = rc102.make_policy

    def make_policy(cell, gpu):
        if cell != "R":
            return base_policy(cell, gpu)
        b = breaker_mod.Breaker(gpu)
        made.append(b)
        return b

    rc102.make_policy = make_policy
    rc102.CELLS = CELLS  # its --cell choices

    from_pretrained = Laya.from_pretrained.__func__

    def hooked(cls, model_id, *args, **kw):  # #94 wraps this one and passes its own trace callback in
        if kw.get("device") == "auto" and kw.get("execution") == "workers" and kw.get("trace") is not None:
            inner = kw["trace"]

            def both(t):
                inner(t)
                if made:
                    made[0].observe(t)

            kw["trace"] = both
        return from_pretrained(cls, model_id, *args, **kw)

    Laya.from_pretrained = classmethod(hooked)

    revision = runtime_revision()
    out = Path(a.output)
    tmpdir = Path(tempfile.mkdtemp(prefix="laya-breaker-"))
    tmp = tmpdir / "base.json.gz"
    argv = sys.argv
    sys.argv = [
        "run_config.py",
        *("--cell", a.cell, "--model", MODEL, "--short", str(SHORT), "--long", str(LONG)),
        *("--output", str(tmp), "--seconds", str(a.seconds), "--cycles", str(a.cycles)),
    ]
    try:
        rc102.main()
    finally:
        sys.argv = argv

    with gzip.open(tmp, "rt") as fh:
        rec = json.load(fh)
    shutil.rmtree(tmpdir)
    res = rec["research"]
    b = made[0] if made else None
    if a.cell == "R" and b is None:
        raise RuntimeError("cell R ran without its breaker")
    res["base102"] = {"experiment": res["experiment"], "cell": res["cell"]}
    res["experiment"] = "coreml-adaptive-breaker"
    res["cell"] = a.cell
    res["breaker"] = {
        "detector": {
            "rule": "C3",
            "threshold_ms": breaker_mod.THRESHOLD_MS,
            "consecutive": breaker_mod.CONSECUTIVE,
            "gap_s": breaker_mod.GAP_S,
            "arm_s": breaker_mod.ARM_S,
        },
        "installed": b is not None,
        "episodes": b.episodes if b else None,
        "final_state": b.state if b else None,
        "trips_columns": ["trip_ns", "episode", "request_id", "prepare_ms"],
        "trips": [list(x) for x in b.trips] if b else [],
        "observations_columns": ["t_ns", "request_id", "prepare_ms", "state_before", "count_after"],
        "observations": [list(x) for x in b.observations] if b else [],
    }
    rec["runtime"] = revision
    rec["args"]["output"] = a.output
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    with gzip.open(part, "wt") as fh:
        json.dump(rec, fh)
    part.rename(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
