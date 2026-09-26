"""One run of the staged-handoff screen: #94's run, in cell A, B, H32 or H64 (../criteria.md).

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run --with pyobjc-framework-CoreML==12.2.2 python \
        research/coreml-staged-handoff/scripts/run_config.py --cell H32 \
        --model laya --short 128 --long 512 --cycles 2 \
        --output research/coreml-staged-handoff/raw/laya-H32-r1.json.gz

Every cell is #94's run_config.py (research/coreml-async-transient/scripts/, loaded by path and
unchanged) with --cell PB-ASYNC: the same load (the ANE backend's coremltools models, then the
stamped PB-ASYNC models built and warmed next to them), the native Core ML completion stamps, the
request trace, the per-thread perf-level counters every 100 ms (recount.py), the per-window thread
CPU snapshots. So every cell has both Core ML paths loaded and warm for every bucket; the cells
differ only in which path each ANE forward of the auto instance takes:

  A    always sync: the coremltools model (today's production path)
  B    always async: PB-ASYNC (#94 / #99's PB-ASYNC)
  H32  handoff.StagedHandoff(guard=32): sync outside a hetero episode and for the first 32 short
       forwards of each hetero episode, async after that
  H64  the same with guard=64

What this script adds around #94's run:
- At the ANE dispatcher's start (DeviceWorker._dispatch on laya-ane-dispatch, the auto instance's
  in-process ANE worker), before its first job: every bucket's model becomes a HandoffModel that
  holds the coremltools model (the one ANEBackend loaded and warmed) and the stamped PB-ASYNC
  model (#94's, loaded and warmed), and backend.forward is wrapped to take one decision per forward
  (policy.decide()) and run that forward's predicts on the chosen model.
- GPU activity for the state machine: the auto instance's GPU worker's submit() marks a job
  started, the job's Future marks it ended (handoff.GpuActivity). The GPU-only instance's worker
  is not counted.
- Per ANE forward of the auto instance: (decision_ns, path, state, guard count). Per run: the
  policy, its episodes count, and the forwards per path.

Nothing else changes: the long stream, the GPU path, the callback queue and the Core ML calls are
#94's. The output is #94's record with research.experiment = "coreml-staged-handoff",
research.cell = A, B, H32 or H64, research.base (what #94 ran) and research.handoff.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ASYNC94 = ROOT / "research" / "coreml-async-transient" / "scripts"

CELLS = ("A", "B", "H32", "H64")
GUARDS = {"H32": 32, "H64": 64}
BASE_CELL = "PB-ASYNC"
PATH_CODE = {"sync": 0, "async": 1}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


handoff = _load("staged_handoff_handoff", HERE / "handoff.py")


def parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", choices=CELLS, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--short", type=int, required=True)
    ap.add_argument("--long", type=int, required=True)
    ap.add_argument("--output", required=True, help=".json.gz")
    ap.add_argument("--seconds", type=float, default=20, help="window length (20 in the campaign)")
    ap.add_argument("--cycles", type=int, default=2, help="cycles per run (2 in the campaign)")
    return ap.parse_args()


def make_policy(cell: str, gpu):
    if cell == "A":
        return handoff.Fixed(handoff.SYNC)
    if cell == "B":
        return handoff.Fixed(handoff.ASYNC)
    return handoff.StagedHandoff(GUARDS[cell], gpu)


class Route:
    """The path of the forward running now; set by the forward wrapper on the dispatcher thread."""

    path = handoff.SYNC


class HandoffModel:
    """One bucket's two Core ML paths; predict() runs the one the current forward chose."""

    def __init__(self, sync_model, async_model, route: Route):
        self._sync, self._async, self._route = sync_model, async_model, route

    def predict(self, feats):
        model = self._async if self._route.path == handoff.ASYNC else self._sync
        return model.predict(feats)

    def __getattr__(self, name):
        return getattr(self._sync, name)


def main():
    a = parse()
    t_start = time.monotonic()
    rc94 = _load("async_transient_run_config", ASYNC94 / "run_config.py")  # #94's, unchanged
    from laya_apple import executor

    gpu = handoff.GpuActivity()
    policy = make_policy(a.cell, gpu)
    route = Route()
    decisions: list = []  # (decision_ns, path code, state, guard count)
    rec_h: dict = {"cell": a.cell, "guard": GUARDS.get(a.cell), "gap_s": handoff.GAP_S, "errors": []}
    sync_models: dict = {}  # bucket -> the coremltools model ANEBackend loaded and warmed
    installed = [False]

    base_init = executor.DeviceWorker.__init__

    def init(self, kind, args, *pos, **kw):  # #94's ResearchWorker calls this through super()
        self._research_auto = bool(rc94._Flags.auto)
        return base_init(self, kind, args, *pos, **kw)

    base_load = executor.DeviceWorker._load_here

    def load_here(self):  # runs before #94's ResearchWorker replaces backend.models
        base_load(self)
        backend = self._loaded.get("backend")
        if getattr(self, "_research_ane", False) and backend is not None:
            sync_models.update(backend.models)

    def research_ane(w) -> bool:  # #94's ResearchWorker marks the auto instance's ANE worker
        return bool(getattr(w, "_research_ane", False)) and w.placement == "thread"

    def install(w) -> None:
        backend = w._backend
        for b, stamped in backend.models.items():  # #94's Stamped(StampedAsyncPrebindModel)
            stamped._model = HandoffModel(sync_models[b], stamped._model, route)
        forward = backend.forward

        def decided_forward(items):
            path, state, count = policy.decide()
            decisions.append((time.monotonic_ns(), PATH_CODE[path], state, count))
            route.path = path
            try:
                return forward(items)
            finally:
                route.path = handoff.SYNC

        backend.forward = decided_forward
        installed[0] = True

    base_dispatch = executor.DeviceWorker._dispatch

    def dispatch(self):  # the dispatcher thread's target: before its first job
        if research_ane(self) and not installed[0]:
            try:
                install(self)
            except Exception as e:
                rec_h["errors"].append(f"install: {e!r}")
                raise
        return base_dispatch(self)

    base_submit = executor.DeviceWorker.submit

    def submit(self, rows, estimate_ms, job_id):
        fut = base_submit(self, rows, estimate_ms, job_id)
        if self.kind == "gpu" and getattr(self, "_research_auto", False):
            gpu.started()
            fut.add_done_callback(lambda _f: gpu.ended())
        return fut

    executor.DeviceWorker.__init__ = init
    executor.DeviceWorker._load_here = load_here
    executor.DeviceWorker._dispatch = dispatch
    executor.DeviceWorker.submit = submit

    tmpdir = Path(tempfile.mkdtemp(prefix="laya-handoff-"))
    tmp = tmpdir / "base.json.gz"
    argv = sys.argv
    sys.argv = [
        "run_config.py",
        *("--cell", BASE_CELL, "--model", a.model, "--short", str(a.short), "--long", str(a.long)),
        *("--output", str(tmp), "--seconds", str(a.seconds), "--cycles", str(a.cycles)),
    ]
    try:
        rc94.main()
    finally:
        sys.argv = argv
    with gzip.open(tmp, "rt") as fh:
        rec = json.load(fh)
    shutil.rmtree(tmpdir)

    if not installed[0]:
        raise RuntimeError("the handoff was never installed on the auto instance's ANE worker")
    res = rec["research"]
    n_async = sum(1 for d in decisions if d[1] == 1)
    rec_h.update(
        {
            "installed": installed[0],
            "episodes": policy.episodes,
            "forwards": len(decisions),
            "forwards_async": n_async,
            "forwards_sync": len(decisions) - n_async,
            "decisions_columns": ["decision_ns", "path", "state", "count"],
            "decisions": decisions,
            "thread": threading.current_thread().name,
        }
    )
    res["base"] = {
        "experiment": res["experiment"],
        "cell": res["cell"],
        "run_config": str((ASYNC94 / "run_config.py").relative_to(ROOT)),
    }
    res["experiment"] = "coreml-staged-handoff"
    res["cell"] = a.cell
    res["handoff"] = rec_h
    res["run_wall_total_s"] = time.monotonic() - t_start
    rec["args"]["output"] = a.output
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")  # a complete file or none: run_all.sh skips finished runs
    with gzip.open(part, "wt") as fh:
        json.dump(rec, fh)
    part.rename(out)
    print("wrote", out, f"({res['run_wall_total_s']:.0f} s)")


if __name__ == "__main__":
    main()
