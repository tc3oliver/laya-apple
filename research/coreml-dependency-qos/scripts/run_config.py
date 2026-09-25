"""One run of the dependency QoS screen: #94's PB-ASYNC run, in cell B, O or Q (../criteria.md).

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run --with pyobjc-framework-CoreML==12.2.2 python \
        research/coreml-dependency-qos/scripts/run_config.py --cell O \
        --model laya --short 128 --long 512 --cycles 2 \
        --output research/coreml-dependency-qos/raw/laya-O-r1.json.gz

Every cell is #94's run_config.py (research/coreml-async-transient/scripts/, loaded by path and
unchanged) with --cell PB-ASYNC: the stamped async prebound path, native Core ML completion stamps,
the request trace, the per-thread perf-level counters every 100 ms (recount.py), the per-window
thread CPU snapshots, the window counter and the IOReport note. This script adds, around it:

- **Every cell:** the ANE dispatcher's pthread_t, captured with pthread_self() on the dispatcher
  thread itself. DeviceWorker._load_here runs on the laya-ane-load thread, not on the dispatcher, so
  the capture is at the start of DeviceWorker._dispatch (the laya-ane-dispatch thread's target),
  before its first job: the auto instance's in-process ANE worker only, the first thread that
  serves its requests. Recorded there: the thread name, threading.get_native_id(),
  pthread_threadid_np() and the requested QoS (pthread_get_qos_class_np). The requested QoS is read
  again at the end of the run, from DeviceWorker.close() before the dispatcher stops, and the moment
  the dispatcher's loop returns (its thread exits) is recorded, so an override's lifetime can be
  checked against its target's. In O the requested QoS stays DEFAULT: an override does not change
  it and is invisible to pthread_get_qos_class_np.
- **Q:** at the same point, on the dispatcher thread, pthread_set_qos_class_self_np(
  QOS_CLASS_USER_INITIATED, 0) once; its return code and the read-back right after are recorded.
- **O:** each short request of the auto instance (the measured closed loop of
  bench_concurrency.closed_loop, short stream only, every window of that instance) goes through a
  proxy whose predict() starts a USER_INITIATED / 0 override on the dispatcher's pthread_t just
  before it calls Laya.predict and ends it in a `finally` once the response has returned. The
  override calls are inside closed_loop's measured latency (intended: they are part of cell O) and
  their own cost is recorded per override. At the end of the run the registry is checked and any
  override left is ended (qos.OverrideRegistry). Per short request: its override token (-1 if the
  start returned NULL) and its route (r.runtime.device).

Nothing else changes: not the long stream, the GPU path, the callback queue or the Core ML call.
The output is #94's record with research.experiment = "coreml-dependency-qos", research.cell = B, O
or Q, research.base (what #94 ran) and research.qos (the records above).
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

CELLS = ("B", "O", "Q")
BASE_CELL = "PB-ASYNC"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


qos = _load("dependency_qos_qos", HERE / "qos.py")


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


class OverrideLaya:
    """Cell O's view of the auto instance for the short client: every predict() runs inside its
    own dependency override on the ANE dispatcher."""

    def __init__(self, laya, registry, target: list, requests: dict):
        self._laya, self._reg, self._target, self._req = laya, registry, target, requests

    def predict(self, *args, **kw):
        token = self._reg.start(self._target[0])
        device = None
        try:
            r = self._laya.predict(*args, **kw)
            device = r.runtime.device
            return r
        finally:
            self._reg.end(token)
            self._req["token"].append(-1 if token is None else token)
            self._req["device"].append(device)

    def __getattr__(self, name):
        return getattr(self._laya, name)


def main():
    a = parse()
    t_start = time.monotonic()
    rc94 = _load("async_transient_run_config", ASYNC94 / "run_config.py")  # #94's, unchanged
    from laya_apple import executor

    import bench_concurrency  # scripts/, on sys.path from #94's module

    rec_q: dict = {
        "cell": a.cell,
        "constants": {"USER_INITIATED": qos.QOS_CLASS_USER_INITIATED, "DEFAULT": qos.QOS_CLASS_DEFAULT},
        "dispatcher": None,
        "at_load": None,
        "set": None,
        "at_end": None,
        "errors": [],
    }
    target: list = [None]  # the dispatcher's pthread_t, once captured
    registry = qos.OverrideRegistry(qos.QOS_CLASS_USER_INITIATED, 0) if a.cell == "O" else None
    short_requests: dict = {"token": [], "device": []}

    def on_dispatcher_start():
        try:
            pt = qos.pthread_self()
            rec_q["dispatcher"] = {
                "thread_name": threading.current_thread().name,
                "native_id": threading.get_native_id(),
                "pthread_threadid": qos.thread_id(pt),
                "captured_ns": time.monotonic_ns(),
            }
            rec_q["at_load"] = qos.get_qos(pt)
            if a.cell == "Q":
                rc = qos.set_qos_self(qos.QOS_CLASS_USER_INITIATED, 0)
                rec_q["set"] = {"qos": qos.QOS_CLASS_USER_INITIATED, "relpri": 0, "rc": rc, "readback": qos.get_qos(pt)}
            target[0] = pt
        except Exception as e:  # recorded; the guards see a missing capture
            rec_q["errors"].append(f"dispatcher start: {e!r}")

    def research_ane(w) -> bool:  # #94's ResearchWorker marks the auto instance's ANE worker
        return bool(getattr(w, "_research_ane", False)) and w.placement == "thread"

    base_dispatch = executor.DeviceWorker._dispatch

    def dispatch(self):  # the dispatcher thread's target: runs on laya-<kind>-dispatch
        if not research_ane(self) or rec_q["dispatcher"] is not None:
            return base_dispatch(self)
        on_dispatcher_start()
        try:
            return base_dispatch(self)
        finally:  # the dispatcher's loop has returned: its thread exits now
            if rec_q["dispatcher"] is not None:
                rec_q["dispatcher"]["exit_ns"] = time.monotonic_ns()

    base_close = executor.DeviceWorker.close

    def close(self, *args, **kw):
        if research_ane(self) and target[0] is not None and rec_q["at_end"] is None:
            try:
                alive = getattr(self, "_thread", None) is not None and self._thread.is_alive()
                rec_q["at_end"] = qos.get_qos(target[0]) if alive else None
                rec_q["at_end_ns"] = time.monotonic_ns()
            except Exception as e:
                rec_q["errors"].append(f"end read: {e!r}")
            if registry is not None:
                registry.end_all()  # before the dispatcher stops
        return base_close(self, *args, **kw)

    executor.DeviceWorker._dispatch = dispatch  # #94's ResearchWorker subclasses this class
    executor.DeviceWorker.close = close

    base_loop = bench_concurrency.closed_loop

    def qos_loop(laya, req, start_at, end_at, out, reference):
        if registry is not None and laya.device == "auto" and req["length"] == a.short:
            laya = OverrideLaya(laya, registry, target, short_requests)
        return base_loop(laya, req, start_at, end_at, out, reference)

    bench_concurrency.closed_loop = qos_loop  # #94's measured_loop, then run_mix.py, wrap this one

    tmpdir = Path(tempfile.mkdtemp(prefix="laya-depqos-"))
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
    if registry is not None:
        registry.end_all()  # no-op when close() already did it
    with gzip.open(tmp, "rt") as fh:
        rec = json.load(fh)
    shutil.rmtree(tmpdir)

    res = rec["research"]
    rec_q["dispatcher_pthread_captured"] = target[0] is not None
    if rec_q["dispatcher"] is not None:
        rec_q["dispatcher"].setdefault("exit_ns", None)  # None: still running when the record was written
    rec_q["override"] = None if registry is None else {**registry.record(), "short_requests": short_requests}
    res["base"] = {
        "experiment": res["experiment"],
        "cell": res["cell"],
        "run_config": str((ASYNC94 / "run_config.py").relative_to(ROOT)),
    }
    res["experiment"] = "coreml-dependency-qos"
    res["cell"] = a.cell
    res["qos"] = rec_q
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
