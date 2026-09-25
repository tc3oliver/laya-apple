"""One run of #57's full product mix (R1's protocol) for one cell of the async-predict screen.

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run --with pyobjc-framework-CoreML==12.2.2 python \
        research/coreml-async-predict/scripts/run_config.py --cell PB-ASYNC \
        --model laya --short 128 --long 512 --output research/coreml-async-predict/raw/laya-PB-ASYNC-r1.json.gz

R1's run_config.py (research/coreml-prebind-full-protocol/scripts/) with a --cell option in
place of --config. The workload, the injection into the heterogeneous instance and every R1
record are R1's code paths, unchanged: run_mix.main() -> bench_concurrency --part a
(solo_short, solo_long, hetero and gpu_only windows in alternating order; 3 cycles of 20 s; 2.0 s
idle and 0.5 s lead before each window; one closed-loop client per stream repeating the
make_request(seed=0) request; answers checked against inline coremltools and MLX references),
one fresh process per run. laya_apple itself is not modified.

  cell      ANE (thread-placed) Core ML predict                               GPU (MLX)
  A         coremltools, GIL held (production)                                worker process
  PB-SYNC   #83's prebind.py: one predict through a ctypes.CDLL shim          worker process
            (GIL released for the whole predict) -- R1's PB
  PB-ASYNC  prebind_async.py: #83's prebound objects, one PyObjC send of      worker process
            predictionFromFeatures:options:completionHandler:, then a wait
            on the bucket's event (GIL released while waiting)

Records, the same in every cell (R1's, plus #89's per-client closing snapshot):
  forwards    per device of the heterogeneous instance, every forward: service start/end and
              the executing thread's CPU
  predicts    in-process ANE: per predict, entry, the binding's four stamps (0 for coremltools),
              exit. PB-SYNC: python_before, native_before, native_after, python_after (#83).
              PB-ASYNC: submit_before, submit_after, callback_entry, wake (prebind_async.py).
  crossings   bridge crossings of one forward per bucket on the calling thread, at load (#83)
  backings    PB-SYNC and PB-ASYNC: each output's mode and the load-time evidence
  trace       every request of the heterogeneous instance submitted in a hetero window
  windows     per auto-instance hetero window: streams, requests per stream, per-thread CPU
              snapshots before the lead and after the window (R1), and each client's own
              closing snapshot (`after_by_stream`, #89: a client that finishes first is gone
              from R1's single closing snapshot)
  callbacks   PB-ASYNC: every completion callback inside a hetero window (entry stamp, native
              thread id, error flag), the queue labels, late or duplicate callbacks. The
              handler names its thread "coreml-callback" (threading registers Core ML's thread
              as a dummy thread on first use), so the per-thread snapshots attribute the
              callback threads' CPU to that name instead of "other".
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MIX77 = ROOT / "research" / "coreml-nogil-product-mix" / "scripts"
PB83 = ROOT / "research" / "coreml-prebind-predict" / "scripts"
sys.path.insert(0, str(ROOT / "benchmarks" / "ane-process-isolation"))  # run_mix.py
sys.path.insert(0, str(ROOT / "scripts"))  # bench_concurrency.py
sys.path.insert(0, str(MIX77))
sys.path.insert(0, str(PB83))
sys.path.insert(0, str(HERE))

import crossings  # noqa: E402  #83's, unchanged
import probe  # noqa: E402  #83's: the Mach per-thread CPU snapshot

CELLS = {
    "A": {"ane_placement": "thread", "ane_predict": "coremltools", "gpu_placement": "process"},
    "PB-SYNC": {"ane_placement": "thread", "ane_predict": "prebind", "gpu_placement": "process"},
    "PB-ASYNC": {"ane_placement": "thread", "ane_predict": "prebind_async", "gpu_placement": "process"},
}
TRACE_COLUMNS = [
    "request_id",
    "target",
    "sequence_length",
    "submit_ns",
    "prepared_ns",
    "routed_ns",
    "queue_enter_ns",
    "dispatch_ns",
    "service_start_ns",
    "service_end_ns",
    "received_ns",
    "response_ns",
]
PREDICT_COLUMNS = {
    "coremltools": ["entry_ns", "_", "_", "_", "_", "exit_ns"],
    "prebind": ["entry_ns", "python_before_ns", "native_before_ns", "native_after_ns", "python_after_ns", "exit_ns"],
    "prebind_async": ["entry_ns", "submit_before_ns", "submit_after_ns", "callback_entry_ns", "wake_ns", "exit_ns"],
}
CALLBACK_COLUMNS = ["callback_entry_ns", "native_thread_id", "error"]


def parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", choices=list(CELLS), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--short", type=int, required=True)
    ap.add_argument("--long", type=int, required=True)
    ap.add_argument("--output", required=True, help=".json.gz")
    ap.add_argument("--seconds", type=float, default=20, help="window length (20 in the campaign)")
    ap.add_argument("--cycles", type=int, default=3, help="cycles per run (3 in the campaign)")
    return ap.parse_args()


class _Flags:
    auto = False  # set while the device="auto", execution="workers" instance is being built


class Stamped:
    """#83's wrapper: stamps predict entry and exit, and picks up the binding's own four stamps
    when it has them. Two clock reads per predict, in every cell."""

    def __init__(self, model, stamps: list | None, out: list):
        self._model, self._stamps, self._out = model, stamps, out

    def predict(self, feats):
        src = self._stamps
        n = len(src) if src is not None else 0
        t0 = time.monotonic_ns()
        result = self._model.predict(feats)
        t1 = time.monotonic_ns()
        s = src[-1] if src is not None and len(src) > n else (0, 0, 0, 0)
        self._out.append((t0, *s, t1))
        return result

    def __getattr__(self, name):
        return getattr(self._model, name)


def main():
    a = parse()
    cfg = CELLS[a.cell]

    import laya_apple
    from laya_apple import Laya, executor

    binding, stamps = None, None
    if cfg["ane_predict"] == "prebind":
        import prebind as binding  # #83's, unchanged

        stamps = binding.STAMPS
    elif cfg["ane_predict"] == "prebind_async":
        import prebind_async as binding

        stamps = binding.STAMPS
    if binding is not None:
        import prebind

        prebind._shim()  # compile now, not inside the ANE loader thread (#83's __init__ holds it)

    auto_workers: list = []
    in_process: dict[str, list] = {"gpu": [], "ane": []}
    predicts: list = []
    load_records: dict = {"crossings": None, "backings": None, "callbacks_per_forward": None}
    cpu_dir = tempfile.mkdtemp(prefix="laya-async-cpu-")
    os.environ["LAYA_NOGIL_CPU_DIR"] = cpu_dir  # #77's worker_entry.py writes there

    class ResearchWorker(executor.DeviceWorker):
        def __init__(self, kind, args, *, placement="process", **kw):
            self._research_ane = False
            if _Flags.auto:
                auto_workers.append(self)
                if kind == "gpu":
                    placement = cfg["gpu_placement"]
                self._research_ane = kind == "ane"
                assert placement == (cfg["ane_placement"] if kind == "ane" else cfg["gpu_placement"])
            super().__init__(kind, args, placement=placement, **kw)

        def _load_here(self):
            super()._load_here()
            backend = self._loaded.get("backend")
            if not self._research_ane or backend is None:
                return
            try:
                if binding is not None:
                    binding.install(backend)
                    executor.warm("ane", backend, self._args["pad_id"])
                pad = self._args["pad_id"]
                n_cb = len(binding.CALLBACKS) if cfg["ane_predict"] == "prebind_async" else 0
                load_records["crossings"] = {
                    str(b): crossings.count(backend.forward, [{"ids": [pad] * b, "markers": [1, 2], "qtype": 0}])[1]
                    for b in backend.buckets
                }
                if cfg["ane_predict"] == "prebind_async":
                    load_records["callbacks_per_forward"] = (len(binding.CALLBACKS) - n_cb) / len(backend.buckets)
                    load_records["async_models"] = list(backend.models.values())
                if binding is not None:
                    load_records["backings"] = {
                        str(b): {"modes": dict(m.modes), "evidence": m.backing_evidence}
                        for b, m in backend.models.items()
                    }
                backend.models = {b: Stamped(m, stamps, predicts) for b, m in backend.models.items()}
            except BaseException as e:
                self._loaded.pop("backend", None)
                self._loaded["error"] = e

    executor.DeviceWorker = ResearchWorker  # model.py imports it from the module at start-up

    forward_timed = executor._forward_timed

    def timed(backend, rows):  # in-process forwards: the dispatcher thread is the executing thread
        c0 = time.thread_time_ns()
        out = forward_timed(backend, rows)
        in_process[backend.device].append((out[2], out[3], time.thread_time_ns() - c0))
        return out

    executor._forward_timed = timed

    class Spawn:  # every process worker starts through #77's worker_entry.py (same executor code)
        def __getattr__(self, name):
            return getattr(subprocess, name)

        @staticmethod
        def Popen(cmd, *args, **kw):
            if list(cmd[1:]) == ["-m", "laya_apple.executor"]:
                cmd = [cmd[0], str(MIX77 / "worker_entry.py")]
            return subprocess.Popen(cmd, *args, **kw)

    executor.subprocess = Spawn()

    trace_rows: list = []

    def record_trace(t):  # every request of the heterogeneous instance, both targets
        trace_rows.append(tuple(getattr(t, c) for c in TRACE_COLUMNS))

    from_pretrained = Laya.from_pretrained.__func__

    def flagged(cls, model_id, *args, **kw):
        _Flags.auto = kw.get("device") == "auto" and kw.get("execution") == "workers"
        if _Flags.auto:
            gpu_return = kw["trace"]  # run_mix.py's GPU-return record, set by its own wrapper

            def both(t):
                record_trace(t)
                gpu_return(t)

            kw["trace"] = both
        try:
            return from_pretrained(cls, model_id, *args, **kw)
        finally:
            _Flags.auto = False

    Laya.from_pretrained = classmethod(flagged)  # run_mix.py wraps this one in turn

    import bench_concurrency  # scripts/, on sys.path above

    windows: dict[tuple[int, int], dict] = {}
    wlock = threading.Lock()
    base_loop = bench_concurrency.closed_loop

    def measured_loop(laya, req, start_at, end_at, out, reference):
        """bench_concurrency's closed loop, with #83's per-thread CPU snapshots of the auto
        instance's windows: one before the lead, one after the window's end."""
        if laya.device != "auto":
            return base_loop(laya, req, start_at, end_at, out, reference)
        stream = "short" if req["length"] == a.short else "long"
        threading.current_thread().name = f"client-{stream}"
        key = (int(start_at * 1e9), int(end_at * 1e9))
        before = probe.snapshot()  # before start_at: outside the window
        with wlock:
            w = windows.setdefault(
                key, {"start_ns": key[0], "end_ns": key[1], "instance": "auto", "streams": {}, "before": before}
            )
        base_loop(laya, req, start_at, end_at, out, reference)
        after = probe.snapshot()  # after end_at
        with wlock:
            w["streams"][stream] = len(out["latency_ms"])
            w.setdefault("after_by_stream", {})[stream] = after  # this client's own closing snapshot
            if w.get("after") is None or after["t_ns"] > w["after"]["t_ns"]:
                w["after"] = after

    bench_concurrency.closed_loop = measured_loop  # run_mix.py wraps this one in turn

    import run_mix

    tmp = Path(tempfile.mkdtemp(prefix="laya-async-run-")) / "run.json"
    sys.argv = [
        "run_mix.py",
        *("--model", a.model, "--short", str(a.short), "--long", str(a.long)),
        *("--ane-placement", cfg["ane_placement"], "--part", "a", "--output", str(tmp)),
        *("--seconds", str(a.seconds), "--cycles", str(a.cycles)),
    ]
    t0 = time.monotonic()
    run_mix.main()
    rec = json.loads(tmp.read_text())
    tmp.unlink()
    tmp.parent.rmdir()

    bounds = hetero_bounds(rec)
    forwards, workers = {}, {}
    for w in auto_workers:
        workers[w.kind] = {"placement": w.placement, "pid": w.pid}
        if w.placement == "thread":
            rows = in_process[w.kind]
        else:
            f = Path(cpu_dir, f"{w.pid}.json")
            rows = json.loads(f.read_text())["forwards"] if f.exists() else None
        forwards[w.kind] = rows
    for f in Path(cpu_dir).iterdir():
        f.unlink()
    os.rmdir(cpu_dir)

    def inside(t):
        return any(lo <= t < hi for lo, hi in bounds)

    trace_rows.sort(key=lambda r: r[3])
    kept = [r for r in trace_rows if inside(r[3])]
    trace = {c: [r[i] for r in kept] for i, c in enumerate(TRACE_COLUMNS)}
    hetero = set(bounds)
    win_list = [w for k, w in sorted(windows.items()) if k in hetero]

    callbacks = None
    if cfg["ane_predict"] == "prebind_async":
        cbs = list(binding.CALLBACKS)
        callbacks = {
            "columns": CALLBACK_COLUMNS,
            "hetero": [list(r) for r in cbs if inside(r[0])],
            "total": len(cbs),
            "submits": sum(m._completion().submits for m in load_records["async_models"]),
            "errors": sum(r[2] for r in cbs),
            "thread_ids": sorted({r[1] for r in cbs}),
            "queue_labels": dict(binding.LABELS),
            "anomalies": list(binding.ANOMALIES),
        }

    rec["research"] = {
        "experiment": "coreml-async-predict",
        "cell": a.cell,
        **cfg,
        "run_wall_s": time.monotonic() - t0,
        "laya_apple": laya_apple.__version__,
        "pyobjc": _version("pyobjc-framework-CoreML"),
        "coremltools": _version("coremltools"),
        "mlx": _version("mlx"),
        "python": sys.version.split()[0],
        "switch_interval_s": sys.getswitchinterval(),
        "workers": workers,
        "forwards_columns": ["service_start_ns", "service_end_ns", "thread_cpu_ns"],
        "forwards": forwards,
        "predicts_columns": PREDICT_COLUMNS[cfg["ane_predict"]],
        "predicts": predicts,
        "crossings": load_records["crossings"],
        "callbacks_per_forward_at_load": load_records["callbacks_per_forward"],
        "backings": load_records["backings"],
        "callbacks": callbacks,
        "trace": trace,
        "windows": win_list,
    }
    rec["args"]["output"] = a.output  # not the temporary path run_mix.py wrote to
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")  # a complete file or none: run_all.sh skips finished runs
    with gzip.open(part, "wt") as fh:
        json.dump(rec, fh)
    part.rename(out)
    print("wrote", out, f"({rec['research']['run_wall_s']:.0f} s)")


def hetero_bounds(rec: dict) -> list[tuple[int, int]]:
    """(start_ns, end_ns) of the hetero windows: the auto instance running both streams at once."""
    seen: dict = {}
    for w in rec["windows_t"]:
        if w["instance"] == "auto":
            seen.setdefault((w["start_ns"], w["end_ns"]), set()).add(w["stream"])
    return sorted(k for k, s in seen.items() if s == {"short", "long"})


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
