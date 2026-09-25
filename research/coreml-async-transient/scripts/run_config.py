"""One run of #92's full product mix for the async transient experiment: A or PB-ASYNC, with the
per-thread counters sampled throughout.

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run --with pyobjc-framework-CoreML==12.2.2 python \
        research/coreml-async-transient/scripts/run_config.py --cell PB-ASYNC \
        --model laya --short 128 --long 512 --cycles 2 \
        --output research/coreml-async-transient/raw/laya-PB-ASYNC-r1.json.gz

#92's run_config.py (research/coreml-async-predict/scripts/), with two changes, the same in all
four runs:

1. PB-ASYNC stamps native completion in every forward. It is #92's prebind_async
   StampedAsyncPrebindModel (loaded by path): the same prebound objects and the same Python
   completion handler, but the send goes through async_stamper.m's C block, which stamps the
   moment Core ML calls the completion and then calls the Python block. Per forward the predicts
   record is entry, submit_before, submit_after, native_completion, callback_entry, wake, exit
   (A: entry and exit, the rest 0). The C block adds one native frame per completion, the same in
   every PB-ASYNC run of this experiment.
2. Per-thread perf-level counters, the primary scheduler instrumentation (recount.py): a
   "laya-recount" thread samples every 100 ms, from before the models load to the end of the
   run, in every cell: MainThread, client-short, client-long, laya-ane-dispatch,
   laya-gpu-dispatch, every completion-callback thread seen so far, and every thread of the
   auto instance's GPU worker process. --no-recount turns it off (research smoke comparison
   only; never in the campaign).
   Thread identities: per window each client's native thread id (its Python thread name is
   client-short / client-long, its OS thread name is also set with pthread_setname_np), the
   completion callbacks' thread ids, the GPU worker's pid.

bench_concurrency.conditions, which part_a calls once per window, is wrapped with a counter over
part_a's deterministic window order (conditions in order, reversed on odd cycles); the run fails if
part_a ran a different number of windows than window_order() expects.

No OS scheduler trace is recorded. IOReport is not sampled (see IOREPORT).

Everything else is #92's: run_mix.main() -> bench_concurrency --part a (solo_short, solo_long,
hetero and gpu_only in alternating order; --cycles 2 of 20 s windows here, so two transitions: the
cycle-0 hetero window follows solo_long, the cycle-1 one gpu_only; 2.0 s idle and 0.5 s lead),
one fresh process per run, and the records: forwards (per-forward executing-thread CPU),
predicts, crossings, backings, the hetero request trace, per-window thread CPU snapshots with
after_by_stream, and the completion callbacks. The part_a per-request latencies are kept for the
per-request analysis.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
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
ASYNC92 = ROOT / "research" / "coreml-async-predict" / "scripts"
sys.path.insert(0, str(ROOT / "benchmarks" / "ane-process-isolation"))  # run_mix.py
sys.path.insert(0, str(ROOT / "scripts"))  # bench_concurrency.py
sys.path.insert(0, str(MIX77))
sys.path.insert(0, str(PB83))
sys.path.insert(0, str(HERE))

import crossings  # noqa: E402  #83's, unchanged
import probe  # noqa: E402  #83's: the Mach per-thread CPU snapshot
import recount  # noqa: E402

CELLS = {
    "A": {"ane_placement": "thread", "ane_predict": "coremltools", "gpu_placement": "process"},
    "PB-ASYNC": {"ane_placement": "thread", "ane_predict": "prebind_async_stamped", "gpu_placement": "process"},
}
CONDITIONS = ("solo_short", "solo_long", "hetero", "gpu_only")  # bench_concurrency.part_a's conds
IOREPORT = (
    "IOReport skipped: research/energy-sampler's sampler costs 6-11 ms of kernel time per sample "
    "(its README), not a small overhead even at 10 Hz, and that track's own results are marked invalid"
)
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
PREDICT_COLUMNS = [
    "entry_ns",
    "submit_before_ns",
    "submit_after_ns",
    "native_completion_ns",
    "callback_entry_ns",
    "wake_ns",
    "exit_ns",
]
CALLBACK_COLUMNS = ["callback_entry_ns", "native_thread_id", "error"]


def parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", choices=list(CELLS), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--short", type=int, required=True)
    ap.add_argument("--long", type=int, required=True)
    ap.add_argument("--output", required=True, help=".json.gz")
    ap.add_argument("--seconds", type=float, default=20, help="window length (20 in the campaign)")
    ap.add_argument("--cycles", type=int, default=2, help="cycles per run (2 in the campaign)")
    ap.add_argument(
        "--no-recount",
        action="store_true",
        help="research smoke only: the counter sampler off, everything else identical",
    )
    return ap.parse_args()


def window_order(cycles: int) -> list[tuple[int, str]]:
    """(cycle, condition) of bench_concurrency.part_a's windows in execution order."""
    return [(k, c) for k in range(cycles) for c in (CONDITIONS if k % 2 == 0 else tuple(reversed(CONDITIONS)))]


class WindowCounter:
    """Advances once per part_a window (one conditions() call each) and names the upcoming one."""

    def __init__(self, cycles: int):
        self.order, self.i = window_order(cycles), 0

    def next(self) -> tuple[int, int, str]:
        if self.i >= len(self.order):
            raise RuntimeError("more windows than bench_concurrency.part_a runs")
        i, (k, c) = self.i, self.order[self.i]
        self.i += 1
        return i, k, c


class _Flags:
    auto = False  # set while the device="auto", execution="workers" instance is being built


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Stamped:
    """#83's wrapper, with the native completion: entry, the binding's four stamps (#92's
    submit_before, submit_after, callback_entry, wake) with the C block's native completion
    inserted before callback_entry, exit. Zeros for coremltools."""

    def __init__(self, model, stamps: list | None, native: list | None, out: list):
        self._model, self._stamps, self._native, self._out = model, stamps, native, out

    def predict(self, feats):
        src = self._stamps
        n = len(src) if src is not None else 0
        t0 = time.monotonic_ns()
        result = self._model.predict(feats)
        t1 = time.monotonic_ns()
        if src is not None and len(src) > n:
            sb, sa, cb, wake = src[-1]
            nat = self._native[-1][0]
            self._out.append((t0, sb, sa, nat, cb, wake, t1))
        else:
            self._out.append((t0, 0, 0, 0, 0, 0, t1))
        return result

    def __getattr__(self, name):
        return getattr(self._model, name)


def main():
    a = parse()
    cfg = CELLS[a.cell]
    order = window_order(a.cycles)

    import laya_apple
    from laya_apple import Laya, executor

    binding, stamps, native = None, None, None
    if cfg["ane_predict"] == "prebind_async_stamped":
        binding = _load("prebind_async", ASYNC92 / "prebind_async.py")  # #92's, unchanged
        import prebind

        prebind._shim()  # #83's __init__ compiles and holds it; compile now, not in the loader thread
        binding._stamper_lib()  # compile the C block now too
        stamps, native = binding.STAMPS, binding.StampedAsyncPrebindModel.NATIVE

    auto_workers: list = []
    in_process: dict[str, list] = {"gpu": [], "ane": []}
    predicts: list = []
    load_records: dict = {"crossings": None, "backings": None, "callbacks_per_forward": None, "models": []}
    cpu_dir = tempfile.mkdtemp(prefix="laya-transient-cpu-")
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
                    from laya_apple.artifacts import COMPILED, artifact_dir

                    backend.models = {
                        b: binding.StampedAsyncPrebindModel(artifact_dir(backend.spec, b) / COMPILED)
                        for b in backend.models
                    }
                    load_records["models"] = list(backend.models.values())
                    executor.warm("ane", backend, self._args["pad_id"])
                pad = self._args["pad_id"]
                n_cb = len(binding.CALLBACKS) if binding is not None else 0
                load_records["crossings"] = {
                    str(b): crossings.count(backend.forward, [{"ids": [pad] * b, "markers": [1, 2], "qtype": 0}])[1]
                    for b in backend.buckets
                }
                if binding is not None:
                    load_records["callbacks_per_forward"] = (len(binding.CALLBACKS) - n_cb) / len(backend.buckets)
                    load_records["backings"] = {
                        str(b): {"modes": dict(m.modes), "evidence": m.backing_evidence}
                        for b, m in backend.models.items()
                    }
                backend.models = {b: Stamped(m, stamps, native, predicts) for b, m in backend.models.items()}
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

    counter = WindowCounter(a.cycles)
    base_conditions = bench_concurrency.conditions

    def conditions():
        out = base_conditions()
        counter.next()
        return out

    bench_concurrency.conditions = conditions  # part_a calls the module global

    windows: dict[tuple[int, int], dict] = {}
    wlock = threading.Lock()
    base_loop = bench_concurrency.closed_loop

    def measured_loop(laya, req, start_at, end_at, out, reference):
        """#92's closed loop with per-thread CPU snapshots and the clients' thread ids."""
        if laya.device != "auto":
            return base_loop(laya, req, start_at, end_at, out, reference)
        stream = "short" if req["length"] == a.short else "long"
        threading.current_thread().name = f"client-{stream}"
        recount.set_os_thread_name(f"client-{stream}")
        key = (int(start_at * 1e9), int(end_at * 1e9))
        before = probe.snapshot()  # before start_at: outside the window
        with wlock:
            w = windows.setdefault(
                key, {"start_ns": key[0], "end_ns": key[1], "instance": "auto", "streams": {}, "before": before}
            )
            w.setdefault("tids", {})[stream] = threading.get_native_id()
        base_loop(laya, req, start_at, end_at, out, reference)
        after = probe.snapshot()  # after end_at
        with wlock:
            w["streams"][stream] = len(out["latency_ms"])
            w.setdefault("after_by_stream", {})[stream] = after  # this client's own closing snapshot
            if w.get("after") is None or after["t_ns"] > w["after"]["t_ns"]:
                w["after"] = after

    bench_concurrency.closed_loop = measured_loop  # run_mix.py wraps this one in turn

    import run_mix

    tmp = Path(tempfile.mkdtemp(prefix="laya-transient-run-")) / "run.json"
    sys.argv = [
        "run_mix.py",
        *("--model", a.model, "--short", str(a.short), "--long", str(a.long)),
        *("--ane-placement", cfg["ane_placement"], "--part", "a", "--output", str(tmp)),
        *("--seconds", str(a.seconds), "--cycles", str(a.cycles)),
    ]
    cb_seen: set[int] = set()
    cb_pos = [0]

    def callback_tids():  # completion-callback thread ids seen so far (appended by the handler)
        if binding is not None:
            rows = binding.CALLBACKS
            for r in rows[cb_pos[0] : len(rows)]:
                cb_seen.add(r[1])
            cb_pos[0] = len(rows)
        return cb_seen

    def gpu_pids():
        return [x.pid for x in auto_workers if x.kind == "gpu" and x.placement == "process" and x.pid]

    sampler = None if a.no_recount else recount.Sampler(extra_tids=callback_tids, extra_pids=gpu_pids)
    if sampler is not None:
        sampler.start()  # from before the models load to the end of the run, every cell
    t0 = time.monotonic()
    run_mix.main()
    if sampler is not None:
        sampler.stop()
    rec = json.loads(tmp.read_text())
    tmp.unlink()
    tmp.parent.rmdir()
    if counter.i != len(counter.order):
        raise RuntimeError(f"part_a ran {counter.i} windows, window_order() expects {len(counter.order)}")

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
    if binding is not None:
        cbs = list(binding.CALLBACKS)
        callbacks = {
            "columns": CALLBACK_COLUMNS,
            "hetero": [list(r) for r in cbs if inside(r[0])],
            "total": len(cbs),
            "submits": sum(m._completion().submits for m in load_records["models"]),
            "errors": sum(r[2] for r in cbs),
            "thread_ids": sorted({r[1] for r in cbs}),
            "queue_labels": dict(binding.LABELS),
            "anomalies": list(binding.ANOMALIES),
        }

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rec["research"] = {
        "experiment": "coreml-async-transient",
        "cell": a.cell,
        **cfg,
        "run_wall_s": time.monotonic() - t0,
        "laya_apple": laya_apple.__version__,
        "pyobjc": _version("pyobjc-framework-CoreML"),
        "coremltools": _version("coremltools"),
        "mlx": _version("mlx"),
        "python": sys.version.split()[0],
        "switch_interval_s": sys.getswitchinterval(),
        "window_order": [list(x) for x in order],
        "recount": None if sampler is None else sampler.record(),
        "ioreport": IOREPORT,
        "workers": workers,
        "forwards_columns": ["service_start_ns", "service_end_ns", "thread_cpu_ns"],
        "forwards": forwards,
        "predicts_columns": PREDICT_COLUMNS,
        "predicts": predicts,
        "crossings": load_records["crossings"],
        "callbacks_per_forward_at_load": load_records["callbacks_per_forward"],
        "backings": load_records["backings"],
        "callbacks": callbacks,
        "trace": trace,
        "windows": win_list,
    }
    rec["args"]["output"] = a.output  # not the temporary path run_mix.py wrote to
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
