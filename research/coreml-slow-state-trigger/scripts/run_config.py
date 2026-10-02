"""One run of #57's full product mix (R1's protocol) for one cell of the slow-state trigger test.

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run --with pyobjc-framework-CoreML==12.2.2 python \
        research/coreml-slow-state-trigger/scripts/run_config.py --cell PB-H \
        --model laya --short 128 --long 512 --output research/coreml-slow-state-trigger/raw/laya-PB-H-r1.json.gz

A copy of R1's run_config.py (research/coreml-prebind-full-protocol/scripts/), with a --cell
option in place of --config. The workload, the injection into the heterogeneous instance and
every R1 record are R1's code paths, unchanged: run_mix.main() -> bench_concurrency --part a
(solo_short, solo_long, hetero and gpu_only windows in alternating order; 3 cycles of 20 s; 2.0 s
idle and 0.5 s lead before each window; one closed-loop client per stream repeating the
make_request(seed=0) request; answers checked against inline coremltools and MLX references),
one fresh process per run. laya_apple itself is not modified.

  cell  R1 config  Core ML predict (ANE thread-placed, GPU in a worker process)   added
  A     A          coremltools, GIL held (production)                             -
  PB-R  PB         #83's prebind, shim called through ctypes.CDLL: GIL released   -
  PB-H  PB         the same, but the shim is called through ctypes.PYFUNCTYPE     -
                   (same address, restype and argtypes): the GIL is held
  PB-W  PB         as PB-R                                                        warm-up
  PB-P  PB         as PB-R                                                        1 ms GIL probe

PB-H: after #83's prebind.install(), each bucket's PrebindModel._call (the function
PrebindModel.predict calls, and nothing else) is replaced by gil_holding(prebind._shim()): the
same native function, the same prebound buffers, provider, outputBackings and options, and the
same four stamps. Only whether ctypes releases the GIL around the call differs.

PB-W: #83's fixed warm-up before each hetero window only. bench_concurrency.part_a calls
conditions() after each window's 2.0 s idle and right before it computes start_at; this script
wraps it with a counter over part_a's deterministic window order (window_order(): conds in order,
reversed on odd cycles). When the upcoming window is hetero, the wrapper runs, after the
original conditions(), both streams closed-loop through the auto instance for WARMUP_S (not
measured; counts, devices and mismatches recorded), exactly as #83's run_config.py did. Then
part_a's normal 0.5 s lead and window follow. The auto instance and each stream's request and
references are the objects bench_concurrency.main() built: they are captured from the first
closed_loop call of the auto instance per stream (the solo_short and solo_long windows of cycle
0 run before the first hetero window). The warm-up uses bench_concurrency's own closed_loop, not
the wrapped ones, so it adds no window record, no windows_t entry and no CPU snapshot; its
requests fall outside every hetero window's bounds, so the trace, forwards and predicts
analyses exclude them as they exclude the solo windows.

PB-P: #83's probe.GilProbe (1 ms grid) is started before the models load and stopped at the
end, as #83's run_config.py did; each hetero window record carries its lateness summary and the
raw per-tick lateness is kept (#83's layout).

Every cell carries R1's records: forwards (per-forward executing-thread CPU), predicts (the
Stamped wrapper: entry, the binding's four stamps, exit), crossings, backings, the hetero trace
and per-window thread CPU snapshots. One addition to R1's window record, with no added work
(the snapshot is already taken): each client's own closing snapshot (`after_by_stream`), because
a client thread that finishes first is gone from R1's single closing snapshot, which left
client-short CPU uncomputable in most windows.
"""

from __future__ import annotations

import argparse
import ctypes
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

import crossings  # noqa: E402  #83's, unchanged
import derive  # noqa: E402  #83's; lateness_summary and in_windows only
import probe  # noqa: E402  #83's: the Mach per-thread CPU snapshot, and GilProbe for PB-P

CONFIGS = {  # R1's
    "A": {"ane_placement": "thread", "ane_predict": "coremltools", "gpu_placement": "process"},
    "PB": {"ane_placement": "thread", "ane_predict": "prebind", "gpu_placement": "process"},
}
CELLS = {  # cell: (R1 config, GIL during the shim call, fixed pre-hetero warm-up, 1 ms GIL probe)
    "A": {"config": "A", "shim_gil": None, "warmup": False, "gil_probe": False},
    "PB-R": {"config": "PB", "shim_gil": "released", "warmup": False, "gil_probe": False},
    "PB-H": {"config": "PB", "shim_gil": "held", "warmup": False, "gil_probe": False},
    "PB-W": {"config": "PB", "shim_gil": "released", "warmup": True, "gil_probe": False},
    "PB-P": {"config": "PB", "shim_gil": "released", "warmup": False, "gil_probe": True},
}
CONDITIONS = ("solo_short", "solo_long", "hetero", "gpu_only")  # bench_concurrency.part_a's conds
WARMUP_S = 2.0  # #83's fixed warm-up: both streams, closed-loop, not measured
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
    "python_before_ns",
    "native_before_ns",
    "native_after_ns",
    "python_after_ns",
    "exit_ns",
]


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


# ------------------------------------------------------------------ PB-H


def gil_holding(fn):
    """The same native function as the ctypes foreign function `fn`, same restype and argtypes,
    called through a PYFUNCTYPE prototype: ctypes keeps the GIL for the call."""
    addr = ctypes.cast(fn, ctypes.c_void_p).value
    return ctypes.PYFUNCTYPE(fn.restype, *fn.argtypes)(addr)


def hold_gil(backend, shim) -> None:
    """PB-H: every prebound bucket calls the shim with the GIL held. PrebindModel.predict calls
    self._call and nothing else native; verify_backings has already run at construction."""
    held = gil_holding(shim)
    for m in backend.models.values():
        assert m._call is shim, "PrebindModel no longer calls the shared shim through _call"
        m._call = held


# ------------------------------------------------------------------ PB-W


def window_order(cycles: int) -> list[tuple[int, str]]:
    """(cycle, condition) of bench_concurrency.part_a's windows in execution order."""
    return [(k, c) for k in range(cycles) for c in (CONDITIONS if k % 2 == 0 else tuple(reversed(CONDITIONS)))]


class WindowCounter:
    """Advances once per part_a window (one conditions() call each) and names the upcoming one."""

    def __init__(self, cycles: int):
        self.order, self.i = window_order(cycles), 0

    def next(self) -> tuple[int, str]:
        if self.i >= len(self.order):
            raise RuntimeError("more windows than bench_concurrency.part_a runs")
        w = self.order[self.i]
        self.i += 1
        return w


class _Flags:
    auto = False  # set while the device="auto", execution="workers" instance is being built


class Stamped:
    """#83's wrapper: stamps predict entry and exit, and picks up the binding's own four stamps
    (#77's layout) when it has them. Two clock reads per predict, in every configuration."""

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
    cell = CELLS[a.cell]
    cfg = CONFIGS[cell["config"]]

    import laya_apple
    from laya_apple import Laya, executor

    binding, stamps = None, None
    if cfg["ane_predict"] == "prebind":
        import prebind as binding  # #83's, unchanged

        stamps = binding.STAMPS
    if binding is not None:
        binding._shim()  # compile now, not inside the ANE loader thread

    auto_workers: list = []
    in_process: dict[str, list] = {"gpu": [], "ane": []}
    predicts: list = []
    load_records: dict = {"crossings": None, "backings": None, "shim_flags": None}
    cpu_dir = tempfile.mkdtemp(prefix="laya-slow-state-cpu-")
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
                    if cell["shim_gil"] == "held":
                        hold_gil(backend, binding._shim())
                    load_records["shim_flags"] = {str(b): int(m._call._flags_) for b, m in backend.models.items()}
                    executor.warm("ane", backend, self._args["pad_id"])
                pad = self._args["pad_id"]
                load_records["crossings"] = {
                    str(b): crossings.count(backend.forward, [{"ids": [pad] * b, "markers": [1, 2], "qtype": 0}])[1]
                    for b in backend.buckets
                }
                if cfg["ane_predict"] == "prebind":
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
    auto_streams: dict[str, tuple] = {}  # stream: (auto instance, request, references), for PB-W

    def measured_loop(laya, req, start_at, end_at, out, reference):
        """bench_concurrency's closed loop, with #83's per-thread CPU snapshots of the auto
        instance's windows: one before the lead, one after the window's end."""
        if laya.device != "auto":
            return base_loop(laya, req, start_at, end_at, out, reference)
        stream = "short" if req["length"] == a.short else "long"
        auto_streams.setdefault(stream, (laya, req, reference))
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
            # each client's own closing snapshot: a thread is gone from later ones (not in R1)
            w.setdefault("after_by_stream", {})[stream] = after
            if w.get("after") is None or after["t_ns"] > w["after"]["t_ns"]:
                w["after"] = after

    bench_concurrency.closed_loop = measured_loop  # run_mix.py wraps this one in turn

    warmups: list = []
    counter = WindowCounter(a.cycles)
    base_conditions = bench_concurrency.conditions

    def conditions():
        """part_a's conditions(), then, for PB-W and a hetero window next, #83's fixed warm-up."""
        out = base_conditions()
        cycle, name = counter.next()
        if cell["warmup"] and name == "hetero":
            if set(auto_streams) != {"short", "long"}:
                raise RuntimeError("warm-up before the auto instance ran both streams")
            outs = {s: {} for s in ("short", "long")}
            w0 = time.monotonic()
            ts = [
                threading.Thread(target=base_loop, args=(laya, req, w0, w0 + WARMUP_S, outs[s], ref))
                for s, (laya, req, ref) in auto_streams.items()
            ]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            warmups.append(
                {
                    "cycle": cycle,
                    "before": name,
                    "start_ns": int(w0 * 1e9),
                    "end_ns": time.monotonic_ns(),
                    "streams": {
                        s: {"n": len(o["latency_ms"]), "devices": o["devices"], "mismatches": o["mismatches"]}
                        for s, o in outs.items()
                    },
                }
            )
        return out

    bench_concurrency.conditions = conditions  # part_a calls the module global

    import run_mix

    tmp = Path(tempfile.mkdtemp(prefix="laya-slow-state-run-")) / "run.json"
    sys.argv = [
        "run_mix.py",
        *("--model", a.model, "--short", str(a.short), "--long", str(a.long)),
        *("--ane-placement", cfg["ane_placement"], "--part", "a", "--output", str(tmp)),
        *("--seconds", str(a.seconds), "--cycles", str(a.cycles)),
    ]
    gil_probe = probe.GilProbe() if cell["gil_probe"] else None
    if gil_probe is not None:
        gil_probe.start()  # before the models load, as #83's run_config.py
    t0 = time.monotonic()
    run_mix.main()
    if gil_probe is not None:
        gil_probe.stop()
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

    probe_rec = None
    if gil_probe is not None:  # #83's per-window summary and raw hetero ticks
        ticks, late = list(gil_probe.ticks), list(gil_probe.late)
        raw_ticks, raw_late = [], []
        for w in win_list:
            lo, hi = w["start_ns"], w["end_ns"]
            sel = [(t, x) for t, x in zip(ticks, late) if lo <= t < hi]
            w["gil_probe"] = derive.lateness_summary([x for _, x in sel])
            raw_ticks.append(_delta_us([t - lo for t, _ in sel]))
            raw_late.append([x // 1000 for _, x in sel])
        probe_rec = {
            "period_ns": gil_probe.period,
            "thread_cpu_ns": gil_probe.cpu_ns,
            "ticks": len(ticks),
            "hetero_tick_delta_us": raw_ticks,
            "hetero_late_us": raw_late,
        }

    rec["research"] = {
        "experiment": "coreml-slow-state-trigger",
        "cell": a.cell,
        **cell,
        **cfg,
        "run_wall_s": time.monotonic() - t0,
        "laya_apple": laya_apple.__version__,
        "pyobjc": _version("pyobjc-framework-CoreML"),
        "coremltools": _version("coremltools"),
        "mlx": _version("mlx"),
        "python": sys.version.split()[0],
        "switch_interval_s": sys.getswitchinterval(),
        "gil_probe": probe_rec,
        "warmup_s": WARMUP_S if cell["warmup"] else None,
        "warmups": warmups if cell["warmup"] else None,
        "window_order": [list(x) for x in counter.order],
        "shim_flags": load_records["shim_flags"],
        "workers": workers,
        "forwards_columns": ["service_start_ns", "service_end_ns", "thread_cpu_ns"],
        "forwards": forwards,
        "predicts_columns": PREDICT_COLUMNS,
        "predicts": predicts,
        "crossings": load_records["crossings"],
        "backings": load_records["backings"],
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


def _delta_us(offsets_ns: list[int]) -> list[int]:
    """#83's: offsets from the window start, in µs, delta-encoded."""
    us = [t // 1000 for t in offsets_ns]
    return [b - a for a, b in zip([0] + us[:-1], us)]


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
