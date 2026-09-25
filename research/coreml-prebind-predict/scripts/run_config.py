"""One run of the hetero-only closed-loop mix for one ANE binding: P (A or B), C or PB.

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run --with pyobjc-framework-CoreML==12.2.2 python \
        research/coreml-prebind-predict/scripts/run_config.py --config PB \
        --model laya --short 128 --long 512 --output research/coreml-prebind-predict/raw/laya-PB-r1.json.gz

Adapted from research/coreml-nogil-product-mix/scripts/run_config.py (#77): the same injection
into the heterogeneous instance (device="auto", execution="workers") and the same process-worker
entry (#77's worker_entry.py, used by path). C loads #77's nogil.py by path, unchanged.
laya_apple itself is not modified.

Workload (criteria.md, "Protocol"): #57's Part A reduced to its hetero windows. The pieces are
scripts/bench_concurrency.py's own (closed_loop, stats, conditions, its request construction
make_request(seed=0) and its inline references), driven here without the solo_short,
solo_long and gpu_only windows, without the gpu_only instance and without Part B:
  per run: load, references, bench_concurrency's 5-request warm-up per stream; then per cycle
  (3): SETTLE_S idle, a fixed WARMUP_S warm-up of both streams (closed-loop, not measured),
  LEAD_S, and one measured hetero window of --seconds (20).
Each stream repeats one fixed request, built by make_request(seed=0): the request content of
window k is the same in every configuration, and a closed-loop client has no arrival schedule
to seed. The output keeps run_mix.py's layout (part_a.windows, windows_t, gpu_return), so #57's
placement_summary reads it unchanged.

  config  ANE placement  Core ML predict                                  GPU (MLX)
  A       thread         coremltools (GIL held)                           worker process
  B       process        coremltools, in the worker                       worker process
  C       thread         #77: PyObjC-built per call, GIL released         worker process
  PB      thread         prebound: one predict crossing (prebind.py)      worker process

Recorded on top of that, non-gating, the same in every configuration (criteria.md):
  trace       every request of the heterogeneous instance submitted in a hetero window: all
              RequestTrace timestamps, both targets (received_ns and response_ns kept)
  forwards    per device, every forward: service start/end and the executing thread's CPU
  predicts    in-process ANE: per predict, entry, the binding's four stamps (0 for coremltools),
              exit
  crossings   in-process ANE: bridge crossings of one forward per bucket, counted at load
  backings    PB: each output's mode and the load-time evidence
  windows     per window: streams, requests per stream, per-thread CPU snapshots taken just
              before its start and just after its end, and the GIL probe's lateness summary
  gil_probe   raw lateness of every 1 ms probe tick inside a hetero window
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
sys.path.insert(0, str(ROOT / "scripts"))  # bench_concurrency.py
sys.path.insert(0, str(MIX77))
sys.path.insert(0, str(HERE))

import crossings  # noqa: E402
import derive  # noqa: E402
import probe  # noqa: E402

CONFIGS = {
    "A": {"ane_placement": "thread", "ane_predict": "coremltools", "gpu_placement": "process"},
    "B": {"ane_placement": "process", "ane_predict": "coremltools", "gpu_placement": "process"},
    "C": {"ane_placement": "thread", "ane_predict": "nogil", "gpu_placement": "process"},
    "PB": {"ane_placement": "thread", "ane_predict": "prebind", "gpu_placement": "process"},
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
STREAMS = ("short", "long")
SETTLE_S = 2.0  # idle before each cycle, as bench_concurrency.part_a
WARMUP_S = 2.0  # the fixed warm-up before every hetero window: both streams, closed-loop, not measured
LEAD_S = 0.5  # from starting the clients to the window's start, as bench_concurrency.part_a
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
    ap.add_argument("--config", choices=sorted(CONFIGS), required=True)
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
    """Wraps one bucket's model: stamps predict entry and exit, and picks up the binding's own
    four stamps (#77's layout) when it has them. Two clock reads per predict, in every
    in-process configuration."""

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
    cfg = CONFIGS[a.config]

    import laya_apple
    from laya_apple import Laya, executor

    binding, stamps = None, None
    if cfg["ane_predict"] == "nogil":
        import nogil as binding  # #77's, unchanged

        stamps = binding.STAMPS
    elif cfg["ane_predict"] == "prebind":
        import prebind as binding

        stamps = binding.STAMPS
    if binding is not None:
        binding._shim()  # compile now, not inside the ANE loader thread

    auto_workers: list = []
    in_process: dict[str, list] = {"gpu": [], "ane": []}
    predicts: list = []
    load_records: dict = {"crossings": None, "backings": None}
    cpu_dir = tempfile.mkdtemp(prefix="laya-prebind-cpu-")
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
    gpu_returns: list = []

    def record_trace(t):  # run_mix.py's GPU-return record, plus every request's full trace
        trace_rows.append(tuple(getattr(t, c) for c in TRACE_COLUMNS))
        if t.target == "gpu":
            gpu_returns.append((t.received_ns, (t.received_ns - t.service_end_ns) // 1000))

    import bench_concurrency  # scripts/, on sys.path above

    from laya_apple.workload import make_request

    windows: dict[int, dict] = {}
    wlock = threading.Lock()
    base_loop = bench_concurrency.closed_loop

    def measured_loop(laya, req, start_at, end_at, out, reference):
        stream = "short" if req["length"] == a.short else "long"
        threading.current_thread().name = f"client-{stream}"
        key = int(start_at * 1e9)
        before = probe.snapshot()  # before start_at: outside the window
        with wlock:
            w = windows.setdefault(
                key,
                {
                    "start_ns": key,
                    "end_ns": int(end_at * 1e9),
                    "instance": "auto",
                    "streams": {},
                    "before": before,
                    "after": None,
                },
            )
        base_loop(laya, req, start_at, end_at, out, reference)
        after = probe.snapshot()  # after end_at
        with wlock:
            w["streams"][stream] = len(out["latency_ms"])
            if w["after"] is None or after["t_ns"] > w["after"]["t_ns"]:
                w["after"] = after

    def both_streams(loop, laya, reqs, refs, start_at, end_at):
        outs = {s: {} for s in STREAMS}
        ts = [threading.Thread(target=loop, args=(laya, reqs[s], start_at, end_at, outs[s], refs[s])) for s in STREAMS]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return outs

    gil_probe = probe.GilProbe()
    gil_probe.start()
    t0 = time.monotonic()

    # bench_concurrency.main()'s set-up, without the gpu_only instance and Part B
    _Flags.auto = True
    try:
        laya = Laya.from_pretrained(
            a.model,
            device="auto",
            execution="workers",
            local_files_only=True,
            ane_placement=cfg["ane_placement"],
            trace=record_trace,
        )
    finally:
        _Flags.auto = False
    load_s = time.monotonic() - t0
    ref_gpu = Laya.from_pretrained(a.model, device="gpu", local_files_only=True)
    ref_ane = Laya.from_pretrained(a.model, device="ane", local_files_only=True)
    tok, model_cfg = laya.tokenizer, laya.config

    def req(length):  # bench_concurrency's request: make_request(seed=0), one question
        state, qs = make_request(tok, model_cfg, length, n_questions=1, seed=0)
        return {"state": state, "questions": qs, "length": length, "q": 1}

    reqs = {"short": req(a.short), "long": req(a.long)}

    def refs_for(r):  # bench_concurrency's inline references per device
        out = {"gpu": ref_gpu.predict(context=r["state"], questions=r["questions"]).answers}
        try:
            out["ane"] = ref_ane.predict(context=r["state"], questions=r["questions"]).answers
        except laya_apple.LayaAppleError:  # longer than the largest ANE bucket
            out["ane"] = None
        return out

    refs = {k: refs_for(r) for k, r in reqs.items()}
    for r in reqs.values():  # bench_concurrency's warm-up of the workers path
        for _ in range(5):
            laya.predict(context=r["state"], questions=r["questions"])

    part_windows, windows_t, warmups = [], [], []
    for cycle in range(a.cycles):
        time.sleep(SETTLE_S)
        cond_before = bench_concurrency.conditions()
        w0 = time.monotonic()  # the fixed warm-up: both streams, closed-loop, not measured
        warm = both_streams(base_loop, laya, reqs, refs, w0, w0 + WARMUP_S)
        warmups.append({s: {"n": len(o["latency_ms"]), "mismatches": o["mismatches"]} for s, o in warm.items()})
        start_at = time.monotonic() + LEAD_S
        end_at = start_at + a.seconds
        outs = both_streams(measured_loop, laya, reqs, refs, start_at, end_at)
        w = {"cycle": cycle, "condition": "hetero", "streams": {}, "conditions_before": cond_before}
        for s, o in outs.items():
            w["streams"][s] = {
                **bench_concurrency.stats(o["latency_ms"]),
                "req_s": len(o["latency_ms"]) / o["wall_s"],
                "devices": o["devices"],
                "mismatches": o["mismatches"],
                "latency_ms": o["latency_ms"],
            }
            windows_t.append(
                {"stream": s, "instance": "auto", "start_ns": int(start_at * 1e9), "end_ns": int(end_at * 1e9)}
            )
        part_windows.append(w)
        print(
            "hetero",
            cycle,
            {
                s: (round(v["req_s"], 1), round(v["p99_ms"], 2), v["devices"], v["mismatches"])
                for s, v in w["streams"].items()
            },
            flush=True,
        )
    auto_info = laya.info()
    laya.close()
    gil_probe.stop()
    rec = {
        "experiment": "coreml-prebind-predict: hetero-only closed-loop mix",
        "args": {
            "model": a.model,
            "short": a.short,
            "long": a.long,
            "seconds": a.seconds,
            "cycles": a.cycles,
            "ane_placement": cfg["ane_placement"],
            "settle_s": SETTLE_S,
            "warmup_s": WARMUP_S,
            "lead_s": LEAD_S,
            "request_seed": 0,
        },
        "laya_apple": laya_apple.__version__,
        "load_s_workers_instance": load_s,
        "auto_info": auto_info,
        "part_a": {"windows": part_windows, "warmups": warmups},
        "windows_t": windows_t,
        "gpu_return": {"received_ns": [r for r, _ in gpu_returns], "return_us": [u for _, u in gpu_returns]},
    }

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

    trace_rows.sort(key=lambda r: r[3])
    in_hetero = derive.in_windows([r[3] for r in trace_rows], bounds)
    trace = {c: [r[i] for r, keep in zip(trace_rows, in_hetero) if keep] for i, c in enumerate(TRACE_COLUMNS)}

    ticks, late = list(gil_probe.ticks), list(gil_probe.late)
    win_list = []
    for w in sorted(windows.values(), key=lambda x: x["start_ns"]):
        lo, hi = w["start_ns"], w["end_ns"]
        sel = [x for t, x in zip(ticks, late) if lo <= t < hi]
        win_list.append({**w, "gil_probe": derive.lateness_summary(sel)})
    raw_ticks, raw_late = [], []
    for lo, hi in bounds:
        sel = [(t, x) for t, x in zip(ticks, late) if lo <= t < hi]
        raw_ticks.append(_delta_us([t - lo for t, _ in sel]))
        raw_late.append([x // 1000 for _, x in sel])

    rec["research"] = {
        "experiment": "coreml-prebind-predict",
        "config": a.config,
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
        "predicts_columns": PREDICT_COLUMNS,
        "predicts": predicts if cfg["ane_placement"] == "thread" else None,
        "crossings": load_records["crossings"],
        "backings": load_records["backings"],
        "trace": trace,
        "windows": win_list,
        "gil_probe": {
            "period_ns": gil_probe.period,
            "thread_cpu_ns": gil_probe.cpu_ns,
            "ticks": len(ticks),
            "hetero_tick_delta_us": raw_ticks,
            "hetero_late_us": raw_late,
        },
    }
    rec["args"]["output"] = a.output
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")  # a complete file or none: run_all.sh skips finished runs
    with gzip.open(part, "wt") as fh:
        json.dump(rec, fh)
    part.rename(out)
    print("wrote", out, f"({rec['research']['run_wall_s']:.0f} s)")


def _delta_us(offsets_ns: list[int]) -> list[int]:
    """Offsets from the window start, in µs, delta-encoded (the 1 ms grid compresses to ~1000s)."""
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
