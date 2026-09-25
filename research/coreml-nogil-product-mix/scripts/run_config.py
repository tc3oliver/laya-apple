"""One run of the #57 product mix for one execution configuration.

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run --with pyobjc-framework-CoreML==12.2.2 python \
        research/coreml-nogil-product-mix/scripts/run_config.py --config C \
        --model laya --short 128 --long 512 --output research/coreml-nogil-product-mix/raw/laya-C-r1.json.gz

The workload is benchmarks/ane-process-isolation/run_mix.py, which runs
scripts/bench_concurrency.py --part a unchanged (solo_short, solo_long, hetero, gpu_only; 3
cycles of 20 s; one closed-loop client per stream; answers checked against inline coremltools
and MLX references). Only the heterogeneous instance (device="auto", execution="workers") is
changed, by injection from this script; laya_apple itself is not modified:

  config  ANE placement  Core ML predict                      GPU (MLX) placement
  A       thread         coremltools (GIL held)               worker process
  B       process        coremltools, in the worker           worker process
  C       thread         PyObjC-built, GIL released (nogil)   worker process
  D       thread         PyObjC-built, GIL released (nogil)   thread in the caller (no IPC)

--ane-predict nogil installs scripts/nogil.py on the ANE backend after it loads, and warms it
again. --gpu-placement thread (D only) builds the GPU DeviceWorker with placement="thread",
the executor's existing in-process mode. The inline references and the gpu_only instance are
the product's in every configuration.

Recorded on top of run_mix.py's output (window bounds, GPU return legs), non-gating:
  forwards    per device of the heterogeneous instance, for each forward that started inside a
              hetero window: service start/end and the executing thread's CPU time (the
              dispatcher thread when in-process; the worker's thread via worker_entry.py)
  gil_wait    C and D: per ANE predict inside a hetero window, the four stamps of nogil.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "ane-process-isolation"))

CONFIGS = {
    "A": {"ane_placement": "thread", "ane_predict": "coremltools", "gpu_placement": "process"},
    "B": {"ane_placement": "process", "ane_predict": "coremltools", "gpu_placement": "process"},
    "C": {"ane_placement": "thread", "ane_predict": "nogil", "gpu_placement": "process"},
    "D": {"ane_placement": "thread", "ane_predict": "nogil", "gpu_placement": "thread"},
}


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


def main():
    a = parse()
    cfg = CONFIGS[a.config]

    import laya_apple
    from laya_apple import Laya, executor

    nogil = None
    if cfg["ane_predict"] == "nogil":
        import nogil  # noqa: F811  compiles the shim now, not inside the ANE loader thread

        nogil._shim()

    auto_workers: list = []
    in_process: dict[str, list] = {"gpu": [], "ane": []}
    cpu_dir = tempfile.mkdtemp(prefix="laya-nogil-cpu-")
    os.environ["LAYA_NOGIL_CPU_DIR"] = cpu_dir

    class ResearchWorker(executor.DeviceWorker):
        def __init__(self, kind, args, *, placement="process", **kw):
            self._nogil = False
            if _Flags.auto:
                auto_workers.append(self)
                if kind == "gpu":
                    placement = cfg["gpu_placement"]
                self._nogil = kind == "ane" and nogil is not None
                assert placement == (cfg["ane_placement"] if kind == "ane" else cfg["gpu_placement"])
            super().__init__(kind, args, placement=placement, **kw)

        def _load_here(self):
            super()._load_here()
            backend = self._loaded.get("backend")
            if self._nogil and backend is not None:
                try:
                    nogil.install(backend)
                    executor.warm("ane", backend, self._args["pad_id"])
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

    class Spawn:  # every process worker starts through worker_entry.py (same executor code)
        def __getattr__(self, name):
            return getattr(subprocess, name)

        @staticmethod
        def Popen(cmd, *args, **kw):
            if list(cmd[1:]) == ["-m", "laya_apple.executor"]:
                cmd = [cmd[0], str(HERE / "worker_entry.py")]
            return subprocess.Popen(cmd, *args, **kw)

    executor.subprocess = Spawn()

    from_pretrained = Laya.from_pretrained.__func__

    def flagged(cls, model_id, *args, **kw):
        _Flags.auto = kw.get("device") == "auto" and kw.get("execution") == "workers"
        try:
            return from_pretrained(cls, model_id, *args, **kw)
        finally:
            _Flags.auto = False

    Laya.from_pretrained = classmethod(flagged)  # run_mix.py wraps this one in turn

    import run_mix

    tmp = Path(tempfile.mkdtemp(prefix="laya-nogil-run-")) / "run.json"
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

    def inside(t):
        return any(lo <= t < hi for lo, hi in bounds)

    forwards, workers = {}, {}
    for w in auto_workers:
        workers[w.kind] = {"placement": w.placement, "pid": w.pid}
        if w.placement == "thread":
            rows = in_process[w.kind]
        else:
            f = Path(cpu_dir, f"{w.pid}.json")
            rows = json.loads(f.read_text())["forwards"] if f.exists() else None
        forwards[w.kind] = None if rows is None else [r for r in rows if inside(r[0])]
    for f in Path(cpu_dir).iterdir():
        f.unlink()
    os.rmdir(cpu_dir)
    gil_wait = None
    if nogil is not None:
        gil_wait = [s for s in nogil.STAMPS if inside(s[2])]

    rec["research"] = {
        "experiment": "coreml-nogil-product-mix",
        "config": a.config,
        **cfg,
        "run_wall_s": time.monotonic() - t0,
        "laya_apple": laya_apple.__version__,
        "pyobjc": _version("pyobjc-framework-CoreML"),
        "coremltools": _version("coremltools"),
        "mlx": _version("mlx"),
        "workers": workers,
        "forwards_columns": ["service_start_ns", "service_end_ns", "thread_cpu_ns"],
        "forwards": forwards,
        "gil_wait_columns": ["python_before_ns", "native_before_ns", "native_after_ns", "python_after_ns"],
        "gil_wait": gil_wait,
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
