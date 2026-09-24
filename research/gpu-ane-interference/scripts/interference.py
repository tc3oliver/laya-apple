"""GPU-ANE interference and tail latency: the measurement driver (issue #16).

    LAYA_APPLE_CACHE=... HF_HUB_OFFLINE=1 uv run python \
        research/gpu-ane-interference/scripts/interference.py \
        --model laya-typed-decisions --ane-placement thread --plan device \
        --out research/gpu-ane-interference/raw/device-laya-typed-decisions-thread.json.gz

The GPU runs in its worker process, the ANE on a thread or in a worker process
(--ane-placement), both in `Laya(execution="workers")` instances of this process. Two plans:

device    Each stream feeds ONE device through `Laya.submit` of an explicit-device instance
          (`device="gpu"` or `device="ane"`, both in this process, so a thread-placed ANE shares
          this interpreter with the GPU's dispatcher exactly as under `auto`). The experimenter,
          not the router, decides which device runs what. Cells:
            solo      one stream alone (the baseline every other cell is compared with)
            matrix    one GPU and one ANE stream, both closed loop
            sweep     a closed-loop victim on one device, an open-loop Poisson aggressor on
                      the other at a fraction of its solo capacity
            control   a closed-loop victim with no device aggressor, but a CPU, memory-
                      bandwidth or GIL aggressor (to tell device from host contention)
            sparse    one device alone, open-loop Poisson at a low fraction of its capacity
                      (--parts sparse): service time when the machine is mostly idle
product   Requests go through Laya.submit, so the v0.2 queue-aware router decides:
            closed    the v0.2 Part A mix (solo_short, solo_long, hetero)
            open      Poisson arrivals of the v0.2 Part B class mix at several rates
            heavy     Poisson arrivals of 90% short / 10% long (--heavy-rates)
          --scheduler both runs every open cell twice per cycle, once with the v0.2
          decide_queued and once with the contention.py prototype (research only).

The request lifecycle comes from the runtime: every instance is created with
`trace=callback`, and each completed request's laya_apple.RequestTrace (submit, prepare,
route, queue, dispatch, service, response, and the queue snapshots the router used) is
stored as it was emitted. The harness adds only what the runtime does not know: the
workload generator's arrival time (the scheduled time in open loop), the request's shape
class and seed, whether the answer matched the inline reference, and the backend phase
split from jobtrace.py (device spans, host work, CPU time), tagged with the request id.
ledger.py joins them on request_id. Every answer is compared with the inline answer for the
same request on the same device; a mismatch is reported, never dropped. All times are
time.monotonic_ns, the runtime's clock.

--no-trace and --no-backend-hooks switch the two layers off, to measure what each costs.

Windows: each cell runs `--cycles` times, in alternating cell order. Within one window the
streams start together; the first `--warmup` seconds are discarded and only requests that
ARRIVE inside [warmup, warmup + window] are measured, while every stream keeps running for a
tail guard after the window, so each measured request ran with its aggressor still active.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
import platform
import queue
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import contention  # noqa: E402
import jobtrace  # noqa: E402

now = time.monotonic_ns

# ----------------------------------------------------------------------------- environment


def conditions() -> dict:
    """Background load before a window. LC_ALL=C: ps prints %cpu with the locale's decimal
    separator otherwise (issue #35)."""
    env = dict(os.environ, LC_ALL="C")
    top = subprocess.run(["ps", "-Ao", "%cpu=,comm="], capture_output=True, text=True, env=env).stdout.splitlines()
    rows = []
    for line in top:
        parts = line.split(None, 1)
        if len(parts) == 2:
            try:
                rows.append((float(parts[0]), parts[1].rsplit("/", 1)[-1]))
            except ValueError:
                pass
    therm = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    return {
        "loadavg": os.getloadavg(),
        "top_cpu": sorted(rows, reverse=True)[:6],
        "thermal_warning": "No thermal warning level has been recorded" not in therm,
        "performance_warning": "No performance warning level has been recorded" not in therm,
    }


def environment(laya) -> dict:
    import coremltools
    import mlx.core as mx
    import numpy

    import laya_apple
    from laya_apple.artifacts import platform_profile

    def sh(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()

    return {
        "laya_apple": laya_apple.__version__,
        "git_commit": sh("git", "-C", str(HERE), "rev-parse", "HEAD"),
        "platform": platform_profile(),
        "python": platform.python_version(),
        "mlx": mx.__version__,
        "coremltools": coremltools.__version__,
        "numpy": numpy.__version__,
        "cpu_perflevels": {
            "p_cores": int(sh("sysctl", "-n", "hw.perflevel0.logicalcpu") or 0),
            "e_cores": int(sh("sysctl", "-n", "hw.perflevel1.logicalcpu") or 0),
        },
        "memory_bytes": int(sh("sysctl", "-n", "hw.memsize") or 0),
        "power": sh("pmset", "-g", "batt").splitlines()[:1],
        "clock": jobtrace.clock_info(),
        "switch_interval_s": sys.getswitchinterval(),
        "laya_info": laya.info(),
    }


# ----------------------------------------------------------------------------- aggressors
# Host-side aggressors, for the controls. Each runs in its own process (so it contends for
# CPU cores or memory bandwidth, not for this interpreter's GIL), except "gil", which is a
# pure-Python thread in this process.

_CPU_BURN = "import time\nx=0\nwhile True:\n    for i in range(100000): x ^= i\n"
_MEM_BW = """
import numpy as np, sys, time, json
a = np.ones(64 << 20, np.float64); b = np.empty_like(a)   # 512 MB each: far beyond any cache
n = 0; t0 = time.perf_counter()
while True:
    np.copyto(b, a); n += 1
    if n % 20 == 0:
        dt = time.perf_counter() - t0
        sys.stdout.write(json.dumps({"gb_s": 2 * a.nbytes * 20 / dt / 1e9}) + "\\n"); sys.stdout.flush()
        t0 = time.perf_counter()
"""


class Aggressor:
    def __init__(self, kind: str, n: int = 1):
        self.kind, self.n = kind, n
        self._procs: list = []
        self._stop = threading.Event()
        self._thread = None
        self.samples: list = []

    def start(self):
        if self.kind == "cpu":
            self._procs = [subprocess.Popen([sys.executable, "-c", _CPU_BURN]) for _ in range(self.n)]
        elif self.kind == "membw":
            self._procs = [
                subprocess.Popen([sys.executable, "-c", _MEM_BW], stdout=subprocess.PIPE, text=True)
                for _ in range(self.n)
            ]
        elif self.kind == "gil":

            def spin():
                x = 0
                while not self._stop.is_set():
                    for i in range(10000):
                        x ^= i

            self._thread = threading.Thread(target=spin, daemon=True)
            self._thread.start()
        else:
            raise ValueError(self.kind)
        time.sleep(1.0)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        for p in self._procs:
            p.terminate()
        for p in self._procs:
            p.wait()
            if p.stdout is not None:
                for line in p.stdout.read().splitlines():
                    try:
                        self.samples.append(json.loads(line)["gb_s"])
                    except (ValueError, KeyError):
                        pass

    def describe(self) -> dict:
        d = {"kind": self.kind, "n": self.n}
        if self.samples:
            d["gb_s_median"] = sorted(self.samples)[len(self.samples) // 2]
        return d


# ----------------------------------------------------------------------------- requests


class Workload:
    """Exact-length requests per shape (several seeds each) and their inline references."""

    def __init__(self, laya, model: str, shapes: dict, seeds: int):
        from laya_apple import Laya, LayaAppleError
        from laya_apple.workload import make_request

        self.laya = laya
        self.shapes = shapes  # name -> (length, questions)
        self.reqs: dict = {}
        ref = {
            "gpu": Laya.from_pretrained(model, device="gpu", local_files_only=True),
            "ane": Laya.from_pretrained(model, device="ane", local_files_only=True),
        }
        self.refs: dict = {}
        for name, (length, q) in shapes.items():
            self.reqs[name] = []
            for seed in range(seeds):
                state, questions = make_request(laya.tokenizer, laya.config, length, n_questions=q, seed=100 + seed)
                self.reqs[name].append({"state": state, "questions": questions, "seed": seed})
                for dev, r in ref.items():
                    try:
                        self.refs[(name, seed, dev)] = r.predict(context=state, questions=questions).answers
                    except LayaAppleError:
                        self.refs[(name, seed, dev)] = None  # not ANE-eligible; never sent there
        for r in ref.values():
            r.close()
        del ref

    def cycle(self, name: str):
        return itertools.cycle(self.reqs[name])


# ----------------------------------------------------------------------------- streams


class DeviceStream:
    """Requests of one shape to one device's worker, closed loop or open-loop Poisson."""

    def __init__(self, run, device: str, shape: str, mode: str = "closed", rate: float = 0.0, seed: int = 0):
        self.run, self.device, self.shape, self.mode, self.rate = run, device, shape, mode, rate
        self.rng = random.Random(seed)
        self.records: list = []
        self.errors: list = []

    @property
    def label(self) -> str:
        return f"{self.device}_{self.shape}"

    def _issue(self, req, arrival):
        """Submit one request to this device's instance (on the calling thread). An explicit
        device="ane" instance applies the product's ANE eligibility gate before queueing."""
        fut = self.run.lays[self.device].submit(context=req["state"], questions=req["questions"])
        rec = {"arrival": arrival, "seed": req["seed"], "shape": self.shape}
        fut.add_done_callback(lambda _f, rec=rec: rec.setdefault("done", now()))  # the collector may lag
        return rec, fut

    def _complete(self, req, rec, fut):
        try:
            r = fut.result()
        except Exception as e:  # an error is a result, not something to skip silently
            self.errors.append(repr(e))
            return
        rec["request_id"] = r.runtime.request_id
        rec["match"] = r.answers == self.run.work.refs[(self.shape, req["seed"], self.device)]
        self.records.append(rec)

    def run_closed(self, start_at, stop_at):
        reqs = self.run.work.cycle(self.shape)
        while now() < start_at:
            time.sleep(0.0002)
        while now() < stop_at:
            req = next(reqs)
            rec, fut = self._issue(req, now())
            self._complete(req, rec, fut)

    def run_open(self, start_at, stop_at):
        reqs = self.run.work.cycle(self.shape)
        pending: queue.Queue = queue.Queue()

        def collect():  # one device is FIFO, so completing in submission order never waits long
            while (item := pending.get()) is not None:
                self._complete(*item)

        collector = threading.Thread(target=collect, daemon=True)
        collector.start()
        t = start_at
        while True:
            t += int(self.rng.expovariate(self.rate) * 1e9)
            if t >= stop_at:
                break
            while (n := now()) < t:
                time.sleep(min(0.0005, (t - n) / 1e9))
            req = next(reqs)
            rec, fut = self._issue(req, t)  # arrival = the scheduled time
            pending.put((req, rec, fut))
        pending.put(None)
        collector.join()

    def run_window(self, start_at, stop_at):
        if self.mode == "closed":
            self.run_closed(start_at, stop_at)
        else:
            self.run_open(start_at, stop_at)


class ProductStream:
    """Requests through Laya.submit (the v0.2 router decides the device)."""

    def __init__(self, run, classes, mode: str = "closed", rate: float = 0.0, label: str = "", seed: int = 0):
        self.run, self.classes, self.mode, self.rate = run, classes, mode, rate  # classes: [(shape, weight)]
        self.label = label or "+".join(c for c, _ in classes)
        self.rng = random.Random(seed)
        self.records: list = []
        self.errors: list = []

    def _issue(self, shape, req, arrival):
        fut = self.run.laya.submit(context=req["state"], questions=req["questions"])
        rec = {"arrival": arrival, "shape": shape, "seed": req["seed"]}
        fut.add_done_callback(lambda _f, rec=rec: rec.setdefault("done", now()))  # the collector may lag
        return rec, fut

    def _complete(self, shape, req, rec, fut):
        try:
            r = fut.result()
        except Exception as e:
            self.errors.append(repr(e))
            return
        rec["request_id"] = r.runtime.request_id
        # device and reason also from RuntimeInfo: they are what a trace-off run is compared on
        rec["device"], rec["reason"] = r.runtime.device, r.runtime.routing_reason
        rec["match"] = r.answers == self.run.work.refs[(shape, req["seed"], r.runtime.device)]
        self.records.append(rec)

    def _pick(self, cycles):
        names = [c for c, _ in self.classes]
        shape = self.rng.choices(names, [w for _, w in self.classes])[0] if len(names) > 1 else names[0]
        return shape, next(cycles[shape])

    def run_window(self, start_at, stop_at):
        cycles = {c: self.run.work.cycle(c) for c, _ in self.classes}
        while now() < start_at:
            time.sleep(0.0002)
        if self.mode == "closed":
            while now() < stop_at:
                shape, req = self._pick(cycles)
                rec, fut = self._issue(shape, req, now())
                self._complete(shape, req, rec, fut)
            return
        pending: queue.Queue = queue.Queue()
        done = []

        def collect():  # two devices complete out of order: wait on each in turn, all finish
            while (item := pending.get()) is not None:
                done.append(item)
                item[3].exception()  # block until this one finished
            for item in done:
                self._complete(*item)

        collector = threading.Thread(target=collect, daemon=True)
        collector.start()
        t = start_at
        while True:
            t += int(self.rng.expovariate(self.rate) * 1e9)
            if t >= stop_at:
                break
            while (n := now()) < t:
                time.sleep(min(0.0005, (t - n) / 1e9))
            shape, req = self._pick(cycles)
            rec, fut = self._issue(shape, req, t)
            pending.put((shape, req, rec, fut))
        pending.put(None)
        collector.join()


# ----------------------------------------------------------------------------- raw layout


def columns(records: list) -> dict:
    """List of dicts -> dict of equal-length lists (keys stored once; None where absent)."""
    keys = list(dict.fromkeys(k for r in records for k in r))
    return {k: [r.get(k) for r in records] for k in keys}


def rows(cols: dict) -> list:
    keys = list(cols)
    return [dict(zip(keys, vals)) for vals in zip(*(cols[k] for k in keys))] if keys else []


# ----------------------------------------------------------------------------- run


class Run:
    def __init__(self, args):
        from laya_apple import Laya

        self.args = args
        self.traces: list = []  # laya_apple.RequestTrace, as the runtime emitted them
        trace = self.traces.append if args.trace else None
        self.trace_dir = Path(tempfile.mkdtemp(prefix="laya-trace-"))
        if args.backend_hooks:
            os.environ["LAYA_TRACE_DIR"] = str(self.trace_dir)
            os.environ["PYTHONPATH"] = os.pathsep.join(
                [str(HERE / "hooks")] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
            )
            jobtrace.install_phase_hooks()  # this process: a thread-placed ANE backend
        common = dict(execution="workers", local_files_only=True, trace=trace)
        t = now()
        if args.plan == "device":  # one instance per device: the harness picks the device
            self.lays = {
                "gpu": Laya.from_pretrained(args.model, device="gpu", **common),
                "ane": Laya.from_pretrained(args.model, device="ane", ane_placement=args.ane_placement, **common),
            }
            self.laya = self.lays["ane"]
        else:
            self.laya = Laya.from_pretrained(args.model, device="auto", ane_placement=args.ane_placement, **common)
            if self.laya.ane is None or self.laya.ane_state.unavailable:
                raise SystemExit(f"ANE path unavailable: {self.laya.info()['auto_ane']}")  # never measure a fallback
            self.lays = {"auto": self.laya}
        self.load_s = (now() - t) / 1e9
        if args.backend_hooks:
            for laya in self.lays.values():
                for w in laya._workers.values():
                    if w.placement == "thread":
                        jobtrace.tag_dispatcher(w)
        spec = self.laya.spec
        ane_short, ane_long = spec.ane_buckets[0], max(self.laya.ane.buckets)
        gpu_long = int(self.laya.config.get("max_len", 512))
        self.shapes = {
            "S": (ane_short, 1),  # ANE short: the smallest bucket
            "M": (128, 1),  # the v0.2 Part A short length (typed-decisions, laya)
            "A": (args.part_a_short, 1),  # the Part A short length for this model
            "B": (ane_long, 1),  # ANE long: the largest validated bucket
            "L": (gpu_long, 1),  # GPU long: the model's maximum length
            "Md": (512 if gpu_long > 512 else 256, 1),  # Part B "medium"
            "S4": (args.part_a_short, 4),  # Part B short 4-question
        }
        self.work = Workload(self.laya, args.model, self.shapes, args.seeds)
        self.env = environment(self.laya)
        jobtrace.clear_phases()  # drop the reference instances' forwards
        self.contention_params = None
        if args.scheduler != "baseline":
            self.contention_params = contention.calibrate(args.model, args.ane_placement)
        self.t_ref = now()
        self.windows: list = []
        self.solo: dict = {}
        self._by_id: dict = {}
        self._seen = 0

    def trace_of(self, request_id):
        """The runtime trace of one request (None with --no-trace)."""
        new = self.traces[self._seen :]  # appended by the dispatcher threads; slicing is safe
        self._seen += len(new)
        for t in new:
            self._by_id[t.request_id] = t
        return self._by_id.get(request_id)

    # ------------------------------------------------------------- one window

    def window(self, cell: dict, cycle: int, streams: list, aggressors: list):
        a = self.args
        time.sleep(a.settle)
        cond = conditions()
        for g in aggressors:
            g.start()
        if cell.get("sched") == "contention":
            contention.install(self.contention_params)
        start_at = now() + 300_000_000
        measure_from = start_at + int(a.warmup * 1e9)
        measure_to = measure_from + int(a.seconds * 1e9)
        stop_at = measure_to + int(a.guard_ms * 1e6)
        threads = [threading.Thread(target=s.run_window, args=(start_at, stop_at)) for s in streams]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for g in aggressors:
            g.stop()
        contention.uninstall()
        rel = lambda x: (x - self.t_ref) / 1e9  # noqa: E731
        out = {
            "cell": cell["name"],
            "kind": cell["kind"],
            "cycle": cycle,
            "spec": {k: v for k, v in cell.items() if k not in ("streams", "aggressors")},
            "conditions_before": cond,
            "t_start": rel(start_at),
            "measure": [rel(measure_from), rel(measure_to)],
            "stop": rel(stop_at),
            "aggressors": [g.describe() for g in aggressors],
            "streams": {},
        }
        brief = {}
        for s in streams:
            recs = [self._rel(r, measure_from, measure_to) for r in s.records]
            out["streams"][s.label] = {
                "device": getattr(s, "device", None),
                "shape": getattr(s, "shape", None),
                "mode": s.mode,
                "rate": s.rate,
                "errors": s.errors,
                "records": columns(recs),
            }
            ms = [r for r in s.records if measure_from <= r["arrival"] < measure_to]
            occ = sorted(
                (t.received_ns - t.dispatch_ns) / 1e6 for r in ms if (t := self.trace_of(r["request_id"])) is not None
            )
            e2e = sorted((r["done"] - r["arrival"]) / 1e6 for r in ms)
            if e2e:
                brief[s.label] = (
                    len(ms),
                    round(occ[len(occ) // 2], 2) if occ else None,
                    round(e2e[int(0.99 * (len(e2e) - 1))], 2),
                    sum(not r["match"] for r in ms),
                )
        self.windows.append(out)
        print(f"[{cell['name']} c{cycle}] n, occupancy P50, client e2e P99, mismatches: {brief}", flush=True)
        return out

    def _us(self, ns: int) -> int:
        """A monotonic_ns timestamp as integer µs after the run's reference."""
        return (ns - self.t_ref) // 1000

    def _rel(self, r: dict, lo: int, hi: int) -> dict:
        return {
            "request_id": r["request_id"],
            "arrival_us": self._us(r["arrival"]),
            "done_us": self._us(r["done"]),
            "seed": r["seed"],
            "shape": r["shape"],
            "match": r["match"],
            "in_window": lo <= r["arrival"] < hi,
            **{k: r[k] for k in ("device", "reason") if k in r},
        }

    # ------------------------------------------------------------- plans

    def solo_service_s(self, device: str, shape: str) -> float:
        return self.solo[(device, shape)]

    def record_solo(self, w: dict):
        """Mean device occupancy (dispatch to result) per solo stream, from the runtime trace;
        the client's own latency when tracing is off."""
        for st in w["streams"].values():
            ms = [r for r in rows(st["records"]) if r["in_window"]]
            svc = []
            for r in ms:
                t = self.trace_of(r["request_id"])
                svc.append((t.received_ns - t.dispatch_ns) / 1e9 if t is not None else None)
            if None in svc:
                svc = [(r["done_us"] - r["arrival_us"]) / 1e6 for r in ms]
            key = (st["device"], st["shape"])
            self.solo.setdefault(key, []).append(sum(svc) / len(svc))

    def device_cells(self) -> list:
        a = self.args
        g_short, g_long, a_short, a_long = "M", "L", "S", "B"
        cells = []
        for dev, shp in (("gpu", g_short), ("gpu", g_long), ("ane", a_short), ("ane", a_long)):
            cells.append({"name": f"solo:{dev}_{shp}", "kind": "solo", "streams": [(dev, shp, "closed", 0)]})
        for gs, ans in itertools.product((g_short, g_long), (a_short, a_long)):
            cells.append(
                {
                    "name": f"matrix:gpu_{gs}+ane_{ans}",
                    "kind": "matrix",
                    "streams": [("gpu", gs, "closed", 0), ("ane", ans, "closed", 0)],
                }
            )
        if "sweep" in a.parts:
            for u in a.utils:
                for vic, agg in ((("gpu", g_long), ("ane", a_long)), (("gpu", g_short), ("ane", a_long)),
                                 (("ane", a_long), ("gpu", g_long)), (("ane", a_long), ("gpu", g_short))):
                    cells.append(
                        {
                            "name": f"sweep:{vic[0]}_{vic[1]}|{agg[0]}_{agg[1]}@{u:g}",
                            "kind": "sweep",
                            "util": u,
                            "streams": [(vic[0], vic[1], "closed", 0), (agg[0], agg[1], "poisson", u)],
                        }
                    )
        if "sparse" in a.parts:  # one device alone at low offered load: no interference possible
            for u in a.sparse_utils:
                for dev, shp in (("gpu", g_short), ("gpu", g_long), ("ane", a_long)):
                    cells.append(
                        {
                            "name": f"sparse:{dev}_{shp}@{u:g}",
                            "kind": "sparse",
                            "util": u,
                            "streams": [(dev, shp, "poisson", u)],
                        }
                    )
            for dev, shp in (("gpu", g_short), ("ane", a_long)):  # same, with the CPU kept busy
                cells.append(
                    {
                        "name": f"sparse:{dev}_{shp}@0.1|cpu1",
                        "kind": "sparse",
                        "util": 0.1,
                        "streams": [(dev, shp, "poisson", 0.1)],
                        "aggressors": [("cpu", 1)],
                    }
                )
        if "control" in a.parts:
            for kind, n in (("cpu", a.cpu_burners), ("membw", 1), ("gil", 1)):
                for dev, shp in (("gpu", g_long), ("ane", a_long)):
                    cells.append(
                        {
                            "name": f"control:{dev}_{shp}|{kind}{n}",
                            "kind": "control",
                            "streams": [(dev, shp, "closed", 0)],
                            "aggressors": [(kind, n)],
                        }
                    )
        return cells

    def product_cells(self) -> list:
        a = self.args
        cells = [
            {"name": "product:solo_short", "kind": "product_closed", "pstreams": [(["A"], "closed", 0)]},
            {"name": "product:solo_long", "kind": "product_closed", "pstreams": [(["L"], "closed", 0)]},
            {"name": "product:hetero", "kind": "product_closed", "pstreams": [(["A"], "closed", 0), (["L"], "closed", 0)]},
        ]
        mix = [("A", 0.6), ("Md", 0.2), ("L", 0.1), ("S4", 0.1)]
        heavy = [("A", 0.9), ("L", 0.1)]  # short-heavy: the ANE saturates first, routing matters
        opens = [(f"product:open@{r:g}", mix, r) for r in a.rates]
        opens += [(f"product:heavy@{r:g}", heavy, r) for r in a.heavy_rates]
        for name, m, rate in opens:
            for sched in ("baseline", "contention") if a.scheduler == "both" else (a.scheduler,):
                cells.append({"name": name + ("#contention" if sched == "contention" else ""), "kind": "product_open",
                              "rate": rate, "sched": sched, "pstreams": [(m, "poisson", rate)]})
        if a.only_open:
            cells = [c for c in cells if c["kind"] == "product_open"]
        return cells

    def make_streams(self, cell: dict) -> list:
        streams = []
        for i, (dev, shp, mode, u) in enumerate(cell.get("streams", [])):
            rate = u / self.solo_service_s(dev, shp) if mode == "poisson" else 0.0
            streams.append(DeviceStream(self, dev, shp, mode, rate, seed=17 + i))
        for i, (classes, mode, rate) in enumerate(cell.get("pstreams", [])):
            if isinstance(classes[0], str):
                classes = [(classes[0], 1.0)]
            streams.append(ProductStream(self, classes, mode, rate, seed=29 + i))
        return streams

    def execute(self):
        a = self.args
        plan = self.device_cells() if a.plan == "device" else self.product_cells()
        if a.plan == "device":  # solo first, once, so the sweep rates are known
            solo = [c for c in plan if c["kind"] == "solo"]
            rest = [c for c in plan if c["kind"] != "solo"]
            for cycle in range(a.cycles):
                for c in solo if cycle % 2 == 0 else solo[::-1]:
                    self.record_solo(self.window(c, cycle, self.make_streams(c), []))
            self.solo = {k: sum(v) / len(v) for k, v in self.solo.items()}
            plan = rest
        for cycle in range(a.cycles):
            for c in plan if cycle % 2 == 0 else plan[::-1]:
                aggs = [Aggressor(k, n) for k, n in c.get("aggressors", [])]
                self.window(c, cycle, self.make_streams(c), aggs)

    def finish(self):
        for laya in self.lays.values():
            laya.close()  # workers exit and dump their phase records
        time.sleep(0.5)
        phases = list(jobtrace.phases())  # this process: a thread-placed ANE
        for f in sorted(self.trace_dir.glob("phases-*.json")):
            phases += json.loads(f.read_text())
        shutil.rmtree(self.trace_dir, ignore_errors=True)
        us = self._us
        backend = [
            {
                "request_id": p["request_id"],
                "device": p["device"],
                "pid": p["pid"],
                "tid": p["tid"],
                "rows": p["rows"],
                "max_len": p["max_len"],
                "t0_us": us(p["t0"]),
                "t1_us": us(p["t1"]),
                "cpu_us": p["cpu_ns"] // 1000,
                "spans": [[us(x), us(y)] for x, y in p["device_spans"]],
            }
            for p in sorted(phases, key=lambda p: p["t0"])
        ]
        traces = []
        for t in sorted(self.traces, key=lambda t: t.submit_ns):
            row = {k: v for k, v in t.to_dict().items() if k not in ("gpu", "ane") and not k.endswith("_ns")}
            for dev in ("gpu", "ane"):
                snap = getattr(t, dev)
                row[f"{dev}_backlog_ms"] = snap.backlog_ms if snap else None
                row[f"{dev}_queued_jobs"] = snap.queued_jobs if snap else None
                row[f"{dev}_running"] = snap.running if snap else None
            for k in ("submit", "prepared", "routed", "queue_enter", "dispatch", "service_start", "service_end",
                      "received", "response"):
                row[k + "_us"] = us(getattr(t, k + "_ns"))
            traces.append(row)
        return {
            "experiment": "gpu-ane-interference",
            "pipeline": "runtime-trace",
            "plan": self.args.plan,
            "args": {k: v for k, v in vars(self.args).items()},
            "time": datetime.now(timezone.utc).isoformat(),
            "environment": self.env,
            "load_s": self.load_s,
            "shapes": {k: {"length": v[0], "questions": v[1]} for k, v in self.shapes.items()},
            "service_model": {"gpu": self.laya._service.gpu, "ane": self.laya._service.ane},
            "contention_params": self.contention_params,
            "solo_mean_service_s": {f"{d}_{s}": v for (d, s), v in self.solo.items()},
            "t_ref_ns": self.t_ref,
            "windows": self.windows,
            "traces": columns(traces),
            "backend": columns(backend),
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="laya-typed-decisions")
    ap.add_argument("--ane-placement", choices=["thread", "process"], required=True)
    ap.add_argument("--plan", choices=["device", "product"], default="device")
    ap.add_argument("--parts", nargs="*", default=["sweep", "control"], help="device plan: extra cell groups")
    ap.add_argument("--utils", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--sparse-utils", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5])
    ap.add_argument("--rates", type=float, nargs="+", default=[15, 25, 35])
    ap.add_argument("--heavy-rates", type=float, nargs="*", default=[], help="product plan: short-heavy mix rates")
    ap.add_argument("--scheduler", choices=["baseline", "contention", "both"], default="baseline",
                    help="product plan: v0.2 decide_queued, the contention.py prototype, or both interleaved")
    ap.add_argument("--only-open", action="store_true", help="product plan: open-loop cells only")
    ap.add_argument("--part-a-short", type=int, default=128, help="Part A short length (v0.2: 96 for multilingual)")
    ap.add_argument("--cpu-burners", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--guard-ms", type=float, default=400.0)
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None, help="run only cells whose name starts with one of these")
    ap.add_argument("--no-trace", dest="trace", action="store_false", help="trace=None (runtime tracing off)")
    ap.add_argument("--no-backend-hooks", dest="backend_hooks", action="store_false",
                    help="no jobtrace.py phase hooks (research instrumentation off)")
    ap.add_argument("--cells", nargs="*", default=None,
                    help="device plan: run only these cells, by exact name (solo cells they need are kept)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = Run(args)
    if args.cells:
        keep_cells = set(args.cells)
        orig_cells = run.device_cells
        run.device_cells = lambda: [c for c in orig_cells() if c["name"] in keep_cells]
    if args.only:
        keep = tuple(args.only)
        for name in ("device_cells", "product_cells"):
            orig = getattr(run, name)
            setattr(run, name, lambda orig=orig: [c for c in orig() if c["kind"] == "solo" or c["name"].startswith(keep)])
    try:
        run.execute()
    finally:
        record = run.finish()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if out.suffix == ".gz" else open
        with opener(out, "wt") as f:
            json.dump(record, f, separators=(",", ":"))
        print("wrote", out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
