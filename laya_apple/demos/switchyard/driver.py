"""Open-loop rounds through one `Laya(execution="workers")` per configuration.

Modelled on part B of scripts/bench_concurrency.py: one submitter thread waits for each
scheduled arrival with the same short-sleep loop and calls `Laya.submit` (which tokenises
inline, as the product does: that time is reported as submit lag); nothing waits for a
response before the next arrival. The trace callback only puts the
RequestTrace on a SimpleQueue; after the drain every trace is joined to its Result on
`request_id`.

All timestamps are `time.monotonic_ns()`, the RequestTrace axis:
  arrival_ns = round_start_ns + scheduled offset
  decision latency = response_ns - arrival_ns (submit lag and queueing included)

Round order (docs/switchyard.md): an even seed runs gpu_only first, an odd seed hybrid
first. One Laya is closed before the next is loaded, so the configurations never share the
machine. A GPU + ANE round that fails, or whose trains stopped reaching the ANE, never fails
the run: the ANE is recorded as unavailable with the reason and the GPU-only round stands.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from collections import Counter
from concurrent.futures import wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, SimpleQueue

from . import world
from .metrics import outcome

GPU_ONLY, HYBRID = "gpu_only", "hybrid"
CONFIGS = {GPU_ONLY: {"device": "gpu"}, HYBRID: {"device": "auto"}}
LEAD_S = 0.5  # the first arrival is this long after the round is armed
# Wait in short sleeps, as scripts/bench_concurrency.part_b does. A Python spin loop would
# hold the GIL against the thread-placed ANE dispatcher in this process and slow it down.
SLEEP_S = 0.0002


class RoundError(RuntimeError):
    """A round (or its warmup) did not complete: a request failed, timed out or left no trace."""


def round_order(seed: int, ane_ready: bool) -> list[str]:
    order = [GPU_ONLY, HYBRID] if seed % 2 == 0 else [HYBRID, GPU_ONLY]
    return order if ane_ready else [GPU_ONLY]


# ----------------------------------------------------------------------------- conditions


def _sh(*cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10, env=dict(os.environ, LC_ALL="C")).stdout
    except Exception:
        return ""


def _is_laya(argv: list[str]) -> bool:
    """argv[0]/argv[1] run laya-apple: the console script, `python -m laya_apple…`, or one of
    the repo's benchmark scripts. Arguments further along (a path, say) never match."""
    names = [Path(a).name for a in argv[:2]]
    if "laya-apple" in names:
        return True
    if len(argv) > 2 and argv[1] == "-m" and argv[2].split(".")[0] == "laya_apple":
        return True
    return any(re.fullmatch(r"(.*/)?scripts/bench_\w+\.py", a) for a in argv[:2])


def other_laya_processes() -> list[dict]:
    """Other processes running laya-apple, excluding this process, its ancestors (the shell
    or `uv run` that started it) and its workers. Only the pid and the executable's name are
    recorded: command lines can carry anything."""
    me = os.getpid()
    procs = {}
    for line in _sh("ps", "-Ao", "pid=,ppid=,command=").splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            procs[int(parts[0])] = (int(parts[1]), parts[2])
    mine, pid = {me}, me
    while pid in procs and procs[pid][0] not in mine and procs[pid][0] > 1:
        pid = procs[pid][0]
        mine.add(pid)
    return [
        {"pid": pid, "name": Path(command.split()[0]).name}
        for pid, (ppid, command) in sorted(procs.items())
        if pid not in mine and ppid != me and _is_laya(command.split())
    ]


def top_cpu(n: int = 5) -> list[list]:
    """The n busiest processes as [%cpu, executable name], as scripts/bench_concurrency does."""
    rows = []
    for line in _sh("ps", "-Ao", "%cpu=,comm=").splitlines():
        parts = line.split(None, 1)
        try:
            rows.append([float(parts[0]), Path(parts[1].strip()).name])
        except (ValueError, IndexError):
            continue
    return sorted(rows, key=lambda r: -r[0])[:n]


def conditions() -> dict:
    """Machine state at one instant: load average, the busiest processes and the
    thermal/performance notes."""
    therm = [x.strip() for x in _sh("pmset", "-g", "therm").splitlines() if x.strip()]
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "loadavg": list(os.getloadavg()),
        "top_cpu": top_cpu(),
        "pmset_therm": therm,
    }


# ----------------------------------------------------------------------------- requests


def payloads(items: list[world.Arrival], tokenizer, config) -> list[tuple]:
    """(context, questions) per arrival, built before the round. Laya.submit still tokenises
    each one on the submitter thread, as the product does; that time is part of submit lag."""
    from ...workload import make_request

    bg = {cls: make_request(tokenizer, config, n, q, seed) for cls, (n, q, seed) in world.BACKGROUND.items()}
    return [world.train_prompt(a.train) if a.train else bg[a.cls] for a in items]


def _wait_until(target_ns: int) -> None:
    while time.monotonic_ns() < target_ns:
        time.sleep(SLEEP_S)


def open_loop(laya, items, reqs, *, timeout_s: float = world.DRAIN_TIMEOUT_S):
    """Submit every arrival on schedule, then wait for all. Returns (round_start_ns, futures)."""
    round_start_ns = time.monotonic_ns() + int(LEAD_S * 1e9)
    futures = []
    for a, (context, questions) in zip(items, reqs):
        _wait_until(round_start_ns + round(a.offset_s * 1e9))
        futures.append(laya.submit(context=context, questions=questions))
    wait(futures, timeout=timeout_s)
    return round_start_ns, futures


def drain(traces: SimpleQueue) -> dict:
    out = {}
    while True:
        try:
            t = traces.get_nowait()
        except Empty:
            return out
        out[t.request_id] = t


def join(items, futures, traces: dict, round_start_ns: int, round_index: int, deadline_ms: float) -> list[dict]:
    """One trace.jsonl record per request, in schedule order. Raises RoundError on any gap."""
    records = []
    for a, fut in zip(items, futures):
        if not fut.done():
            raise RoundError(f"request {a.index} ({a.cls}) did not finish within the drain timeout")
        if fut.exception() is not None:
            e = fut.exception()
            raise RoundError(f"request {a.index} ({a.cls}) failed: {type(e).__name__}: {e}") from e
        result = fut.result()
        trace = traces.get(result.runtime.request_id)
        if trace is None:
            raise RoundError(f"request {a.index} (id {result.runtime.request_id}) has no trace")
        arrival_ns = round_start_ns + round(a.offset_s * 1e9)
        rec = trace.to_dict() | {"round": round_index, "index": a.index, "class": a.cls, "arrival_ns": arrival_ns}
        if a.train is not None:
            answer = result.answers[world.QUESTION_ID]
            latency_ms = (trace.response_ns - arrival_ns) / 1e6
            rec |= {
                "line": a.train.line,
                "train_id": a.train.train_id,
                "pattern": a.train.pattern,
                "answer": answer["choice"],
                "probabilities": answer["probabilities"],
                "oracle": a.train.oracle,
                "outcome": outcome(latency_ms, answer["choice"], a.train.oracle, deadline_ms),
            }
        records.append(rec)
    return records


# ----------------------------------------------------------------------------- rounds


@dataclass
class Round:
    index: int
    config: str
    started_utc: str
    round_start_ns: int
    records: list
    conditions: dict = field(default_factory=dict)
    ane_verified: bool | None = None  # hybrid: its trains really ran on the ANE; gpu_only: None


def warmup(laya, traces: SimpleQueue, items, reqs) -> Counter:
    """Run the warmup schedule open-loop; return the device count of its trains."""
    _, futures = open_loop(laya, items, reqs)
    for a, f in zip(items, futures):
        if not f.done():
            raise RoundError(f"warmup request {a.index} ({a.cls}) did not finish within the drain timeout")
        if f.exception() is not None:
            e = f.exception()
            raise RoundError(f"warmup request {a.index} ({a.cls}) failed: {type(e).__name__}: {e}") from e
    seen = drain(traces)
    by_id = {f.result().runtime.request_id: a for a, f in zip(items, futures)}
    return Counter(t.target for rid, t in seen.items() if rid in by_id and by_id[rid].cls == world.TRAIN)


def run_round(laya, traces: SimpleQueue, index: int, config: str, items, reqs, deadline_ms: float) -> Round:
    drain(traces)  # nothing from an earlier window joins this one
    start = conditions()
    started_utc = datetime.now(timezone.utc).isoformat()
    round_start_ns, futures = open_loop(laya, items, reqs)
    records = join(items, futures, drain(traces), round_start_ns, index, deadline_ms)
    return Round(
        index=index,
        config=config,
        started_utc=started_utc,
        round_start_ns=round_start_ns,
        records=records,
        conditions={"start": start, "end": conditions(), "other_laya_processes": other_laya_processes()},
    )


def load(config: str, traces: SimpleQueue, *, local_files_only: bool = False):
    from ...model import Laya

    return Laya.from_pretrained(
        world.MODEL,
        device=CONFIGS[config]["device"],
        execution="workers",
        local_files_only=local_files_only,
        trace=traces.put,
    )


def soc_slug(soc: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (soc or "unknown-soc").lower()).strip("-") or "unknown-soc"


def ane_lost(records: list[dict]) -> str | None:
    """Why a GPU + ANE round did not really use the ANE, or None: no train ran there, or the
    router reported the ANE runtime gone (a dead worker) for any train."""
    from ...routing import RUNTIME_UNAVAILABLE

    trains = [r for r in records if r["class"] == world.TRAIN]
    on_ane = sum(r["target"] == "ane" for r in trains)
    lost = sum(r["routing_reason"] == RUNTIME_UNAVAILABLE for r in trains)
    if on_ane and not lost:
        return None
    return f"{on_ane} of {len(trains)} trains ran on the ANE; {lost} were routed {RUNTIME_UNAVAILABLE}"


@dataclass
class Run:
    seed: int
    duration_s: float
    schedule: list
    warmup_schedule: list
    ane: object  # ane_state.AneStatus, final (after the hybrid warmup and round)
    rounds: list = field(default_factory=list)
    configs: dict = field(default_factory=dict)  # label -> {device, execution, ane_placement}
    ane_artifacts: dict = field(default_factory=dict)
    error: str | None = None  # the GPU-only round failed: what happened (completed rounds are kept)


def _config_round(out: Run, config: str, order: list, cache: dict, local_files_only: bool, step):
    """Load, warm up and run one configuration's round; always closes its Laya.
    Returns the Round, or None when the warmup shows the ANE is not used."""
    from .ane_state import READY, after_warmup

    traces: SimpleQueue = SimpleQueue()
    step("loading", config)
    laya = load(config, traces, local_files_only=local_files_only)
    try:
        if "reqs" not in cache:
            cache["reqs"] = payloads(out.schedule, laya.tokenizer, laya.config)
            cache["warm"] = payloads(out.warmup_schedule, laya.tokenizer, laya.config)
        step("warmup", config)
        devices = warmup(laya, traces, out.warmup_schedule, cache["warm"])
        if config == HYBRID:
            out.ane = after_warmup(out.ane, devices.get("ane", 0))
            if out.ane.state != READY:
                step("ane_skipped", out.ane)
                return None
            if laya.ane is not None:
                out.ane_artifacts = {str(b): h for b, h in sorted(laya.ane.artifact_sha256.items())}
        info = laya.info()
        out.configs[config] = {
            "device": info["device"],
            "execution": info["execution"],
            "ane_placement": info["ane_placement"] if config == HYBRID else None,
        }
        total = len(out.rounds) + len(order) - order.index(config)
        step("round", len(out.rounds) + 1, total, config, out.duration_s)
        return run_round(laya, traces, len(out.rounds), config, out.schedule, cache["reqs"], world.DEADLINE_MS)
    finally:
        laya.close()


def run(seed: int, duration_s: float, ane, *, local_files_only: bool = False, step=lambda *args: None) -> Run:
    """Every round of one benchmark. `ane` is the pre-warmup ane_state.AneStatus.

    `step(kind, *details)` reports progress: ("loading", config), ("warmup", config),
    ("round", number, total, config, duration_s), ("round_done", Round), ("ane_skipped", status)
    and ("ane_failed", status) when the GPU + ANE round is dropped or not verified,
    ("round_failed", message) when the GPU-only round fails (the run stops; `Run.error`).
    """
    from dataclasses import replace

    from .ane_state import ANE_FAILED, ANE_LOST, READY, UNAVAILABLE, AneStatus

    out = Run(seed, duration_s, world.schedule(seed, duration_s), world.warmup_schedule(seed), ane)
    out.ane_artifacts = dict(ane.artifacts)
    order = round_order(seed, ane.state == READY)
    cache: dict = {}
    for config in order:
        try:
            rnd = _config_round(out, config, order, cache, local_files_only, step)
        except Exception as e:
            detail = f"{type(e).__name__}: {e}"
            out.configs.pop(config, None)  # a configuration without a finished round is not reported
            if config != HYBRID:
                out.error = detail
                step("round_failed", detail)
                break
            out.ane = AneStatus(UNAVAILABLE, ANE_FAILED, detail=detail, artifacts=out.ane.artifacts)
            step("ane_failed", out.ane)
            continue
        if rnd is None:
            continue
        if config == HYBRID:
            why = ane_lost(rnd.records)
            rnd.ane_verified = why is None
            if why is not None:
                out.ane = replace(out.ane, state=UNAVAILABLE, reason=ANE_LOST, detail=why)
                step("ane_failed", out.ane)
        out.rounds.append(rnd)
        step("round_done", rnd)
    return out
