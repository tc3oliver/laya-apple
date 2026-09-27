"""The candidate: split one multi-question request across the ANE and the GPU (research only).

`plan_split` is the preregistered policy (criteria.json, "policy"), a pure function:

- A row is ANE-eligible if some loaded ANE bucket fits it (length <= bucket, options <= 32).
- The ANE takes the k longest eligible rows (removing them shrinks the GPU batch's padded
  length most), one after another at B=1; the GPU batches the rest.
- For every k in 0..eligible, the predicted makespan is
      max(ane_backlog + sum(ane_ms(bucket(row)) for the ANE rows),
          gpu_backlog + gpu_ms(longest GPU row, GPU row count))
  with the runtime's own service model (laya_apple.scheduling.ServiceModel, i.e. the shipped
  routing.json forward P50s) and the device backlogs read once from the workers' queues. A side
  with no rows costs 0 (its backlog does not delay this request).
- The smallest makespan wins; ties go to the smaller k (the GPU, today's behaviour).
- Only requests with 2 or more questions are split. A 1-question request is never touched.

`SplitSubmitter` runs a plan on a live `Laya(device="auto", execution="workers")` instance: one
job on the ANE worker (its k rows) and one on the GPU worker (the rest), queued at the same time
on the product's own FIFOs, then the product's `format_answers`. It never falls back: a device
error fails the request.

Nothing here is imported by `laya_apple`.
"""

from __future__ import annotations

import itertools
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass

import numpy as np

ANE_MAX_OPTIONS = 32  # laya_apple.registry.ANE_MAX_OPTIONS; checked against it at run time
NEG = -1e4  # the padding value the model uses for absent options


@dataclass(frozen=True)
class Plan:
    ane_rows: tuple  # row indices, in the order the ANE runs them
    gpu_rows: tuple
    buckets: tuple  # ANE bucket per ane_rows entry
    makespan_ms: float  # predicted
    gpu_only_ms: float  # predicted makespan of k = 0 (today's path)
    gpu_backlog_ms: float
    ane_backlog_ms: float

    @property
    def k(self) -> int:
        return len(self.ane_rows)


def bucket_for(length: int, buckets) -> int | None:
    return next((b for b in sorted(buckets) if length <= b), None)


def plan_split(
    lengths,
    options,
    buckets,
    gpu_ms,
    ane_ms,
    *,
    gpu_backlog_ms: float = 0.0,
    ane_backlog_ms: float = 0.0,
) -> Plan:
    """lengths/options: per row; buckets: loaded ANE buckets; gpu_ms(length, n) and ane_ms(bucket)
    are the service estimates (ms)."""
    n = len(lengths)
    if n != len(options):
        raise ValueError("lengths and options differ in length")
    eligible = []
    for r in range(n):
        b = bucket_for(lengths[r], buckets)
        if b is not None and options[r] <= ANE_MAX_OPTIONS:
            eligible.append((r, b))
    eligible.sort(key=lambda rb: (-lengths[rb[0]], rb[0]))  # longest first; stable by row index
    all_rows = list(range(n))

    def gpu_cost(rows) -> float:
        return gpu_backlog_ms + gpu_ms(max(lengths[r] for r in rows), len(rows)) if rows else 0.0

    gpu_only = gpu_cost(all_rows)
    best = None
    max_k = len(eligible) if n >= 2 else 0
    for k in range(max_k + 1):
        ane = eligible[:k]
        taken = {r for r, _ in ane}
        gpu_rows = [r for r in all_rows if r not in taken]
        t_ane = ane_backlog_ms + sum(ane_ms(b) for _, b in ane) if ane else 0.0
        span = max(t_ane, gpu_cost(gpu_rows))
        if best is None or span < best[0]:  # strict: ties keep the smaller k
            best = (span, ane, gpu_rows)
    span, ane, gpu_rows = best
    return Plan(
        ane_rows=tuple(r for r, _ in ane),
        gpu_rows=tuple(gpu_rows),
        buckets=tuple(b for _, b in ane),
        makespan_ms=float(span),
        gpu_only_ms=float(gpu_only),
        gpu_backlog_ms=float(gpu_backlog_ms),
        ane_backlog_ms=float(ane_backlog_ms),
    )


def merge_outputs(n: int, parts) -> tuple[np.ndarray, np.ndarray]:
    """parts: [(rows, logits, act)]; returns (logits [n, width] padded with NEG, act [n, 2])."""
    width = max(max(2, lg.shape[1]) for _, lg, _ in parts)
    logits = np.full((n, width), NEG, np.float32)
    act = np.zeros((n, 2), np.float32)
    seen = set()
    for rows, lg, ac in parts:
        if len(rows) != lg.shape[0] or len(rows) != ac.shape[0]:
            raise ValueError("a device returned a different number of rows than it was given")
        for j, r in enumerate(rows):
            logits[r, : lg.shape[1]] = lg[j]
            act[r] = ac[j]
            seen.add(r)
    if seen != set(range(n)):
        raise ValueError("some rows have no output")
    return logits, act


@dataclass
class Outcome:
    """What one request did, beside its answers."""

    answers: dict
    plan: Plan | None  # None: the request went through Laya.submit unchanged (1 question)
    t0_ns: int
    done_ns: int
    devices: dict  # device -> {"rows", "queue_enter_ns", "dispatch_ns", "start_ns", "end_ns"}
    handoff_path: str | None = None

    @property
    def latency_ms(self) -> float:
        return (self.done_ns - self.t0_ns) / 1e6


class SplitSubmitter:
    """Runs plans on one Laya(device="auto", execution="workers") instance.

    mode "split": the candidate. mode "gpu": all rows as one GPU job (what Laya.submit does for a
    multi-question request). mode "ane": all rows as one ANE job (every row must fit a bucket).
    Returns a Future resolving to an Outcome. The request's clock starts before tokenisation, as
    Laya.submit's does."""

    _ids = itertools.count(1 << 40)  # job ids on the worker protocol, disjoint from Laya's

    def __init__(self, laya, *, buckets=None):
        from laya_apple.prompt import format_answers
        from laya_apple.registry import ANE_MAX_OPTIONS as product_max

        if product_max != ANE_MAX_OPTIONS:
            raise RuntimeError(f"ANE_MAX_OPTIONS changed ({product_max}); update split.py")
        if laya.execution != "workers" or laya.device != "auto":
            raise ValueError('the split needs Laya(device="auto", execution="workers")')
        self.laya = laya
        self.gpu = laya._workers.get("gpu")
        self.ane = laya._workers.get("ane")
        if self.gpu is None or self.ane is None or laya.ane is None:
            raise RuntimeError("both device workers must be running (is the ANE ready on this machine?)")
        loaded = tuple(laya.ane.buckets)
        self.buckets = tuple(sorted(buckets)) if buckets is not None else loaded
        missing = set(self.buckets) - set(loaded)
        if missing:
            raise RuntimeError(f"ANE buckets {sorted(missing)} are not loaded (loaded: {list(loaded)})")
        service = laya._service
        for b in self.buckets:
            service.ane_ms(b)  # KeyError now, not mid-run, if the service model lacks a bucket
        self._format = format_answers
        self._lock = threading.Lock()  # one plan at a time: backlogs read and jobs queued together

    def _estimate_gpu(self, length: int, n: int) -> float:
        return self.laya._service.gpu_ms(length, n)

    def _estimate_ane(self, b: int) -> float:
        return self.laya._service.ane_ms(b)

    def submit(self, context, questions, mode: str = "split") -> Future:
        t0 = time.monotonic_ns()
        prep = self.laya.prepare(context, questions)
        items = prep.items
        n = len(items)
        if n == 0:
            raise ValueError("no questions")
        prepared_ms = (time.monotonic_ns() - t0) / 1e6  # Laya.submit's prepare time (the breaker's signal)
        lengths = [len(it["ids"]) for it in items]
        options = [len(it["markers"]) for it in items]
        with self._lock:
            gq, aq = self.gpu.read_queue(), self.ane.read_queue()
            gb, ab = (gq[0] if gq else 0.0), (aq[0] if aq else 0.0)
            if mode == "split":
                plan = plan_split(
                    lengths,
                    options,
                    self.buckets,
                    self._estimate_gpu,
                    self._estimate_ane,
                    gpu_backlog_ms=gb,
                    ane_backlog_ms=ab,
                )
            elif mode == "gpu":
                plan = Plan((), tuple(range(n)), (), 0.0, 0.0, gb, ab)
            elif mode == "ane":
                bs = [bucket_for(x, self.buckets) for x in lengths]
                if any(b is None for b in bs) or any(o > ANE_MAX_OPTIONS for o in options):
                    raise ValueError("a row does not fit any loaded ANE bucket")
                plan = Plan(tuple(range(n)), (), tuple(bs), 0.0, 0.0, gb, ab)
            else:
                raise ValueError(f"unknown mode {mode!r}")
            if plan.ane_rows:
                # the product's own eligibility check, before anything is queued
                got = tuple(self.laya.ane.check([items[r] for r in plan.ane_rows]))
                if got != plan.buckets:
                    raise RuntimeError(f"bucket mismatch: plan {plan.buckets}, runtime {got}")
            out: Future = Future()
            planned = []
            if plan.gpu_rows:
                est = self._estimate_gpu(max(lengths[r] for r in plan.gpu_rows), len(plan.gpu_rows))
                planned.append(("gpu", plan.gpu_rows, est, self.gpu))
            if plan.ane_rows:
                est = sum(self._estimate_ane(b) for b in plan.buckets)
                planned.append(("ane", plan.ane_rows, est, self.ane))
            handoff = self.laya._handoff if self.laya._handoff is not None and self.laya._handoff.enabled else None
            results: dict = {}
            state = {"pending": len(planned)}
            lock = threading.Lock()

            def complete():
                try:
                    parts, devices = [], {}
                    for dev, (rows, res) in results.items():
                        if isinstance(res, BaseException):
                            raise res
                        lg, ac, enq, disp, start, end = res
                        parts.append((rows, lg, ac))
                        devices[dev] = {
                            "rows": list(rows),
                            "queue_enter_ns": enq,
                            "dispatch_ns": disp,
                            "start_ns": start,
                            "end_ns": end,
                        }
                    logits, act = merge_outputs(n, parts)
                    answers = self._format(prep, logits, act, self.laya.calibration)
                    oc = Outcome(answers, plan, t0, time.monotonic_ns(), devices)
                    if not out.cancelled():
                        out.set_result(oc)
                except BaseException as e:
                    if not out.cancelled():
                        out.set_exception(e)

            def done(dev, rows, f):
                # Runs on the device's dispatcher thread before it takes its next job (the callback
                # is passed to submit, as Laya.submit does).
                try:
                    res = f.result()
                except BaseException as e:
                    res = e
                if dev == "ane" and handoff is not None and not isinstance(res, BaseException):
                    try:  # as Laya.submit feeds the breaker for an ANE request
                        handoff.observe(prepared_ms)
                    except Exception:
                        pass
                with lock:
                    results[dev] = (rows, res)
                    state["pending"] -= 1
                    last = state["pending"] == 0
                if last:
                    complete()

            for dev, rows, est, worker in planned:
                worker.submit(
                    [items[r] for r in rows],
                    est,
                    next(self._ids),
                    callback=lambda f, dev=dev, rows=rows: done(dev, rows, f),
                )
        return out
