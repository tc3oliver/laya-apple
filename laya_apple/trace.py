"""Opt-in request lifecycle tracing for execution="workers".

    def on_trace(trace):
        print(trace.request_id, trace.target, trace.routing_reason, trace.e2e_ms)

    laya = Laya.from_pretrained(model_id, execution="workers", trace=on_trace)

Every request that completes produces exactly one immutable `RequestTrace`. It is passed to
the callback once the answers are formatted and before the request's Future resolves, on the
thread that resolves it (the target device's dispatcher thread): a slow callback delays that
device's next job, so keep it cheap (append to a list or a queue). An exception raised by the
callback is reported once as a RuntimeWarning and never fails the request. Failed requests
produce no trace, and neither do requests with no questions, which never reach a device.

Timestamps are `time.monotonic_ns()`. On macOS that clock is system-wide, so the worker
process's service timestamps share the parent's time axis. Every duration is derived from
the raw timestamps; none is stored twice.

With trace=None (the default) nothing here is created: no RequestTrace, no extra message, no
extra lock. The queue snapshots below are what the router reads either way.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import NamedTuple


class QueueSnapshot(NamedTuple):
    """One device's queue state, read under that device's lock in one step.

    backlog_ms: service estimates of the queued jobs plus the running job's remaining estimate
    (the router's input). queued_jobs: jobs waiting, not counting the running one.
    running: the dispatcher holds a job it has not finished.
    """

    backlog_ms: float
    queued_jobs: int
    running: bool


@dataclass(frozen=True, slots=True)
class RequestTrace:
    """The lifecycle of one completed request (execution="workers").

    `request_id` is unique within the process, across Laya instances. It is also
    `RuntimeInfo.request_id` and the job id on the worker protocol, so anything recorded per
    request, in this process or a worker, joins on it.

    `gpu` / `ane` are the exact snapshot objects the router decided on (None: no such worker,
    or not ready yet; the router then counts a backlog of 0). `service_estimate_ms` is the
    estimate charged to the target queue for this request, the same number later requests see
    in that queue's backlog.

    Lifecycle, in order (all `time.monotonic_ns()`):
      submit_ns         Laya.submit accepted the request (after argument checks)
      prepared_ns       tokenised and laid out (Laya.prepare)
      routed_ns         the routing decision is made
      queue_enter_ns    the job entered the target device's FIFO queue
      dispatch_ns       the device's dispatcher took the job from the queue
      service_start_ns  the backend's forward started, in the process that runs it
      service_end_ns    the backend's forward returned, in that process
      received_ns       the dispatcher resolved the device job with the result (after the
                        reply IPC, if any)
      response_ns       answers formatted; the Future is about to resolve
    """

    request_id: int
    sequence_length: int
    question_count: int
    target: str  # "gpu" | "ane"
    routing_reason: str  # routing.REASONS / scheduling.QUEUE_REASONS
    service_estimate_ms: float
    gpu: QueueSnapshot | None
    ane: QueueSnapshot | None
    submit_ns: int
    prepared_ns: int
    routed_ns: int
    queue_enter_ns: int
    dispatch_ns: int
    service_start_ns: int
    service_end_ns: int
    received_ns: int
    response_ns: int
    # target "ane": the Core ML predict binding that served it ("nogil" | "coremltools"),
    # as RuntimeInfo.ane_predict. None for the GPU.
    ane_predict: str | None = None

    # --------------------------------------------------------------- router inputs, flat

    @property
    def gpu_backlog_ms(self) -> float:
        return self.gpu.backlog_ms if self.gpu else 0.0

    @property
    def ane_backlog_ms(self) -> float:
        return self.ane.backlog_ms if self.ane else 0.0

    @property
    def gpu_queue_depth(self) -> int:
        return self.gpu.queued_jobs if self.gpu else 0

    @property
    def ane_queue_depth(self) -> int:
        return self.ane.queued_jobs if self.ane else 0

    @property
    def gpu_running(self) -> bool:
        return bool(self.gpu and self.gpu.running)

    @property
    def ane_running(self) -> bool:
        return bool(self.ane and self.ane.running)

    # --------------------------------------------------------------- durations (ms)

    @property
    def prepare_ms(self) -> float:
        return (self.prepared_ns - self.submit_ns) / 1e6

    @property
    def routing_ms(self) -> float:
        return (self.routed_ns - self.prepared_ns) / 1e6

    @property
    def enqueue_ms(self) -> float:
        """Routed to queued: the ANE shape check and the service estimate."""
        return (self.queue_enter_ns - self.routed_ns) / 1e6

    @property
    def queue_ms(self) -> float:
        """Waiting in the FIFO; the same interval as RuntimeInfo.queue_wait_ms."""
        return (self.dispatch_ns - self.queue_enter_ns) / 1e6

    @property
    def dispatch_ms(self) -> float:
        """Dispatcher to forward start: the request IPC and the worker's receive (process
        placement); the call overhead only (thread placement)."""
        return (self.service_start_ns - self.dispatch_ns) / 1e6

    @property
    def service_ms(self) -> float:
        """The backend's forward; the same interval as RuntimeInfo.device_ms."""
        return (self.service_end_ns - self.service_start_ns) / 1e6

    @property
    def return_ms(self) -> float:
        """Forward end to the dispatcher holding the result (the reply IPC, if any)."""
        return (self.received_ns - self.service_end_ns) / 1e6

    @property
    def postprocess_ms(self) -> float:
        """Result received to response: answer formatting and RuntimeInfo."""
        return (self.response_ns - self.received_ns) / 1e6

    @property
    def e2e_ms(self) -> float:
        """Submit to response; the same interval as RuntimeInfo.latency_ms."""
        return (self.response_ns - self.submit_ns) / 1e6

    def to_dict(self) -> dict:
        """The raw fields as plain JSON-compatible values (snapshots as dicts or None)."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = v._asdict() if isinstance(v, QueueSnapshot) else v
        return out


TraceCallback = Callable[[RequestTrace], None]
