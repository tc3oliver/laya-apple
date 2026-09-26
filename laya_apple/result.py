"""Backend-normalised result and per-call runtime diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class RuntimeInfo:
    """Everything needed to explain how one request was executed."""

    backend: str  # "mlx" | "coreml" | "none" (questions={}: nothing ran)
    device: str  # "gpu" | "ane" | "none"
    model: str
    model_revision: str
    sequence_length: int  # longest prompt row, tokens
    question_count: int
    routing_reason: str
    artifact_revision: str  # ANE: artifact tree hash; MLX: "mlx:<weights sha256[:12]>"
    latency_ms: float
    compute_units: str | None = None  # Core ML only
    buckets: tuple = ()  # Core ML only: fixed length used per question
    dtype: str | None = None
    execution: str = "inline"  # "inline" | "workers"
    queue_wait_ms: float | None = None  # workers: time queued before the device started it
    device_ms: float | None = None  # workers: time inside the backend's forward, in the worker
    gpu_backlog_ms: float | None = None  # workers: backlog estimates seen by the router
    ane_backlog_ms: float | None = None
    request_id: int | None = None  # this instance's id for the request; RequestTrace.request_id
    truncated: bool = False  # the state was cut to fit the model's max_len (as upstream does)
    # Language routing (Laya.from_pretrained("auto"), laya_apple/router.py): why this checkpoint
    # was chosen, e.g. "language_english"; None when the caller named the checkpoint.
    model_routing: str | None = None

    def __str__(self) -> str:
        return (
            f"backend={self.backend} device={self.device} reason={self.routing_reason} "
            f"model={self.model} L={self.sequence_length} q={self.question_count} "
            f"latency_ms={self.latency_ms:.2f}"
        )


@dataclass(frozen=True)
class Result:
    answers: dict
    usage: dict
    runtime: RuntimeInfo
    model: str = "laya-rl-agent"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Upstream-compatible dict (model/answers/usage) plus a `runtime` block. Entries of
        `extra` (such as the language router's `routing`) are added at the top level, as
        upstream adds them to its result dict; they never replace a key above."""
        out = {"model": self.model, "answers": self.answers, "usage": self.usage}
        for key, value in self.extra.items():
            out.setdefault(key, value)
        out["runtime"] = asdict(self.runtime)
        out["runtime"]["buckets"] = list(self.runtime.buckets)
        return out
