"""Core ML / Neural Engine backend: fixed-shape BC1S artifacts, pinned to CPU_AND_NE.

Host runtime (embedding gather, additive masks, marker selector, FP32 action head)
adapted from mizorewww/laya-coreml@4619e04 laya_coreml/ane.py (Apache-2.0; see NOTICE),
as validated in research/phase-0-feasibility/scripts/backends.py.

Each request row is padded to the smallest offered bucket that holds it. A row that fits
no offered bucket raises UnsupportedShapeError: nothing is truncated, and nothing runs on
another device.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ..artifacts import COMPILED, artifact_dir, load_verified
from ..errors import ArtifactError, ArtifactMissingError, ComputeUnitMismatchError, UnsupportedShapeError
from ..registry import ANE_COMPUTE_UNITS, ANE_MAX_OPTIONS, ModelSpec


def ane_features(rows, L, B, embedding, type_embedding, window, pad_id):
    """Fixed-shape feature dict for up to B prepared rows (B,C,1,L layout, FP16)."""
    if len(rows) > B:
        raise ValueError("too many rows for this export")
    ids = np.full((B, L), pad_id, np.int64)
    valid = np.zeros((B, L), bool)
    valid[:, 0] = True  # dummy rows still need one valid key
    qtype = np.zeros(B, np.int64)
    marker_map = np.zeros((B, L, 1, ANE_MAX_OPTIONS), np.float16)
    for r, it in enumerate(rows):
        m = len(it["ids"])
        if m > L:
            raise UnsupportedShapeError(f"row has {m} tokens; bucket holds {L} (no truncation)")
        ids[r, :m] = it["ids"]
        valid[r, :m] = True
        qtype[r] = it["qtype"]
        marker_map[r, it["markers"], 0, np.arange(len(it["markers"]))] = 1
    emb = embedding[ids].transpose(0, 2, 1)[:, :, None, :]
    full = np.broadcast_to(valid[:, None, :], (B, L, L))
    local = (window[None] | ~valid[:, :, None]) & full
    feats = {  # scores are laid out [B, key, 1, query]
        name: np.where(value.transpose(0, 2, 1)[:, :, None, :], 0, -1e4).astype(np.float16)
        for name, value in (("full_mask", full), ("local_mask", local))
    }
    feats["embeddings"] = np.ascontiguousarray(emb, dtype=np.float16)
    feats["type_vectors"] = np.ascontiguousarray(type_embedding[qtype][:, :, None, None], dtype=np.float16)
    feats["marker_map"] = marker_map
    return feats


# Runtime placement probe. CPU_AND_NE lets Core ML use the CPU, and the static compute plan
# (checked at load) cannot see where the *loaded* model runs. The probe times the loaded model
# against a CPU_ONLY instance of the same artifact on the same input, back to back, so machine
# load slows both alike. On the ANE the ratio measured 0.32-0.50 for every shipped artifact;
# a model running on the CPU would measure ~1.0 (benchmarks/v1.0.md, benchmarks/v1.0/probe.json).
PROBE_MAX_RATIO = 0.8
PROBE_RUNS = 5


def _fastest_ms(model, feats, runs: int) -> float:
    import time

    model.predict(feats)  # the first call may still pay one-off set-up
    best = math.inf
    for _ in range(runs):
        t = time.perf_counter()
        model.predict(feats)
        best = min(best, (time.perf_counter() - t) * 1e3)
    return best


def probe_placement(spec: ModelSpec, bucket: int, model, compiled_path: Path, feats: dict) -> dict:
    """Refuse a loaded model that runs no faster than the same artifact on the CPU.

    Returns {"ane_ms", "cpu_ms", "ratio"} or raises ComputeUnitMismatchError."""
    import coremltools as ct

    cpu = ct.models.CompiledMLModel(str(compiled_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    ane_ms, cpu_ms = _fastest_ms(model, feats, PROBE_RUNS), _fastest_ms(cpu, feats, PROBE_RUNS)
    del cpu
    ratio = ane_ms / cpu_ms
    if ratio > PROBE_MAX_RATIO:
        raise ComputeUnitMismatchError(
            f"{spec.name} L{bucket}: the loaded model takes {ane_ms:.1f} ms against {cpu_ms:.1f} ms on CPU_ONLY "
            f"(ratio {ratio:.2f} > {PROBE_MAX_RATIO}), so it is not running on the Neural Engine (the on-device "
            f"ANE compile may have failed). Run `laya-apple artifacts warm {spec.name}` and retry; if it "
            "persists, rebuild the artifact."
        )
    return {"ane_ms": round(ane_ms, 3), "cpu_ms": round(cpu_ms, 3), "ratio": round(ratio, 3)}


class HostWeights:
    """Embedding table, type embedding and action head, read from the original checkpoint."""

    def __init__(self, checkpoint: Path, local_attention: int):
        from safetensors import safe_open

        with safe_open(str(checkpoint / "model.safetensors"), framework="numpy") as w:
            self.embedding = w.get_tensor("encoder.embeddings.tok_embeddings.weight")
            self.type_embedding = w.get_tensor("type_emb.weight")
            self.action = {
                k: w.get_tensor("act_head." + k).astype(np.float32)
                for k in ("0.weight", "0.bias", "2.weight", "2.bias")
            }
        self.radius = local_attention // 2
        self._windows = {}
        self._erf = np.vectorize(math.erf, otypes=[np.float64])

    def window(self, L):
        if L not in self._windows:
            pos = np.arange(L)
            self._windows[L] = np.abs(pos[:, None] - pos[None, :]) <= self.radius
        return self._windows[L]

    def tail(self, outputs, rows):
        """Mask inactive options, derive action features, run the FP32 action head."""
        logits = pooled = None
        for v in outputs.values():
            if v.shape[1] == 1:
                logits = v.reshape(v.shape[0], -1)[: len(rows)].astype(np.float32)
            else:
                pooled = v.reshape(v.shape[0], -1)[: len(rows)].astype(np.float32)
        mm = np.zeros((len(rows), ANE_MAX_OPTIONS), bool)
        for r, it in enumerate(rows):
            mm[r, : len(it["markers"])] = True
        logits = np.where(mm, logits, -1e4)
        p = np.exp(logits - logits.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        k = np.maximum(mm.sum(-1), 2).astype(np.float32)
        ent = -(p * np.log(np.maximum(p, 1e-9))).sum(-1) / np.log(k)
        top = np.sort(p, -1)[:, -2:]
        feats = np.stack((top[:, 1], top[:, 1] - top[:, 0], ent, k / 255.0), -1)
        # NumPy's Accelerate matmul raises spurious FP-exception warnings on macOS even for
        # finite inputs; outputs are checked for finiteness by format_answers and the parity gate.
        with np.errstate(all="ignore"):
            h = np.concatenate((pooled, feats), -1) @ self.action["0.weight"].T + self.action["0.bias"]
            h = h * (1 + self._erf(h / math.sqrt(2)).astype(np.float32)) / 2  # exact erf GELU
            act = h @ self.action["2.weight"].T + self.action["2.bias"]
        return logits, act.astype(np.float32)


class ANEShapes:
    """Which rows an ANE backend can serve, and with which artifact.

    Shared by the in-process backend and the parent-side view of an ANE worker process
    (laya_apple.executor), so both refuse the same requests with the same errors before
    any work runs.
    """

    name = "coreml"
    device = "ane"
    compute_units = ANE_COMPUTE_UNITS

    def __init__(self, spec: ModelSpec, offered, artifact_sha256: dict, load_errors: dict):
        self.spec = spec
        self.offered = tuple(sorted(offered))
        self.artifact_sha256 = dict(artifact_sha256)
        self.load_errors = dict(load_errors)
        self.probes: dict = getattr(self, "probes", {})  # runtime placement probe results per bucket

    @property
    def buckets(self) -> tuple:
        return tuple(sorted(self.artifact_sha256))

    def bucket_for(self, n_tokens: int) -> int:
        for b in self.offered:
            if n_tokens <= b:
                if b not in self.artifact_sha256:
                    raise self.load_errors[b]
                return b
        raise UnsupportedShapeError(
            f"{n_tokens} tokens exceeds the largest validated ANE bucket for {self.spec.name} "
            f"({self.offered[-1]}). Use device='gpu' or device='auto'."
        )

    def check(self, items) -> list[int]:
        buckets = []
        for it in items:
            if len(it["markers"]) > ANE_MAX_OPTIONS:
                raise UnsupportedShapeError(
                    f"{len(it['markers'])} options exceeds the ANE artifact's {ANE_MAX_OPTIONS}"
                )
            buckets.append(self.bucket_for(len(it["ids"])))
        return buckets

    def artifact_revision(self, items) -> str:
        return ",".join(f"L{b}:{self.artifact_sha256[b][:12]}" for b in sorted(set(self.check(items))))


class ANEBackend(ANEShapes):
    def __init__(
        self,
        spec: ModelSpec,
        checkpoint: Path,
        pad_id: int,
        local_attention: int,
        buckets,
        *,
        strict=True,
    ):
        """Load and verify every offered bucket.

        strict (explicit device="ane"): a missing bucket is recorded and raised when a
        request needs it; any other verification failure raises now, as does having no
        usable bucket at all. Non-strict (device="auto"): every artifact failure is recorded
        and the bucket is simply not offered to auto routing.

        Every loaded bucket also passes the runtime placement probe (probe_placement).
        """
        self.pad_id = pad_id
        self.host = HostWeights(checkpoint, local_attention)
        self.models, self.manifests, load_errors = {}, {}, {}
        self.probes: dict = {}
        tolerated = ArtifactMissingError if strict else ArtifactError
        for b in sorted(buckets):
            try:
                self.models[b], self.manifests[b] = load_verified(spec, b)
                self.probes[b] = probe_placement(
                    spec, b, self.models[b], artifact_dir(spec, b) / COMPILED, self._probe_features(b)
                )
            except tolerated as e:
                self.models.pop(b, None)
                self.manifests.pop(b, None)
                load_errors[b] = e
        if strict and not self.models:
            raise next(iter(load_errors.values()))
        super().__init__(spec, buckets, {b: m.artifact_sha256 for b, m in self.manifests.items()}, load_errors)

    def _probe_features(self, b: int) -> dict:
        row = {"ids": [self.pad_id] * b, "markers": [1, 2], "qtype": 0}
        return ane_features(
            [row], b, 1, self.host.embedding, self.host.type_embedding, self.host.window(b), self.pad_id
        )

    def forward(self, items):
        buckets = self.check(items)
        logits = np.full((len(items), ANE_MAX_OPTIONS), -1e4, np.float32)
        acts = []
        for r, (it, b) in enumerate(zip(items, buckets)):
            feats = ane_features(
                [it], b, 1, self.host.embedding, self.host.type_embedding, self.host.window(b), self.pad_id
            )
            lg, ac = self.host.tail(self.models[b].predict(feats), [it])
            logits[r] = lg[0]
            acts.append(ac[0])
        return logits, np.stack(acts)
