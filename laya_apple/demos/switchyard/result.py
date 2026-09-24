"""result.json and trace.jsonl for one Switchyard run.

result.json (`schema: "laya-apple/switchyard-result"`, validated by
laya_apple/data/switchyard-result.schema.json) holds every derived number; trace.jsonl holds
one raw record per request (RequestTrace.to_dict() plus round, index, class, arrival_ns and,
for trains, the decision). Every number in result.json can be recomputed from trace.jsonl
with laya_apple.demos.switchyard.metrics. Machine paths are written as <LAYA_APPLE_CACHE>,
<HF_HOME> or ~.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from ... import __version__
from ...schema import errors, load_schema
from . import metrics, world

SCHEMA = "laya-apple/switchyard-result"
SCHEMA_VERSION = 1
SCHEMA_FILE = "switchyard-result.schema.json"
TOOL = "laya-apple switchyard"
RESULT_FILE, TRACE_FILE, REPLAY_FILE = "result.json", "trace.jsonl", "replay.html"


def _roots() -> list[tuple[str, str]]:
    """(path, placeholder), most specific first: the laya-apple cache, the Hugging Face home,
    then the home directory."""
    import os

    from ...hub import cache_root

    hf = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    return [(str(cache_root()), "<LAYA_APPLE_CACHE>"), (hf, "<HF_HOME>"), (str(Path.home()), "~")]


def redact(obj, roots: list[tuple[str, str]] | None = None):
    """Replace machine paths (see _roots) with placeholders in every string, recursively
    (keys included)."""
    if roots is None:
        roots = _roots()
    if isinstance(obj, str):
        for path, placeholder in roots:
            if path:
                obj = obj.replace(path, placeholder)
        return obj
    if isinstance(obj, dict):
        return {redact(k, roots): redact(v, roots) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(v, roots) for v in obj]
    return obj


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def machine() -> dict:
    from ...artifacts import _sh, platform_profile
    from ...model import platform_validated
    from ...profiles import load_local

    memsize = _sh("sysctl", "-n", "hw.memsize")
    return {
        "platform": platform_profile() | {"machine": platform.machine()},
        "memory_gb": round(int(memsize) / 2**30) if memsize.isdigit() else None,
        "python": platform.python_version(),
        "mlx": _version("mlx"),
        "routing_profile_validated": platform_validated(),
        "calibrated_profile": load_local(world.MODEL) is not None,
    }


def model_info(ane_artifacts: dict) -> dict:
    from ...registry import resolve

    spec = resolve(world.MODEL)
    return {
        "name": spec.name,
        "revision": spec.revision,
        "weights_sha256": spec.weights_sha256,
        "ane_artifacts": dict(ane_artifacts),
    }


def workload(seed: int, duration_s: float, items) -> dict:
    return {
        "id": world.WORKLOAD_ID,
        "version": world.WORKLOAD_VERSION,
        "seed": seed,
        "duration_s": duration_s,
        "warmup_s": world.WARMUP_S,
        "warmup_seed": seed + world.WARMUP_SEED_OFFSET,
        "arrivals": world.ARRIVALS,
        "burst_period_s": world.BURST_PERIOD_S,
        "burst_on_s": world.BURST_ON_S,
        "nominal_rate_req_s": world.RATE_REQ_S,
        "deadline_ms": world.DEADLINE_MS,
        "thresholds_ms": list(world.THRESHOLDS_MS),
        "mix": dict(world.MIX),
        "lengths": dict(world.LENGTHS),
        "prompt_template_sha256": world.prompt_template_sha256(),
        "schedule_sha256": world.schedule_sha256(items),
        "offered": world.offered(items, duration_s),
    }


def is_standard(seed: int, duration_s: float) -> bool:
    return seed == world.DEFAULT_SEED and float(duration_s) == world.DURATION_S


def build(run, *, machine_info: dict | None = None, model: dict | None = None, created_utc: str | None = None) -> dict:
    """The result.json object for a driver.Run (machine and model are looked up if not given)."""
    rounds, configs = [], {}
    for rnd in run.rounds:
        s = metrics.summarize([(rnd.records, rnd.round_start_ns)])
        rounds.append(
            {
                "index": rnd.index,
                "config": rnd.config,
                "started_utc": rnd.started_utc,
                "round_start_ns": rnd.round_start_ns,
                "timetable_sha256": metrics.timetable_sha256(rnd.records, rnd.round_start_ns),
                "ane_verified": rnd.ane_verified,
                "conditions": rnd.conditions,
                **s,
            }
        )
    for label, cfg in run.configs.items():
        mine = [r for r in run.rounds if r.config == label]
        pooled = metrics.summarize([(r.records, r.round_start_ns) for r in mine])
        # summary has the shape of a round's `systems`; `game` is pooled the same way
        configs[label] = {
            **cfg,
            "rounds": [r.index for r in mine],
            "game": pooled["game"],
            "summary": pooled["systems"],
        }
    sequence = [r.config for r in run.rounds]
    standard = is_standard(run.seed, run.duration_s)
    # A comparison needs both configurations and every GPU + ANE round verified on the ANE.
    available = set(configs) >= {"gpu_only", "hybrid"} and all(
        r.ane_verified for r in run.rounds if r.config == "hybrid"
    )
    timetables = {r["timetable_sha256"] for r in rounds}
    result = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "laya_apple": __version__,
        "created_utc": created_utc or datetime.now(timezone.utc).isoformat(),
        "mode": "standard",
        "standard": standard,
        # A standard run is submittable whether or not the GPU + ANE round ran (GPU-only Macs too).
        "submission_eligible": standard,
        "workload": workload(run.seed, run.duration_s, run.schedule),
        "machine": machine_info if machine_info is not None else machine(),
        "model": model if model is not None else model_info(run.ane_artifacts),
        "ane": run.ane.to_dict(),
        "design": {"ordering": "seed_parity", "counterbalance": "none", "sequence": sequence},
        "rounds": rounds,
        "configs": configs,
        "comparison": {
            "available": available,
            "same_schedule_sha256": len(timetables) == 1 if len(rounds) >= 2 else None,
            "decision_disagreements": metrics.decision_disagreements([r.records for r in run.rounds]),
        },
        "trace_file": TRACE_FILE,
    }
    return redact(result)


def validate(result: dict) -> list[str]:
    return errors(load_schema(SCHEMA_FILE), result)


def write(out_dir: Path, result: dict, rounds) -> tuple[Path, Path]:
    """Write result.json and trace.jsonl (one record per request, rounds in order)."""
    problems = validate(result)
    if problems:
        raise ValueError("result.json does not match its schema: " + "; ".join(problems[:5]))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path, trace_path = out_dir / RESULT_FILE, out_dir / TRACE_FILE
    with open(trace_path, "w", encoding="utf-8") as f:
        for rnd in rounds:
            for rec in rnd.records:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    result_path.write_text(json.dumps(result, indent=1) + "\n")
    return result_path, trace_path


def read(out_dir: Path) -> tuple[dict, list[dict]]:
    """(result, trace records) from a run directory."""
    out_dir = Path(out_dir)
    result = json.loads((out_dir / RESULT_FILE).read_text())
    trace_path = out_dir / result.get("trace_file", TRACE_FILE)
    with open(trace_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return result, records
