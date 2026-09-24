"""`laya-apple switchyard`: run the benchmark headless, write the results, open the replay.

    laya-apple switchyard [--seed N] [--duration S] [--out DIR] [--no-open] [--setup-ane] [--replay DIR]

Every line the command prints is in messages.py, so the wording can be reviewed in one place.
A Mac without a usable Neural Engine path runs the GPU-only round and exits 0.
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import world
from .messages import CONFIG_NAMES, HELP, MESSAGES


def say(key: str, **fields) -> None:
    print(MESSAGES[key].format(**fields), flush=True)


def default_out(now: datetime | None = None) -> Path:
    from ...artifacts import platform_profile
    from .driver import soc_slug

    now = now or datetime.now(timezone.utc)
    return Path("switchyard-results") / f"{now.strftime('%Y%m%dT%H%M%SZ')}-{soc_slug(platform_profile().get('soc'))}"


def _open(path: Path, no_open: bool) -> None:
    if no_open:
        say("replay_path", path=path)
        return
    say("replay")
    if not _launch(path.resolve()):
        say("replay_open_failed", path=path)


def _launch(path: Path) -> bool:
    """Open a local file in the default browser with /usr/bin/open. webbrowser.open() goes
    through AppleScript `open location`, which reports success for a file:// URL without
    opening anything."""
    try:
        return subprocess.run(["/usr/bin/open", str(path)], capture_output=True).returncode == 0
    except OSError:
        return False


def _summary(result: dict) -> None:
    configs, available = result["configs"], result["comparison"]["available"]
    if "gpu_only" not in configs:  # the GPU-only round failed; that was already said
        return
    # GPU only first, whatever the run order; GPU + ANE only when it is a real comparison
    shown = [c for c in CONFIG_NAMES if c in configs and (c == "gpu_only" or available)]
    if len(shown) == 2:
        say("summary", configs=" vs ".join(CONFIG_NAMES[c] for c in shown))
    elif "hybrid" in configs:
        say("summary_unverified", reason=result["ane"]["reason_text"])
    else:
        say("summary_single", reason=result["ane"]["reason_text"])
    for label in shown:
        cfg = configs[label]
        say(
            "summary_row",
            config=CONFIG_NAMES[label],
            late=cfg["summary"]["late"]["count"],
            trains=cfg["game"]["trains"],
            p99=cfg["summary"]["decision_latency"]["p99_ms"],
            queue=cfg["summary"]["queue_wait"]["p99_ms"],
        )
    say("late_definition", deadline=result["workload"]["deadline_ms"])
    if result["comparison"]["decision_disagreements"]:
        say("disagreements", count=result["comparison"]["decision_disagreements"])


def replay_only(directory: Path, no_open: bool) -> int:
    from . import replay
    from .result import SCHEMA, read

    try:
        result, records = read(directory)
    except (OSError, ValueError, KeyError):  # missing folder or files, or not a Switchyard run
        say("no_result", path=directory)
        return 2
    if result.get("schema") != SCHEMA:
        say("no_result", path=directory)
        return 2
    path = replay.write(directory, result, records)
    say("rebuilt", path=path)
    _open(path, no_open)
    return 0


def mlx_available() -> bool:
    """Apple silicon macOS with MLX importable: the GPU round cannot run anywhere else."""
    import importlib.util
    import platform
    import sys

    return sys.platform == "darwin" and platform.machine() == "arm64" and importlib.util.find_spec("mlx") is not None


def run(a) -> int:
    from ...registry import resolve
    from . import ane_state, driver, replay
    from . import result as result_mod

    if a.replay:  # rebuilding a replay needs no model, so it works on any machine
        return replay_only(Path(a.replay), a.no_open)
    if not world.whole_burst_cycles(a.duration):
        say("duration_not_whole_cycles", period=world.BURST_PERIOD_S, standard=world.DURATION_S)
        return 2
    if not mlx_available():
        say("needs_apple_silicon")
        return 2
    spec = resolve(world.MODEL)
    failed = None
    if a.setup_ane:
        say("setup_start")
        failed = ane_state.setup(
            spec,
            local_files_only=a.offline,
            step=lambda kind, bucket=None: say(f"setup_{kind}", bucket=bucket),
            log=lambda m: print(f"    {m}", flush=True),
        )
        if failed is None:
            say("setup_done")
        else:
            say("setup_failed", reason=failed.reason_text)
    say("checking_ane")
    status = failed or ane_state.detect(spec)
    if status.state == ane_state.READY:
        say("ane_ready")
    else:
        say("ane_missing_setup" if status.setup_command else "ane_missing", reason=status.reason_text)
    if not result_mod.is_standard(a.seed, a.duration):
        say("not_standard", seed=world.DEFAULT_SEED, duration=world.DURATION_S)

    def step(kind, *details):
        if kind == "loading":
            say("loading", model=world.MODEL, config=CONFIG_NAMES[details[0]])
        elif kind == "warmup":
            say("warmup", seconds=world.WARMUP_S)
        elif kind == "round":
            number, total, config, seconds = details
            say("round", number=number, total=total, config=CONFIG_NAMES[config], seconds=seconds)
        elif kind == "round_done":
            from .metrics import summarize

            rnd = details[0]
            s = summarize([(rnd.records, rnd.round_start_ns)])
            say(
                "round_done",
                late=s["game"]["late"],
                trains=s["game"]["trains"],
                p99=s["systems"]["decision_latency"]["p99_ms"],
            )
        elif kind == "ane_skipped":
            say("ane_skipped", reason=details[0].reason_text)
        elif kind == "ane_failed":
            say("ane_failed", reason=details[0].reason_text)
        elif kind == "round_failed":
            say("round_failed", detail=details[0])

    out = driver.run(a.seed, a.duration, status, local_files_only=a.offline, step=step)
    if not out.rounds:  # only possible when the GPU-only round failed; already reported
        return 1
    result = result_mod.build(out)
    out_dir = Path(a.out) if a.out else default_out()
    result_mod.write(out_dir, result, out.rounds)
    say("partial_saved" if out.error else "wrote", path=out_dir)
    _summary(result)
    if out.ane.setup_command:
        say("ane_setup_hint", command=out.ane.setup_command)
    records = [rec for rnd in out.rounds for rec in rnd.records]
    _open(replay.write(out_dir, result, records), a.no_open)
    # A missing or failed ANE never fails the command; a failed GPU-only round does.
    return 1 if out.error else 0


def _positive(value: str) -> float:
    x = float(value)
    if not x > 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return x


def add_parser(sub) -> None:
    s = sub.add_parser("switchyard", help=HELP["command"], description=HELP["description"])
    s.add_argument("--seed", type=int, default=world.DEFAULT_SEED, help=HELP["seed"])
    s.add_argument("--duration", type=_positive, default=world.DURATION_S, help=HELP["duration"])
    s.add_argument("--out", help=HELP["out"])
    s.add_argument("--no-open", action="store_true", help=HELP["no_open"])
    s.add_argument("--setup-ane", action="store_true", help=HELP["setup_ane"])
    s.add_argument("--replay", metavar="DIR", help=HELP["replay"])
    s.set_defaults(fn=run)
