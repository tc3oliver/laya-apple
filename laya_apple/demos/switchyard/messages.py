"""Every user-facing string of `laya-apple switchyard`, in one place for review.

MESSAGES are terminal lines (str.format templates). REASON_TEXT is the plain-language text
for each ane_state reason code: result.json records the code as `ane.reason` and this text
as `ane.reason_text`, and the replay card shows the text. HELP is the --help text. Numbers
come from the workload constants in world.py, never typed in here.
"""

from __future__ import annotations

from . import ane_state, world

MESSAGES = {
    "needs_apple_silicon": (
        "Switchyard needs an Apple silicon Mac with MLX, laya-apple's GPU backend. "
        "To view an existing run anywhere, use --replay DIR."
    ),
    "checking_ane": "Checking the Neural Engine…",
    "ane_ready": "Neural Engine ready. Running the same timetable twice: GPU only and GPU + ANE.",
    "ane_missing": "GPU + ANE unavailable: {reason}. Running the GPU-only round.",
    "ane_missing_setup": (
        "GPU + ANE unavailable: {reason}. Running the GPU-only round; the end of the run says how to add GPU + ANE."
    ),
    "ane_setup_hint": "To add the GPU + ANE round, set up the Neural Engine (several minutes, once per Mac):\n  {command}",
    "ane_skipped": "Skipping the GPU + ANE round: {reason}.",
    "ane_failed": "GPU + ANE round not counted: {reason}.",
    "round_failed": "The GPU-only round could not finish ({detail}).",
    "partial_saved": "Saved the rounds that finished in {path}",
    "duration_not_whole_cycles": (
        "--duration must be a whole number of {period:g} s burst cycles (3, 6, 9, … s), so the offered load "
        "matches the timetable's mean rate (standard: {standard:g} s)."
    ),
    "setup_failed": "Neural Engine setup did not finish: {reason}.",
    "setup_start": "Setting up the Neural Engine (this can take several minutes)…",
    "setup_verified": "  {bucket}-token model: already built and checked",
    "setup_building": "  {bucket}-token model: building, then checking it gives the same answers as the GPU…",
    "setup_calibrating": "  Measuring this Mac to decide which requests go to the Neural Engine…",
    "setup_done": "Neural Engine setup finished.",
    "loading": "Loading {model} for the {config} round…",
    "warmup": "Warming up for {seconds:g} s (not counted)…",
    "round": "Round {number} of {total}: {config}, {seconds:g} s of trains…",
    "round_done": "  {late:,} of {trains:,} trains late · P99 decision latency {p99:,.1f} ms",
    "not_standard": (
        "Note: custom seed or duration. Results are not comparable with standard runs (seed {seed}, {duration:g} s)."
    ),
    "wrote": "Results saved in {path}",
    "summary": "Same timetable. {configs}.",
    "summary_single": "GPU only (GPU + ANE did not run: {reason}).",
    "summary_unverified": "GPU only (the GPU + ANE round is not counted: {reason}).",
    "summary_row": (
        "  {config:<10} {late:>5,} of {trains:,} trains late   P99 decision {p99:>9,.1f} ms   "
        "P99 queue wait {queue:>9,.1f} ms"
    ),
    "late_definition": "  Late = decision more than {deadline:g} ms after the train arrived.",
    "disagreements": (
        "Warning: {count} trains were sent to a different platform in the two rounds (see comparison in result.json)."
    ),
    "replay": "Opening the replay in your browser…",
    "replay_path": "Replay saved: {path} (open it in a browser)",
    "rebuilt": "Rebuilt the replay: {path}",
    "no_result": "No result.json in {path}. Pass the folder a previous `laya-apple switchyard` run created.",
}
CONFIG_NAMES = {"gpu_only": "GPU only", "hybrid": "GPU + ANE"}
REASON_TEXT = {
    ane_state.NO_COREMLTOOLS: "The Neural Engine runtime (Core ML Tools) is not installed",
    ane_state.NO_ARTIFACTS: "The Neural Engine model is not built on this Mac yet",
    ane_state.NOT_CALIBRATED: "This Mac has not been calibrated for the Neural Engine yet",
    ane_state.NOT_APPLE_SILICON: "This machine is not an Apple silicon Mac",
    ane_state.REBUILD_ARTIFACTS: "The Neural Engine model needs to be rebuilt on this Mac",
    ane_state.PARITY_FAILED: "A Neural Engine model failed the correctness check on this Mac",
    ane_state.NOT_ON_ANE: "Core ML could not run the model on the Neural Engine",
    ane_state.NO_ANE_WARMUP: "The router sent no trains to the Neural Engine during warmup",
    ane_state.ANE_LOST: "The Neural Engine stopped being used during the round",
    ane_state.ANE_FAILED: "The GPU + ANE round failed on this Mac",
    ane_state.NO_CONVERT: "The Neural Engine build tools (laya-apple[convert]) are not installed",
    ane_state.CALIBRATION_FAILED: "Measuring this Mac for the Neural Engine failed",
}
HELP = {
    "command": "Benchmark GPU only vs GPU + ANE on one timetable, then replay it as a rail junction",
    "description": (
        f"Runs one seeded timetable of trains (one Laya routing decision each) and background requests "
        f"twice, GPU only and GPU + ANE, {world.DURATION_S:g} s of trains per round, and counts the trains "
        f"whose decision takes more than {world.DEADLINE_MS:g} ms. Takes about 3 minutes once the model is "
        "downloaded (first run downloads ~800 MB). Writes result.json and trace.jsonl, then opens a replay "
        "of the recorded run in your browser."
    ),
    "seed": f"timetable seed (standard: {world.DEFAULT_SEED}; other seeds are not comparable)",
    "duration": f"seconds of trains per round (standard: {world.DURATION_S:g})",
    "out": "folder for result.json, trace.jsonl and replay.html (default: ./switchyard-results/<utc>-<soc>/)",
    "no_open": "write replay.html but do not open a browser",
    "setup_ane": (
        "first build and check the Neural Engine models and measure this Mac "
        "(several minutes, needs laya-apple[convert])"
    ),
    "replay": "rebuild replay.html from a previous run's folder and open it (no model needed)",
}
