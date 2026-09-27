"""Re-run the placement and parity gates on an EXISTING W8 artifact (addendum 2), no rebuild.

    uv run --extra ane python research/ane-w8/scripts/replicate_gate.py \
        --config w8-pt --model laya-typed-decisions --length 64 --raw research/ane-w8/raw/replication-l64

Not timing-sensitive. Uses the same production calls as build_w8.py steps 4-5, unchanged:
`compute_plan_summary` + `check_ane_placement`, and common.parity (production `ane_parity` plus
the recording pass). Also records the artifact's tree SHA-256 and sizes, and the shipped FP16
artifact's sizes. Writes <raw>/<model>/<cell>/gate.json and parity_rows.jsonl; never overwrites.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CELLS, cell_name, environment, log, now, parity, research_root, tree_bytes, weight_bytes, write_json  # noqa: E402


def main() -> None:
    from laya_apple.artifacts import COMPILED, artifact_dir, check_ane_placement, compute_plan_summary, tree_sha256
    from laya_apple.hub import checkpoint_path
    from laya_apple.registry import ANE_COMPUTE_UNITS, resolve

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", choices=sorted(CELLS), required=True)
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--root")
    ap.add_argument("--raw", required=True)
    args = ap.parse_args()
    root = research_root(args.root)
    spec = resolve(args.model)
    out = Path(args.raw).resolve() / args.model / cell_name(args.length, args.config)
    if (out / "gate.json").exists() or (out / "parity_rows.jsonl").exists():
        sys.exit(f"{out} already has gate data; refusing to overwrite raw data")
    compiled = root / spec.name / spec.revision[:12] / f"bc1s-masked-{args.config}-L{args.length}-B1" / COMPILED
    base = artifact_dir(spec, args.length) / COMPILED
    rec = {
        "experiment": "ane-w8/replication-gate",
        "model": spec.name,
        "length": args.length,
        "config": args.config,
        "started_at": now(),
        "environment": environment(),
        "candidate": {
            "tree_sha256": tree_sha256(compiled),
            "compiled_bytes": tree_bytes(compiled),
            "weight_bin_bytes": weight_bytes(compiled),
        },
        "baseline": {
            "tree_sha256": tree_sha256(base),
            "compiled_bytes": tree_bytes(base),
            "weight_bin_bytes": weight_bytes(base),
        },
    }
    log("placement")
    rec["placement"] = placement = compute_plan_summary(compiled, ANE_COMPUTE_UNITS)
    try:
        check_ane_placement(placement)
        rec["placement_gate"] = "PASS"
    except Exception as e:  # ComputeUnitMismatchError
        rec["placement_gate"] = "FAIL"
        rec["placement_error"] = str(e)
    log("parity")
    ckpt = checkpoint_path(spec, local_files_only=True)
    rec["parity"] = parity(spec, compiled, args.length, ckpt, out / "parity_rows.jsonl")
    rec["parity_gate"] = "PASS" if rec["parity"]["passed"] else "FAIL"
    rec["finished_at"] = now()
    write_json(out / "gate.json", rec)
    log(f"placement {rec['placement_gate']}, parity {rec['parity_gate']}")


if __name__ == "__main__":
    main()
