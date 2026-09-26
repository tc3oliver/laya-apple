"""Write raw/fixtures-<model>.json: every workload request (state, questions, token lengths).

    HF_HUB_OFFLINE=1 uv run --extra ane --extra convert python research/intra-request-split/scripts/fixtures.py --model laya

Tokenises only (no model load, no device). Refuses to overwrite an existing file that differs:
the fixtures are generated once, before any measurement, and every later step reads them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    a = p.parse_args(argv)
    fx = common.make_fixtures(a.model)
    for name, w in fx["workloads"].items():
        for r in w["requests"]:
            if max(r["lengths"]) != w["tokens"] or len(r["lengths"]) != w["questions"]:
                raise SystemExit(f"{a.model} {name} seed {r['seed']}: rows {r['lengths']} are not the workload shape")
    path = common.fixtures_path(a.model)
    text = json.dumps(fx, indent=1) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise SystemExit(f"{path} exists and differs from what this revision generates; not overwritten")
        print(f"{path} is up to date")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    for name, w in fx["workloads"].items():
        print(a.model, name, [(r["seed"], min(r["lengths"]), max(r["lengths"])) for r in w["requests"]])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
