"""Send the same requests to `laya-apple serve` and to upstream `laya.serve`, and compare.

    uv run python scripts/compare_serve_upstream.py \
        --ours http://127.0.0.1:8642 --upstream http://127.0.0.1:8643 \
        --model typed-decisions --out benchmarks/serve-compat/typed-decisions.json

The requests are the shipped parity golden cases for the model (state + questions). For each
response it checks:

- wire shape: the same top-level keys (ours adds only `laya_apple`), the same `routing`
  keys, the same answer ids, and the same keys in every answer;
- values: the same `choice` and `usage`, and every probability, score, noul, confidence
  and act_probability within the FP16 parity gate's 0.02 (docs/correctness.md).

With --auto the requests name no model, as Jev clients do, and each server routes by
language; `routing.model`, `reason`, `detection` and `workflow` must then agree too.
`routing.repo` is not compared: ours names the pinned standalone repository, and the
pinned upstream launcher reports a local path.

Upstream runs PyTorch; ours runs MLX or the ANE, so values are compared with the gate's
tolerance, never for equality. Exit status 1 if any case fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from laya_apple.parity import load_goldens  # noqa: E402

UPSTREAM_TO_OURS = {"english": "laya", "multilingual": "laya-multilingual", "typed-decisions": "laya-typed-decisions"}
TOL = 0.02
NUMERIC = ("confidence", "answer_confidence", "score", "noul")


def post(base: str, body: dict) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer placeholder"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def compare(ours: dict, theirs: dict, auto: bool = False) -> list[str]:
    problems = []
    if auto:
        for k in ("model", "reason", "detection", "workflow"):
            if ours["routing"][k] != theirs["routing"][k]:
                problems.append(f"routing.{k}: ours {ours['routing'][k]!r}, upstream {theirs['routing'][k]!r}")
    extra = set(ours) - set(theirs)
    if extra != {"laya_apple"} or set(theirs) - set(ours):
        problems.append(f"top-level keys: ours {sorted(ours)}, upstream {sorted(theirs)}")
    if set(ours.get("routing") or {}) != set(theirs.get("routing") or {}):
        problems.append(f"routing keys: ours {sorted(ours.get('routing') or {})}, upstream {sorted(theirs['routing'])}")
    if ours.get("usage") != theirs.get("usage"):
        problems.append(f"usage: ours {ours.get('usage')}, upstream {theirs.get('usage')}")
    extra_answers = set(ours.get("answers", {})) - set(theirs.get("answers", {}))
    if extra_answers:
        problems.append(f"answers only in ours: {sorted(extra_answers)}")
    for qid, want in theirs.get("answers", {}).items():
        got = ours.get("answers", {}).get(qid)
        if got is None:
            problems.append(f"{qid}: missing")
            continue
        if set(got) != set(want):
            problems.append(f"{qid}: answer keys ours {sorted(got)}, upstream {sorted(want)}")
        if got.get("choice") != want.get("choice"):
            problems.append(f"{qid}: choice ours {got.get('choice')!r}, upstream {want.get('choice')!r}")
        for k in NUMERIC:
            if k in want and k in got and abs(got[k] - want[k]) > TOL:
                problems.append(f"{qid}: {k} ours {got[k]}, upstream {want[k]}")
        for label, p in (want.get("probabilities") or {}).items():
            q = (got.get("probabilities") or {}).get(label)
            if q is None or abs(q - p) > TOL:
                problems.append(f"{qid}: probabilities[{label}] ours {q}, upstream {p}")
        a, b = (got.get("action") or {}).get("act_probability"), (want.get("action") or {}).get("act_probability")
        if a is None or b is None or abs(a - b) > TOL:
            problems.append(f"{qid}: act_probability ours {a}, upstream {b}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ours", required=True)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--model", required=True, choices=sorted(UPSTREAM_TO_OURS))
    ap.add_argument("--auto", action="store_true", help="send no model field; compare the language routing too")
    ap.add_argument("--out", help="write the per-case report here (JSON)")
    a = ap.parse_args()

    cases = load_goldens(UPSTREAM_TO_OURS[a.model])["cases"]
    report = {"model": a.model, "auto": a.auto, "tolerance": TOL, "cases": []}
    for case in cases:
        body = {"state": case["state"], "questions": case["questions"]}
        if not a.auto:
            body["model"] = a.model
        ours, theirs = post(a.ours, body), post(a.upstream, body)
        problems = compare(ours, theirs, a.auto)
        report["cases"].append(
            {
                "name": case["name"],
                "passed": not problems,
                "problems": problems,
                "ours_device": ours.get("laya_apple", {}).get("device"),
                "routed_to": ours.get("routing", {}).get("model"),
            }
        )
        print(f"{'PASS' if not problems else 'FAIL'}  {case['name']}  ({ours.get('laya_apple', {}).get('device')})")
        for p in problems:
            print(f"      {p}")
    report["passed"] = all(c["passed"] for c in report["cases"])
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(f"{sum(c['passed'] for c in report['cases'])}/{len(report['cases'])} cases pass")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
