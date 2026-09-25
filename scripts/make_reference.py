"""Record PyTorch CPU FP32 references from unmodified upstream Laya for the parity goldens.

    uv sync --extra dev --extra reference
    HF_HUB_OFFLINE=1 uv run python scripts/make_reference.py            # all three models
    uv run python scripts/make_goldens.py                               # then derive the goldens

Writes research/upstream-reference/raw/laya-<version>/<model>.json. Every case is run through
the installed upstream `laya` (the `[reference]` extra) exactly as `Agent.system_one` runs it:
upstream validates the questions, tokenizes the prompt and computes the answers. The script
records the prompt items, the unrounded decision and action logits and the public answers.

Cases: every Phase -1 fixture (read back from research/phase-0-feasibility/raw/reference), plus
the DRIFT_CASES below, which exercise what upstream changed after 0.3.5. For each Phase -1
case the new items and logits are compared with the Phase -1 record and the result is stored
as `phase0_identical`; make_goldens.py refuses to derive goldens when one of them differs.
`tokenizer_match` records whether laya_apple.prompt.prepare reproduces upstream's items.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHASE0 = ROOT / "research/phase-0-feasibility/raw/reference"
OUT = ROOT / "research/upstream-reference/raw"
MODELS = ("laya", "laya-multilingual", "laya-typed-decisions")
PACKAGES = ("torch", "transformers", "laya", "tokenizers", "numpy", "safetensors", "huggingface-hub")

_REFUND = "My card was charged two times for order 5521. I want the second charge refunded."
_NOUL = {"type": "noul", "instructions": "Does the customer explicitly ask for a refund?"}
_LONG_DESCRIPTION = (
    "charges, invoices, refunds, duplicate payments, failed card authorisations, currency conversion "
    "fees, annual plan renewals, proration after a plan change, tax and VAT questions, receipts for "
    "expense reports, payment method updates, chargebacks and disputed transactions"
)

# Written for this project. Each case exercises one upstream change after 0.3.5.
DRIFT_CASES = [
    {
        # e0557cb (#224): a conversation list that overflows max_len keeps its newest turns.
        "name": "drift-conversation-left-truncated",
        "state": [{"role": "user", "content": _REFUND}, {"role": "agent", "content": "Checking that for you now."}] * 60
        + [{"role": "user", "content": "Forget the refund. Please cancel my subscription today."}],
        "questions": {
            "q0": {"type": "noul", "instructions": "Is the customer asking to cancel the subscription?"},
            "q1": _NOUL,
            "q2": {
                "type": "choice",
                "instructions": "What does the customer want now?",
                "criteria": ["refund", "cancellation", "information"],
            },
        },
    },
    {
        # 8dacc39 (#228): non-string instructions are serialized with ensure_ascii=False.
        "name": "drift-structured-instructions-non-ascii",
        "state": {"message": "Meine Karte wurde für Bestellung 5521 zweimal belastet. Bitte erstatten Sie den Betrag."},
        "questions": {
            "q0": {"type": "noul", "instructions": {"frage": "Ist eine Rückerstattung fällig?", "kanal": "E-Mail"}},
            "q1": {
                "type": "choice",
                "instructions": {"aufgabe": "Welches Team übernimmt das Ticket?"},
                "criteria": {"billing": "Zahlungen, Rückerstattungen", "technical": "Fehler, Störungen"},
            },
            "q2": {"type": "score", "instructions": ["Dringlichkeit", "bewerten"], "criteria": ["niedrig", "hoch"]},
        },
    },
    {
        # 158398a (#163): noul `labels` replace the literal false:/true: option prefixes.
        "name": "drift-noul-labels",
        "state": _REFUND,
        "questions": {
            "q0": {**_NOUL, "labels": {"false": "B", "true": "A"}},
            "q1": {
                **_NOUL,
                "criteria": {"true": "a refund is requested", "false": "no refund is requested"},
                "labels": {"false": " no ", "true": " yes "},
            },
            "q2": {
                "type": "noul",
                "instructions": "Is the tone polite?",
                "labels": {"true": "polite", "false": "rude"},
            },
        },
    },
    {
        # 7913866 (#146): noul criteria keys are matched case-insensitively.
        "name": "drift-noul-criteria-key-case",
        "state": _REFUND,
        "questions": {
            "q0": {**_NOUL, "criteria": {"True": "a refund is requested", "FALSE": "no refund is requested"}},
            "q1": {**_NOUL, "criteria": {"TRUE": "money is owed back"}},
        },
    },
    {
        # 34d27df (#109): option text is capped at 48 tokens by the tokenizer, not by a slice.
        "name": "drift-long-option-text",
        "state": _REFUND,
        "questions": {
            "q0": {
                "type": "choice",
                "instructions": "Which team should own this ticket?",
                "criteria": {"billing": _LONG_DESCRIPTION, "technical": "errors, outages", "other": "anything else"},
            },
            "q1": {"type": "score", "instructions": "How urgent is this?", "criteria": [_LONG_DESCRIPTION, "urgent"]},
        },
    },
]


def environment() -> dict:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    soc = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
    return {
        "soc": soc.stdout.strip() or None,
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "packages": packages,
    }


def isolated_checkpoint(src: Path, dest: Path) -> Path:
    """Upstream may rewrite tokenizer_config.json in place; keep the shared HF cache untouched."""
    dest.mkdir(parents=True)
    (dest / "model.safetensors").symlink_to((src / "model.safetensors").resolve())
    shutil.copy(src / "rl_agent_config.json", dest)
    shutil.copytree(src / "encoder", dest / "encoder")
    shutil.copytree(src / "tokenizer", dest / "tokenizer")
    return dest


def run(model: str, version: str) -> dict:
    import laya
    import torch
    from laya.common import collate_items

    from laya_apple.hub import checkpoint_path, sha256_file
    from laya_apple.prompt import Tokenizer, prepare
    from laya_apple.registry import resolve

    torch.set_grad_enabled(False)
    spec = resolve(model)
    ckpt = checkpoint_path(spec, local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1")
    phase0 = json.loads((PHASE0 / f"{model}.json").read_text())
    base = [{k: c[k] for k in ("name", "length", "state", "questions") if k in c} for c in phase0["cases"]]
    old = {c["name"]: c for c in phase0["cases"]}

    with tempfile.TemporaryDirectory() as tmp:
        path = isolated_checkpoint(ckpt, Path(tmp) / "checkpoint")
        agent = laya.load(str(path), device="cpu")
        if agent.device.type != "cpu" or agent.amp_enabled:
            raise SystemExit(f"upstream did not select CPU FP32: {agent.device} amp={agent.amp_enabled}")
        tok = Tokenizer(path / "tokenizer")
        out = {
            "model": model,
            "reference": f"upstream NandhaKishorM/laya {version} Agent, torch CPU float32",
            "source_weights_sha256": sha256_file(path / "model.safetensors"),
            "environment": environment(),
            "cases": [],
        }
        for case in base + DRIFT_CASES:
            ids = list(case["questions"])
            for qid in ids:
                agent._check_question(qid, case["questions"][qid])
            internal = {qid: agent._to_internal(case["questions"][qid]) for qid in ids}
            items = agent._encode_state(case["state"], ids, internal)
            b = collate_items([items], agent.tok.pad_token_id)
            logits, act = agent.model(
                b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"]
            )
            logits, act = logits.float().numpy(), act.float().numpy()
            result = agent.system_one(case["state"], case["questions"])
            record = {
                **case,
                "items": items,
                "tokenizer_match": prepare(tok, agent.cfg, case["state"], case["questions"]).items == items,
                "logits": [row[: len(it["markers"])].tolist() for row, it in zip(logits, items)],
                "action_logits": act.tolist(),
                "result": result,
            }
            if case["name"] in old:
                ref = old[case["name"]]
                record["phase0_identical"] = all(record[k] == ref[k] for k in ("items", "logits", "action_logits"))
            out["cases"].append(record)
            print(
                model,
                case["name"],
                max(len(it["ids"]) for it in items),
                "tok-match",
                record["tokenizer_match"],
                "phase0-identical",
                record.get("phase0_identical", "n/a"),
                flush=True,
            )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, action="append", help="default: all three")
    args = ap.parse_args()
    version = importlib.metadata.version("laya")
    dest = OUT / f"laya-{version}"
    dest.mkdir(parents=True, exist_ok=True)
    for model in args.model or MODELS:
        data = run(model, version)
        (dest / f"{model}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
