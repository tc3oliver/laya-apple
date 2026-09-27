"""The 1.6 path: language routing, prebuilt ANE artifacts, `predict` and `predict_shortlist`.

`Laya.from_pretrained("auto")` returns a `LayaRouter` that loads `laya` and
`laya-multilingual` and picks one per request by the language of the context. This script
sends an English and a non-English request through `predict`, then one high-cardinality
`choice` question through `predict_shortlist`, and prints which checkpoint and device answered
each and why.

This script never downloads ANE artifacts. The first run downloads the two pinned checkpoints,
as every `from_pretrained` does. To serve short requests on the Neural Engine, fetch the
prebuilt artifacts first (the `ane` extra; no `convert` extra or PyTorch needed):

    uv run --extra ane laya-apple artifacts fetch laya
    uv run --extra ane laya-apple artifacts fetch laya-multilingual

Each fetched archive is checked on this machine before it is registered: its SHA-256 against
the repository index, the manifest and platform profile, the compute plan, the full FP16 parity
gate and the placement probe. Prebuilt artifacts exist for one platform profile only (Apple M4
Max, macOS 26, coremltools 9.0); on any other Mac, `fetch` raises `ArtifactMissingError` naming
the build command, and this script still runs, on the MLX GPU.

The first load of a fetched artifact still runs Core ML's on-device compile, which takes
minutes. Until then, or without artifacts, every request runs on the GPU and
`routing_reason` says why.

Run:

    uv run --extra ane python examples/auto_fetch_shortlist.py
"""

from __future__ import annotations

from laya_apple import Laya, embed_fn_from_laya

URGENCY = {
    "urgency": {
        "type": "choice",
        "instructions": "How urgent is this?",
        "criteria": ["low", "medium", "high"],
    }
}

# A `choice` question with many labels: the case predict_shortlist is for.
INTENTS = [
    "card_lost_or_stolen",
    "card_not_working",
    "card_payment_declined",
    "cash_withdrawal_charge",
    "change_pin",
    "contactless_not_working",
    "duplicate_charge",
    "exchange_rate",
    "failed_transfer",
    "pending_card_payment",
    "refund_not_showing",
    "request_refund",
    "top_up_failed",
    "transfer_not_received",
    "verify_identity",
    "wrong_amount_charged",
]
INTENT = {"intent": {"type": "choice", "instructions": "What does the customer want?", "criteria": INTENTS}}


def show(label: str, result) -> None:
    rt = result.runtime
    answer = next(iter(result.answers.values()))["choice"]
    print(f"{label:<11} {answer:<22} {rt.model:<18} {rt.model_routing:<26} {rt.device:<4} {rt.routing_reason}")


def main():
    with Laya.from_pretrained("auto") as router:
        print(f"{'request':<11} {'answer':<22} {'checkpoint':<18} {'model_routing':<26} {'dev':<4} routing_reason")

        english = "The customer was charged twice for the same invoice and is frustrated."
        show("english", router.predict(context=english, questions=URGENCY))

        german = "Der Kunde wurde für dieselbe Rechnung zweimal belastet und ist verärgert."
        show("german", router.predict(context=german, questions=URGENCY))

        # predict_shortlist keeps the k labels closest to the context, then runs one predict.
        # embed_fn_from_laya mean-pools the chosen checkpoint's own MLX encoder (inline only).
        checkpoint = router.route(context=english)["checkpoint"]
        embed_fn = embed_fn_from_laya(router.instances[checkpoint])
        result = router.predict_shortlist(context=english, questions=INTENT, embed_fn=embed_fn, k=5)
        show("shortlist", result)
        print("kept labels:", result.extra["shortlist"]["intent"]["labels"])


if __name__ == "__main__":
    main()
