# Upstream reference outputs

This directory holds raw PyTorch CPU FP32 outputs of unmodified upstream Laya. The shipped
parity goldens (`laya_apple/parity/goldens/`) are derived from them.

## laya-0.3.20

`raw/laya-0.3.20/<model>.json` was recorded with `scripts/make_reference.py`:

- upstream `laya==0.3.20` (NandhaKishorM/laya@23a1752), torch 2.7.0 and transformers 5.17.0,
  from the `[reference]` extra;
- the pinned checkpoint revisions in `laya_apple/registry.py`, loaded offline from the
  HF cache (`HF_HUB_OFFLINE=1`);
- CPU, FP32, no autocast (the script refuses to run if upstream selects anything else);
- Apple M4 Max, macOS 26.6.2, Python 3.12.14 (`environment` in each file).

Upstream builds each case's prompt with the same code path as `Agent.system_one`: it
validates the questions, normalises them and tokenizes the state once. The script then runs
one forward pass over all of the case's rows and records:

- the prompt items;
- the unrounded decision logits and action logits;
- `Agent.system_one`'s public result;
- `tokenizer_match`: whether `laya_apple.prompt.prepare` reproduces upstream's items.

### Cases

- **Every Phase -1 fixture,** read back from `research/phase-0-feasibility/raw/reference/`.
  That record was made with upstream 0.3.5. For each case, `phase0_identical` records whether
  the items, decision logits and action logits under 0.3.20 are identical to it.
  `scripts/make_goldens.py` refuses to derive goldens when any is false. In this recording
  all 93 are identical (29 + 32 + 32), and `tokenizer_match` holds for every case.
- **Five `drift-*` cases,** written for this project. Each exercises one upstream change
  after 0.3.5:

  | Case | Upstream change |
  |---|---|
  | `drift-conversation-left-truncated` | a conversation list over `max_len` keeps its newest turns (e0557cb, #224) |
  | `drift-structured-instructions-non-ascii` | non-string instructions use `json.dumps(..., ensure_ascii=False)` (8dacc39, #228) |
  | `drift-noul-labels` | noul `labels` replace the `false:` / `true:` prefixes (158398a, #163) |
  | `drift-noul-criteria-key-case` | noul criteria keys are lowercased (7913866, #146) |
  | `drift-long-option-text` | option text is capped at 48 tokens by the tokenizer, not by a slice (34d27df, #109) |

  laya-apple's 0.3.5 prompt code does not reproduce the first four cases' token ids. The
  fifth case confirms that the tokenizer cap and the slice give the same ids. Its option
  text is 50 to 53 tokens long, depending on the tokenizer.

The Phase -1 raw record is not modified. This is a new measurement next to it.
