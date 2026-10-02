# Plan (written before the full calibration ran)

Recorded 2026-10-02, after the screen (`raw/screen-*`) and before pass 1.

- Procedure: `laya-apple --offline calibrate laya-typed-decisions laya laya-multilingual
  --warmup 10 --iters 300` (pass 1), then the same command with the model order reversed
  (pass 2). One process per pass; nothing else measured or tested while a pass runs.
- Rule: the unchanged `laya_apple.derivation.derive_model` (the rule behind the shipped
  table), applied by `calibrate` to each pass.
- Ship criterion: a profile ships only if, for every model, both passes give the same
  `auto_ane_buckets` and every offered bucket's artifact passed parity. The shipped data is
  pass 1's profile, unedited. If the passes disagree for any model, nothing ships and the
  disagreement is recorded.
