---
title: "Validate another macOS / coremltools version"
labels: ["help wanted", "coreml", "hardware", "correctness"]
---

## Problem

The shipped routing table and the release validation come from one profile: macOS 26.6.2
with coremltools 9.0 (`docs/support-matrix.md`). Two community hardware results have since
run laya-apple's own build-and-parity pipeline on other macOS versions, both with
coremltools 9.0 and `laya-typed-decisions` only:

- macOS 27.0 on an M4 Pro (#32): ANE artifacts built, placement and parity passed. Prior
  third-party work had observed different Core ML placement on macOS 27 (enumerated shapes
  running on the GPU instead of the ANE); this one machine did not reproduce it.
- macOS 26.2 on an M4 (#41): ANE artifacts built, placement and parity passed.

No one has run the pipeline against a coremltools release other than 9.0, so it is still
unknown whether the same BC1S graph places 100% on the ANE after a Core ML compiler
version change. More macOS 27 results, on other SoCs and with the other two models, are
also wanted.

## Why it matters

`profiles.py`'s docstring says the shipped routing table is valid only for the profile
it was measured on, and a mismatched profile falls back to MLX-only
(`platform_not_validated`, `docs/no-silent-fallback.md` row 18). Knowing what actually
happens on the next macOS/coremltools combination — rather than just falling back safely
— tells maintainers whether the BC1S rewrite is robust to a Core ML compiler version
change or whether it needs its own follow-up fix.

## Expected output

A report of what `laya-apple artifacts build` and `laya-apple parity --device ane`
produce on a coremltools release other than 9.0, or on a macOS 27 profile not covered above:
compute plan (ANE % and transition count), parity pass/fail, and the placement probe
ratio. Include exact `sw_vers` and `pip show coremltools` output. If it fails, describe
the failure mode (build error, wrong device placement, parity regression) — that is a
complete and useful result on its own.

## How to validate

1. `uv sync --extra dev --extra ane --extra convert` on the target macOS/coremltools combination.
2. `laya-apple artifacts build laya-typed-decisions`.
3. `laya-apple parity laya-typed-decisions --device ane`.
4. `laya-apple info` to capture `platform_profile()` and `platform_validated_for_auto_ane`.
5. Write up pass/fail with the raw compute-plan and parity output attached.

## Relevant files

- `docs/support-matrix.md` ("Platform scope")
- `laya_apple/artifacts.py` (`platform_profile`, `profile_matches`)
- `laya_apple/profiles.py`
- `docs/no-silent-fallback.md` (row 18, unvalidated platform)

## Difficulty / scope

Needs access to a Mac running a macOS version or coremltools release other than the
tested combination, plus the `ane`/`convert` extras. Investigative; a documented failure
is as valuable as a pass.
