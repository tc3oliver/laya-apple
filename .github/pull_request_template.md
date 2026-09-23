<!-- Title format: <type>(<scope>): <imperative summary>, for example
     bench(hardware): add M3 Max benchmark result. See CONTRIBUTING.md, "Pull request titles". -->

## What and why

<!-- What changed, and why it was wrong or missing before. -->

## Checklist

- [ ] Fast test suite passes (`uv run pytest -q -m "not integration and not parity and not ane and not stress"`)
- [ ] `uv run ruff check .` and `uv run ruff format --check laya_apple scripts tests` pass
- [ ] Parity tolerances are unchanged (or the change is specifically about parity, with why explained above)
- [ ] If this touches a device/bucket/precision decision or an `except` around backend selection, `docs/no-silent-fallback.md` has a new row and a test
- [ ] `CHANGELOG.md` has an entry under `[Unreleased]` for any user-visible change
- [ ] No generated Core ML artifacts or model weights are committed
