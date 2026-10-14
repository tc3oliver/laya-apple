"""Generate the labelled dev (annotation) and eval (measurement-only) sets.

Every request is synthesised from the templates below with a fixed seed; nothing is copied
from a third-party prompt or benchmark. dev and eval draw slot values from the same pools but
use different seeds, and eval adds templates dev never saw (confusable and OOD slices).
"""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

LABELS = ["lint-fix", "small-bugfix", "refactor", "docs", "needs-human", "needs-more-info"]
POOLS = {
    "file": ["src/api/client.py", "lib/auth/session.ts", "app/models/order.rb", "pkg/cache/lru.go",
             "src/utils/dates.js", "internal/billing/invoice.go", "web/components/Cart.tsx",
             "tools/export_report.py", "core/scheduler.rs", "src/parser/tokens.py"],
    "dir": ["src/", "lib/", "app/", "pkg/", "web/components/", "tools/"],
    "tool": ["ruff", "eslint", "prettier", "mypy", "clippy", "golangci-lint", "rubocop"],
    "rule": ["E501", "no-unused-vars", "F401", "W291", "unused_must_use", "no-shadow", "D100", "S1192"],
    "func": ["parse_date", "load_config", "merge_orders", "render_cart", "retry_request", "flush_buffer",
             "compute_total", "normalize_path", "refresh_token_cache", "split_batches"],
    "func2": ["parse_timestamp", "read_settings", "combine_orders", "draw_cart", "send_with_retry", "drain_buffer"],
    "err": ["a KeyError", "an IndexError", "a TypeError", "a NullPointerException", "a ZeroDivisionError"],
    "cond": ["the list is empty", "the input has a trailing slash", "the user has no email set",
             "the timezone is UTC+13", "the cache is cold", "the value is negative"],
    "test": ["test_parse_empty", "test_total_rounding", "test_retry_backoff", "CartTest.testEmpty", "test_merge_dupes"],
    "field": ["user.email", "order.items", "config.timeout", "payload.meta"],
    "flag": ["--dry-run", "--since", "--format json", "--no-cache"],
    "svc": ["billing-worker", "mailer", "search-indexer", "gateway", "report-service"],
    "table": ["orders", "users", "payments", "audit_log", "sessions"],
    "role": ["support agents", "contractors", "interns", "partner accounts"],
    "feature": ["export", "search", "checkout", "dark mode", "notifications"],
    "section": ["Installation", "Usage", "Troubleshooting", "FAQ"],
    "n": ["1800", "2400", "3100", "950"],
    "cls": ["ReportBuilder", "OrderManager", "SessionStore", "EventBus"],
    "line": ["12", "88", "141", "307"],
}
NOISE = [
    "Traceback (most recent call last):\n  File \"{file}\", line {line}, in {func}\n    return handler(ctx)\n"
    "RuntimeError: unexpected state while processing batch 17",
    "$ git log --oneline -5\n3f2a1c9 wip\n91be0d4 update deps\nc02d7aa add retry\n88e1f03 tweak CI\nd41a9b2 init",
    "diff --git a/{file} b/{file}\n@@ -{line},7 +{line},9 @@\n-    items = fetch()\n+    items = fetch() or []\n"
    "     for it in items:\n         process(it)",
    "CI log: step 'test' started 10:02:11 ... 214 passed, 3 skipped in 41.2s ... step 'package' ok",
    "Background: the team shipped v2.4 last week; the on-call rotation changes on Monday; "
    "the staging cluster was resized and the nightly job now takes 40 minutes.",
]

# (label, template). DEV = used to design rules / prompts / thresholds.
DEV = [
    ("lint-fix", "Fix the {tool} warning `{rule}` in {file}."),
    ("lint-fix", "CI is red on {tool}: {rule} at {file}:{line}. Make it pass."),
    ("lint-fix", "Run the formatter over {dir} and commit the whitespace-only changes."),
    ("lint-fix", "Remove the unused imports {tool} reports in {file}."),
    ("small-bugfix", "{func} in {file} throws {err} when {cond}. Fix it."),
    ("small-bugfix", "Off-by-one in {func}: the last item is skipped. Failing test: {test}."),
    ("small-bugfix", "{test} fails since the date format became ISO; make {func} parse it."),
    ("small-bugfix", "Missing null check for {field} in {func} causes {err}."),
    ("refactor", "Extract the duplicated validation logic in {file} and {dir} into one shared helper."),
    ("refactor", "Rename {func} to {func2} across the codebase; behaviour must not change."),
    ("refactor", "Split {file} ({n} lines) into smaller modules without changing the public API."),
    ("refactor", "Replace the nested callbacks in {func} with async/await."),
    ("docs", "Update the README to describe the new {flag} flag."),
    ("docs", "Add docstrings to the public functions in {file}."),
    ("docs", "Fix the typo in the {section} section of CONTRIBUTING.md."),
    ("docs", "Document the environment variables {svc} reads in docs/config.md."),
    ("needs-human", "Migrate the {table} table to a new schema in production with no downtime."),
    ("needs-human", "Rotate the credentials {svc} uses and roll the change out to every deployment."),
    ("needs-human", "Decide whether {svc} should move from REST to gRPC and plan the cutover."),
    ("needs-human", "Change the permission model so {role} can read billing data."),
    ("needs-more-info", "It's broken, please fix."),
    ("needs-more-info", "Make it better."),
    ("needs-more-info", "The {feature} isn't working like before."),
    ("needs-more-info", "Look at the failing build."),
    ("needs-more-info", "Something is off with {svc}."),
    ("needs-more-info", "Please take a look at {func}."),
    ("needs-more-info", "Handle the {table} thing."),
    ("needs-more-info", "{file} is weird, can you check?"),
    ("needs-more-info", "Fix the {feature} problem."),
]
DEV_CONFUSABLE = [
    ("lint-fix", "Fix the markdown lint errors in README.md."),
    ("docs", "The example output in docs/usage.md no longer matches what {func} prints; update the docs."),
    ("small-bugfix", "Fix the typo in the error message of {func}; {test} asserts the old text."),
    ("needs-human", "Fix the SQL injection in {func} in {file}."),
    ("refactor", "Add type hints to {func} in {file}."),
]
# Eval-only: new confusable templates (cross-class cues) and OOD phrasing (terse, non-English, odd framing).
EVAL_CONFUSABLE = DEV_CONFUSABLE + [
    ("lint-fix", "Rename the variable that {tool} flags as `{rule}` in {file}."),
    ("lint-fix", "Delete the dead code in {file} that {tool} reports as unused."),
    ("small-bugfix", "Refactor {func} so it stops raising {err} when {cond}."),
    ("small-bugfix", "Rounding bug: {func} returns 0.30000000000000004; {test} expects 0.3."),
    ("refactor", "Fix the misspelled name of {func} everywhere it is used."),
    ("refactor", "Convert {cls} from inheritance to composition; tests must stay green."),
    ("docs", "Fix the failing doctest in {file} by correcting the expected output in the comment."),
    ("docs", "Rewrite the comments in {file} so they explain why, not what."),
    ("needs-human", "The auth middleware lets expired tokens through; patch it."),
    ("needs-human", "Delete the old {table} rows from the production database."),
    ("needs-human", "Rewrite the session handling in {file} to be cleaner and safer."),
    ("needs-more-info", "Clean up {file}, it has gotten messy."),
    ("needs-more-info", "Fix the {feature} bug we discussed."),
    ("needs-more-info", "Fix the test."),
    ("needs-more-info", "Can you check {svc}? Something is up."),
    ("needs-more-info", "Why is {func} like that?"),
    ("needs-more-info", "Update the config."),
]
EVAL_OOD = [
    ("lint-fix", "pre-commit rejects my push: trailing whitespace and missing EOL in {file}."),
    ("lint-fix", "{tool} 報了 {rule},幫我修掉。"),
    ("lint-fix", "isort complains about import order in {file}, tidy it."),
    ("small-bugfix", "users say the {feature} button does nothing on the second click, stale closure in {file} I think."),
    ("small-bugfix", "{func} 在 {cond} 時會出錯,{test} 可以重現。"),
    ("small-bugfix", "hotfix: {func} divides by zero if the batch is empty."),
    ("refactor", "tidy up {cls}: pull the retry loop out into its own function."),
    ("refactor", "把 {file} 裡重複的程式碼抽成共用函式。"),
    ("docs", "write a changelog entry for the new {feature} behaviour."),
    ("docs", "補上 {file} 的使用說明。"),
    ("docs", "add an architecture diagram note to the {section} section."),
    ("needs-human", "turn on the {feature} flag for all users in prod."),
    ("needs-human", "請決定是否把 {svc} 的資料庫換成別的引擎。"),
    ("needs-human", "refund every customer charged twice last month."),
    ("needs-more-info", "ok do the thing"),
    ("needs-more-info", "還是不行,再看一下。"),
    ("needs-more-info", "same issue as before"),
]


def fill(rng, tpl):
    out = tpl
    for k, vals in POOLS.items():
        while "{" + k + "}" in out:
            out = out.replace("{" + k + "}", rng.choice(vals), 1)
    return out


def longify(rng, text):
    pre = "\n\n".join(fill(rng, rng.choice(NOISE)) for _ in range(rng.randint(2, 4)))
    post = fill(rng, rng.choice(NOISE))
    return f"{pre}\n\nRequest: {text}\n\n{post}"


def build(seed, plan, exclude=()):
    """Round-robin over each slice's templates; redraw slots until the text is new."""
    rng = random.Random(seed)
    rows, seen = [], set(exclude)
    for slice_, pool, per in plan:
        n = tries = 0
        while n < per:
            tries += 1
            assert tries < per * 100, f"{slice_}: templates exhausted at {n}/{per}"
            label, tpl = rng.choice(pool) if slice_ == "long" else pool[tries % len(pool)]
            text = fill(rng, tpl)
            if slice_ == "long":
                text = longify(rng, text)
            if text in seen:
                continue
            seen.add(text)
            rows.append({"id": f"{slice_}-{n:03d}", "slice": slice_, "label": label, "text": text})
            n += 1
    return rows


def main():
    d = Path(__file__).parent / "data"
    abstain = [t for t in DEV if t[0] == "needs-more-info"]
    ident = [t for t in DEV if t[0] != "needs-more-info"]
    dev = build(1001, [("id", ident, 120), ("confusable", list(DEV_CONFUSABLE), 30),
                       ("abstain", abstain, 24), ("long", ident, 12)])
    ev_abstain = [t for t in EVAL_CONFUSABLE + EVAL_OOD if t[0] == "needs-more-info"]
    ev = build(2002, [("id", ident, 120), ("confusable", list(EVAL_CONFUSABLE), 80),
                      ("ood", [t for t in EVAL_OOD if t[0] != "needs-more-info"], 60),
                      ("abstain", abstain + ev_abstain, 34), ("long", ident, 50)],
               exclude={r["text"] for r in dev})
    assert not {r["text"] for r in dev} & {r["text"] for r in ev}, "dev/eval overlap"
    for name, rows in (("dev", dev), ("eval", ev)):
        (d / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    lines = ["# Split", "", "Generated by `gen_data.py`; seeds fixed (dev 1001, eval 2002).", ""]
    for name, rows in (("dev", dev), ("eval", ev)):
        h = hashlib.sha256((d / f"{name}.jsonl").read_bytes()).hexdigest()[:16]
        lines += [f"## {name} ({len(rows)} rows, sha256 {h})", "",
                  "label counts: " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(r["label"] for r in rows).items())),
                  "slice counts: " + ", ".join(f"{k}={v}" for k, v in sorted(Counter(r["slice"] for r in rows).items())), ""]
    (d / "SPLIT.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
