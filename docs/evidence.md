# Raw evidence and the size cap

laya-apple keeps its raw measurements so that every number can be re-derived. Raw traces are
also the one thing in this repository that grows without bound, and Git keeps every version of
every file forever, so each clone and fetch pays for all of it. This page is the rule that
keeps the evidence and stops the growth.

## The rule

- **No tracked file may be larger than 1 MiB (1,048,576 bytes).** This applies to the whole
  tree, and it matters most under `research/` and `benchmarks/`, where raw per-request traces,
  profiler output and datasets land. A file of exactly 1 MiB is allowed.
- **Small evidence stays in the repository:** reports, tables, summaries and reduced results
  (`results.json`, `tables.md`), the scripts that produce them, environment and machine
  descriptions, manifests (below), and any raw file within the cap.
- **Larger raw data goes to external evidence:** an immutable location, referenced from an
  in-repo manifest that gives its URL, SHA-256 and size. Compress it first. A `.json.gz` is
  the repository's existing convention, and the cap applies to the committed file.
- **A report names where its numbers come from.** For external data it names the manifest
  entry (path and SHA-256), the same way it names a committed file today. The headline
  numbers and the reduced results stay in the repository, so a report reads and its claims
  check without downloading anything. The raw data is needed only to re-derive them.
- **A total is a review rule.** The cap is per file, so it does not bound a campaign made of
  many files just under it. A pull request that adds more than 5 MiB of raw data in total
  publishes the bulk externally even if each file is under the cap.

`scripts/check_file_sizes.py` enforces the per-file cap in CI (the `test` job). The 5 MiB total
is checked by the reviewer.

## Why 1 MiB

Measured on `main` at `25c4bb3`, the commit the allowlist was generated from, using the size of
each file's blob, which is what the check uses (`git ls-files -s`, then
`git cat-file --batch-check`). Every percentile on this page is nearest-rank: the value at rank
⌈q·n⌉ of the sorted sizes, so each one is a size that a real file has.

| | Files | Bytes |
|---|---:|---:|
| Every tracked file | 1,455 | 290,245,586 |
| Under `research/` and `benchmarks/` | 1,175 | 266,462,661 |
| of which over 1 MiB | 58 | 124,234,420 |
| of which over 2 MiB | 14 | 59,811,704 |
| Over 1 MiB anywhere (`docs/media`: 6, `tests/fixtures`: 2, the rest above) | 66 | 139,718,211 |

- In `research/` and `benchmarks/` the median file is 9,041 bytes, the 90th percentile 888,676
  bytes (868 KiB), the 95th 1,041,473 bytes and the 99th 2,459,572 bytes. The largest is
  12,251,382 bytes.
- 1 MiB sits at the 95th percentile. 95.1% of the files are under it, and the 4.9% above it
  hold 46.6% of the bytes. The files that make the repository heavy are few and easy to name,
  and ordinary reports, tables and small raw files are untouched.
- 2 MiB would leave 44 files of 1 to 2 MiB uncapped. They hold 64,422,716 bytes, many of them
  gzipped raw run files, which is what a typical campaign produces.
- 512 KiB would put 174 existing `research/` and `benchmarks/` files over the cap and make
  the cap hit routine small files.
- The cap does not catch many small files. `research/coreml-staged-handoff` is 41,217,418
  bytes in 113 files, none over 1 MiB. Across the 35 research and benchmark directories the
  median total is 3.3 MiB, the 75th percentile 7.0 MiB, and 13 are over 5 MiB. That is where
  the 5 MiB review rule comes from.

## Publishing external evidence

Publish a campaign's large raw files once, then reference them from a manifest.

**Where.** Either of these, and nothing mutable:

- **A GitHub release on a dedicated evidence tag.** The tag is `evidence-<track>-<yyyymmdd>`.
  It must not start with `v`, because every `v*` tag publishes to PyPI
  ([`docs/publishing.md`](publishing.md)). Create the release as a pre-release that is not
  `latest`, so it never reads as a product release. One tag per campaign. Never move, delete
  or reuse a tag, and never replace an asset. A correction is a new tag and a new manifest
  entry.
- **A Hugging Face dataset revision.** The URL pins the full 40-character commit hash, never
  a branch such as `main`:
  `https://huggingface.co/datasets/<owner>/<name>/resolve/<commit>/<path>`.

Only a maintainer can create the release. If you are not a maintainer, say so in the pull
request: attach or link the file, state its SHA-256, and the maintainer publishes it and gives
you the final URL for the manifest before the merge.

For a maintainer:

```bash
shasum -a 256 raw/run-003/trace.jsonl.gz        # the SHA-256 for the manifest
gh release create evidence-<track>-<yyyymmdd> --prerelease --latest=false \
  --title "Evidence: <track>" --notes "Raw data for research/<track>. See its evidence.json." \
  raw/run-003/trace.jsonl.gz
```

The SHA-256 is what makes the reference safe. A release asset can be replaced by someone with
write access, so a downloaded file counts only if its hash matches the manifest.

## The manifest

Each directory that cites external evidence has an `evidence.json` beside its report. The
paths in it are relative to that directory.

```json
{
  "format": "laya-apple-evidence-manifest",
  "format_version": 1,
  "files": [
    {
      "path": "raw/run-003/trace.jsonl.gz",
      "url": "https://github.com/tc3oliver/laya-apple/releases/download/evidence-example-20261001/run-003-trace.jsonl.gz",
      "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "size": 4817231,
      "description": "Per-request trace, run 3"
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `format`, `format_version` | `laya-apple-evidence-manifest` and `1`. A change to the layout raises the version. |
| `files[].path` | Where the file belongs, relative to the manifest's directory. Fetching it there restores the layout the report and the analysis scripts expect. Do not also commit a file at this path. |
| `files[].url` | The immutable URL from the section above: a release asset on an `evidence-*` tag, or a Hugging Face URL pinned to a commit. |
| `files[].sha256` | Lower-case hex SHA-256 of the file exactly as published. For a `.gz` file, that is the compressed file. |
| `files[].size` | Size in bytes of the file as published. |
| `files[].description` | Optional, one line. |

The example's URL and hash are placeholders, not a published file.

To fetch and verify a file:

```bash
cd research/<track>
curl -fL --create-dirs -o raw/run-003/trace.jsonl.gz "<url from evidence.json>"
echo "<sha256 from evidence.json>  raw/run-003/trace.jsonl.gz" | shasum -a 256 -c -
```

Do not run the analysis on a file that fails the check.

## The check

```bash
uv run python scripts/check_file_sizes.py
```

It reads the blob size of every file in the Git index, and the allowlist from the index too
(the working tree copy only if the index has none), so it measures what a commit would add,
and a local run agrees with CI. It exits 1, and names each file, when:

- a tracked file is over 1 MiB and is not in
  [`scripts/file_size_allowlist.txt`](../scripts/file_size_allowlist.txt);
- a file in the allowlist has grown past the size recorded there;
- a line in the allowlist is stale: the file is no longer tracked or is now under the cap.

It exits 2, with a one-line error, when it could not run: git failed, or the allowlist has a
malformed line.

When it fails on your new file, in this order:

1. Compress it (`gzip`) if it is JSON or JSONL and is not already.
2. If it is still over 1 MiB, publish it as external evidence and commit the manifest entry
   instead of the file.
3. Do not add it to the allowlist. Never add research or benchmark data to the allowlist. Any
   other addition needs explicit maintainer agreement in the pull request. For a non-evidence
   file, such as a README demo video, shrink it first.

## Files already in the repository

The 66 files that were over the cap when it was introduced are listed in the allowlist, each
with its size on that day. They stay where they are, unchanged: raw benchmark and research data
is historical record and is never edited to change a result, and the published history is not
rewritten. The allowlist is the grandfather list and never grows for evidence data. Moving those
files out is a separate migration, reviewed on its own (see the discussion in
[#136](https://github.com/tc3oliver/laya-apple/issues/136)). If it happens, each file's line
leaves the allowlist in the same pull request, and a manifest takes the file's place.

Renaming or moving a grandfathered file is not free: the check sees the old path removed and a
new path over the cap, so the allowlist line must be updated to the new path in the same pull
request, with the explicit maintainer agreement that any other allowlist change needs.
