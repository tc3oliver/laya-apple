"""Stage this machine's validated ANE artifacts for the prebuilt-artifact repository.

Writes, under --out, the layout `laya-apple artifacts fetch` reads (laya_apple/prebuilt.py):

    index.json
    <model>/<revision[:12]>/<profile-slug>/<model>-L<bucket>.tar.gz

Each archive is `laya-apple artifacts export` output, so it is exported only after it
verifies here (file hash, compute plan). An existing --out/index.json is merged: entries for
other profiles, models or revisions are kept, and the same path is replaced. This script
never uploads and needs no Hugging Face credentials; it prints the upload command for the
maintainer. A receiving machine re-validates every archive (docs/guide.md, "Prebuilt
artifacts").

    LAYA_APPLE_CACHE=<cache> HF_HUB_OFFLINE=1 \\
        uv run python scripts/publish_prebuilt.py --out <staging> [MODEL ...] [--length L ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="*", help="default: every supported model")
    p.add_argument("--length", type=int, action="append", help="default: every offered bucket")
    p.add_argument("--out", required=True, help="staging directory (mirrors the repository)")
    p.add_argument("--repo", default=None, help="repository id to print in the upload command")
    a = p.parse_args(argv)

    from laya_apple.artifacts import artifact_dir, platform_profile, read_manifest
    from laya_apple.lifecycle import export_artifact
    from laya_apple.prebuilt import (
        DEFAULT_PREBUILT_REPO,
        INDEX,
        archive_path,
        make_index_entry,
        merge_index,
        validate_index,
    )
    from laya_apple.registry import models, resolve

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index_file = out / INDEX
    index = validate_index(json.loads(index_file.read_text())) if index_file.exists() else None
    profile = platform_profile()
    if not profile.get("coremltools"):
        raise SystemExit("coremltools is not importable: install the [ane] extra on the build machine")
    specs = [resolve(m) for m in a.models] if a.models else list(models().values())
    entries = []
    for spec in specs:
        for bucket in a.length or spec.ane_buckets:
            if bucket not in spec.ane_buckets:
                raise SystemExit(f"{spec.name} does not offer L{bucket}; offered: {list(spec.ane_buckets)}")
            manifest = read_manifest(spec, bucket)
            if manifest.get("status") != "validated":
                raise SystemExit(f"{artifact_dir(spec, bucket)} is not validated ({manifest.get('status')!r})")
            rel = archive_path(spec, bucket, manifest["platform"])
            dest = out / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            written = export_artifact(spec, bucket, dest.with_name(dest.name[: -len(".tar.gz")]))
            entry = make_index_entry(manifest, written, rel)
            entries.append(entry)
            print(f"{rel}  {entry['bytes'] / 1e6:.0f} MB  sha256 {entry['archive_sha256'][:16]}…", flush=True)
    index_file.write_text(json.dumps(merge_index(index, entries), indent=1) + "\n")
    repo = a.repo or DEFAULT_PREBUILT_REPO
    print(f"\nwrote {len(entries)} archives and {index_file}")
    print("Upload (maintainer, logged in with `hf auth login`):")
    print(f"  hf upload {repo} {out} . --repo-type model --commit-message 'Add prebuilt ANE artifacts'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
