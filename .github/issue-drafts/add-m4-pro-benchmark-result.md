---
title: "Add an M4 Pro benchmark result"
labels: ["good first issue", "help wanted", "benchmark", "hardware"]
---

> **Have this Mac? You can contribute useful benchmark data without changing any code.**

Every number laya-apple publishes comes from one Apple M4 Max. The M4 Pro has the same
Neural Engine generation but fewer GPU cores and less memory bandwidth, so where the GPU
and the ANE cross over has never been measured on it.

| | |
|---|---|
| **Hardware needed** | An M4 Pro Mac (Mac mini or MacBook Pro, any memory size) |
| **Estimated time** | About 10 minutes for an MLX-only result, about 25 minutes with the optional ANE step |
| **Code changes** | **None.** You add one generated directory under `hardware-results/`. |
| **Also needed** | `git`, [uv](https://docs.astral.sh/uv/) (`brew install uv`), about 5 GB of free disk with the ANE step |

## 1. Install (about 5 minutes)

```bash
git clone https://github.com/tc3oliver/laya-apple
cd laya-apple
uv sync
uv run laya-apple download laya-typed-decisions
```

`download` fetches the pinned 800 MB checkpoint and verifies its SHA-256.

## 2. Optional, and the most valuable part: build ANE artifacts (10–15 minutes)

```bash
uv sync --extra ane --extra convert
uv run laya-apple artifacts build laya-typed-decisions
```

This converts the model for each Neural Engine bucket and keeps an artifact only if it
passes the placement and parity gates on your Mac. A rejected artifact is a result too:
the report records why. If you skip this step, the report records the ANE column as
`untested`, and an MLX-only result is still useful.

## 3. Run the benchmark (1–3 minutes)

```bash
uv run python scripts/hardware_report.py --quick
```

When it finishes, it prints your matrix row and the directory it wrote, for example:

```
| Apple M4 Pro (<memory> GB, macOS <version>) | <MLX> | <ANE> | no | untested |

wrote hardware-results/apple-m4-pro-macos<major>/bundle.json and hardware-results/apple-m4-pro-macos<major>/summary.md (<seconds> s)
```

`Auto uses ANE: no` and `Heterogeneous: untested` are expected: `device="auto"` stays on
the GPU on a Mac whose routing profile is neither shipped nor calibrated
(`platform_not_validated`). That is the runtime working as designed, not a failure.

If you built ANE artifacts and have a few more minutes, also run
`uv run laya-apple calibrate laya-typed-decisions` **after** the report and paste its
output into the pull request. It shows whether this machine gets its own local routing
profile.

## 4. Check the result before you share it

```bash
NEW=$(git ls-files --others --exclude-standard hardware-results/ | xargs -n1 dirname | sort -u)
echo "$NEW"                    # exactly one new directory
head -20 "$NEW/summary.md"     # the matrix row and your environment
grep -rn -e "$USER" -e "$(hostname -s)" "$NEW" || echo "OK: no user name or host name in the bundle"
```

The script records no host name or serial number. It replaces your home directory with
`~` and the checkout path with `.`. Do not edit `bundle.json` or `summary.md`. If
something looks wrong, say so in the pull request.

## 5. Open the pull request

With the [GitHub CLI](https://cli.github.com/):

```bash
gh repo fork --remote              # your fork becomes origin, this repo becomes upstream
git switch -c bench/add-my-mac
git add hardware-results/
git commit -m "Add hardware report for my Mac"
git push -u origin HEAD
gh pr create --fill
```

Without the GitHub CLI: fork the repository on GitHub, then run
`git remote add fork https://github.com/<you>/laya-apple`, the same `switch`, `add` and
`commit` commands, and `git push -u fork HEAD`. Open the pull request from the link that
`git push` prints.

In the pull request description, paste the matrix row the script printed and write
`Closes #3`.

## What the reviewer checks

- The PR adds only the new `hardware-results/<soc>-macos<major>/` directory, with
  `bundle.json` and `summary.md`.
- The files are unedited. The reviewer regenerates the summary with
  `uv run python scripts/hardware_report.py --render hardware-results/<dir>/bundle.json`
  and compares it.
- Parity: MLX, and ANE if you built artifacts, shows `passed`. A failure is still a
  useful result, so submit it anyway.
- Provenance: the laya-apple revision is a published commit and the dirty flag is false.

Full details: [`docs/community-benchmarks.md`](https://github.com/tc3oliver/laya-apple/blob/main/docs/community-benchmarks.md#add-your-mac).
Questions are welcome as comments on this issue.
