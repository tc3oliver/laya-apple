---
title: "Add an M1/M2 MLX-only benchmark result"
labels: ["good first issue", "help wanted", "benchmark", "hardware"]
---

> **Have this Mac? You can contribute useful benchmark data without changing any code.**

Every number laya-apple publishes comes from one Apple M4 Max. An M1 or M2 result turns
"MLX is expected to work on older Apple silicon" into a measured fact. This is the
easiest hardware contribution in the project.

| | |
|---|---|
| **Hardware needed** | Any M1 or M2 Mac: M1, M1 Pro, M1 Max, M1 Ultra, M2, M2 Pro, M2 Max or M2 Ultra |
| **Estimated time** | About 10 minutes, mostly installing packages and downloading an 800 MB checkpoint |
| **Code changes** | **None.** You add one generated directory under `hardware-results/`. |
| **Also needed** | `git`, [uv](https://docs.astral.sh/uv/) (`brew install uv`), about 2 GB of free disk |

## 1. Install (about 5 minutes)

```bash
git clone https://github.com/tc3oliver/laya-apple
cd laya-apple
uv sync
uv run laya-apple download laya-typed-decisions
```

`uv sync` installs the MLX-only base package. You do not need the `ane` or `convert`
extras for this issue. `download` fetches the pinned checkpoint and verifies its SHA-256.

## 2. Run the benchmark (1–3 minutes)

```bash
uv run python scripts/hardware_report.py --quick
```

When it finishes, it prints your matrix row and the directory it wrote, for example:

```
| Apple M1 Pro (<memory> GB, macOS <version>) | <MLX> | untested | no | untested |

wrote hardware-results/apple-m1-pro-macos<major>/bundle.json and hardware-results/apple-m1-pro-macos<major>/summary.md (<seconds> s)
```

`ANE: untested`, `Auto uses ANE: no` and `Heterogeneous: untested` are the **expected**
result of an MLX-only run. They are not failures. `MLX` is the column this issue fills in.

## 3. Check the result before you share it

```bash
NEW=$(git ls-files --others --exclude-standard hardware-results/ | xargs -n1 dirname | sort -u)
echo "$NEW"                    # exactly one new directory
head -20 "$NEW/summary.md"     # the matrix row and your environment
grep -rn -e "$USER" -e "$(hostname -s)" "$NEW" || echo "OK: no user name or host name in the bundle"
```

The script records no host name or serial number. It replaces your home directory with
`~` and the checkout path with `.`. Do not edit `bundle.json` or `summary.md`. If
something looks wrong, say so in the pull request.

## 4. Open the pull request

With the [GitHub CLI](https://cli.github.com/):

```bash
gh repo fork --remote              # your fork becomes origin, this repo becomes upstream
git switch -c bench/add-my-mac
git add hardware-results/
git commit -m "bench(hardware): add <SoC> benchmark result"
git push -u origin HEAD
gh pr create --fill
```

Replace `<SoC>` with your chip, for example `M1 Pro` or `M2 Max`. `gh pr create --fill`
uses the commit subject as the pull request title.

Without the GitHub CLI: fork the repository on GitHub, then run
`git remote add fork https://github.com/<you>/laya-apple`, the same `switch`, `add` and
`commit` commands, and `git push -u fork HEAD`. Open the pull request from the link that
`git push` prints.

In the pull request description, paste the matrix row the script printed and write
`Closes #1`.

## What the reviewer checks

- The PR adds only the new `hardware-results/<soc>-macos<major>/` directory, with
  `bundle.json` and `summary.md`.
- The files are unedited. The reviewer regenerates the summary with
  `uv run python scripts/hardware_report.py --render hardware-results/<dir>/bundle.json`
  and compares it.
- MLX parity: `passed` for the model measured. A failure is still a useful result, so
  submit it anyway.
- Provenance: the laya-apple revision is a published commit and the dirty flag is false.

Full details: [`docs/community-benchmarks.md`](https://github.com/tc3oliver/laya-apple/blob/main/docs/community-benchmarks.md#add-your-mac).
Questions are welcome as comments on this issue.
