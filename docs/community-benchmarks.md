# Community benchmarks: the Apple Silicon matrix

Every number laya-apple ships was measured on one machine, an Apple M4 Max
([`compatibility.md`](compatibility.md)). This page collects results from other Macs,
each one produced by the same script and submitted as a pull request, so the matrix
fills in from measurements rather than expectations.

## Priority contribution slots

These Macs are the current priority, and each has a `good first issue` with
copy-and-paste steps from `git clone` to the pull request. You can add a result in
10–25 minutes without changing any code. The [matrix](#matrix) below records what has
been measured; any Mac without a result is wanted, with or without an issue.

| Your Mac | Contribute | What you run |
|---|---|---|
| M1 or M2, any variant | [#1: M1/M2 MLX-only result](https://github.com/tc3oliver/laya-apple/issues/1) | MLX only, about 10 minutes |
| M3 Max | [#2: M3 Max result](https://github.com/tc3oliver/laya-apple/issues/2) | MLX, plus ANE if you have 15 more minutes |
| Any other Apple silicon Mac, including M5 and newer | No issue needed: follow [Add your Mac](#add-your-mac) | Same steps |

## Matrix

| SoC | MLX | ANE | Auto uses ANE | Heterogeneous | Evidence |
|---|---|---|---|---|---|
| M1 | – | – | – | – | **Wanted** — [submit a benchmark](#add-your-mac) |
| M1 Pro | – | – | – | – | **Wanted** — [submit a benchmark](#add-your-mac) |
| M1 Max | – | – | – | – | **Wanted** — [submit a benchmark](#add-your-mac) |
| M1 Ultra | – | – | – | – | **Wanted** — [submit a benchmark](#add-your-mac) |
| M2 | – | – | – | – | **Wanted** — [submit a benchmark](#add-your-mac) |
| M2 Pro | – | – | – | – | **Wanted** — [submit a benchmark](#add-your-mac) |
| M2 Max | – | – | – | – | **Wanted** — [submit a benchmark](#add-your-mac) |
| M2 Ultra | – | – | – | – | **Wanted** — [submit a benchmark](#add-your-mac) |
| M3 | – | – | – | – | **Wanted** — [submit a benchmark](#add-your-mac) |
| M3 Pro | – | – | – | – | **Wanted** — [submit a benchmark](#add-your-mac) |
| M3 Max | – | – | – | – | **Wanted** — [submit a benchmark](#add-your-mac) |
| M3 Ultra | – | – | – | – | **Wanted** — [submit a benchmark](#add-your-mac) |
| M4 | – | – | – | – | **Wanted** — [submit a benchmark](#add-your-mac) |
| M4 Pro (48 GB, macOS 27.0) | ✓ | ✓ | yes, local calibrated profile | ✓ | [bundle](../hardware-results/apple-m4-pro-macos27/summary.md) ([#32](https://github.com/tc3oliver/laya-apple/pull/32); `--quick`, laya-typed-decisions only) |
| M4 Max (macOS 26.6.2) | ✓ | ✓ | yes | ✓ | [`benchmarks/v1.0.md`](../benchmarks/v1.0.md), [bundle](../hardware-results/apple-m4-max-macos26/summary.md) |
| M5 | – | – | – | – | **Wanted** — [submit a benchmark](#add-your-mac) |
| M5 Pro | – | – | – | – | **Wanted** — [submit a benchmark](#add-your-mac) |
| M5 Max | – | – | – | – | **Wanted** — [submit a benchmark](#add-your-mac) |
| M5 Ultra | – | – | – | – | **Wanted** — [submit a benchmark](#add-your-mac) |
| Any other or newer Apple silicon | – | – | – | – | **Wanted** — [submit a benchmark](#add-your-mac) |

What each column means:

- **MLX**: MLX FP16 passes the parity gate against the shipped PyTorch FP32 goldens for
  every model measured.
- **ANE**: the ANE artifacts built on that machine pass the same gate. `untested` means
  no artifacts were built there.
- **Auto uses ANE**: `device="auto"` sends at least one request to the ANE. On a profile
  that is not shipped and not calibrated, the answer is `no` with `routing_reason`
  `platform_not_validated`. That is the runtime working as designed, not a failure.
  `yes, local calibrated profile` means the submitter ran `laya-apple calibrate` on
  that Mac and `auto` used the resulting local profile. The shipped routing table does
  not include that profile, so on another Mac with the same SoC `auto` stays on MLX
  until that Mac builds its own ANE artifacts and runs `laya-apple calibrate`
  ([`compatibility.md`](compatibility.md), "What happens on an untested profile").
- **Heterogeneous**: a short closed-loop mix of short and long requests has zero answer
  mismatches and more aggregate throughput with GPU + ANE than GPU alone.

`–` with **Wanted** means nobody has submitted a result yet. It says nothing about
whether that Mac works.

A filled row records one submitted run on one machine, with the memory, macOS version
and scope shown in the row. It is not a claim about every configuration of that SoC.

## Add your Mac

You need a Mac with Apple silicon, `git`, [uv](https://docs.astral.sh/uv/)
(`brew install uv`) and about 2 GB of free disk, or 5 GB with the optional ANE step. An
MLX-only result takes about 10 minutes.

1. Clone the repository, install it and download the pinned checkpoint (800 MB, hash
   verified):

   ```bash
   git clone https://github.com/tc3oliver/laya-apple
   cd laya-apple
   uv sync
   uv run laya-apple download laya-typed-decisions
   ```

2. Optional: build the ANE artifacts on your machine.

   ```bash
   uv sync --extra ane --extra convert
   uv run laya-apple artifacts build laya-typed-decisions
   ```

   This converts the model for each ANE bucket and registers an artifact only if it
   passes the placement and parity gates on your Mac. It takes about 5–10 minutes. You
   can skip it: results for MLX alone are useful too, and the report then records the
   ANE column as `untested` together with this command.

3. Run the report:

   ```bash
   uv run python scripts/hardware_report.py --quick
   ```

   The script records your SoC, memory, macOS version and build, the package versions
   and the laya-apple revision on its own. Then it runs MLX parity, ANE parity when
   artifacts exist, warm latency, the routing decisions of `device="auto"`, and a short
   heterogeneous mix when `auto` routes to the ANE. `--quick` measures
   `laya-typed-decisions` with fewer iterations. Without `--quick` it measures all
   three models with more iterations, so download them all first
   (`uv run laya-apple download`) or pass `--models laya-typed-decisions`.

4. Check the new directory before you share it:

   ```bash
   NEW=$(git ls-files --others --exclude-standard hardware-results/ | xargs -n1 dirname | sort -u)
   echo "$NEW"                    # exactly one new directory
   head -20 "$NEW/summary.md"     # the matrix row and your environment
   grep -rn -e "$USER" -e "$(hostname -s)" "$NEW" || echo "OK: no user name or host name in the bundle"
   ```

   The script records no host name or serial number, and it replaces your home
   directory with `~` and the checkout path with `.`.

5. Open a pull request that adds the directory the script prints,
   `hardware-results/<soc>-macos<major>/`, unchanged. With the
   [GitHub CLI](https://cli.github.com/):

   ```bash
   gh repo fork --remote              # your fork becomes origin, this repo becomes upstream
   git switch -c bench/add-my-mac
   git add hardware-results/
   git commit -m "bench(hardware): add <SoC> benchmark result"
   git push -u origin HEAD
   gh pr create --fill
   ```

   Replace `<SoC>` with your chip, for example `M1 Pro` or `M2 Max`. `gh pr create --fill`
   uses the commit subject as the pull request title, for example
   `bench(hardware): add M3 Max benchmark result`. A different title is fine; the
   maintainer can adjust it when merging.

   Paste the matrix row the script printed into the description, and name the issue it
   closes if there is one ([#1](https://github.com/tc3oliver/laya-apple/issues/1),
   [#2](https://github.com/tc3oliver/laya-apple/issues/2),
   [#3](https://github.com/tc3oliver/laya-apple/issues/3)).

## What reviewers check

- **The bundle is complete.** It has `bundle.json` and `summary.md`, every requested
  model has a section, and any step that did not run says why (`unavailable` with the
  build command, or `skipped` with a reason). Errors stay in the bundle; they are
  results too.
- **Parity.** The MLX and ANE parity summaries are the verdicts of the same gate
  `laya-apple parity` applies, with the per-row maxima recorded. A ✓ needs `passed:
  true` for every model measured.
- **No manual edits.** `summary.md` must render from `bundle.json`, the latency records
  must keep their raw samples, and the environment must match the directory name.
  Reviewers regenerate it with
  `uv run python scripts/hardware_report.py --render hardware-results/<dir>/bundle.json`
  and compare. Do not edit either file. If something looks wrong, say so in the pull
  request instead.
- **Provenance.** The laya-apple revision is a published commit, and the dirty flag is
  false or explained.

## How a result changes the matrix

A merged bundle fills in that SoC's row from its `summary.md`. If two bundles for the
same SoC disagree, the row shows both until the reason is understood.

A result does **not** change what `device="auto"` does on anyone's machine. The shipped
routing table (`laya_apple/data/routing.json`) applies only to the profiles it was
derived on, and a validated ANE result on another profile does not add that profile to
it. On your own machine, `laya-apple calibrate` writes a local profile that lets `auto`
use the ANE there ([`compatibility.md`](compatibility.md), "What happens on an untested
profile"). Adding a new shipped profile is a separate change that re-derives the table
from committed measurements. It is not done through this matrix.

The bundle format is described in [`hardware-results/README.md`](../hardware-results/README.md).
