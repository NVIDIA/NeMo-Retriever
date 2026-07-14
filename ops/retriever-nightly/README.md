<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Retriever Nightly Launcher

This directory provides a manual launcher and a latest-main controller. Both
run the library and ViDoRe v3 benchmark suite through the portable harness
interface. The manual launcher never changes Git state; the controller fetches
and manages immutable detached worktrees. Neither tool installs a scheduler or
distributes datasets.

## One-Command Full Run

Export the two supported secrets and invoke the launcher from a clean checkout:

```bash
export HF_TOKEN=...
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
./ops/retriever-nightly/run-nightly.sh
```

That foreground command runs all twelve benchmarks and posts one terminal
Slack summary. `SLACK_WEBHOOK_URL` is optional; omit it to run without posting.
The launcher writes artifacts outside the checkout and prints the terminal
session directory. No `sudo`, systemd service, or configuration file is
required.

## Manual Teammate Kickoff

### Prerequisites

The supported v1 host is a Linux NVIDIA workstation with:

- a clean NeMo Retriever Git checkout;
- `git`, Bash, `uv`, `flock`, `realpath`, and NVIDIA drivers available;
- access to the twelve benchmark datasets through `/datasets` or other local
  paths;
- a Hugging Face read token in `HF_TOKEN` plus outbound HTTPS access to `huggingface.co`,
  `cas-server.xethub.hf.co`, and `cas-bridge.xethub.hf.co` for ViDoRe
  queries, qrels, and corpus metadata; and
- enough system RAM, local model cache, and artifact storage for the selected
  runfiles. The complete batch suite is not validated on 128 GiB hosts.

`uv` may be set with `RETRIEVER_UV_BIN`, discovered from `PATH`, or installed at
`$HOME/.local/bin/uv`. The locked `nemo_retriever` project selects Python 3.12
and the repository dependencies.

Batch mode starts its models locally, so it needs no model-provider API keys.
On a host with the standard `/datasets/nv-ingest` layout, the environment or
optional `nightly.env` accepts only two secrets and one optional path override:

| Setting | Required | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | yes | Read-only access to ViDoRe evaluation data. |
| `SLACK_WEBHOOK_URL` | no | Enables one terminal Slack post for real runs. |
| `RETRIEVER_DATASET_PATHS` | nonstandard hosts only | Replaces the checked-in `/datasets/nv-ingest` map. |

On hosts with a writable `/raid/$USER`, the launcher automatically keeps its
private configuration, artifacts, and managed latest-main checkouts there.
Other hosts use `$HOME`.

Direct exports are the smallest configuration interface. A private file is
optional for operators who do not want to export the same values in every
shell. To create it:

```bash
if [[ -d /raid/$USER && -w /raid/$USER ]]; then
  RETRIEVER_NIGHTLY_ROOT=/raid/$USER
else
  RETRIEVER_NIGHTLY_ROOT=$HOME
fi
RETRIEVER_NIGHTLY_CONFIG_DIR="$RETRIEVER_NIGHTLY_ROOT/.config/nemo-retriever/nightly"
mkdir -p "$RETRIEVER_NIGHTLY_CONFIG_DIR"
chmod 700 "$RETRIEVER_NIGHTLY_CONFIG_DIR"
test -e "$RETRIEVER_NIGHTLY_CONFIG_DIR/nightly.env" || \
  cp ops/retriever-nightly/nightly.env.example "$RETRIEVER_NIGHTLY_CONFIG_DIR/nightly.env"
chmod 600 "$RETRIEVER_NIGHTLY_CONFIG_DIR/nightly.env"
${EDITOR:-vi} "$RETRIEVER_NIGHTLY_CONFIG_DIR/nightly.env"
```

The checked-in `dataset_paths.datasets.yaml` already describes the standard
`/datasets/nv-ingest` layout. Only hosts with a different layout need to copy
and edit `nemo_retriever/harness/dataset_paths.example.yaml`, then set
`RETRIEVER_DATASET_PATHS` in `nightly.env`.

Both launchers source the detected file only when it exists; otherwise they use
the current environment. An existing secrets file must be owned by the
invoking user with mode `600`. The launchers do not discover a repository
`.env` file. `RETRIEVER_CONFIG_FILE` remains an optional advanced path override.

Verify the token and read one byte from one remote parquet object in every
ViDoRe evaluation partition before starting GPU work:

```bash
./ops/retriever-nightly/run-nightly.sh --check-vidore-access
```

The access check does not download full parquet objects. A redirect failure
such as `302 -> 403 at cas-bridge.xethub.hf.co` is a Hugging Face/CAS delivery
failure; do not start the full suite until the check exits zero.

Then preflight the complete twelve-benchmark suite without starting ingest or
query:

```bash
./ops/retriever-nightly/run-nightly.sh --dry-run
```

Inspect the resulting `session_summary.json` and child plans. Dry-runs never
post to Slack. A real run posts when `SLACK_WEBHOOK_URL` is configured; use
`--no-slack` for a real functional test that must not post. The launcher prints
the timestamped session directory on success or terminal harness failure.

Use one positional runfile for a smaller real canary before the full run:

```bash
./ops/retriever-nightly/run-nightly.sh \
  --no-slack \
  nemo_retriever/harness/runfiles/jp20_beir.json
```

Run the complete suite from the current checkout with no positional runfiles:

```bash
./ops/retriever-nightly/run-nightly.sh
```

If `SLACK_WEBHOOK_URL` is configured, that real run posts its terminal summary.
Add `--no-slack` only when the full run is itself a functional test that must
not post.

### Select the Code Under Test

The launcher runs the current clean checkout and records its commit in the
session artifacts. It never changes that checkout. To run the latest fetched
`upstream/main` without disturbing another worktree:

```bash
git fetch upstream main
git worktree add --detach ../NeMo-Retriever-benchmark-main upstream/main
cd ../NeMo-Retriever-benchmark-main
```

To run an exact fetched commit, replace `upstream/main` with its SHA and choose
a distinct worktree directory:

```bash
git worktree add --detach ../NeMo-Retriever-benchmark-abc1234 abc1234
cd ../NeMo-Retriever-benchmark-abc1234
```

Run the same launcher command from that worktree. Tracked, staged, and untracked
changes are rejected so a result is attributable to the recorded commit.
Ignored cache files do not make the checkout dirty; datasets, configuration,
and artifacts remain outside the checkout.

### Latest-Main Selection

Manual `run-nightly.sh` invocations deliberately run the current clean
checkout. `run-latest-main.sh` is the corresponding one-command controller for
a checkout where this feature is already present; it:

1. fetches `main` from the `upstream` remote;
2. resolves the fetched commit before doing any GPU work;
3. creates or reuses an immutable detached worktree named `main-<full SHA>`;
4. runs the ViDoRe access check from that selected commit; and
5. invokes that commit's `run-nightly.sh` only when fetch and access preflight
   both succeed.

A fetch failure is fail-closed: the controller does not fall back to yesterday's
commit. It never runs `git pull`, merges into the controller checkout, or moves
the reviewed controller branch. Session artifacts still record the exact
selected SHA in `run_commit`. The fetched source is also recorded locally at
`refs/retriever-nightly/latest-main` for inspection.

Immutable worktrees and one shared `uv` project environment live under the
detected nightly root at `retriever-nightly-checkouts`; on `/raid` hosts this is
`/raid/$USER/retriever-nightly-checkouts`. The seven most recently used SHA
worktrees are retained. Modified managed worktrees are never deleted
automatically.

The one-time checkout setup must provide an `upstream` remote. Operators do not
choose a branch or commit after that:

```bash
git remote get-url upstream >/dev/null 2>&1 || \
  git remote add upstream https://github.com/NVIDIA/NeMo-Retriever.git
./ops/retriever-nightly/run-latest-main.sh --dry-run
```

The dry-run fetches and selects the latest commit but skips remote access and
GPU execution. Use `run-latest-main.sh --check-vidore-access` to validate the
selected latest-main commit and its machine credentials without starting a
session. Once that selected main commit contains this launcher, the complete
latest-main suite is also one command:

```bash
./ops/retriever-nightly/run-latest-main.sh
```

### Slack Report

To enable Slack for real runs, export the incoming-webhook URL or place it in
the optional mode-`600` `nightly.env`:

```bash
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

The URL itself is the Slack switch. If it is unset or empty, real runs complete
without posting. If it is set, the launcher validates it before expensive work
and posts once after `session_summary.json` exists. An invalid configured URL
fails preflight. Pass `--no-slack` for canaries or other real functional tests;
that flag suppresses webhook validation and posting. Dry-runs and access checks
never post. The launcher removes the URL from the benchmark child environment
and exposes it only to the final Slack command.

Command-line flags override values loaded from `RETRIEVER_CONFIG_FILE`; those
values override the launcher defaults. Run either launcher with `--help` for
its supported interface.

## Runtime Contract

The launcher takes a nonblocking host-local lock, forces batch mode, and runs
12 checked-in runfiles as one session: JP20, BO767, Earnings, FinanceBench, and
all eight public ViDoRe v3 domains. Real sessions execute each child in a fresh
spawned process so Ray and materialized dataframe memory are released before
the next benchmark; the parent still writes one terminal session summary.
Dry-runs stay in the parent process because they do not materialize datasets. A
configured Slack report runs once after a terminal session summary exists. A
`.slack_post_attempted` marker prevents a second attempt for the same session.
Incoming webhooks do not provide an idempotency key, so ambiguous transport
failures require human inspection.

The Slack report keeps the library benchmarks detailed and collapses the full
ViDoRe v3 suite into total ingest time, aggregate pages/sec, macro-average
Recall@5 and nDCG@10 for the English and complete suites, and one accuracy row
per domain. Per-domain throughput and timing remain in the session artifacts.

If one runtime child fails, `run-files` continues the remaining datasets and
writes a failed session summary. When Slack is configured, the launcher still
attempts one report and returns the harness status. If the harness succeeds but
Slack fails, it returns the Slack command's nonzero status.

The launcher defaults `VLLM_DEEP_GEMM_WARMUP=skip` unless the caller explicitly
sets another vLLM-supported mode. This skips the optional compatibility-sensitive
warmup without disabling DeepGEMM kernels. It intentionally does not set
`VLLM_USE_DEEP_GEMM=0` or `VLLM_MOE_USE_DEEP_GEMM=0`. Set the warmup variable
explicitly, for example to `full`, only when validating another mode. This
matches the reliability direction under discussion in
[NVIDIA/NeMo-Retriever PR #2292](https://github.com/NVIDIA/NeMo-Retriever/pull/2292).

## Troubleshooting Preflight And Host Memory

`--check-vidore-access` validates the configured token, reads repository
metadata, and follows the same Hugging Face redirects used by `datasets` while
reading one byte from one parquet object in each of the queries, qrels, and
corpus partitions. If the token is valid but the final CAS host returns `403`, compare
the same check
from another network before rotating credentials. Success elsewhere points to
host proxy, firewall, or egress policy; failure from multiple networks should
be escalated with the named dataset object to Hugging Face or ViDoRe.

Batch ingest currently materializes each terminal Ray dataset in Python.
High-resolution page payloads can therefore consume substantially more system
RAM than the final LanceDB table. The nightly's per-run process boundary
prevents that memory from accumulating across the twelve children, but an
individual large benchmark must still fit on the host. If Ray reports the
dataset and VDB write complete while a child remains idle at high RSS, capture
`run.log`, `status.json`, process RSS, and the Ray task summary. Retry only that
runfile as a focused reproduction; do not classify the symptom as GPU OOM
unless the GPU process or kernel logs show an actual allocation failure.
