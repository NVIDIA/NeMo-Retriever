<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Retriever Nightly Launcher

This directory provides one public launcher for the library and ViDoRe v3
benchmark suite:

| Workflow | Command | Code that runs |
| --- | --- | --- |
| Test the current checkout | `run-nightly.sh` | The current branch, including local changes. |
| Regular latest-main nightly | `run-nightly.sh --ref upstream/main` | Freshly fetched `upstream/main` in a clean worktree. |
| Reproduce an exact revision | `run-nightly.sh --ref <SHA>` | A clean worktree at an available local commit. |

With no `--ref`, the launcher runs the checkout that contains the script and
does not fetch, switch branches, or reject local changes. With `--ref`, it
resolves one commit and creates or reuses an immutable detached worktree. It
exits after one terminal session summary. Recurrence is deliberately kept
outside its interface; the [daily `tmux` workflow](#daily-runs-with-tmux) is a
small shell loop rather than an installed scheduler. The launcher never merges
into or moves the invoking checkout, and it does not distribute datasets.

## Quick Start On A Standard Host

A workstation with `/datasets/nv-ingest` and writable `/raid/$USER` needs only
a Hugging Face token and, optionally, a Slack webhook to run its current
checkout:

```bash
export HF_TOKEN=...
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
./ops/retriever-nightly/run-nightly.sh
```

That foreground command runs the current branch exactly as it exists, all
twelve benchmarks, and one terminal Slack summary. `SLACK_WEBHOOK_URL` is
optional; omit it to run without posting. Before GPU work, it checks ViDoRe
access. It then uses the checked-in `/datasets/nv-ingest` map, writes artifacts
under `/raid/$USER/retriever-nightly-artifacts`, and prints the terminal session
directory. No model-provider API key, `nightly.env`, `sudo`, systemd service,
or timer is required.

The only required operator input on that host is `HF_TOKEN`. Set
`SLACK_WEBHOOK_URL` when the terminal summary should post to Slack. A custom
dataset map or artifact root is needed only when the host does not have the
standard paths. Use `--ref upstream/main` only when the operator deliberately
wants the newest clean upstream commit instead of the checked-out code.

## Validate The Current PR Checkout

### Prerequisites

The supported v1 host is a Linux NVIDIA workstation with:

- a NeMo Retriever Git checkout;
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
| `RETRIEVER_DATASET_PATHS` | nonstandard hosts only | Path to a YAML file that replaces the checked-in `/datasets/nv-ingest` map. |

On hosts with a writable `/raid/$USER`, the launcher automatically keeps its
private configuration, artifacts, and managed Git checkouts there.
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

### Dataset Paths On A Nonstandard Host

The checked-in `dataset_paths.datasets.yaml` describes the standard
`/datasets/nv-ingest` layout. Do not pass `--dataset-paths` on a host with that
layout.

On any other host, `--dataset-paths` takes the path to a YAML configuration
file, not a dataset directory. Copy the complete twelve-dataset template
outside the checkout, replace its paths, and pass the resulting file:

```bash
cp nemo_retriever/harness/dataset_paths.example.yaml \
  /raid/$USER/retriever-dataset-paths.yaml
${EDITOR:-vi} /raid/$USER/retriever-dataset-paths.yaml
./ops/retriever-nightly/run-nightly.sh \
  --dataset-paths /raid/$USER/retriever-dataset-paths.yaml \
  --dry-run
```

For a JP20-only canary, the YAML file may contain only its local dataset and
ground-truth query paths:

```yaml
schema_version: 1
datasets:
  jp20:
    path: /raid/data/jp20
    query_file: /raid/data/jp20_query_gt.csv
```

Run that one benchmark by also supplying its runfile:

```bash
./ops/retriever-nightly/run-nightly.sh \
  --dataset-paths /raid/data/retriever-dataset-paths.yaml \
  --no-slack \
  nemo_retriever/harness/runfiles/jp20_beir.json
```

Keep the machine-local YAML outside the repository. For repeated runs, export
`RETRIEVER_DATASET_PATHS=/raid/$USER/retriever-dataset-paths.yaml` or set the
same value in the optional `nightly.env`; the command-line flag is simplest for
a one-off run.

The launcher sources the detected file only when it exists; otherwise it uses
the current environment. An existing secrets file must be owned by the
invoking user with mode `600`. The launcher does not discover a repository
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

## Git Selection

With no `--ref`, `run-nightly.sh`:

1. selects the Git checkout that contains the launcher;
2. runs its current branch and working tree without fetching or switching;
3. permits tracked, staged, and untracked changes; and
4. runs the ViDoRe access check before real GPU work.

Every session records the checkout's HEAD in `run_commit` and whether it had
local changes in `working_tree_dirty`. Dirty runs also write
`source_worktree_status.txt` in the session directory and prefix the Slack
title with `[LOCAL CHANGES]`. The status artifact records paths and Git state,
not file contents, so a dirty run is intentionally identifiable but not fully
reproducible.

`--ref REF` requests a clean committed run. A local branch, tag, or SHA is
resolved without fetching. A remote branch such as `upstream/main` is fetched
first, then resolved fail-closed; a fetch failure never falls back to a stale
remote-tracking commit. The selected commit runs from an immutable detached
worktree named `commit-<full SHA>`. The launcher never runs `git pull`, merges
into the invoking checkout, or moves its current branch.

Immutable worktrees and one shared `uv` project environment live under the
detected nightly root at `retriever-nightly-checkouts`; on `/raid` hosts this is
`/raid/$USER/retriever-nightly-checkouts`. The seven most recently used SHA
worktrees are retained. Modified managed worktrees are never deleted
automatically.

The one-time latest-main setup must provide an `upstream` remote. Request a
clean latest-main preflight explicitly:

```bash
git remote get-url upstream >/dev/null 2>&1 || \
  git remote add upstream https://github.com/NVIDIA/NeMo-Retriever.git
./ops/retriever-nightly/run-nightly.sh --ref upstream/main --dry-run
```

The dry-run fetches and selects the latest commit but skips remote access and
GPU execution. Use `run-nightly.sh --ref upstream/main --check-vidore-access`
to validate that commit and the machine credentials without starting a
session. The complete latest-main suite is:

```bash
./ops/retriever-nightly/run-nightly.sh --ref upstream/main
```

Use `--ref HEAD` when local changes should be ignored and only the current
commit should run, or `--ref <full-SHA>` to reproduce an earlier run. The
invoking checkout may itself be dirty because the selected ref always runs in
a separate clean worktree. Ignored cache files do not mark a run dirty.

## Daily Runs With `tmux`

The launcher remains a one-shot developer tool. Use a transparent shell loop
inside `tmux` when a workstation should start the latest `upstream/main`
nightly approximately every 24 hours:

```bash
tmux new -s retriever-nightly

export HF_TOKEN=...
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

interval=86400
while true; do
  started="$(date +%s)"
  ./ops/retriever-nightly/run-nightly.sh --ref upstream/main
  elapsed=$(( $(date +%s) - started ))
  if (( elapsed < interval )); then
    sleep "$(( interval - elapsed ))"
  fi
done
```

Enter the exports inside the new `tmux` session so they do not depend on an
older tmux server's saved environment. Detach with `Ctrl-b d`, inspect it with
`tmux attach -t retriever-nightly`, and stop it with `tmux kill-session -t
retriever-nightly`.

The loop is serial: runs never overlap. It targets a 24-hour start-to-start
interval; if one run exceeds 24 hours, the next begins only after it finishes.
The loop survives an SSH disconnect but not a workstation reboot. This is an
operator-owned development workflow, not an installed service or timer.

While testing an unmerged branch, omit `--ref upstream/main` to run that
checkout on every iteration. Keep `--ref upstream/main` for the production
loop so every iteration fetches and runs the newest clean upstream commit.

## Slack Report

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
values override the launcher defaults. Run the launcher with `--help` for its
supported interface.

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
