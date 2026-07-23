<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Retriever Nightly Launcher

`run-nightly.sh` runs the checked-in library and ViDoRe benchmark suite once.
It adds host locking, Git selection, standard dataset paths, artifact placement,
and optional Slack reporting around
[`retriever harness run-files`](../../nemo_retriever/harness/README.md).

It does not install a scheduler or system service.

## Choose the Source

| Goal | Command | Source used |
| --- | --- | --- |
| Test the current checkout | `./ops/retriever-nightly/run-nightly.sh` | Current branch and local changes |
| Run the newest upstream code | `./ops/retriever-nightly/run-nightly.sh --ref upstream/main` | Freshly fetched `upstream/main` |
| Reproduce one commit | `./ops/retriever-nightly/run-nightly.sh --ref <SHA>` | Clean detached worktree |

With no positional runfiles, the launcher runs twelve benchmarks: JP20, BO767,
Earnings, FinanceBench, and all eight ViDoRe v3 domains. Pass one or more
runfiles to run a smaller selection.

## First Run

On a standard host with `/datasets/nv-ingest` and writable `/raid/$USER`:

```bash
export HF_TOKEN=...
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...  # optional
```

Validate access and configuration before starting GPU work:

```bash
./ops/retriever-nightly/run-nightly.sh --check-vidore-access
./ops/retriever-nightly/run-nightly.sh --dry-run
./ops/retriever-nightly/run-nightly.sh \
  --no-slack \
  nemo_retriever/harness/runfiles/jp20_beir.json
```

All three commands should exit zero. Then run the full current checkout:

```bash
./ops/retriever-nightly/run-nightly.sh
```

Use `--ref upstream/main` instead when the result must represent the newest
clean upstream commit.

## Host Requirements

The launcher expects:

- Linux with NVIDIA drivers
- Git, Bash, `uv`, `flock`, and `realpath`
- local access to the selected datasets
- `HF_TOKEN` and outbound Hugging Face access for real ViDoRe runs
- enough RAM, model cache, and artifact storage for the selected runfiles

The full batch suite is not validated on 128 GiB hosts. It starts models locally
and does not require a model-provider API key.

Common settings:

| Setting | When needed |
| --- | --- |
| `HF_TOKEN` | Every real launcher run; a read token is sufficient |
| `SLACK_WEBHOOK_URL` | Only when the terminal result should post to Slack |
| `RETRIEVER_HARNESS_REFERENCE_FILE` | When Slack should show the current RC beside matching results |
| `RETRIEVER_DATASET_PATHS` | Hosts without the standard dataset layout |
| `RETRIEVER_NIGHTLY_ROOT` | Hosts that need a nondefault config, artifact, and checkout root |

Direct exports are the smallest setup. Persistent values may be copied from
[`nightly.env.example`](nightly.env.example) into:

```text
<nightly-root>/.config/nemo-retriever/nightly/nightly.env
```

The launcher uses `/raid/$USER` as the nightly root when it is writable and
`$HOME` otherwise. The optional file must be owned by the current user with mode
`600`. Existing exported values take precedence. The launcher never loads a
repository `.env`.

## Use Nonstandard Dataset Paths

The default map,
[`dataset_paths.datasets.yaml`](dataset_paths.datasets.yaml), uses
`/datasets/nv-ingest`. On another host, copy the portable template outside the
repository:

```bash
cp nemo_retriever/harness/dataset_paths.example.yaml \
  /local/path/to/dataset_paths.yaml
${EDITOR:-vi} /local/path/to/dataset_paths.yaml
```

Pass the YAML file—not a dataset directory:

```bash
./ops/retriever-nightly/run-nightly.sh \
  --dataset-paths /local/path/to/dataset_paths.yaml \
  --dry-run
```

For repeated runs, export the same path as `RETRIEVER_DATASET_PATHS`.

## Git Selection

Without `--ref`, the launcher runs the checkout containing the script exactly
as it exists. It records the commit and dirty state in the session. Dirty Slack
reports are prefixed with `[LOCAL CHANGES]`.

With `--ref`:

1. Local branches, tags, and SHAs resolve without fetching.
2. Remote branches such as `upstream/main` are fetched first.
3. Fetch failure stops the run instead of using a stale remote-tracking commit.
4. The selected commit runs in a clean detached worktree.

Managed worktrees and a shared `uv` environment live under the nightly root.
The launcher retains the seven most recently used clean worktrees and never
moves or merges the invoking checkout.

Configure `upstream` once if needed:

```bash
git remote get-url upstream >/dev/null 2>&1 || \
  git remote add upstream https://github.com/NVIDIA/NeMo-Retriever.git
```

Use `--ref HEAD` to ignore local changes and run only the current commit.

## Run Daily with `tmux`

Start a session:

```bash
tmux new -s retriever-nightly
```

Inside it, export the environment and run a serial 24-hour loop:

```bash
export HF_TOKEN=...
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
export RETRIEVER_HARNESS_REFERENCE_FILE=/path/to/current-release.json

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

Each iteration fetches the newest `upstream/main`. Runs do not overlap. If one
exceeds 24 hours, the next starts after it finishes.

Detach with `Ctrl-b d`, reconnect with `tmux attach -t retriever-nightly`, and
stop the loop with `tmux kill-session -t retriever-nightly`. The session
survives an SSH disconnect but not a host reboot.

To exercise an unmerged branch repeatedly, omit `--ref upstream/main`.

## Slack and the Current Release

When `SLACK_WEBHOOK_URL` is set, a real run posts once after
`session_summary.json` exists. Dry-runs and access checks never post. Use
`--no-slack` for a real canary.

Set `RETRIEVER_HARNESS_REFERENCE_FILE` to show the current release beside
matching nightly results. The
[harness reporting guide](../../nemo_retriever/harness/README.md#report-completed-results)
defines the small external JSON format.

The report presents observed values with their GPU context. It does not enforce
a score, assign a verdict, append history, or modify the release file. To move
to a new RC, replace the external file's label and values.

## Runtime Behavior

- A nonblocking host-local lock prevents overlapping launcher processes.
- Every run uses batch mode; each benchmark runs in a fresh child process.
- A failed child is recorded and later benchmarks still run.
- Each child has a six-hour wall-time limit.
- A configured Slack report is attempted once after a terminal session exists.
- `VLLM_DEEP_GEMM_WARMUP` defaults to `skip` unless the caller sets it.

The command returns the harness status. If the harness succeeds but Slack
posting fails, it returns the Slack command's nonzero status.

## Troubleshooting

**ViDoRe access fails:** rerun `--check-vidore-access`. A final `403` from a
Hugging Face CAS host can indicate proxy, firewall, or egress policy rather than
an invalid token. Compare from another network before rotating credentials.

**A large run stalls with high system memory:** batch ingest can materialize
page payloads in Python. Capture the child `run.log`, `status.json`, process RSS,
and Ray task summary, then reproduce only that runfile. Do not classify it as a
GPU OOM without GPU or kernel allocation evidence.

**Configuration is unclear:** run
`./ops/retriever-nightly/run-nightly.sh --help`. CLI flags override exported
values, which override the optional config file, which overrides launcher
defaults.
