<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Portable Nightly Second-Host Validation

Use this checklist on a separate Linux NVIDIA workstation before handing the
launcher to additional teammates. The review uses the pushed feature branch,
keeps datasets and artifacts outside the repository, validates the functional
path first, and finishes with one foreground full-suite command and terminal
Slack post. It does not install or start an operating-system service.

## 1. Create a Local Review Branch

In an existing clone whose `origin` points to the contributor fork:

```bash
git fetch origin jioffe502/retriever-nightly-vidore-v3
git switch --create review/portable-nightly \
  --track origin/jioffe502/retriever-nightly-vidore-v3
git status --short
git rev-parse HEAD
```

`git status --short` must be empty. If this is a new clone, add the NVIDIA
repository as `upstream` for later comparisons:

```bash
git remote add upstream https://github.com/NVIDIA/NeMo-Retriever.git
```

Every validation command below passes `--ref HEAD`, so the launcher runs
exactly the checked-out review commit from an immutable detached worktree. It
does not fetch or move the review branch. The no-`--ref` production workflow
fetches `upstream/main` only after this feature is merged.

## 2. Prepare the Host

Confirm that `uv` and the NVIDIA driver are available:

```bash
uv --version
nvidia-smi
```

This host has the standard `/datasets/nv-ingest` layout and `/raid/$USER`, so
the checked-in dataset map and launcher path defaults apply. Export the two
secrets; a read Hugging Face token is sufficient:

```bash
export HF_TOKEN=...
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

The launcher writes artifacts under `/raid/$USER/retriever-nightly-artifacts`.
No configuration file is required. On a host without the standard dataset
layout, copy and edit `nemo_retriever/harness/dataset_paths.example.yaml`, then
export `RETRIEVER_DATASET_PATHS` with that path.

## 3. Verify ViDoRe Evaluation Access

Before starting GPU work, validate the configured token and read one byte from
one remote parquet object in each of the queries, qrels, and corpus partitions:

```bash
./ops/retriever-nightly/run-nightly.sh --ref HEAD --check-vidore-access
```

The command should exit zero and report access for all eight ViDoRe v3
datasets. It does not download the full objects. Do not start the complete
suite if this check reports a Hugging Face or CAS redirect failure.

## 4. Preflight All Twelve Benchmarks

```bash
./ops/retriever-nightly/run-nightly.sh --ref HEAD --dry-run
```

The command should exit zero, report a new timestamped session directory, and
write a `session_summary.json` with `dry_run: true`, twelve runs, and a
`run_commit` matching `git rev-parse HEAD`. `isolate_runs` is `false` because
the dry-run does not materialize batch data.

## 5. Run the JP20 Canary

```bash
./ops/retriever-nightly/run-nightly.sh \
  --ref HEAD \
  --no-slack \
  nemo_retriever/harness/runfiles/jp20_beir.json
```

Confirm that the command exits zero and the session summary contains one
successful run with `isolate_runs: true`. The launcher defaults the optional
DeepGEMM warmup to `skip`; no host setting is needed.

## 6. Capture the Handoff Evidence

For the dry-run and JP20 sessions, record:

- the launcher exit code and printed session directory;
- `success`, `exit_code`, `dry_run`, `isolate_runs`, `run_commit`, and the
  number of `runs` in `session_summary.json`;
- any failed child name and its artifact directory; and
- the GPU model and driver from `nvidia-smi`.

The functional validation is complete when the ViDoRe access check, twelve-run
dry-run, and real JP20 canary succeed with terminal summaries attributed to the
review branch commit. These steps do not post to Slack.

## 7. Run the Complete Nightly and Slack Report

From the clean draft checkout, start all twelve benchmarks with one command:

```bash
./ops/retriever-nightly/run-nightly.sh --ref HEAD
```

The launcher runs the four library benchmarks and all eight ViDoRe v3 domains.
Each child runs in a fresh process; failures do not prevent later children from
running, and the parent writes one terminal `session_summary.json`. Because
`SLACK_WEBHOOK_URL` is exported, that terminal summary posts once. Confirm the
full `run_commit`, twelve child results, final command exit status, and Slack
message. The process remains attached to the invoking shell; use the host's
normal session manager if it must survive a disconnected terminal.

## 8. Start The Post-Merge Daily Workflow

After the launcher is merged to `upstream/main`, keep the one-shot launcher in
a transparent 24-hour loop inside `tmux`:

```bash
tmux new -s retriever-nightly

export HF_TOKEN=...
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

interval=86400
while true; do
  started="$(date +%s)"
  ./ops/retriever-nightly/run-nightly.sh
  elapsed=$(( $(date +%s) - started ))
  if (( elapsed < interval )); then
    sleep "$(( interval - elapsed ))"
  fi
done
```

The default command fetches the newest `upstream/main` on each iteration,
preflights ViDoRe access, runs the suite, and posts once when
`SLACK_WEBHOOK_URL` is set. Enter the exports inside the new tmux session,
detach with `Ctrl-b d`, and inspect it later with `tmux attach -t
retriever-nightly`. The serial loop does not overlap runs. It survives SSH
disconnects but must be restarted after a workstation reboot; no service or
timer is installed.
