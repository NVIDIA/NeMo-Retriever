<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Portable Nightly Second-Host Validation

Use this checklist on a separate Linux NVIDIA workstation before handing the
launcher to additional teammates. The review uses the pushed feature branch,
keeps datasets and artifacts outside the repository, validates the functional
path before installing recurrence, and finishes with one complete service run
and terminal Slack post.

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

The launcher runs exactly the checked-out commit. It does not fetch, pull, or
switch refs.

## 2. Prepare the Host

Confirm that `uv` and the NVIDIA driver are available:

```bash
uv --version
nvidia-smi
```

This host has the standard `/datasets/nv-ingest` layout and `/raid/$USER`, so
the checked-in dataset map and launcher path defaults apply. Create one private
configuration file and populate `HF_TOKEN` and `SLACK_WEBHOOK_URL`. A read
Hugging Face token is sufficient:

```bash
RETRIEVER_NIGHTLY_CONFIG_DIR="/raid/$USER/.config/nemo-retriever/nightly"
mkdir -p "$RETRIEVER_NIGHTLY_CONFIG_DIR"
chmod 700 "$RETRIEVER_NIGHTLY_CONFIG_DIR"
test -e "$RETRIEVER_NIGHTLY_CONFIG_DIR/nightly.env" || \
  cp ops/retriever-nightly/nightly.env.example "$RETRIEVER_NIGHTLY_CONFIG_DIR/nightly.env"
chmod 600 "$RETRIEVER_NIGHTLY_CONFIG_DIR/nightly.env"
${EDITOR:-vi} "$RETRIEVER_NIGHTLY_CONFIG_DIR/nightly.env"
```

The launcher discovers that file and writes artifacts under
`/raid/$USER/retriever-nightly-artifacts`; no shell exports are required. It
does not read a repository `.env`. On a host without the standard dataset
layout, copy and edit `nemo_retriever/harness/dataset_paths.example.yaml` and
set only `RETRIEVER_DATASET_PATHS` in `nightly.env`.

## 3. Verify ViDoRe Evaluation Access

Before starting GPU work, validate the configured token and read one byte from
one remote parquet object in each of the queries, qrels, and corpus partitions:

```bash
./ops/retriever-nightly/run-nightly.sh --check-vidore-access
```

The command should exit zero and report access for all eight ViDoRe v3
datasets. It does not download the full objects. Do not start the complete
suite if this check reports a Hugging Face or CAS redirect failure.

## 4. Preflight All Twelve Benchmarks

```bash
./ops/retriever-nightly/run-nightly.sh --dry-run
```

The command should exit zero, report a new timestamped session directory, and
write a `session_summary.json` with `dry_run: true`, twelve runs, and a
`run_commit` matching `git rev-parse HEAD`. `isolate_runs` is `false` because
the dry-run does not materialize batch data.

## 5. Run the JP20 Canary

```bash
./ops/retriever-nightly/run-nightly.sh \
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

## 7. Install the Portable Daily Timer

Install the root-managed service and daily timer for the current service user,
absolute controller checkout, and current branch's tracked remote branch:

```bash
./ops/retriever-nightly/install-systemd.sh --test-current-branch
systemctl list-timers nrl-harness-batch-hf.timer --all
systemctl cat nrl-harness-batch-hf.service
```

The rendered service must name the current user and checkout, not a checked-in
host identity. Test mode embeds the tracking remote and branch in the service,
so the scheduled controller exercises the unmerged PR instead of selecting an
older `upstream/main`. The timer runs daily at midnight in
`America/New_York`; it does not run immediately when enabled.

## 8. Start the Complete Nightly and Slack Report

Start the installed service once without waiting for the next timer event:

```bash
sudo systemctl start --no-block nrl-harness-batch-hf.service
journalctl -fu nrl-harness-batch-hf.service
```

The controller checks ViDoRe access, then runs the four library benchmarks and
all eight ViDoRe v3 domains. Each child runs in a fresh process; failures do not
prevent later children from running, and the parent writes one terminal
`session_summary.json`. If `SLACK_WEBHOOK_URL` is configured, that terminal
summary posts once. Confirm the full `run_commit`, twelve child results, final
service exit status, and Slack message. The already-enabled timer will launch
the next run at the following scheduled midnight.

## 9. Switch the Timer to Production After Merge

After the PR merges, validate production latest-main selection and reinstall
without the test override:

```bash
git remote get-url upstream
./ops/retriever-nightly/run-latest-main.sh --dry-run
./ops/retriever-nightly/install-systemd.sh
systemctl cat nrl-harness-batch-hf.service
```

Confirm that the selected SHA matches
`git rev-parse refs/retriever-nightly/latest-main`, the summary records that SHA
as `run_commit`, and the reinstalled service no longer contains
`RETRIEVER_LATEST_SOURCE` or `RETRIEVER_LATEST_REF`. Future timer events now
fetch and run the latest `upstream/main` automatically. A second complete suite
is not required solely to make this production switch.
