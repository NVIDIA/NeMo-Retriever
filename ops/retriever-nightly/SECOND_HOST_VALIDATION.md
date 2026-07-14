<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Portable Nightly Second-Host Validation

Use this checklist on a separate Linux NVIDIA workstation before handing the
launcher to additional teammates. The review uses the pushed feature branch,
keeps datasets and artifacts outside the repository, and does not install a
timer or post to Slack.

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
configuration file and populate `HF_TOKEN`. A read token is sufficient:

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

The pre-handoff validation is complete when the ViDoRe access check, twelve-run
dry-run, and real JP20 canary succeed with terminal summaries attributed to the
review branch commit. A second full suite is not required before handoff. Do
not enable the timer or post to Slack as part of this functional review.

## 7. Validate Latest-Main Selection After Merge

Do not use the latest-main controller while validating the feature branch; its
purpose is to select `upstream/main`. After the PR merges, validate the
scheduled selection path without GPU work:

```bash
git remote get-url upstream
./ops/retriever-nightly/run-latest-main.sh --dry-run
```

Confirm that the printed selected SHA matches
`git rev-parse refs/retriever-nightly/latest-main`, the controller checkout did
not move, and the dry-run summary has that SHA as `run_commit`. The installed
systemd service uses this controller automatically, so scheduled users do not
fetch or select a commit themselves.

## 8. Run the Lead's First Complete Nightly

After the latest-main dry-run succeeds, run the controller without positional
runfiles or `--no-slack`:

```bash
./ops/retriever-nightly/run-latest-main.sh
```

The controller checks ViDoRe access, then runs the four library benchmarks and
all eight ViDoRe v3 domains. Each child runs in a fresh process; failures do not
prevent later children from running, and the parent writes one terminal
`session_summary.json`. If `SLACK_WEBHOOK_URL` is configured, that terminal
summary posts once. Confirm the full `run_commit`, twelve child results, final
exit code, and Slack message before enabling the timer described in `README.md`.
