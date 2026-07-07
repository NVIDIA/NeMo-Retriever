<!-- SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Retriever Harness Maintainer Notes

The user and agent contract is documented in [`README.md`](README.md). Keep this
file limited to implementation boundaries that maintainers need when changing
the harness.

## Product Boundary

- `retriever ingest` and `retriever query` are the direct product workflows.
- `retriever harness` is a developer benchmark/evaluation runner built on the
  same planning and workflow APIs.
- `run` executes one registered benchmark using registry paths or explicit
  overrides.
- `run-set` executes a code-owned benchmark group using registry paths.
- `run-files` executes one or more checked-in runfiles and can apply a
  machine-local dataset path map. It is the portable suite entry point and owns
  dry-run behavior for the complete session.
- `post-slack` only reads completed artifacts. It does not execute benchmarks
  or mutate their results.
- Scheduling, deployment, retries, locking, and secret distribution are outside
  this harness surface.

The harness must not route these commands through `retriever pipeline run` or
`nemo_retriever.examples.graph_pipeline`.

## Implementation Map

- `runfile.py` parses one portable run request; `dataset_paths.py` resolves the
  machine-local dataset map.
- `execution.py` preflights and executes one benchmark.
- `runsets.py` converts runsets or runfiles into prepared runs, then executes
  both through one session loop.
- `json_io.py` atomically publishes JSON artifacts; `artifact_writer.py` owns
  per-run status, events, logs, and artifact cleanup.
- `slack.py` reads completed artifacts and renders or posts a report. It does
  not participate in benchmark execution.

## Configuration Ownership

The Python registry owns benchmark and dataset semantics. Checked-in runfiles
own concrete modes, metric gates, and narrow overrides. A local
`dataset_paths.yaml` owns machine-specific document and query locations and
must remain outside source control.

Resolution precedence is:

1. Registry defaults.
2. Runfile overrides.
3. Machine-local dataset paths.
4. CLI `--set` overrides.

Large checked-in benchmark runfiles use batch ingest. Do not silently change
their mode or hardware-sensitive worker tuning without fresh validation.

## Artifact Contract

Poll `status.json` while a run is active. Read `results.json` after one run is
terminal and `session_summary.json` after a multi-run session is terminal.
Those files contain summary metrics and relative pointers to detailed evidence.

Detailed run evidence can include:

- `events.jsonl`
- `runfile.json`
- `resolved_benchmark.json`
- `ingest_plan.json`
- `query_plan.json`
- `environment.json`
- `run.log`
- `beir_metrics.json`
- `beir_run.trec`
- `query_results.jsonl`
- `lancedb/`

New runs do not write `summary_metrics.json`. Compatibility readers may still
accept that file in older artifact directories. Failure summaries stay concise;
full tracebacks belong in `run.log`.

## Validation

At minimum, changes should cover:

- CLI help and dry-run behavior for `run`, `run-set`, and `run-files`.
- One-run and multi-run terminal artifact shapes.
- Missing inputs, invalid overrides, and metric-gate failures.
- Dataset path precedence and secret redaction.
- Slack preview without a webhook and transport errors without secret leakage.
- A real benchmark smoke run when execution behavior changes.

Use `EXPECTED_RESULTS.md` for dataset facts and observed metric ranges. Do not
turn hardware-sensitive reference numbers into implicit global pass/fail policy.
