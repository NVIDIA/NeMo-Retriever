<!-- SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Retriever Harness

Developer benchmark harness for Retriever ingest/query evaluation.

The harness is artifact-first. Humans may read CLI output, but agents and
orchestrators should read `results.json`, `status.json`, and
`summary_metrics.json`.

## Quick Start

Run commands from the repository root through the `nemo_retriever` project:

```bash
uv run --project nemo_retriever retriever harness list --runsets
uv run --project nemo_retriever retriever harness show jp20_beir --json
```

Resolve a benchmark without executing ingest or queries:

```bash
uv run --project nemo_retriever retriever harness run jp20_beir \
  --dry-run \
  --output-dir /tmp/retriever-harness-dry-run \
  --json
```

Run the cheap JP20 ingest smoke check:

```bash
uv run --project nemo_retriever retriever harness run jp20_smoke \
  --output-dir /tmp/retriever-harness-jp20-smoke \
  --require 'files==20' \
  --require 'pages==1940' \
  --json
```

Run the full JP20 BEIR benchmark:

```bash
uv run --project nemo_retriever retriever harness run jp20_beir \
  --output-dir /tmp/retriever-harness-jp20-beir \
  --require 'files==20' \
  --require 'pages==1940' \
  --require 'query_count==115' \
  --require 'recall_5>=0.85' \
  --require 'ndcg_10>=0.75' \
  --json
```

Run the same JP20 BEIR request from a checked-in runfile:

```bash
uv run --project nemo_retriever retriever harness run \
  --runfile nemo_retriever/harness/runfiles/jp20_beir.json \
  --output-dir /tmp/retriever-harness-jp20-beir \
  --json
```

Large checked-in BEIR runfiles such as BO767, FinanceBench, Earnings, and
ViDoRe use `mode: batch`. Keep JP20 local for quick smoke validation, and use
batch mode for larger canonical quality runs so Ray-backed ingest owns worker
parallelism and memory pressure.

## Commands

- `list`: list code-owned benchmarks and optional runsets.
- `show`: inspect one benchmark definition.
- `run`: run one benchmark.
- `run-set`: expand and run a code-owned runset.
- `run-files`: execute one or more checked-in runfiles as one session.
- `post-slack`: post an existing session or run artifact set to Slack.
- `diff`: compare two run artifact directories by `summary_metrics.json`.

Legacy graph-pipeline harness execution, sweep, nightly, runner, reporting, and
portal commands are not part of the phase-one CLI surface. Portal and Helm
support files are intentionally preserved for follow-on owner work.

## Reviewer Guide

Review the PR in this order:

1. Start with this README for the user-facing harness contract.
2. Read `benchmark_registry.py` for code-owned datasets, benchmarks, and
   runsets.
3. Read `resolution.py` for how registry specs, runfiles, CLI `--set`
   overrides, and mode selection become ingest/query requests.
4. Read `execution.py` for the artifact-first run lifecycle and exit-code
   behavior.
5. Read `beir_runner.py` and `metrics.py` for query evaluation and
   `summary_metrics.json` construction.
6. Read `artifact_writer.py` for artifact names, status updates, and `run.log`
   capture.
7. Read `json_io.py` for shared artifact JSON read/write helpers used by the
   harness, diff, runset, Slack, and artifact-writing paths.

Intentional removals:

- old `run.py` and `runner.py`: subprocess-oriented graph-pipeline harness
  execution and portal runner agent
- old `parsers.py`: regex parsing of stdout/progress logs
- old `nightly.py`, `reporting.py`, and nightly/sweep YAML: previous session
  reporting and scheduled-run machinery
- old harness pytests: this harness is validated by functional benchmark
  execution and artifact/exit-code checks

Intentional preserves:

- `portal/`, `history.py`, and `scheduler.py`: retained for upcoming portal
  repurpose work; `slack.py` now owns artifact replay and Slack payloads
- `helm_manager.py`, `helm-profiles/`, and harness Helm examples: retained for
  Helm owner follow-up work

## Runfiles

Runfiles are a small reproducibility helper for agents, handoffs, and
orchestrators. They describe one concrete run request:

- registered `benchmark`
- optional `name`, `mode`, `run_id`, and `output_dir`
- optional `set` overrides
- optional `require` metric gates

Runfiles cannot define new datasets or benchmarks. Add recurring benchmark
definitions to the Python registry instead.

The harness accepts JSON, YAML, or YML runfiles. Runfiles use
`schema_version: 1`; unknown top-level runfile keys fail during resolution with
exit code `2`. The checked-in JP20 example is
[`runfiles/jp20_beir.json`](runfiles/jp20_beir.json).

### Configure Machine-Local Dataset Paths

Dataset locations vary between developer systems. Keep benchmark definitions
and checked-in runfiles independent of one machine's mount layout. Copy
[`dataset_paths.example.yaml`](dataset_paths.example.yaml) to an untracked
location, then set the document and query paths available on the machine that
runs the harness.

Pass the local file with `--dataset-paths`. Relative paths in the file resolve
relative to the file itself. The harness writes the resolved absolute paths to
`expanded_runs.json` and each run's `resolved_benchmark.json`.

Settings resolve in this order, from lowest to highest precedence:

1. Benchmark registry defaults.
2. Checked-in runfile overrides.
3. Machine-local dataset paths.
4. Command-line `--set` overrides.

### Run Several Runfiles as One Session

The following command runs the canonical library benchmarks and writes all
artifacts beneath one session directory:

```bash
export VLLM_USE_DEEP_GEMM=0
export RETRIEVER_SESSION_DIR=/local/path/to/retriever-artifacts/library-nightly-$(date -u +%Y%m%d_%H%M%S_UTC)

uv run --project nemo_retriever retriever harness run-files \
  --session-name library_nightly \
  --output-dir "$RETRIEVER_SESSION_DIR" \
  --dataset-paths /local/path/to/dataset_paths.yaml \
  --json \
  nemo_retriever/harness/runfiles/jp20_beir.json \
  nemo_retriever/harness/runfiles/bo767_beir.json \
  nemo_retriever/harness/runfiles/earnings_beir.json \
  nemo_retriever/harness/runfiles/financebench_beir.json
```

On vLLM environments affected by the current DeepGEMM availability check,
`VLLM_USE_DEEP_GEMM=0` keeps the local embedding backend enabled while using
fallback kernels. Remove the variable after the vLLM and DeepGEMM dependency
stack is repaired. Because this setting can affect throughput, the harness
records it in `environment.json`.

`run-files` owns the session layout. Runfiles passed to this command cannot set
their own `output_dir` or `run_id`. The session uses the following paths and
identifiers:

```text
<session-output-dir>/
  expanded_runs.json
  session_summary.json
  001_<runfile-name>/
  002_<runfile-name>/

run ID: <session-name>_<index>_<runfile-name>
```

Session names and runfile names can contain letters, numbers, periods,
underscores, and hyphens. Other characters fail validation before execution.

## Post Results to Slack

Harness execution and Slack reporting are separate operations. `run-files`
writes local artifacts and never contacts Slack. `post-slack` reads an existing
session or run artifact, builds a summary, and sends that summary without
rerunning ingestion or queries.

This separation lets you inspect a completed session before reporting it and
reuse the same artifacts when report formatting changes.

### Prerequisites

Before you post a report, verify the following:

- The run completed far enough to write `session_summary.json` or
  `results.json`.
- The environment includes the `requests` package.
- `SLACK_WEBHOOK_URL` contains an incoming webhook for the destination channel.

Set the webhook in the process environment:

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
```

Do not put the webhook URL in a runfile, dataset paths file, shell argument, or
artifact. For scheduled runs, load it from a permissions-restricted environment
file outside the repository.

### Post a Completed Session

Pass the session directory to `post-slack`:

```bash
uv run --project nemo_retriever retriever harness post-slack \
  --title "nemo-retriever library nightly" \
  "$RETRIEVER_SESSION_DIR"
```

You can also pass one or more run artifact directories or `results.json` files.
Each invocation sends a new Slack message; it does not modify the completed
harness artifacts.

By default, the report includes file and page counts, ingest time, ingest
pages/sec, query count, recall, nDCG, environment details, and local artifact
paths when those values are available. Use repeated `--metric-key` options to
select a different metric set. Use `--no-artifact-paths` to omit local paths.

### Preview Report Formatting

Use `--preview` to render the exact Slack payload without reading
`SLACK_WEBHOOK_URL` or making an HTTP request:

```bash
uv run --project nemo_retriever retriever harness post-slack \
  --preview \
  --title "nemo-retriever library nightly" \
  "$RETRIEVER_SESSION_DIR"
```

Preview the same completed session as often as needed while adjusting the
title, metric selection, or artifact-path setting. When the payload is ready,
run the command again without `--preview` to post it. Preview and posting use
the same artifact loader and payload formatter.

### Preserve Run and Report Status

A failed benchmark session normally still writes a summary that can be posted.
When you schedule the harness, save the run exit code, attempt Slack reporting,
then return the run failure after reporting:

```bash
uv run --project nemo_retriever retriever harness run-files \
  --session-name library_nightly \
  --output-dir "$RETRIEVER_SESSION_DIR" \
  --dataset-paths /local/path/to/dataset_paths.yaml \
  nemo_retriever/harness/runfiles/jp20_beir.json \
  nemo_retriever/harness/runfiles/bo767_beir.json \
  nemo_retriever/harness/runfiles/earnings_beir.json \
  nemo_retriever/harness/runfiles/financebench_beir.json
run_status=$?

uv run --project nemo_retriever retriever harness post-slack "$RETRIEVER_SESSION_DIR"
slack_status=$?

if [ "$run_status" -ne 0 ]; then
  exit "$run_status"
fi
exit "$slack_status"
```

Use a host-local scheduler such as cron and a nonblocking `flock` lock to avoid
overlapping daily runs. Keep the session directory on runner-owned storage; the
harness does not upload full artifacts to Slack or external artifact storage.

## Controls And Overrides

Benchmarks are code-owned defaults. Use `--set KEY=VALUE` for one-off
ablations, or put the same keys under `set` in a runfile for reproducible
agent/orchestrator runs.

Examples:

```bash
retriever harness run jp20_beir \
  --set query.top_k=20 \
  --set query.rerank=true \
  --set ingest.extract.batch.page_elements_workers=1
```

Runfile equivalent:

```json
{
  "schema_version": 1,
  "benchmark": "bo767_beir",
  "mode": "batch",
  "set": {
    "query.top_k": 10,
    "ingest.extract.batch.pdf_extract_workers": 8,
    "ingest.embed.batch.embed_batch_size": 64
  }
}
```

Supported override namespaces:

- `dataset.*`: dataset path, query/qrels file, input type, BEIR loader, and
  BEIR doc ID settings.
- `ingest.*`: profile, input type, Ray mode/address, extraction/media/caption,
  dedup, chunk, embedding, image-store, storage, and batch worker settings.
- `query.*`: top-k, candidate-k, page dedup, content types, retrieval mode,
  embedding endpoint/model, reranking, LanceDB URI, and table name.
- `evaluation.*`: evaluation mode, BEIR loader/dataset/split/language/doc ID
  field, and metric cutoffs.

Unknown override keys fail during resolution with exit code `2`. Values are
parsed as YAML scalars/lists/maps, so booleans, numbers, nulls, and lists can be
passed naturally.

Use `retriever harness show <benchmark> --json` and `retriever harness run
<benchmark> --dry-run --json` to inspect the exact resolved benchmark and
plans before launching an expensive run.

## Implementation Boundary

The harness does not shell out to `retriever ingest`, `retriever query`, or
`retriever pipeline run`. It calls the same Python workflow/planning APIs used
by the CLI:

- ingest: `resolve_ingest_plan(...)` and `run_ingest_workflow(...)`
- query: `resolve_query_plan(...)` and shared query workflow objects
- BEIR: harness-owned query iteration over the resolved query plan

This keeps benchmark execution in-process at the Python boundary while still
reusing the CLI-owned request/plan/workflow seams. Stdout remains diagnostic
only; artifacts and exit codes are the contract.

## Artifacts

Read these files instead of scraping stdout:

- `results.json`: authoritative run result and artifact manifest.
- `status.json`: current/final run status, phase, and failure payload.
- `summary_metrics.json`: compact metrics for gates, dashboards, and agents.
- `events.jsonl`: phase transitions and harness events.
- `resolved_benchmark.json`: exact resolved benchmark spec.
- `ingest_plan.json`: redacted ingest dry-run plan.
- `query_plan.json`: resolved query plan.
- `environment.json`: commit and runtime context.
- `run.log`: captured lower-level stdout/stderr for non-dry execution.
- `beir_metrics.json`: BEIR metrics when BEIR evaluation executes.
- `beir_run.trec`: TREC runfile when BEIR evaluation executes.
- `query_results.jsonl`: per-query results when queries execute.

Dry-runs write only planning artifacts. They do not create empty `run.log`,
`beir_metrics.json`, `beir_run.trec`, or `query_results.jsonl` files.

## Gates

Use explicit `--require` gates. Gate expressions compare keys from
`summary_metrics.json`:

```bash
--require 'files==20'
--require 'recall_5>=0.85'
--require 'query_latency_p95_ms<=1200'
```

Gate failures exit with code `20` and still write artifacts.

During `--dry-run`, gates for unavailable execution metrics are skipped and
listed in `results.json` as `skipped_metric_gates`. Static gates such as
`files==20` and `pages==1940` are still evaluated.

Known dataset facts, observed result ranges, and suggested gates live in
[`EXPECTED_RESULTS.md`](EXPECTED_RESULTS.md). Keep threshold knowledge there,
not in benchmark Python code.

## Agent Instructions

For automated harness work:

1. Start with `retriever harness list --runsets --json`.
2. Use `retriever harness show <benchmark> --json` to inspect a benchmark.
3. Use `--output-dir` so artifact paths are deterministic.
4. Use `--dry-run` before expensive runs when changing paths, overrides, or
   gates.
5. Use explicit `--require` gates from `EXPECTED_RESULTS.md`.
6. Decide success from the process exit code and `results.json`.
7. Read `summary_metrics.json` for benchmark metrics.
8. Read `run.log` only when lower-level ingest/query logs are needed.
9. Do not parse progress bars, human CLI formatting, or raw stdout as the API.
10. Do not use `retriever pipeline run` for phase-one harness validation.

## Exit Codes

- `0`: success
- `2`: invalid benchmark/config/override/gate syntax
- `3`: dataset or input missing
- `10`: ingest failure
- `11`: query failure
- `12`: evaluation failure
- `20`: metric gate failure
- `30`: artifact write failure
- `70`: unexpected internal error

## More Detail

- [`EXPECTED_RESULTS.md`](EXPECTED_RESULTS.md): dataset facts, observed metrics,
  and suggested explicit gates.
- [`HANDOFF.md`](HANDOFF.md): current implementation notes and validation
  history for this revamp.
