<!-- SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Retriever Harness

Developer benchmark harness for repeatable Retriever ingest/query evaluation.

Use `retriever ingest` and `retriever query` when you want to operate Retriever
directly on your own inputs. Use `retriever harness` when you want to run a
registered benchmark with reproducible settings, metric gates, and a stable
artifact contract. The harness does not install or operate a scheduler.

The harness is artifact-first. Poll `status.json` while one run is active, read
`results.json` when that run is terminal, and read `session_summary.json` for a
multi-run session. Those terminal files point to detailed evidence; agents do
not need to scan every file in the artifact directory.

## Quick Start

Run commands from the repository root through the `nemo_retriever` project:

```bash
uv run --project nemo_retriever retriever harness list --runsets
uv run --project nemo_retriever retriever harness show jp20_beir --json
```

The registry contains canonical benchmark definitions, but dataset mounts vary
by machine. Before executing a checked-in runfile, copy
[`dataset_paths.example.yaml`](dataset_paths.example.yaml) outside the repository
and replace the example document and query paths with paths available locally.

Dry-run one checked-in dataset with that path map:

```bash
uv run --project nemo_retriever retriever harness run-files \
  --session-name jp20_check \
  --output-dir /tmp/retriever-harness-jp20-check \
  --dataset-paths /local/path/to/dataset_paths.yaml \
  --dry-run \
  --json \
  nemo_retriever/harness/runfiles/jp20_beir.json
```

After inspecting `session_summary.json` and the child run's resolved plans,
remove `--dry-run` to execute it. The same `run-files` command accepts multiple
runfiles for a collection.

If the registry's default paths already exist on the machine, `run` is the
short form for one registered benchmark:

```bash
uv run --project nemo_retriever retriever harness run jp20_beir \
  --output-dir /tmp/retriever-harness-jp20-beir \
  --require 'files==20' \
  --require 'pages==1940' \
  --require 'query_count==115' \
  --json
```

Large checked-in BEIR runfiles such as BO767, FinanceBench, Earnings, and
ViDoRe use `mode: batch`. Keep JP20 local for quick smoke validation, and use
batch mode for larger canonical quality runs so Ray-backed ingest owns worker
parallelism and memory pressure.

## Commands

- `list`: list code-owned benchmarks and optional runsets.
- `show`: inspect one benchmark definition.
- `run`: run one registered benchmark using registry paths or explicit `--set`
  overrides.
- `run-set`: expand a code-owned benchmark group using registry paths.
- `run-files`: execute one or more runfiles with an optional machine-local
  dataset path map. This is the portable entry point for the checked-in suite.
- `post-slack`: preview or post existing artifacts; it never executes a run.
- `diff`: compare two run artifact directories by `results.json` summary metrics.

Legacy graph-pipeline, sweep, recurring-job, runner, reporting-UI, and portal
commands are not part of this CLI surface. Scheduling and deployment belong to
separate infrastructure, not the benchmark harness.

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

The harness does not distribute or download private benchmark corpora or qrels.
The operator must have access to the datasets referenced by the selected
runfiles.

Pass the local file with `--dataset-paths`. Relative paths in the file resolve
relative to the file itself. The harness writes the resolved absolute paths to
`expanded_runs.json` and each run's `resolved_benchmark.json`.

Settings resolve in this order, from lowest to highest precedence:

1. Benchmark registry defaults.
2. Checked-in runfile overrides.
3. Machine-local dataset paths.
4. Command-line `--set` overrides.

### Run One Or More Runfiles As A Session

For a single dataset, pass one runfile:

```bash
uv run --project nemo_retriever retriever harness run-files \
  --session-name jp20_beir \
  --output-dir /local/path/to/retriever-artifacts/jp20-beir \
  --dataset-paths /local/path/to/dataset_paths.yaml \
  nemo_retriever/harness/runfiles/jp20_beir.json
```

For the four-dataset library suite, pass all four runfiles. This is still an
ordinary user-invoked harness session; the repository does not schedule it:

```bash
export RETRIEVER_SESSION_DIR=/local/path/to/retriever-artifacts/library-beir-$(date -u +%Y%m%d_%H%M%S_UTC)

uv run --project nemo_retriever retriever harness run-files \
  --session-name library_beir \
  --output-dir "$RETRIEVER_SESSION_DIR" \
  --dataset-paths /local/path/to/dataset_paths.yaml \
  --json \
  nemo_retriever/harness/runfiles/jp20_beir.json \
  nemo_retriever/harness/runfiles/bo767_beir.json \
  nemo_retriever/harness/runfiles/earnings_beir.json \
  nemo_retriever/harness/runfiles/financebench_beir.json
```

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
artifact. Load it from the process environment or a permissions-restricted
secret file outside the repository.

### Post a Completed Session

Pass the session directory to `post-slack`:

```bash
uv run --project nemo_retriever retriever harness post-slack \
  --title "nemo-retriever library benchmarks" \
  "$RETRIEVER_SESSION_DIR"
```

You can also pass one or more run artifact directories or `results.json` files.
Each invocation sends a new Slack message; it does not modify the completed
harness artifacts.

By default, the report includes file and page counts, ingest time, ingest
pages/sec, query count, recall, nDCG, and environment details when those values
are available. Use repeated `--metric-key` options to select a different metric
set. Use `--artifact-paths` when recipients can access the runner's local paths.

### Preview Report Formatting

Use `--preview` to render the exact Slack payload without reading
`SLACK_WEBHOOK_URL` or making an HTTP request:

```bash
uv run --project nemo_retriever retriever harness post-slack \
  --preview \
  --title "nemo-retriever library benchmarks" \
  "$RETRIEVER_SESSION_DIR"
```

Preview the same completed session as often as needed while adjusting the
title, metric selection, or artifact-path setting. When the payload is ready,
run the command again without `--preview` to post it. Preview and posting use
the same artifact loader and payload formatter.

`post-slack` has its own exit status and never changes the completed session's
status or artifacts. Any policy that combines run status, report status,
retries, locking, or recurrence belongs to the caller.

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

The harness controller calls those APIs in its Python process; this does not
force the ingest workload into local/in-process mode. A runfile with `mode:
batch` still resolves to Ray-backed batch ingest, while `mode: local` resolves
to local in-process ingest. Stdout remains diagnostic only; artifacts and exit
codes are the contract.

## Artifacts

Use one entrypoint for each lifecycle level instead of scanning the directory:

- `status.json`: current phase and concise failure state while a run is active.
- `results.json`: authoritative terminal result, summary metrics, and relative
  pointers to detailed run evidence.
- `session_summary.json`: authoritative terminal result for `run-set` and
  `run-files` sessions, with relative pointers to each child run.

The terminal files are deliberately compact. A successful run has this shape:

```json
{
  "benchmark": "jp20_beir",
  "dataset": "jp20",
  "success": true,
  "exit_code": 0,
  "summary_metrics": {"files": 20, "pages": 1940, "recall_5": 0.887},
  "failure": null,
  "artifacts": {"log": "run.log", "lancedb": "lancedb"}
}
```

A multi-run session summarizes its children without embedding their detailed
results:

```json
{
  "session_type": "runfiles",
  "session_name": "library_beir",
  "success": true,
  "exit_code": 0,
  "runs": [
    {
      "benchmark": "jp20_beir",
      "success": true,
      "results_path": "001_jp20_beir/results.json"
    }
  ]
}
```

Follow the pointers in `results.json` only when deeper evidence is needed:

- `events.jsonl`: phase transitions and harness events.
- `runfile.json`: original runfile payload, when a runfile was used.
- `resolved_benchmark.json`: exact effective benchmark spec.
- `ingest_plan.json`: redacted executable ingest plan.
- `query_plan.json`: executable query plan.
- `environment.json`: commit and runtime context.
- `run.log`: captured lower-level stdout/stderr and full exception tracebacks.
- `beir_metrics.json`: full BEIR metric family when evaluation executes.
- `beir_run.trec`: standard TREC runfile when evaluation executes.
- `query_results.jsonl`: per-query latency and ranked hits.
- `lancedb/`: the ingested table and index used by the run.

Artifact manifest paths are relative to the run directory so a copied session
remains readable. Failure messages in `status.json` and `results.json` are kept
concise; use the listed debug artifacts, normally `run.log`, for full traces.
When an output directory is reused, the harness removes only its known generated
artifacts before starting; unrelated files in that directory are preserved.

`environment.json` records an allowlisted set of reproducibility diagnostics,
including source revision, Python and package versions, accelerator information,
and selected Hugging Face, Ray, CUDA, and vLLM settings. Credentials and webhook
values are not recorded.

New runs keep summary metrics inside `results.json`; they do not emit a separate
`summary_metrics.json`. `diff` and `post-slack` retain read compatibility with
older harness artifacts.

Dry-runs write only planning artifacts. They do not create empty `run.log`,
`beir_metrics.json`, `beir_run.trec`, `query_results.jsonl`, or `lancedb/`.

## Gates

Use explicit `--require` gates. Gate expressions compare keys from
`results.json.summary_metrics`:

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

1. Use this harness only for registered benchmark/evaluation work. Use
   `retriever ingest` and `retriever query` for direct product workflows.
2. Start with `retriever harness list --runsets --json`, then inspect the target
   with `retriever harness show <benchmark> --json`.
3. Copy `dataset_paths.example.yaml` outside the repository and set the paths
   available on the current machine.
4. Use `run-files --dataset-paths ...` with one runfile for one dataset or
   multiple runfiles for a suite.
5. Always set `--output-dir`, and use `--dry-run` before expensive execution
   when changing paths, overrides, or gates.
6. Use explicit `--require` gates from `EXPECTED_RESULTS.md`.
7. Decide success from the process exit code and `results.json` for one run or
   `session_summary.json` for a session.
8. Read `summary_metrics` from the applicable terminal JSON file. Follow its
   pointers to `run.log` or other detailed evidence only when needed.
9. Do not parse progress bars, human CLI formatting, or raw stdout as the API.
10. Treat `post-slack` as optional post-processing. Previewing or posting never
    executes a benchmark.

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
- [`HANDOFF.md`](HANDOFF.md): concise maintainer-oriented implementation notes.
