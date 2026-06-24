<!-- SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Retriever harness handoff

Operator notes for the revamped internal `retriever harness`.

## Scope

- Developer-only benchmark harness for Retriever engineers.
- Artifact-first: stdout is for humans; agents and orchestrators should read
  JSON artifacts.
- Uses shared Retriever ingest/query workflow code.
- Does not route phase-one commands through `retriever pipeline run` or
  `nemo_retriever.examples.graph_pipeline`.

## Core commands

Run from the repository root through the `nemo_retriever` project:

```bash
uv run --project nemo_retriever retriever harness list --json
uv run --project nemo_retriever retriever harness show jp20_beir --json
uv run --project nemo_retriever retriever harness run jp20_beir \
  --dry-run \
  --output-dir /tmp/retriever-harness-dry-run \
  --json

uv run --project nemo_retriever retriever harness run jp20_beir \
  --output-dir /tmp/retriever-harness-jp20-beir \
  --require 'files==20' \
  --require 'pages==1940' \
  --require 'query_count==115' \
  --require 'recall_5>=0.85' \
  --require 'ndcg_10>=0.75' \
  --json
```

The primary command surface is intentionally small:

- `list`
- `show`
- `run`
- `run-set`
- `diff`

Legacy graph-pipeline harness execution, sweep, nightly, runner, reporting, and
portal paths have been removed.

Useful negative-path checks:

```bash
uv run --project nemo_retriever retriever harness run jp20_beir \
  --dry-run \
  --output-dir /tmp/retriever-harness-invalid \
  --set query.nope=1 \
  --json

uv run --project nemo_retriever retriever harness run jp20_beir \
  --dry-run \
  --output-dir /tmp/retriever-harness-missing \
  --set dataset.path=/tmp/retriever-harness-does-not-exist \
  --json

uv run --project nemo_retriever retriever harness run jp20_beir \
  --dry-run \
  --output-dir /tmp/retriever-harness-gate-fail \
  --require 'files>20' \
  --json
```

Expected exit codes:

- `0`: success
- `2`: invalid benchmark/config/override
- `3`: dataset or input missing
- `10`: ingest failure
- `11`: query failure
- `12`: evaluation failure
- `20`: metric gate failure
- `30`: artifact write failure
- `70`: unexpected internal error

## Built-in benchmarks

The current code-owned registry includes:

- `jp20_smoke`: cheap JP20 fast-text ingest check; no BEIR queries.
- `jp20_beir`: full JP20 ingest plus 115 BEIR queries and recall/NDCG metrics.
- `bo20_smoke`
- `bo767_beir`
- `financebench_beir`
- `bo10k_beir_fast_text`

`earnings_consulting` corpus exists locally, but the old
`data/earnings_consulting_multimodal.csv` query/qrels file is not present in
this checkout. Do not treat `earnings_beir` as phase-one-ready until that file
is restored or replaced.

## Artifact contract

Per run, read these files instead of scraping CLI text:

- `status.json`
- `events.jsonl`
- `resolved_benchmark.json`
- `ingest_plan.json`
- `query_plan.json`
- `summary_metrics.json`
- `environment.json`
- `results.json`
- `run.log` when ingest/query execution runs
- `beir_metrics.json` when BEIR evaluation executes
- `beir_run.trec` when BEIR evaluation executes
- `query_results.jsonl` when queries execute

Dry-runs resolve the benchmark, ingest plan, query plan, and summary metrics,
but intentionally do not execute ingest or BEIR queries. Dry-run artifact
payloads list only files that were actually written; they do not create or
advertise empty BEIR result files.

Non-dry harness runs use the same quiet capture behavior as `retriever ingest
--quiet`: noisy lower-level model/progress output is suppressed on success and
flushed on failure. Suppressed output is persisted to `run.log` for both
successful and failed non-dry runs.

## Expectations And Gates

Use explicit `--require` gates. The harness intentionally does not encode
recommended thresholds in Python; known results live in
`nemo_retriever/harness/EXPECTED_RESULTS.md` so humans and agents can inspect
and update them as benchmarks mature.

Current JP20 examples:

- `jp20_smoke`: `files==20`, `pages==1940`
- `jp20_beir`: `files==20`, `pages==1940`, `query_count==115`,
  `recall_5>=0.85`, `ndcg_10>=0.75`

Dry-runs can only evaluate gates for metrics available without execution, such
as `files` and `pages`.

## Current validated smoke

Validated in `/raid/jioffe/nv-ingest-harness-revamp`:

- `retriever harness list --json` exits `0`.
- `retriever harness show jp20_beir --json` exits `0`.
- `retriever harness run jp20_beir --dry-run --output-dir /tmp/retriever-harness-dry-run --json` exits `0`.
- Invalid override `--set query.nope=1` exits `2`.
- Missing dataset override exits `3`.
- Passing gate `--require 'files>=20'` exits `0`.
- Failing gate `--require 'files>20'` exits `20` with
  `failure_reason=metric_gate_failed`.
- Invalid gate syntax exits `2` with `failure_reason=invalid_metric_gate`.
- `run-set jp20_core --dry-run --require 'files>=20'` exits `0`.
- `run-set jp20_core --dry-run --require 'files>20'` exits `20`.
- `retriever harness run jp20_beir --require 'recall_5>=0.85' --require 'ndcg_10>=0.75' --json`
  exits `0` on the current local dataset/hardware.
