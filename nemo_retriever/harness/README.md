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
- `diff`: compare two run artifact directories by `summary_metrics.json`.

Legacy graph-pipeline harness execution, sweep, nightly, runner, reporting, and
portal commands were removed.

## Runfiles

Runfiles are a small reproducibility helper for agents, handoffs, and
orchestrators. They describe one concrete run request:

- registered `benchmark`
- optional `name`, `mode`, `run_id`, and `output_dir`
- optional `set` overrides
- optional `require` metric gates

Runfiles cannot define new datasets or benchmarks. Add recurring benchmark
definitions to the Python registry instead.

The harness accepts JSON, YAML, or YML runfiles. The checked-in JP20 example is
[`runfiles/jp20_beir.json`](runfiles/jp20_beir.json).

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
