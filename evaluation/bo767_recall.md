# Evaluate BO767 Retrieval with the Retriever Harness

Use the Retriever harness for the canonical BO767 end-to-end ingest, query,
and BEIR evaluation. The checked-in runfile selects batch execution and carries
the benchmark's worker tuning and required file, page, and query counts.

Run these commands from the repository root.

## Configure the Dataset Paths

Copy the example path map to an untracked location:

```bash
cp nemo_retriever/harness/dataset_paths.example.yaml /tmp/retriever-dataset-paths.yaml
```

Edit the `bo767` entry so `path` points to the directory containing the 767
documents and `query_file` points to the BO767 query/qrels CSV available on the
machine running the benchmark.

## Validate the Resolved Run

Resolve the run without launching ingestion or evaluation:

```bash
uv run --project nemo_retriever retriever harness run-files \
  --session-name bo767_beir_check \
  --output-dir /tmp/retriever-harness-bo767-check \
  --dataset-paths /tmp/retriever-dataset-paths.yaml \
  --dry-run \
  --json \
  nemo_retriever/harness/runfiles/bo767_beir.json
```

Inspect `session_summary.json`, `expanded_runs.json`, and the child run's
`resolved_benchmark.json` before starting the full run.

## Run the Evaluation

Remove `--dry-run` and choose a durable artifact directory:

```bash
uv run --project nemo_retriever retriever harness run-files \
  --session-name bo767_beir \
  --output-dir /local/path/to/retriever-artifacts/bo767-beir \
  --dataset-paths /tmp/retriever-dataset-paths.yaml \
  --json \
  nemo_retriever/harness/runfiles/bo767_beir.json
```

BO767 is a large batch benchmark. Keep the terminal process alive until the
harness reaches a terminal state; model startup and ingestion can be quiet for
extended periods.

## Read the Results

The harness artifacts are the evaluation contract:

- `status.json` reports the current phase and concise failure state.
- `results.json` is the authoritative terminal result and summary metrics.
- `session_summary.json` is the terminal result for the runfile session.
- `beir_metrics.json` contains the complete BEIR metric family.
- `query_results.jsonl` contains per-query latency and ranked hits.
- `environment.json` records the commit and runtime context.

Refer to
[`nemo_retriever/harness/EXPECTED_RESULTS.md`](../nemo_retriever/harness/EXPECTED_RESULTS.md#bo767)
for the expected BO767 counts and current reference metrics. For the complete
harness contract and troubleshooting guidance, refer to
[`nemo_retriever/harness/README.md`](../nemo_retriever/harness/README.md).
