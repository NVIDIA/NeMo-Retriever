# Evaluate BO767 Retrieval

BO767 is a checked-in batch benchmark. Configure the host's dataset paths, dry
run the exact request, and then execute it.

```bash
cp nemo_retriever/harness/dataset_paths.example.yaml \
  /local/path/to/dataset_paths.yaml
${EDITOR:-vi} /local/path/to/dataset_paths.yaml

uv run --project nemo_retriever retriever harness run-files \
  --session-name bo767_beir \
  --output-dir /local/path/to/retriever-artifacts/bo767-beir \
  --dataset-paths /local/path/to/dataset_paths.yaml \
  --dry-run \
  nemo_retriever/harness/runfiles/bo767_beir.json
```

Confirm that `session_summary.json` succeeds and inspect the child
`resolved_benchmark.json`. Remove `--dry-run` to execute the benchmark.

Read `session_summary.json` first and the child `results.json` for terminal
metrics. Use `run.log` only when deeper diagnostics are needed.

- [Library harness guide](../nemo_retriever/harness/docs/library.md)
- [BO767 dataset facts and observations](../nemo_retriever/harness/docs/expected-results.md#bo767-observations)
- [Shared artifact contract](../nemo_retriever/harness/README.md#results-and-artifacts)
