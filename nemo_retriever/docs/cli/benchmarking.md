# Benchmarking with the `retriever` CLI

This page covers benchmark workflows for NeMo Retriever Library. See also
`docs/docs/extraction/benchmarking.md` for published extraction benchmark notes and
[`nemo_retriever/harness/HANDOFF.md`](../../harness/HANDOFF.md) for operator-oriented
notes on `retriever harness`.

`retriever harness` is the single CLI entry point for benchmark orchestration.

## Harness (development / experimental)

Run from the repository root or any directory. The harness uses code-owned
benchmark names from `nemo_retriever.harness.benchmark_registry`; use
`retriever harness list` to discover the available benchmarks and runsets.

```bash
# List benchmark registry entries, optionally including runsets
retriever harness list
retriever harness list --runsets

# Inspect one concrete benchmark spec
retriever harness show jp20_beir

# Run one benchmark and write stable artifacts
retriever harness run jp20_beir

# Run one benchmark in batch mode
retriever harness run bo767_beir --mode batch

# Override a resolved config key for this run
retriever harness run bo767_beir --set query.top_k=5

# Expand and run a code-owned benchmark runset
retriever harness run-set jp20_core
```

Related commands:

```bash
retriever harness --help
retriever harness list --help
retriever harness show --help
retriever harness run --help
retriever harness run-set --help
retriever harness diff --help
```

### Agentic BEIR evaluation

Harness runs use the standard dense retrieval path unless agentic retrieval is
enabled in the resolved benchmark query config. Set `query.agentic: true` in a
code-owned benchmark or runfile, or use repeatable `--set` overrides on the CLI.
The agentic harness path runs the same ReAct retrieval graph used by root query,
but only after ingest and only for BEIR evaluation (`evaluation.mode: beir`).

Minimal BEIR override example:

```bash
retriever harness run jp20_beir \
  --set query.agentic=true \
  --set query.agentic_llm_model=nvidia/llama-3.3-nemotron-super-49b-v1.5
```

Useful agentic query overrides:

- `query.agentic_llm_model` — chat model used by the ReAct and selection agents;
  required when `query.agentic=true`.
- `query.agentic_invoke_url` — OpenAI-compatible chat-completions endpoint. Omit
  to use the built-in NVIDIA endpoint.
- `query.agentic_backend_top_k` — backend candidate pool per ReAct retrieval
  call. Must be at least the final requested metric depth (`max(evaluation.ks)`).
- `query.agentic_react_max_steps` — maximum ReAct loop iterations per query
  (defaults to `50`).
- `query.agentic_text_truncation` — max characters of each candidate shown to
  the agent; `0` disables truncation.
- `query.agentic_num_concurrent` — number of queries the agent batch runs
  concurrently (defaults to `1`).
- `query.agentic_temperature` — defaults to `0.0`; hosted/default NVIDIA
  endpoints are validated as `0.0..1.0`, while other OpenAI-compatible endpoints
  allow `0.0..2.0`.
- `query.agentic_reasoning_effort` — optional provider-specific field forwarded
  only when configured.

### Image storage

For normal ingest, configure image persistence on `retriever ingest` with
`--store-images-uri <uri>` (local path or fsspec URI). Stored assets follow
`--embed-granularity` (page vs element images). Benchmark-specific ingest
settings belong in the harness benchmark spec or a small `--set` override.

## Notes

- **Configuration:** `retriever harness` uses code-owned benchmarks/runsets from
  `nemo_retriever.harness.benchmark_registry`; use `--set KEY=VALUE` for small
  per-run config overrides.
- **Launcher:** for internal benchmarking, `retriever harness run BENCHMARK` and
  `retriever harness run-set RUNSET` are the benchmark orchestration entry points
  (development / experimental; no guarantees).
