# Workflow: Agentic retrieval

Use this workflow after you have ingested documents into a LanceDB table. Agentic retrieval does not ingest files. It queries the same table, embedding model, and storage flags as one-pass `retriever query`.

**Agentic retrieval** runs a large language model (LLM) Reason and Act (ReAct) loop: the agent issues several retrieval sub-queries, fuses candidates with reciprocal rank fusion, and selects a final document ranking. **One-pass retrieval** sends a single dense or hybrid query and returns text-enriched chunk hits. For the concept distinction, refer to [Agentic retrieval (concept)](agentic-retrieval-concept.md).

## Query with the CLI { #query-with-the-cli }

`retriever query --agentic` is the one-shot CLI path. It searches the LanceDB table built by `retriever ingest`. Reuse the same `--lancedb-uri`, `--table-name`, and embedding model that you used at ingest. When `--embed-model-name` is omitted, agentic retrieval uses the selected table's model.

### Local in-process vLLM { #local-in-process-vllm }

The CLI and NRB agentic benchmark paths default to an in-process local vLLM agent
LLM. If you omit `--agentic-llm-model` and `--agentic-invoke-url`, the library
loads `nemotron-8b` (`nvidia/Llama-3.1-Nemotron-Nano-8B-v1`) on the local CUDA
host. This requires a Linux CUDA GPU and the `[local]` extra.

GPU placement follows process-level vLLM behavior. Set `CUDA_VISIBLE_DEVICES` before you start the command.

```bash
CUDA_VISIBLE_DEVICES=0 retriever query "find documents about parser behavior" --agentic
```

The larger `super-49b` profile is also supported. Pass `--agentic-local-tensor-parallel-size 2` with two visible GPUs for that profile.

```bash
CUDA_VISIBLE_DEVICES=0,1 retriever query "find documents about parser behavior" \
  --agentic \
  --agentic-llm-model super-49b \
  --agentic-local-tensor-parallel-size 2
```

Custom in-process LLMs are not supported. The agent loop depends on OpenAI-style tool-call messages. Use an OpenAI-compatible endpoint for custom models.

### Remote OpenAI-compatible NIM or hosted endpoint { #remote-openai-compatible-endpoint }

Providing `--agentic-invoke-url` routes the agent to that remote chat-completions endpoint. `--agentic-llm-model` is required on the remote path and is sent as the remote model ID. The LLM client defaults to `callable`, which calls the endpoint over the shared chat-completions HTTP client and needs no extra LLM SDK.

Self-hosted NIM or a local OpenAI-compatible server:

```bash
retriever query "find documents about parser behavior" \
  --agentic \
  --agentic-llm-model nvidia/llama-3.3-nemotron-super-49b-v1.5 \
  --agentic-invoke-url http://localhost:9000/v1/chat/completions
```

NVIDIA-hosted Build endpoint (requires `NVIDIA_API_KEY`; `NGC_API_KEY` is the fallback):

```bash
retriever query "find documents about parser behavior" \
  --agentic \
  --agentic-llm-model nvidia/llama-3.3-nemotron-super-49b-v1.5 \
  --agentic-invoke-url https://integrate.api.nvidia.com/v1/chat/completions
```

`--agentic-local-tensor-parallel-size` is ignored when `--agentic-invoke-url` is set. For hosted model IDs, refer to [Default NVCF endpoints](prerequisites-support-matrix.md#default-nvcf-endpoints). For key setup, refer to [Authentication and API keys](api-keys.md).

### CLI options { #cli-options }

The following options apply only with `--agentic`. For the full flag list, refer to [Agentic retrieval](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/docs/cli/README.md#agentic-retrieval) in the CLI reference.

| Option | Default | Notes |
|---|---|---|
| `--agentic-llm-model` | `nemotron-8b` when no invoke URL is set | Local profile alias (`nemotron-8b` or `super-49b`) or remote model ID when `--agentic-invoke-url` is set. |
| `--agentic-invoke-url` | unset (local vLLM) | OpenAI-compatible `/v1/chat/completions` endpoint. Required together with `--agentic-llm-model` for remote runs. |
| `--agentic-local-tensor-parallel-size` | `1` | vLLM `tensor_parallel_size` for the in-process agent LLM. Set to `2` for local `super-49b`. Ignored when `--agentic-invoke-url` is set. |
| `--agentic-react-max-steps` | `50` | Maximum ReAct loop iterations. |
| `--agentic-reasoning-effort` | `high` | Forwarded on OpenAI-compatible agent LLM calls. Ignored by the local adapter. |

Embedding credentials use `NVIDIA_API_KEY` or `NGC_API_KEY` when you call a remote embedding endpoint. The CLI also reuses `--embed-invoke-url`, `--top-k`, `--lancedb-uri`, and `--table-name` from standard retrieval.

## Enable agentic retrieval in the service { #enable-agentic-retrieval-in-the-service }

Retriever Service exposes agentic retrieval on `POST /v1/query` when `agentic.enabled` is true. Service mode requires remote OpenAI-compatible LLM and embedding endpoints. Local in-process vLLM remains available on the one-shot CLI and harness paths only.

Enable agentic retrieval in `retriever-service.yaml`:

```yaml
agentic:
  enabled: true
  llm_model: nvidia/llama-3.3-nemotron-super-49b-v1.5
  invoke_url: https://your-llm.example/v1/chat/completions
  reasoning_effort: high
  backend_top_k: 20
  react_max_steps: 50
  request_timeout_s: 1800
```

`agentic.invoke_url` and `agentic.llm_model` are required when `agentic.enabled` is true. The VectorDB process owns the LanceDB volume and executes the agentic workflow. Start it with matching `--agentic`, `--agentic-llm-model`, and `--agentic-invoke-url` options. LLM and embedding credentials are resolved from the service process environment (`NVIDIA_API_KEY`, then `NGC_API_KEY`).

Agentic service requests use the configured remote embedding endpoint for retrieval. The result-selection graph does not require a local embedding model or Hugging Face cache.

On Kubernetes, the Helm chart maps the same knobs under `serviceConfig.agentic`. Refer to the [Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md).

The VectorDB service runs up to four non-agentic queries concurrently by default.
Set `--max-concurrent-queries` when starting `nemo_retriever.service.vectordb_app`
to use a different positive limit.

REST clients set the flag on `/v1/query`:

```bash
curl -X POST http://localhost:7670/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "find documents about parser behavior", "top_k": 5, "agentic": true}'
```

When service auth is enabled, send `Authorization: Bearer <token>` (`NEMO_RETRIEVER_API_TOKEN`). Requests with `agentic: true` return HTTP `400` when agentic retrieval is not configured on the service.

`top_k` cannot exceed the configured `agentic.backend_top_k` (default 20). Agentic queries are capped at 4,096 characters.

## Query with MCP { #query-with-mcp }

`retriever service start` mounts a FastMCP HTTP endpoint at `/mcp` by default. Model Context Protocol (MCP) agents can use that endpoint to call the running service for health checks, pipeline introspection, document ingestion, job status, VectorDB query, agentic retrieval, and answer generation. If service auth is enabled, the MCP endpoint uses the same bearer-token middleware as the REST API.

Plain and agentic retrieval share `POST /v1/query` and the same hits response envelope. They are separate MCP tools so agents can choose explicitly:

- `query` calls `POST /v1/query` with `agentic=false` for one-pass dense or hybrid retrieval.
- `agentic_query` calls `POST /v1/query` with `agentic=true` and runs the ReAct retrieval workflow. It is added to MCP when `agentic.enabled` is true.

Use `--query-methods classic` (default), `agentic`, or `all` to choose which retrieval tools the MCP server registers. Mounted `/mcp` uses the same knob through `mcp.query_methods` in the service config. Agentic tools are omitted unless `agentic.enabled` is also true.

For local stdio-based agents, run the MCP server as a shim that points at an existing retriever service:

```bash
retriever service mcp-stdio \
  --service-url http://localhost:7670 \
  --query-methods agentic \
  --api-token "$NEMO_RETRIEVER_API_TOKEN"
```

For remote agents, expose the retriever service URL and configure the agent to connect to:

```text
https://<retriever-service-host>/mcp
```

The `ingest_documents` MCP tool accepts either paths visible to the MCP server process or inline `content_base64` document bytes. Use inline base64 for remote agents whose local files are not present on the service host.

## Result contract { #result-contract }

One-pass retrieval returns text-enriched chunk hits. Agentic retrieval returns a document-level ranking.

CLI `retriever query --agentic` prints JSON objects with `rank`, `doc_id`, and `result_source`. `result_source` is `final_results`, `rrf`, or `selection_agent`, depending on which stage produced the ranked ID.

Service `POST /v1/query` with `agentic=true` uses the same hits envelope as classic retrieval:

- Successful responses set `query_mode` to `"agentic"`. Classic dense or hybrid `/v1/query` (including `format=evidence`) sets `query_mode` to `"classic"`.
- Each hit places the selected `doc_id` in `source`.
- `metadata` carries `result_source` and `rank`.
- Chunk-level fields (`text`, `page_number`, scores, and related) are unset.

## Failure and retry behavior { #failure-and-retry-behavior }

Operational failures from the agent LLM or retrieval tool, including embedding, vector database, and reranker endpoint failures, terminate the query with an error instead of returning a successful empty result.

On the service:

- HTTP `400` when `agentic.enabled` is false.
- HTTP `422` when the query exceeds 4,096 characters, `format` is not `hits`, or `rerank` is combined with `agentic`.
- HTTP `501` when agentic service queries have no remote embedding endpoint.
- HTTP `503` with a `Retry-After: 30` header when every dedicated agentic worker is busy. The service sheds load instead of queueing behind a multi-minute run.
- HTTP `502` when the gateway cannot reach the VectorDB process.

Agentic runs use a dedicated worker pool in the VectorDB process so they cannot exhaust the capacity used by plain queries. A ReAct run cannot be interrupted once started, so a worker stays occupied until it finishes even if the caller times out or disconnects.

## Limitations and resource requirements { #limitations-and-resource-requirements }

- Local in-process agent LLMs are limited to the tested `nemotron-8b` and `super-49b` profiles. Custom in-process models require an OpenAI-compatible endpoint instead.
- Local CLI and harness runs need a CUDA GPU host and the `[local]` extra. `super-49b` needs two visible GPUs and `--agentic-local-tensor-parallel-size 2`.
- Retriever Service agentic queries require a remote chat-completions URL, a remote embedding endpoint, and matching credentials in the process environment.
- Agentic results are document IDs, not chunk text. Downstream answer generation must load source documents by those IDs if it needs passage text.
- Service agentic queries accept a single query string, `format=hits` only, and cannot combine `rerank=true` on the same `/v1/query` request. On the CLI, `--rerank` applies to each agent retrieve hop.

## Related Topics { #related-topics }

- [Agentic retrieval (concept)](agentic-retrieval-concept.md)
- [Semantic retrieval](vdbs.md#semantic-retrieval)
- [Metadata and filtering](vdbs.md#metadata-and-filtering)
- [Evaluate on your data](evaluate-on-your-data.md)
- [Authentication and API keys](api-keys.md)
- [CLI reference: Agentic retrieval](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/docs/cli/README.md#agentic-retrieval)
- [Release notes](releasenotes.md#retrieval-and-rag)
