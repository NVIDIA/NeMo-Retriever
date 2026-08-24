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

This self-hosted NIM configuration gap does not apply to NVIDIA-hosted Build endpoints. A Helm-deployed Super-49B NIM rejects tool-call requests until you add the passthrough arguments. Refer to [Self-hosted Helm Super-49B](#self-hosted-helm-super-49b).

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

## Self-hosted Helm Super-49B { #self-hosted-helm-super-49b }

Use this path when the agent LLM is the Helm-deployed Super-49B NIM rather than local in-process vLLM or an NVIDIA-hosted Build endpoint.

`nimOperator.answer_llm.enabled=true` deploys Super-49B and auto-wires it only to `serviceConfig.llm` for `POST /v1/answer`. That answer path sends a plain text-generation request and does not require tool calling. `serviceConfig.agentic` is a separate block and stays empty unless you set it.

The chart starts that NIM with `NIM_PASSTHROUGH_ARGS=--disable-custom-all-reduce`. The agentic ReAct loop sends OpenAI-style tool-call messages with `tool_choice=auto`. A self-hosted vLLM-backed Super-49B NIM rejects those requests with HTTP 400 unless you also pass `--enable-auto-tool-choice` and `--tool-call-parser llama3_json`.

You can reuse the same Super-49B NIM for agentic retrieval after you add those arguments. `POST /v1/answer` continues to work.

If you set `nimOperator.answer_llm.env` in a values file, include the full list. Change only the `NIM_PASSTHROUGH_ARGS` value.

```yaml
nimOperator:
  answer_llm:
    enabled: true
    env:
      - name: NIM_HTTP_API_PORT
        value: "8000"
      - name: NIM_TENSOR_PARALLEL_SIZE
        value: "2"
      - name: NIM_PASSTHROUGH_ARGS
        value: "--disable-custom-all-reduce --enable-auto-tool-choice --tool-call-parser llama3_json"
      - name: NCCL_IB_DISABLE
        value: "1"
      - name: NCCL_P2P_DISABLE
        value: "1"
```

Equivalent `--set` override when you do not use a values file. Helm `--set` replaces the `env` list, so include every Super-49B environment entry and change only the `NIM_PASSTHROUGH_ARGS` value:

```bash
helm upgrade --install retriever ./nemo_retriever/helm \
  --set nimOperator.answer_llm.enabled=true \
  --set nimOperator.answer_llm.env[0].name=NIM_HTTP_API_PORT \
  --set-string nimOperator.answer_llm.env[0].value=8000 \
  --set nimOperator.answer_llm.env[1].name=NIM_TENSOR_PARALLEL_SIZE \
  --set-string nimOperator.answer_llm.env[1].value=2 \
  --set nimOperator.answer_llm.env[2].name=NIM_PASSTHROUGH_ARGS \
  --set-string nimOperator.answer_llm.env[2].value="--disable-custom-all-reduce --enable-auto-tool-choice --tool-call-parser llama3_json" \
  --set nimOperator.answer_llm.env[3].name=NCCL_IB_DISABLE \
  --set-string nimOperator.answer_llm.env[3].value=1 \
  --set nimOperator.answer_llm.env[4].name=NCCL_P2P_DISABLE \
  --set-string nimOperator.answer_llm.env[4].value=1
```

After the NIM is Ready, confirm the passthrough arguments:

```bash
kubectl exec -n <namespace> deploy/answer-llm -- printenv NIM_PASSTHROUGH_ARGS
```

The value must include `--enable-auto-tool-choice` and `--tool-call-parser llama3_json`.

Forward the answer LLM for CLI use:

```bash
kubectl port-forward -n <namespace> service/answer-llm 9000:8000
```

Then run the remote command in [Remote OpenAI-compatible NIM or hosted endpoint](#remote-openai-compatible-endpoint). Point `--agentic-invoke-url` at `http://localhost:9000/v1/chat/completions` and set `--agentic-llm-model` to `nvidia/llama-3.3-nemotron-super-49b-v1.5`. Reuse the same embedding invoke URL and model name that you used at ingest.

For service-mode `POST /v1/query` with `agentic=true` and the MCP `agentic_query` tool, also set `serviceConfig.agentic`. The chart does not copy `answer_llm` into this block.

```yaml
serviceConfig:
  agentic:
    enabled: true
    llmModel: nvidia/llama-3.3-nemotron-super-49b-v1.5
    invokeUrl: http://answer-llm:8000/v1/chat/completions
```

`invokeUrl` uses the in-cluster Super-49B service. Change the hostname if you override `nimOperator.answer_llm.nimServiceName`. `llmModel` is the model ID advertised by the NIM, not the LiteLLM `openai/` prefix used by `serviceConfig.llm.model`.

If you register MCP retrieval tools, set `serviceConfig.mcp.queryMethods` to `agentic` or `all` as well. Agentic MCP tools are omitted unless `serviceConfig.agentic.enabled` is true.

For other self-hosted OpenAI-compatible NIMs, enable automatic tool choice and the parser that model requires. The `llama3_json` parser is the verified Super-49B setting.

For chart keys, refer to [Agentic retrieval (self-hosted Super-49B)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#agentic-retrieval-llm) in the Helm chart README.

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

On Kubernetes, the Helm chart maps the same knobs under `serviceConfig.agentic`. Enabling `nimOperator.answer_llm` does not populate this block. Refer to [Self-hosted Helm Super-49B](#self-hosted-helm-super-49b) and the [Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#agentic-retrieval-llm).

The VectorDB service runs up to four non-agentic queries concurrently by default.
Set `--max-concurrent-queries` when starting `nemo_retriever.service.vectordb_app`
to use a different positive limit.

REST clients set the flag on `/v1/query`:

```bash
curl -X POST http://localhost:7670/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "find documents about parser behavior", "top_k": 5, "agentic": true}'
```

The service-query CLI sends the same request when you pass `--agentic`:

```bash
retriever query service "find documents about parser behavior" \
  --service-url http://localhost:7670 \
  --agentic
```

This command uses the service's configured agentic LLM and embedding endpoints.
It does not start a local agent LLM or accept the local query command's
`--agentic-llm-model` and `--agentic-invoke-url` options.

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

One-pass retrieval returns text-enriched chunk hits. Agentic retrieval ranks documents, and returns the same hit fields as one-pass retrieval for each selected document. The agent works with reduced candidate records internally, but the selected documents are rehydrated at the end of the loop from the retrieval hop that returned them.

Every agentic hit carries the one-pass hit fields (`text`, `metadata`, `source`, `source_id`, `path`, `page_number`, `pdf_basename`, `pdf_page`, scores, and related) plus these agentic annotations:

- `doc_id` — the document identifier the agent selected.
- `rank` — the position in the final ranking.
- `result_source` — `final_results`, `rrf`, or `selection_agent`, depending on which stage produced the ranked ID.

CLI `retriever query --agentic` prints those hits as JSON objects.

Service `POST /v1/query` with `agentic=true` uses the same hits envelope as classic retrieval. Successful responses set `query_mode` to `"agentic"`. Classic dense or hybrid `/v1/query` (including `format=evidence`) sets `query_mode` to `"classic"`. For backward compatibility with the previous agentic service contract, service and MCP hits also copy `rank` and `result_source` under `metadata`; the top-level fields are authoritative and carry the same values.

An agent can name a document that no retrieval hop returned, which leaves nothing to rehydrate. Those hits report null one-pass fields, and `source` falls back to `doc_id`.

## Failure and retry behavior { #failure-and-retry-behavior }

Operational failures from the agent LLM or retrieval tool, including embedding, vector database, and reranker endpoint failures, terminate the query with an error instead of returning a successful empty result.

An HTTP `400` from the chat-completions NIM with `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set` means the self-hosted endpoint is not tool-call ready. Refer to [Self-hosted Helm Super-49B](#self-hosted-helm-super-49b). The CLI then exits with `Agentic retrieval failed (llm_call_failed)`.

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
- The default Helm `answer_llm` Super-49B NIM is limited to `POST /v1/answer` until you add the tool-call passthrough arguments. Enabling `nimOperator.answer_llm` does not configure `serviceConfig.agentic`.
- Agentic results are document IDs, not chunk text. Downstream answer generation must load source documents by those IDs if it needs passage text.
- Service agentic queries accept a single query string, `format=hits` only, and cannot combine `rerank=true` on the same `/v1/query` request. On the CLI, `--rerank` applies to each agent retrieve hop.

## Related Topics { #related-topics }

- [Agentic retrieval (concept)](agentic-retrieval-concept.md)
- [Semantic retrieval](vdbs.md#semantic-retrieval)
- [Metadata and filtering](vdbs.md#metadata-and-filtering)
- [Evaluate on your data](evaluate-on-your-data.md)
- [Authentication and API keys](api-keys.md)
- [CLI reference: Agentic retrieval](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/docs/cli/README.md#agentic-retrieval)
- [Helm chart README: Agentic retrieval (self-hosted Super-49B)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#agentic-retrieval-llm)
- [Release notes](releasenotes.md#retrieval-and-rag)
