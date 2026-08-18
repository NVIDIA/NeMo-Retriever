# Workflow: Agentic retrieval

**Agentic retrieval** describes patterns where a planner or tool-using agent queries retrieval systems in a loop (often combining multiple searches, filters, and rerankers) instead of sending a single static query.

NeMo Retriever Library provides ingestion, embedding, storage, and retrieval building blocks (jobs, chunking, vector stores, reranking) that you orchestrate in application code or frameworks.


## Local agentic retrieval

The `retriever query --agentic` and harness BEIR agentic paths default to an
in-process local vLLM agent LLM. If no agent model is provided, the library loads
`nemotron-8b` (`nvidia/Llama-3.1-Nemotron-Nano-8B-v1`) on the local CUDA host.
The larger `super-49b` profile is also supported. Pass
`--agentic-local-tensor-parallel-size 2` with two visible GPUs for that
profile (and the same pattern for other multi-GPU local TP runs).
Tensor-parallel startup automatically disables NVLink multicast collectives
(`NCCL_NVLS_ENABLE=0`, `TORCH_SYMM_MEM_DISABLE_MULTICAST=1`) when the TP
device group has no NVLink — common on dual-GPU PCIe workstations — where
those collectives abort vLLM startup. Other custom in-process LLMs are not supported yet because the agent
loop depends on OpenAI-style tool-call messages; use an OpenAI-compatible
endpoint for custom models.

```bash
retriever query "find documents about parser behavior" --agentic
```

For custom or already deployed chat models, opt into the endpoint path:

```bash
retriever query "find documents about parser behavior" \
  --agentic \
  --agentic-llm-model custom-remote-model \
  --agentic-invoke-url http://localhost:9000/v1/chat/completions
```

Providing `--agentic-invoke-url` routes the agent to that remote endpoint; the LLM
client defaults to `callable`, which calls the endpoint over the shared
chat-completions HTTP client and needs no LLM SDK installed.

Operational failures from the agent LLM or retrieval tool, including embedding,
vector database, and reranker endpoint failures, terminate the query with an
error instead of returning a successful empty result.

## MCP access for agents

`retriever service start` mounts a FastMCP HTTP endpoint at `/mcp` by default.
Agents can use that endpoint to call the running service for health checks,
pipeline introspection, document ingestion, job status, VectorDB query, agentic
retrieval, and answer generation. If service auth is enabled, the MCP endpoint
uses the same bearer-token middleware as the REST API.

Plain and agentic retrieval share `POST /v1/query` and the same hits response
envelope. They are separate MCP tools so agents can choose explicitly:

- `query` calls `POST /v1/query` with `agentic=false` for one-pass dense or hybrid retrieval.
- `agentic_query` calls `POST /v1/query` with `agentic=true` and runs the ReAct
  retrieval workflow. It is added to MCP when `agentic.enabled` is true.
  Agentic results are document-level: `source` is the selected `doc_id`,
  `metadata` carries `result_source` and `rank`, and chunk-level fields
  (`text`, `page_number`, scores, …) are unset.

Enable agentic retrieval in `retriever-service.yaml`:

```yaml
agentic:
  enabled: true
  llm_model: your-openai-compatible-model
  invoke_url: https://your-llm.example/v1/chat/completions
  reasoning_effort: high
  backend_top_k: 20
  react_max_steps: 50
  request_timeout_s: 1800
```

The VectorDB process owns the LanceDB volume and executes the agentic workflow.
Start it with matching `--agentic`, `--agentic-llm-model`, and
`--agentic-invoke-url` options. The LLM and embedding credentials are resolved
from the service process environment. Service mode requires remote
OpenAI-compatible LLM and embedding endpoints; local in-process models remain
available through the one-shot CLI and harness paths.

REST clients set the flag on `/v1/query`:

```bash
curl -X POST http://localhost:7670/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "find documents about parser behavior", "top_k": 5, "agentic": true}'
```

Successful responses include ``query_mode``: ``"agentic"`` for this path and
``"classic"`` for dense/hybrid ``/v1/query`` (including ``format=evidence``).

Requests with `agentic: true` return HTTP `400` when agentic retrieval is not
configured on the service. Agentic runs use a small dedicated worker pool in the
VectorDB process so they cannot exhaust the capacity used by plain queries. A
ReAct run cannot be interrupted once started, so a worker stays occupied until
it finishes even if the caller times out or disconnects. When every worker is
busy the endpoint sheds load with `503` and a `Retry-After` header instead of
queueing behind a multi-minute run. Agentic queries are capped at 4096
characters to bound prompt size and cost across the multi-step loop.

For local stdio-based agents, run the MCP server as a shim that points at an existing retriever service:

```bash
retriever service mcp-stdio \
  --service-url http://localhost:7670 \
  --query-methods agentic \
  --api-token "$NEMO_RETRIEVER_API_TOKEN"
```

Use `--query-methods classic` (default), `agentic`, or `all` to choose which retrieval
tools the MCP server registers. Mounted `/mcp` uses the same knob via
`mcp.query_methods` in the service config; agentic tools are omitted unless
`agentic.enabled` is also true.

For remote agents, expose the retriever service URL and configure the agent to connect to:

```text
https://<retriever-service-host>/mcp
```

The `ingest_documents` MCP tool accepts either paths visible to the MCP server process or inline `content_base64` document bytes. Use inline base64 for remote agents whose local files are not present on the service host.

**Where to go next**

Use these pages together with your orchestration layer:

- [Semantic retrieval](vdbs.md#semantic-retrieval), [Metadata and filtering](vdbs.md#metadata-and-filtering), and [Evaluate on your data](evaluate-on-your-data.md) for retrieval quality, reranking, and evaluation guidance
- [Agentic retrieval (concept)](agentic-retrieval-concept.md)
- [Release notes](releasenotes.md), which may mention agentic retrieval updates
