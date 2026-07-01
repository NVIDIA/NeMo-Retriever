# Workflow: Agentic retrieval

**Agentic retrieval** describes patterns where a planner or tool-using agent queries retrieval systems in a loop (often combining multiple searches, filters, and rerankers) instead of sending a single static query.

NeMo Retriever Library provides ingestion, embedding, storage, and retrieval building blocks (jobs, chunking, vector stores, reranking) that you orchestrate in application code or frameworks.

## MCP access for agents

`retriever service start` mounts a FastMCP HTTP endpoint at `/mcp` by default. Agents can use that endpoint to call the running service for health checks, pipeline introspection, document ingestion, job status, VectorDB query, and answer generation. If service auth is enabled, the MCP endpoint uses the same bearer-token middleware as the REST API.

For local stdio-based agents, run the MCP server as a shim that points at an existing retriever service:

```bash
retriever service mcp-stdio \
  --service-url http://localhost:7670 \
  --api-token "$NEMO_RETRIEVER_API_TOKEN"
```

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
