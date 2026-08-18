# Agentic retrieval (concept)

**Agentic retrieval** is iterative, tool-driven retrieval. A large language model (LLM) agent plans steps, issues search tool calls, fuses candidates, and ranks documents until it has enough context. **One-pass retrieval** sends a single static query through dense or hybrid search and returns chunk-level hits.

NeMo Retriever Library includes a first-class agentic query path. `retriever query --agentic`, `POST /v1/query` with `agentic=true`, and the `agentic_query` Model Context Protocol (MCP) tool run a Reason and Act (ReAct) loop over the same LanceDB table that one-pass retrieval uses. You do not have to implement the agent loop in application code.

The agentic path returns ranked document IDs rather than text-enriched chunks. Local CLI and harness runs default to an in-process vLLM agent LLM. Retriever Service requires a remote OpenAI-compatible chat-completions endpoint.

For commands, service configuration, request and response contracts, and failure behavior, refer to [Workflow: Agentic retrieval](workflow-agentic-retrieval.md).

## Related Topics { #related-topics }

- [Workflow: Agentic retrieval](workflow-agentic-retrieval.md)
- [Semantic retrieval](vdbs.md#semantic-retrieval)
- [Starter kits](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/README.md)
