# Workflow: Agentic retrieval

**Agentic retrieval** describes patterns where a planner or tool-using agent queries retrieval systems in a loop (often combining multiple searches, filters, and rerankers) instead of sending a single static query.

NeMo Retriever Library provides ingestion, embedding, storage, and retrieval building blocks (jobs, chunking, vector stores, reranking) that you orchestrate in application code or frameworks.

Wire agentic loops by calling those building blocks from your framework or application: ingest and index with the [Python API](nemo-retriever-api-reference.md) or [Workflow: Ingest documents](workflow-document-ingestion.md), then expose [semantic retrieval](vdbs.md#semantic-retrieval) and [metadata filtering](vdbs.md#metadata-and-filtering) as agent tools. Refer to [Starter kits](starter-kits.md) for LangChain and LlamaIndex multimodal RAG notebooks.

**Where to go next**

Use these pages together with your orchestration layer:

- [Semantic retrieval](vdbs.md#semantic-retrieval), [Metadata and filtering](vdbs.md#metadata-and-filtering), and [Evaluate on your data](evaluate-on-your-data.md) for retrieval quality, reranking, and evaluation guidance
- [Agentic retrieval (concept)](agentic-retrieval-concept.md)
- [Release notes](releasenotes.md), which may mention agentic retrieval updates
