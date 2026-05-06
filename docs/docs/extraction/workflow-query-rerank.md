# Workflow: Query and rerank

After documents are [ingested](workflow-document-ingestion.md), [chunked](chunking.md), and [indexed](vdbs.md), applications run retrieval (semantic and optionally hybrid search) and optional reranking to improve top-K quality.

Follow these steps:

1. **Query.** Run searches against your vector store with filters as needed. Refer to [Semantic and hybrid retrieval](vdbs.md#semantic-and-hybrid-retrieval) and [Custom metadata and filtering](custom-metadata.md).
2. **Combine hybrid patterns.** Merge dense vectors with sparse or full-text signals where supported, such as LanceDB hybrid mode in [Vector databases](vdbs.md).
3. **Rerank.** Apply a reranker NIM for a second-stage score on candidates. Refer to the [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md#model-hardware-requirements) for reranker NIM options and GPU notes.

**Bridge from extraction**

Pipeline stages are summarized in [Concepts](concepts.md). Retrieval quality evaluation appears in [Evaluate on your data](evaluate-on-your-data.md).
