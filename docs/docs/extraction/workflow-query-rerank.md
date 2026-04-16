# Workflow: Query and rerank

After documents are [ingested](v2-api-guide.md), [chunked](chunking.md), and [indexed](data-store.md), applications run retrieval (semantic and optionally hybrid search) and optional reranking to improve top-K quality.

Follow these steps:

1. **Query.** Run searches against your vector store with filters as needed. See [Semantic and hybrid retrieval](semantic-hybrid-retrieval.md) and [Custom metadata and filtering](custom-metadata.md).
2. **Combine hybrid patterns.** Merge dense vectors with sparse or full-text signals where supported, such as LanceDB hybrid mode in [Vector databases](data-store.md).
3. **Rerank.** Apply a reranker NIM for a second-stage score on candidates. See [Reranking](reranking.md) and the [Support matrix](support-matrix.md) for reranker NIM options and GPU notes.

**Bridge from extraction**

Pipeline stages are summarized in [Concepts](concepts.md). Retrieval quality evaluation appears in [Benchmarking](benchmarking.md) and [Evaluate on your data](evaluate-on-your-data.md).
