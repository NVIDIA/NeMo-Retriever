# Semantic and hybrid retrieval

**Semantic retrieval** uses dense embeddings to find content that is similar in meaning to a query. **Hybrid retrieval** combines dense vectors with sparse or lexical signals (for example, BM25-style full-text) and fuses ranked lists for better recall on keyword-heavy queries.

In NeMo Retriever Library, use these resources:

- [Concepts](concepts.md) for pipeline and search patterns
- [Vector databases](vdbs.md) for LanceDB hybrid mode (dense, BM25, and RRF)
- [Environment variables](environment-config.md) for hybrid-related flags where documented
- [Custom metadata and filtering](custom-metadata.md) for filtering

**Evaluation**

For evaluation and metrics, refer to [Evaluate on your data](evaluate-on-your-data.md).
