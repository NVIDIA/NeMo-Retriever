# Reranking

!!! note

    NVIDIA Ingest (nv-ingest) has been renamed NeMo Retriever Library.


A **reranker** model re-scores the top candidates from initial retrieval so the final top-K better matches the query. NeMo Retriever Library evaluation and deployments can integrate reranker NIMs (see the [Support matrix](support-matrix.md) for availability and GPU considerations).

**When to enable reranking**

Enable reranking when both of the following apply:

- Recall at K from vector or hybrid search is acceptable, but ordering of the top results still needs improvement.
- Benchmarking shows measurable gains for your domain (see reranker modes in [Benchmarking](benchmarking.md)).

**Configuration pointers**

Reranker options appear in benchmarking and recall configurations. Start with [Benchmarking](benchmarking.md) and your vector store setup in [Vector databases](data-store.md).

**Related**

- [Workflow: Query and rerank](workflow-query-rerank.md)
