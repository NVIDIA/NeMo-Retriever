# Data Upload for NeMo Retriever Library

Use this documentation to learn how [NeMo Retriever Library](overview.md) handles and uploads data.

## On this page {#on-this-page}

- [Overview](#overview)
- [Why LanceDB?](#why-lancedb)
- [Upload to LanceDB](#upload-to-lancedb)
- [Semantic and hybrid retrieval](#semantic-and-hybrid-retrieval)
- [Hybrid search (LanceDB)](#hybrid-search-lancedb)
- [LanceDB deployment characteristics](#lancedb-deployment-characteristics)
- [Upload to a Custom Data Store](#upload-to-a-custom-data-store)
- [Vector database partners](#vector-database-partners)
    - [Backends with `VDB` implementations](#vdb-backends-implementations)
- [Related Topics](#related-topics)

## Overview {#overview}

NeMo Retriever Library supports extracting text representations of various forms of content,
and ingesting to a vector database. [LanceDB](https://lancedb.com/) is the vector database backend for storing and retrieving extracted embeddings.

The data upload task (`vdb_upload`) pulls extraction results to the Python client,
and then pushes them to LanceDB (embedded, in-process).

The vector database stores only the extracted text representations of ingested data.
It does not store the embeddings for images.

!!! tip "Storing Extracted Images"

    To persist extracted images, tables, and chart renderings to disk or object storage, use the `store` task in addition to `vdb_upload`. The `store` task supports any fsspec-compatible backend (local filesystem, S3, GCS, etc.). For details, refer to [Store Extracted Images](nemo-retriever-api-reference.md).

NeMo Retriever Library supports uploading data by using the [Ingestor.vdb_upload API](nemo-retriever-api-reference.md).
Currently, data upload is not supported through the [CLI](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/docs/cli).



## Why LanceDB? {#why-lancedb}

LanceDB is optimized for low-latency retrieval in this stack:

- **Lance columnar format** — Data is stored in Lance files, an Arrow/Parquet-style analytics layout optimized for fast local scans and indexed retrieval. This reduces serialization overhead compared with a separate database server.
- **IVF_HNSW_SQ index** — Vectors are scalar-quantized (SQ) within an IVF-HNSW index, compressing them for faster search with lower memory bandwidth cost.
- **Embedded runtime** — LanceDB runs in-process, so you do not run extra vector-database containers for the default path. Fewer moving parts to start, configure, and maintain.

This combination of file format, index strategy, and in-process runtime supports the latency characteristics described in benchmarks.



## Upload to LanceDB {#upload-to-lancedb}

LanceDB uses the `LanceDB` operator class from the client library. You can configure it via the Python API.

### Programmatic API (Python)

Pass `vdb_op="lancedb"` to `vdb_upload`, or construct a `LanceDB` instance and pass it as `vdb_op`:

```python
from nv_ingest_client.util.vdb.lancedb import LanceDB

vdb = LanceDB(
    uri="./lancedb_data",    # Path to LanceDB database directory
    table_name="nemo-retriever",  # Table name
    index_type="IVF_HNSW_SQ",  # Index type (default)
    hybrid=False,            # True = also build FTS for hybrid (see Hybrid search below)
)

# Ingest
vdb.run(results)

# Dense-only retrieve when hybrid=False (default)
docs = vdb.retrieval(queries, top_k=10)
```

With `hybrid=False`, `vdb.retrieval()` runs dense vector search. With **`hybrid=True`**, `vdb.run(results)` also builds the BM25/FTS index for hybrid ingest, but **`LanceDB.retrieval()` does not implement hybrid queries** and raises `NotImplementedError` if the operator was created with `hybrid=True`. For hybrid (dense + BM25 + RRF) **queries**, import and call **`lancedb_hybrid_retrieval()`** from `nv_ingest_client.util.vdb.lancedb` (same `table_path` / `table_name` as the `LanceDB` instance)—see [Hybrid search (LanceDB)](#hybrid-search-lancedb).

When using the `Ingestor` with `vdb_upload`, pass `vdb_op="lancedb"` or a `LanceDB` instance so uploads target LanceDB. If you omit `vdb_op`, nv-ingest-client still defaults the string argument to `"milvus"` for backward compatibility, which is not the LanceDB operator—always pass `vdb_op="lancedb"` when you intend LanceDB.

## Semantic and hybrid retrieval {#semantic-and-hybrid-retrieval}

**Semantic retrieval** uses dense embeddings to find content that is similar in meaning to a query. **Hybrid retrieval** combines dense vectors with sparse or lexical signals (for example, BM25-style full-text) and fuses ranked lists for better recall on keyword-heavy queries.

In NeMo Retriever Library, the default vector path is LanceDB. Use these resources together with the sections on this page:

- [Hybrid search (LanceDB)](#hybrid-search-lancedb) for LanceDB hybrid mode (dense vectors, BM25, and RRF) and query APIs
- [Concepts](concepts.md) for broader pipeline and search patterns
- [Environment variables](environment-config.md) for hybrid-related flags where documented
- [Custom metadata and filtering](custom-metadata.md) for query-time filtering

**Evaluation** — For evaluation and metrics, refer to [Evaluate on your data](evaluate-on-your-data.md).

## Hybrid search (LanceDB) {#hybrid-search-lancedb}

LanceDB supports **hybrid retrieval**, combining dense vector similarity with BM25 full-text search. Results are fused using Reciprocal Rank Fusion (RRF) reranking.

Hybrid search improves recall by approximately +0.5% to +3.5% over vector-only retrieval with negligible latency impact:

| Dataset            | Vector-Only Recall@5 | Hybrid Recall@5 | Delta  |
|--------------------|----------------------|-----------------|--------|
| bo767 (76K rows)   | 84.5%                | 85.0%           | +0.5%  |
| bo767 (reranked)   | 90.7%                | 91.8%           | +1.1%  |
| earnings (19K rows)| 61.5%                | 65.0%           | +3.5%  |
| earnings (reranked)| 74.5%                | 76.4%           | +1.9%  |

Hybrid search latency is typically 28–57 ms/query (vs. 31–37 ms/query for vector-only). The one-time FTS index build adds approximately 6.5 seconds for a 76K-row dataset.

Enable hybrid **ingest** by setting `hybrid=True` when creating the `LanceDB` operator so `vdb.run(results)` builds the BM25-friendly FTS index alongside vectors.

!!! note "Hybrid queries use `lancedb_hybrid_retrieval`, not `LanceDB.retrieval()`"

    `LanceDB.retrieval()` only supports dense vector search. If the operator was created with `hybrid=True`, calling `vdb.retrieval(...)` raises `NotImplementedError` (“hybrid retrieval with precomputed vectors is not implemented yet”). For hybrid (dense + BM25 + RRF) **queries**, use **`lancedb_hybrid_retrieval()`** from the same module, with the same `table_path` / `table_name` as the `LanceDB` instance:

    ```python
    from nv_ingest_client.util.vdb.lancedb import LanceDB, lancedb_hybrid_retrieval

    vdb = LanceDB(uri="./lancedb_data", table_name="nemo-retriever", hybrid=True)
    vdb.run(results)

    docs = lancedb_hybrid_retrieval(
        queries,
        table_path="./lancedb_data",
        table_name="nemo-retriever",
        top_k=10,
    )
    ```

## LanceDB deployment characteristics {#lancedb-deployment-characteristics}

| Aspect              | LanceDB                                      |
|---------------------|----------------------------------------------|
| Runtime model       | Embedded (in-process)                        |
| External services   | None for the vector store itself             |
| Helm / extra stack  | Not required for LanceDB (default path)      |
| Index type          | IVF_HNSW_SQ (default)                        |
| Hybrid search       | BM25 FTS + vector (RRF) when enabled         |
| Persistence         | Lance files on disk under your configured URI |



## Upload to a Custom Data Store {#upload-to-a-custom-data-store}

You can ingest to other data stores by using the `Ingestor.vdb_upload` method;
however, you must configure other data stores and connections yourself.
NeMo Retriever Library does not provide connections to other data sources.

## Vector database partners {#vector-database-partners}

NeMo Retriever Library integrates with vector databases used for RAG collections. The sections above focus on LanceDB as used in the library. This section summarizes other client `VDB` implementations and how they plug into NeMo Retriever Library graph operators. For chunking before ingest, see [Chunking and splitting](chunking.md).

### Backends with `VDB` implementations (retriever adapters) {#vdb-backends-implementations}

NeMo Retriever graph operators [`IngestVdbOperator`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/vdb/operators.py) and [`RetrieveVdbOperator`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/vdb/operators.py) wrap concrete classes that implement the [`nv_ingest_client.util.vdb.VDB`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/client/src/nv_ingest_client/util/vdb/adt_vdb.py) interface (`run` for ingest, `retrieval` for search). The following external vector databases have implementations in the client library you can pass as `vdb` / configure via `vdb_op` where supported:

| Backend | Project | Implementation |
|---------|---------|----------------|
| **LanceDB** | [LanceDB](https://lancedb.com/) | [`lancedb.py`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/client/src/nv_ingest_client/util/vdb/lancedb.py) — pass `vdb_op="lancedb"` (recommended). |
| **OpenSearch** | [OpenSearch](https://opensearch.org/) | [`opensearch.py`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/client/src/nv_ingest_client/util/vdb/opensearch.py) — reference operator; wire your own `OpenSearch` instance as `vdb` and see [Build a Custom Vector Database Operator](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/building_vdb_operator.ipynb). |

On nv-ingest-client `Ingestor.vdb_upload`, omitting `vdb_op` does not select LanceDB; see [Upload to LanceDB](#upload-to-lancedb).

For LanceDB, pass `vdb_op="lancedb"` (or a `LanceDB` instance). For other `VDB` subclasses, construct the client class and pass it as the graph operator’s `vdb` argument.

**Related**

- [Embedding NIMs and models](embedding-nims-models.md)
- [NVIDIA NIM catalog](https://build.nvidia.com/) for embedding and retrieval-related NIMs

!!! important

    NVIDIA documents and validates the first-party LanceDB operator for this library. If you integrate a different vector store, you are responsible for testing and maintaining that integration.

To implement a custom operator, follow the `VDB` abstract interface described in [Build a Custom Vector Database Operator](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/building_vdb_operator.ipynb).



## Related Topics {#related-topics}

- [Use the NeMo Retriever Library Python API](nemo-retriever-api-reference.md)
- [Store Extracted Images](nemo-retriever-api-reference.md)
- [Environment Variables](environment-config.md)
- [Troubleshoot Nemo Retriever Extraction](troubleshoot.md)
