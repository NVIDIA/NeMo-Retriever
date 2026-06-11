# Vector databases

Use this documentation to learn how [NeMo Retriever Library](overview.md) stores extracted embeddings and uploads data to vector databases.

## On this page { #on-this-page }

- [Overview](#overview)
- [Why LanceDB?](#why-lancedb)
- [Upload to LanceDB](#upload-to-lancedb)
- [Semantic retrieval](#semantic-retrieval)
- [Metadata and filtering](#metadata-and-filtering)
- [LanceDB deployment characteristics](#lancedb-deployment-characteristics)
- [Upload to a Custom Data Store](#upload-to-a-custom-data-store)
- [Vector database partners](#vector-database-partners)
    - [Backends with `VDB` implementations](#vdb-backends-implementations)
    - [RAG Blueprint and partner vector stores](#rag-blueprint-and-partner-vector-stores)
    - [More information (embeddings & custom `VDB`)](#vector-database-partners-more-info)
- [Related Topics](#related-topics)

## Overview { #overview }

NeMo Retriever Library supports extracting text representations of various forms of content,
and ingesting to a vector database. [LanceDB](https://lancedb.com/) is the vector database backend for storing and retrieving extracted embeddings.

The data upload task (`vdb_upload`) pulls extraction results to the Python client,
and then pushes them to LanceDB (embedded, in-process).

The vector database stores only the extracted text representations of ingested data.
It does not store the embeddings for images.

!!! tip "Storing Extracted Images"

    To persist extracted images, tables, and chart renderings to disk or object storage, use the `store` task in addition to `vdb_upload`. The `store` task supports any fsspec-compatible backend (local filesystem, S3, GCS, and other object stores). For details, refer to [Store Extracted Images](nemo-retriever-api-reference.md).

NeMo Retriever Library supports uploading data through `.vdb_upload()` on `create_ingestor(...)` ([Python API guide](nemo-retriever-api-reference.md)).
Currently, data upload is not supported through the [CLI](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/docs/cli).



## Why LanceDB? { #why-lancedb }

LanceDB is optimized for low-latency retrieval in this stack:

- **Lance columnar format** — Data is stored in Lance files, an Arrow/Parquet-style analytics layout optimized for fast local scans and indexed retrieval. This reduces serialization overhead compared with a separate database server.
- **IVF_HNSW_SQ index** — Vectors are scalar-quantized (SQ) within an IVF-HNSW index, compressing them for faster search with lower memory bandwidth cost.
- **Embedded runtime** — LanceDB runs in-process, so you do not run extra vector-database containers for the default path. Fewer moving parts to start, configure, and maintain.

This combination of file format, index strategy, and in-process runtime supports the latency characteristics described in benchmarks.



## Upload to LanceDB { #upload-to-lancedb }

LanceDB uses the `LanceDB` operator class from the client library. You can configure it through the Python API.

### Programmatic API (Python)

Pass `vdb_op="lancedb"` to `vdb_upload`, or construct a `LanceDB` instance and pass it as `vdb_op`.

For parameter details, refer to the [Python API guide](nemo-retriever-api-reference.md).

```python
from nemo_retriever.common.vdb.lancedb import LanceDB

vdb = LanceDB(
    uri="./lancedb_data",    # Path to LanceDB database directory
    table_name="nemo-retriever",  # Table name
    index_type="IVF_HNSW_SQ",  # Index type (default)
)

# Ingest
vdb.run(results)

# Retrieve with precomputed query vectors
docs = vdb.retrieval(queries, top_k=10)
```

Query ingested tables with `LanceDB.retrieval()` (precomputed vectors) or with [`Retriever.query`](nemo-retriever-api-reference.md) (embeds the query string for you). Optional `where` predicates and client-side filters are documented under [Metadata and filtering](#metadata-and-filtering).

When using the `Ingestor` with `vdb_upload`, pass `vdb_op="lancedb"` or a `LanceDB` instance so uploads target LanceDB. If you omit `vdb_op`, the ingestion Python client still defaults the string argument to `"milvus"` for backward compatibility, which is not the LanceDB operator—always pass `vdb_op="lancedb"` when you intend LanceDB.

## Semantic retrieval { #semantic-retrieval }

Semantic retrieval uses dense embeddings to find content that is similar in meaning to a query. In NeMo Retriever Library, the default vector path is LanceDB. Use these resources together with the sections on this page:

- [Metadata and filtering](#metadata-and-filtering) for sidecar metadata at ingest and query-time filters
- [Concepts](concepts.md) for broader pipeline and search patterns
- [Use the NeMo Retriever Library Python API](nemo-retriever-api-reference.md) for `Retriever.query` and `LanceDB.retrieval` parameters

**Evaluation** — For evaluation and metrics, refer to [Evaluate on your data](evaluate-on-your-data.md).

## Metadata and filtering { #metadata-and-filtering }

This page covers LanceDB upload and retrieval. **Metadata is not duplicated here.**

- **Published guide** — [Custom metadata and filtering](custom-metadata.md) (sidecar `meta_*` on `vdb_upload`, compact JSON in LanceDB, server-side `where` on `Retriever.query`, and client-side `filter_hits_by_content_metadata`).
- **Canonical reference** — [Vector DB operators and LanceDB — Metadata filtering](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/src/nemo_retriever/vdb#metadata-filtering) in `nemo_retriever/src/nemo_retriever/vdb/README.md` (operator behavior and examples).

## LanceDB deployment characteristics { #lancedb-deployment-characteristics }

| Aspect              | LanceDB                                      |
|---------------------|----------------------------------------------|
| Runtime model       | Embedded (in-process)                        |
| External services   | None for the vector store itself             |
| Helm / extra stack  | Not required for LanceDB (default path)      |
| Index type          | IVF_HNSW_SQ (default)                        |
| Persistence         | Lance files on disk under your configured URI |



## Upload to a Custom Data Store { #upload-to-a-custom-data-store }

You can ingest to other data stores through `.vdb_upload()` on `create_ingestor(...)`;
however, you must configure other data stores and connections yourself.
NeMo Retriever Library does not provide connections to other data sources.

## Vector database partners { #vector-database-partners }

NeMo Retriever Library integrates with vector databases used for RAG collections. The sections above focus on LanceDB as the shipped backend. This section lists that backend and how partner or custom `VDB` subclasses plug into graph operators. For chunking behavior, refer to [Chunking](concepts.md#chunking).

### Backends with `VDB` implementations (retriever adapters) { #vdb-backends-implementations }

NeMo Retriever graph operators [`IngestVdbOperator`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/vdb/operators.py) and [`RetrieveVdbOperator`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/vdb/operators.py) wrap concrete classes that implement the [`VDB`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/vdb/adt_vdb.py) interface (`run` for ingest, `retrieval` for search). The library ships one first-party backend:

| Backend | Project | Implementation |
|---------|---------|----------------|
| **LanceDB** | [LanceDB](https://lancedb.com/) · [documentation](https://lancedb.github.io/lancedb/) | [`lancedb.py`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/vdb/lancedb.py) — pass `vdb_op="lancedb"` (recommended). |

On `GraphIngestor.vdb_upload`, omitting `vdb_op` does not select LanceDB; refer to [Upload to LanceDB](#upload-to-lancedb).

Pass `vdb_op="lancedb"` or a `LanceDB` instance. To integrate another vector database, subclass [`VDB`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/vdb/adt_vdb.py) and pass your operator instance as `vdb` (refer to [Build a Custom Vector Database Operator](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/building_vdb_operator.ipynb)).

### RAG Blueprint and partner vector stores { #rag-blueprint-and-partner-vector-stores }

Some deployments use a different vector store than the default LanceDB path on this page—for example the [NVIDIA RAG Blueprint](https://docs.nvidia.com/rag/latest/index.html) (Docker Compose or Helm) or a partner package that subclasses the same [`VDB`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/vdb/adt_vdb.py) interface. Use the following public references when you wire those stacks to ingestion and retrieval:

| Vector store | Where to configure or implement |
|--------------|--------------------------------|
| **[Elasticsearch](https://www.elastic.co/elasticsearch)** | [Configure Elasticsearch as Your Vector Database for NVIDIA RAG Blueprint](https://docs.nvidia.com/rag/latest/change-vectordb.html) — compose profiles, environment variables, and Helm notes for the RAG Blueprint. |
| **[Pinecone](https://www.pinecone.io/)** | [Customize your vector database (Pinecone + NVIDIA RAG)](https://github.com/pinecone-io/nvidia-pinecone-rag/blob/main/docs/vector-database.md) in the [`pinecone-io/nvidia-pinecone-rag`](https://github.com/pinecone-io/nvidia-pinecone-rag) repository. |
| **[Teradata](https://www.teradata.com/)** | [TeradataVDB (NVIDIA NIM Ingest integration)](https://docs.teradata.com/r/VMware/Teradata-Package-for-Generative-AI-Function-Reference/Vector-Store/NVIDIA-NIM-Ingest-Integration/TeradataVDB) — `teradatagenai.vector_store.teradataVDB.TeradataVDB` implements the NeMo Retriever ingestion `VDB` abstract class for Teradata Vector Store. |

Testing and release cadence for these integrations follow the owning project (RAG Blueprint, Pinecone sample repo, or Teradata Generative AI package), not the first-party LanceDB operator validated for NeMo Retriever Library on this page.

### More information (embeddings & custom `VDB`) { #vector-database-partners-more-info }

- [Custom metadata and filtering](custom-metadata.md) and the package [VDB README (metadata filtering)](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/src/nemo_retriever/vdb#metadata-filtering)
- [Multimodal embeddings (VLM)](embedding.md)
- [NeMo Retriever Text Embedding NIM](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/overview.html)
- [NVIDIA NIM catalog](https://build.nvidia.com/) for embedding and retrieval-related NIMs

!!! important

    NVIDIA documents and validates the first-party LanceDB operator for this library. If you integrate a different vector store, you are responsible for testing and maintaining that integration.

To implement a custom operator, follow the `VDB` abstract interface described in [Build a Custom Vector Database Operator](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/building_vdb_operator.ipynb).

## Related Topics { #related-topics }

- [Custom metadata and filtering](custom-metadata.md)
- [Vector DB operators and LanceDB (source)](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/src/nemo_retriever/vdb)
- [Use the NeMo Retriever Library Python API](nemo-retriever-api-reference.md)
- [Store Extracted Images](nemo-retriever-api-reference.md)
- [Environment Variables](environment-config.md)
- [Troubleshoot Nemo Retriever Extraction](troubleshoot.md)
