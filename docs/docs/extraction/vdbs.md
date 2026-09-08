# Vector databases

Use this documentation to learn how [NeMo Retriever Library](overview.md) stores extracted embeddings and uploads data to vector databases.

## On this page { #on-this-page }

- [Overview](#overview)
- [LanceDB Overview](#why-lancedb)
- [Upload to LanceDB](#upload-to-lancedb)
    - [Direct LanceDB ingest and retrieval](#direct-lancedb-ingest-and-retrieval)
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

NeMo Retriever Library supports uploading data through `.vdb_upload()` on `create_ingestor(...)` ([Python API guide](nemo-retriever-api-reference.md)) and through the public `retriever ingest` CLI.

- **Python SDK ingest** (`.vdb_upload()` on `create_ingestor(...)`) persists embeddings to LanceDB with default URI `lancedb` and default table `nemo-retriever`. Default `Retriever()` queries that same table.
- **Local and batch CLI ingest** (`retriever ingest`, `retriever ingest local`, `retriever ingest batch`) persist embeddings to LanceDB (default URI `lancedb`, table `nemo-retriever`).
- **Service CLI ingest** (`retriever ingest service`) writes to service-configured storage.

The Python SDK and the local CLI share the same LanceDB default table. Pass an explicit URI and table name at ingest and at query time only when you need a non-default location.

For supported modes and target storage, refer to the [Retriever CLI](https://github.com/NVIDIA/NeMo-Retriever/tree/26.08.1/nemo_retriever/docs/cli).

`.vdb_upload()` does not generate embeddings. For dense SDK ingestion, include `.embed()` in the pipeline:

```python
from nemo_retriever import create_ingestor


result = (
    create_ingestor(run_mode="inprocess")
    .files(["document.pdf"])
    .extract(extract_text=True)
    .embed()
    .vdb_upload()
    .ingest()
)
```

Bare `.vdb_upload()` writes to table `nemo-retriever`. Default `Retriever()`
queries that table.

You can omit `.embed()` if a custom stage provides an embedding in `metadata["embedding"]` or `text_embeddings_1b_v2["embedding"]`. Dense upload fails closed before the backend write if any searchable row in a nonempty batch is missing an embedding. This includes a mixed batch where other rows have embeddings: the library raises `VdbUploadError`, a `ValueError` subclass, instead of writing the embedded subset. An extraction that produces no content completes without uploading records. For automatic handling of overlength text before upload, refer to [Text inputs that exceed the model limit](embedding.md#text-input-overflow).

## LanceDB Overview { #why-lancedb }

LanceDB is optimized for low-latency retrieval in this stack:

- **Lance columnar format** — Data is stored in Lance files, an Arrow/Parquet-style analytics layout optimized for fast local scans and indexed retrieval. This reduces serialization overhead compared with a separate database server.
- **IVF_HNSW_SQ index** — Vectors are scalar-quantized (SQ) within an IVF-HNSW index, compressing them for faster search with lower memory bandwidth cost.
- **Embedded runtime** — LanceDB runs in-process, so you do not run extra vector-database containers for the default path. Fewer moving parts to start, configure, and maintain.

This combination of file format, index strategy, and in-process runtime supports the latency characteristics described in benchmarks.



## Upload to LanceDB { #upload-to-lancedb }

LanceDB uses the `LanceDB` operator class from the client library. You can configure it through the Python API or the CLI.

### CLI

The following command runs local ingest into the default LanceDB table `nemo-retriever`:

```bash
retriever ingest ./data/multimodal_test.pdf
```

Use `--lancedb-uri` and `--table-name` on the local and batch commands when you need a non-default LanceDB location. For modes and flags, refer to the [Retriever CLI](https://github.com/NVIDIA/NeMo-Retriever/tree/26.08.1/nemo_retriever/docs/cli).

### Programmatic API (Python)

`GraphIngestor.vdb_upload()` selects LanceDB when you omit `vdb_op`. `VdbUploadParams.vdb_op` defaults to `"lancedb"`. Passing `vdb_op="lancedb"` is optional explicitness, not a requirement.

For URI, table name, and other parameters, refer to the [Python API guide](nemo-retriever-api-reference.md).

### Direct LanceDB ingest and retrieval { #direct-lancedb-ingest-and-retrieval }

You can also construct a `LanceDB` instance and call `run` and `retrieval` directly. This is the optional low-level path. Prefer `.vdb_upload()` for typical ingest.

Graph ingest returns a pandas `DataFrame` of flat rows. Use the following input shapes:

- `LanceDB.run()` expects nested client record batches: a `list` of batches, and each batch is a `list` of record dictionaries. Convert graph or `DataFrame` rows with `to_client_vdb_records()` before you call `run()`.
- `LanceDB.run()` does not accept the graph `DataFrame` or a flat `list` of dictionaries from `DataFrame.to_dict("records")`.
- `LanceDB.retrieval()` takes precomputed query vectors. Pass a `list` of embedding vectors whose length matches `vector_dim`. For query strings, use [`Retriever.query`](nemo-retriever-api-reference.md).
- `IngestVdbOperator` accepts the same flat `DataFrame` or graph rows. It converts them with `to_client_vdb_records()` and then calls `run()`.

The following example uses a two-dimensional fixture so you can copy it without a GPU or embedding NIM:

- Replace `graph_rows` with your `.ingest()` `DataFrame` when you already have embeddings.
- Set `vector_dim` to your embedding length. The `LanceDB` default `vector_dim` is 2048.
- The fixture sets `create_index=False` so the one-row table is written without building the default `IVF_HNSW_SQ` index. Default ingest builds that index.

For the ingestion contract, refer to the [Vector DB operators and LanceDB README](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/src/nemo_retriever/common/vdb/README.md#ingestvdboperator-ingestion).

Copy this example to write one row and retrieve at least one hit.

```python
import pandas as pd

from nemo_retriever.common.vdb.lancedb import LanceDB
from nemo_retriever.common.vdb.records import to_client_vdb_records
from nemo_retriever.operators.vdb import IngestVdbOperator

# Graph ingest rows after extract and embed. Substitute your .ingest() DataFrame.
graph_rows = pd.DataFrame(
    [
        {
            "text": "hello from graph",
            "text_embeddings_1b_v2": {"embedding": [1.0, 0.0]},
            "path": "graph.pdf",
            "page_number": 2,
            "metadata": {},
        }
    ]
)

vdb = LanceDB(
    uri="./lancedb_data",
    table_name="nemo-retriever",
    vector_dim=2,
    create_index=False,
)

records = to_client_vdb_records(graph_rows)
vdb.run(records)

queries = [[1.0, 0.0]]
docs = vdb.retrieval(queries, top_k=10)

operator = IngestVdbOperator(
    vdb=LanceDB(
        uri="./lancedb_data_operator",
        table_name="nemo-retriever",
        vector_dim=2,
        create_index=False,
    )
)
operator(graph_rows)
```

Query ingested tables with `LanceDB.retrieval()` (precomputed vectors) or with [`Retriever.query`](nemo-retriever-api-reference.md) (embeds the query string for you). Optional `where` predicates and client-side filters are documented under [Metadata and filtering](#metadata-and-filtering).

To use a custom operator, pass a `VDB` instance as `vdb` to `IngestVdbOperator` (refer to [Build a Custom Vector Database Operator](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/examples/building_vdb_operator.ipynb)).

## Semantic retrieval { #semantic-retrieval }

Semantic retrieval uses dense embeddings to find content that is similar in meaning to a query. In NeMo Retriever Library, the default vector path is LanceDB. Use these resources together with the sections on this page:

- [Metadata and filtering](#metadata-and-filtering) for custom metadata at ingest and filtered retrieval
- [Concepts](concepts.md) for broader pipeline and search patterns
- [Use the NeMo Retriever Library Python API](nemo-retriever-api-reference.md) for `Retriever.query` and `LanceDB.retrieval` parameters
- [Workflow: Agentic retrieval](workflow-agentic-retrieval.md) for the LLM-driven ReAct query path over the same LanceDB table

**Evaluation** — For evaluation and metrics, refer to [Evaluate on your data](evaluate-on-your-data.md).

## Metadata and filtering { #metadata-and-filtering }

Refer to the [metadata filtering notebook](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/examples/nemo_retriever_retriever_query_metadata_filter.ipynb) for an end-to-end example of adding custom metadata fields to your documents and filtering retrieval results with that metadata.

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

NeMo Retriever graph operators [`IngestVdbOperator`](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/src/nemo_retriever/operators/vdb.py) and [`RetrieveVdbOperator`](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/src/nemo_retriever/operators/vdb.py) wrap concrete classes that implement the [`VDB`](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/src/nemo_retriever/common/vdb/adt_vdb.py) interface (`run` for ingest, `retrieval` for search). The library ships one first-party backend:

| Backend | Project | Implementation |
|---------|---------|----------------|
| **LanceDB** | [LanceDB](https://lancedb.com/) · [documentation](https://lancedb.github.io/lancedb/) | [`lancedb.py`](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/src/nemo_retriever/common/vdb/lancedb.py) — default `vdb_op` is `"lancedb"`. |

`GraphIngestor.vdb_upload()` selects LanceDB when `vdb_op` is omitted. Refer to [Upload to LanceDB](#upload-to-lancedb).

To integrate another vector database, subclass [`VDB`](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/src/nemo_retriever/common/vdb/adt_vdb.py) and pass your operator instance as `vdb` (refer to [Build a Custom Vector Database Operator](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/examples/building_vdb_operator.ipynb)).

### RAG Blueprint and partner vector stores { #rag-blueprint-and-partner-vector-stores }

Some deployments use a different vector store than the default LanceDB path on this page—for example the [NVIDIA RAG Blueprint](https://docs.nvidia.com/rag/latest/index.html) (Docker Compose or Helm) or a partner package that subclasses the same [`VDB`](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/nemo_retriever/src/nemo_retriever/common/vdb/adt_vdb.py) interface. Use the following public references when you wire those stacks to ingestion and retrieval:

| Vector store | Where to configure or implement |
|--------------|--------------------------------|
| **[Elasticsearch](https://www.elastic.co/elasticsearch)** | [Configure Elasticsearch as Your Vector Database for NVIDIA RAG Blueprint](https://docs.nvidia.com/rag/latest/change-vectordb.html) — compose profiles, environment variables, and Helm notes for the RAG Blueprint. |
| **[Pinecone](https://www.pinecone.io/)** | [Pinecone Configuration for Pinecone Enterprise RAG Blueprint](https://github.com/pinecone-io/nvidia-rag/blob/main/docs/pinecone-configuration.md) in the [`pinecone-io/nvidia-rag`](https://github.com/pinecone-io/nvidia-rag) repository. |
| **[Teradata](https://www.teradata.com/)** | [TeradataVDB (NVIDIA NIM Ingest integration)](https://docs.teradata.com/r/VMware/Teradata-Package-for-Generative-AI-Function-Reference/Vector-Store/NVIDIA-NIM-Ingest-Integration/TeradataVDB) — `teradatagenai.vector_store.teradataVDB.TeradataVDB` implements the NeMo Retriever ingestion `VDB` abstract class for Teradata Vector Store. |

Testing and release cadence for these integrations follow the owning project (RAG Blueprint, Pinecone sample repo, or Teradata Generative AI package), not the first-party LanceDB operator validated for NeMo Retriever Library on this page.

### More information (embeddings & custom `VDB`) { #vector-database-partners-more-info }

- [Metadata filtering notebook](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/examples/nemo_retriever_retriever_query_metadata_filter.ipynb) and the package [VDB README (metadata filtering)](https://github.com/NVIDIA/NeMo-Retriever/tree/26.08.1/nemo_retriever/src/nemo_retriever/common/vdb#metadata-filtering)
- [Multimodal embeddings (VLM)](embedding.md)
- [NeMo Retriever Text Embedding NIM](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/overview.html)
- [NVIDIA NIM catalog](https://build.nvidia.com/) for embedding and retrieval-related NIMs

!!! important

    NVIDIA documents and validates the first-party LanceDB operator for this library. If you integrate a different vector store, you are responsible for testing and maintaining that integration.

To implement a custom operator, follow the `VDB` abstract interface described in [Build a Custom Vector Database Operator](https://github.com/NVIDIA/NeMo-Retriever/blob/26.08.1/examples/building_vdb_operator.ipynb). For an overview of all customization paths (UDFs, graph pipelines, and embeddings), refer to [Customize & extend](customize-extend.md).

## Related Topics { #related-topics }

- [Metadata and filtering](#metadata-and-filtering)
- [Workflow: Agentic retrieval](workflow-agentic-retrieval.md)
- [Customize & extend](customize-extend.md)
- [Vector DB operators and LanceDB (source)](https://github.com/NVIDIA/NeMo-Retriever/tree/26.08.1/nemo_retriever/src/nemo_retriever/common/vdb)
- [Use the NeMo Retriever Library Python API](nemo-retriever-api-reference.md)
- [Retriever CLI](https://github.com/NVIDIA/NeMo-Retriever/tree/26.08.1/nemo_retriever/docs/cli)
- [Store Extracted Images](nemo-retriever-api-reference.md)
- [Environment Variables](environment-config.md)
- [Troubleshoot NeMo Retriever Extraction](troubleshoot.md)
