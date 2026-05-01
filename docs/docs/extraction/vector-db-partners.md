# Vector database partners

NeMo Retriever Library integrates with vector databases used for RAG collections. Documentation here focuses on LanceDB as used in the library, and on NVIDIA cuVS where it applies to indexing. Refer to [Vector databases](vdbs.md) and [Chunking and splitting](chunking.md).

## Backends with `VDB` implementations (retriever adapters)

NeMo Retriever graph operators [`IngestVdbOperator`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/vdb/operators.py) and [`RetrieveVdbOperator`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/vdb/operators.py) wrap concrete classes that implement the [`nv_ingest_client.util.vdb.VDB`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/client/src/nv_ingest_client/util/vdb/adt_vdb.py) interface (`run` for ingest, `retrieval` for search). The following external vector databases have implementations in the client library you can pass as `vdb` / configure via `vdb_op` where supported:

| Backend | Project | Implementation |
|---------|---------|----------------|
| **LanceDB** | [LanceDB](https://lancedb.com/) | [`lancedb.py`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/client/src/nv_ingest_client/util/vdb/lancedb.py) — use `vdb_op="lancedb"` (recommended default). |
| **OpenSearch** | [OpenSearch](https://opensearch.org/) | [`opensearch.py`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/client/src/nv_ingest_client/util/vdb/opensearch.py) — reference operator; wire your own `OpenSearch` instance as `vdb` and see [Build a Custom Vector Database Operator](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/building_vdb_operator.ipynb). |

For LanceDB, pass `vdb_op="lancedb"` (or a `LanceDB` instance). For other `VDB` subclasses, construct the client class and pass it as the graph operator’s `vdb` argument.

## Indexing with NVIDIA cuVS

Accelerated IVF-style indexing used in this stack is documented alongside NVIDIA RAPIDS; see the [cuVS project](https://github.com/rapidsai/cuvs) for GPU-accelerated vector search building blocks. Product-level indexing guidance for NeMo Retriever also appears in the top-level docs ([What is NVIDIA NeMo Retriever?](../index.md)).

**Related**

- [Embedding NIMs and models](embedding-nims-models.md)
- [NVIDIA NIM catalog](https://build.nvidia.com/) for embedding and retrieval-related NIMs
