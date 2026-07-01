# Customize & extend

NeMo Retriever Library ships with defaults tuned for strong recall on common document types. When those defaults are not enough, you can extend the library at several levels—from task keyword arguments on the fluent ingestor API through custom graph operators and vector-database adapters.

Use this page to choose an extension path and find the detailed guides in the repository.

## On this page { #on-this-page }

- [Start with task configuration](#start-with-task-configuration)
- [User-defined functions (UDFs)](#user-defined-functions-udfs)
- [Custom graph pipelines](#custom-graph-pipelines)
- [Custom vector databases](#custom-vector-databases)
- [Custom embedding models](#custom-embedding-models)
- [Related topics](#related-topics)

## Start with task configuration { #start-with-task-configuration }

Most customization does not require new code. Chain tasks on `create_ingestor(...)` and pass keyword arguments to control extraction, chunking, embedding, and storage—for example `extract_method`, chunking and splitting options on `.extract()`, `embed_modality` on `.embed()`, and `vdb_op` / `vdb_kwargs` on `.vdb_upload()`.

For parameter details, refer to the [Python API guide](nemo-retriever-api-reference.md). For chunking behavior and pipeline concepts, refer to [Concepts](concepts.md).

## User-defined functions (UDFs) { #user-defined-functions-udfs }

A **user-defined function (UDF)** wraps your Python logic as a first-class pipeline stage. In the graph model, `UDFOperator` turns a plain callable into an operator you can chain with built-in stages—for example to normalize HTML, apply a custom split, or call an external service between extract and embed steps.

Use UDFs when you need a small, self-contained transformation that is not covered by task keyword arguments.

**Repository guides and examples**

- [NeMo Retriever graph README — `UDFOperator`](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/src/nemo_retriever/graph#using-udfoperator) — API, lifecycle, and when to use `UDFOperator` versus a custom operator class
- [UDF example scripts](https://github.com/NVIDIA/NeMo-Retriever/tree/main/examples/udfs) — sample implementations such as HTML-to-Markdown conversion and structural splitting
- [NimClient and custom NIM endpoints](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/developer_docs/nimclient.md#nimclient-and-custom-nim-endpoints) — call custom or self-hosted NIM microservices from UDF stages

## Custom graph pipelines { #custom-graph-pipelines }

When you need to compose pipelines stage-by-stage, reuse operators across workflows, or run the same graph in-process or with Ray Data, use the **graph execution model** instead of (or alongside) the fluent `GraphIngestor` API.

The graph package provides `AbstractOperator`, executors (`InprocessExecutor`, `RayDataExecutor`), and operator chaining with `>>`. Built-in ingestion operators live under `nemo_retriever.operators`; you can add your own operators or UDF stages anywhere in the chain.

For the full guide—including custom operator classes, executors, and graph shape constraints—refer to the [NeMo Retriever graph README](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/src/nemo_retriever/graph#nemo-retriever-graph).

## Custom vector databases { #custom-vector-databases }

The supported user path for vector storage is **[LanceDB](vdbs.md)** (`vdb_op="lancedb"`). That page covers upload, semantic retrieval, metadata filtering, and LanceDB deployment characteristics.

To integrate a different vector store, implement the [`VDB`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/common/vdb/adt_vdb.py) interface and wire it through graph [`IngestVdbOperator`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/operators/vdb.py) / [`RetrieveVdbOperator`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/operators/vdb.py). NVIDIA validates the first-party LanceDB operator; you are responsible for testing and maintaining other backends.

**Repository guides**

- [Build a custom vector database operator (notebook)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/building_vdb_operator.ipynb) — step-by-step walkthrough
- [Vector DB package (source)](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/src/nemo_retriever/common/vdb) — `VDB` abstract base and LanceDB reference implementation

Partner and blueprint integrations (Elasticsearch, Pinecone, Teradata, and others) are summarized on [Vector databases — Vector database partners](vdbs.md#vector-database-partners).

## Custom embedding models { #custom-embedding-models }

!!! note "Coming soon"

    Dedicated documentation for wiring custom embedding models into ingestion graphs is in progress. Today, configure embedding through task parameters and environment variables:

    - [Multimodal embeddings (VLM)](embedding.md) — default and multimodal embed flows
    - [Environment variables](environment-config.md) — `*_ENDPOINT` variables for self-hosted or hosted NIM embed services
    - [NeMo Retriever Text Embedding NIM](https://docs.nvidia.com/nim/nemo-retriever/text-embedding/latest/overview.html) — OpenAI-compatible text embedding NIM

## Related topics { #related-topics }

- [Concepts — Pipeline and tasks](concepts.md#pipeline-and-tasks)
- [Vector databases](vdbs.md)
- [Multimodal embeddings (VLM)](embedding.md)
- [Python API guide](nemo-retriever-api-reference.md)
- [Starter kits and notebooks](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/README.md)
