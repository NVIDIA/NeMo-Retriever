# Workflow: Document ingestion

**Document ingestion** is the step where NeMo Retriever Library reads your files (PDFs, Office documents, images, and other [supported formats](supported-file-types.md)), runs extraction and optional enrichment, and returns structured content you can chunk, embed, and index.

Follow these steps:

1. **Choose how you call the library.** Use the [Python API](python-api-reference.md) or [CLI](cli-reference.md) from application code, or run a deployment (for example [Quickstart: NeMo Retriever Library (local)](quickstart-library-mode.md), [Deploy (Docker Compose)](quickstart-guide.md), or [Quickstart: Kubernetes (Helm)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md)) and send jobs over the network.
2. **Use the V2 processing pipeline.** Parallel PDF handling and the default ingest path are described in the [HTTP API (V2)](v2-api-guide.md) and [API guide](nemo-retriever-api-reference.md). Set `message_client_kwargs={"api_version": "v2"}` when using the client if you need to be explicit.
3. **Tune extraction for your content.** Refer to [Multimodal extraction](supported-file-types.md) for formats, [Text and layout extraction](text-layout-extraction.md), [Tables](extraction-tables.md), [OCR](extraction-ocr-scanned.md), and related topics in the navigation.

Pipeline concepts and stage overview appear in [Key concepts](concepts.md).

**Next:** [Workflow: Build a searchable collection](workflow-build-searchable-collection.md).
