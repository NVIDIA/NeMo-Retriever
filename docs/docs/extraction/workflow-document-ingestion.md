# Workflow: Ingest documents into a searchable VDB collection

This page covers **extracting content from documents** and **turning that content into a searchable vector collection** in one place so you can scroll and search (for example with Ctrl+F) instead of jumping across multiple short workflow stubs.

## Ingest and extract

**Document ingestion** is the step where NeMo Retriever Library reads your files (PDFs, Office documents, images, and other [supported formats](multimodal-extraction.md#supported-file-types-and-formats)), runs extraction and optional enrichment, and returns structured content you can chunk, embed, and index.

Follow these steps:

1. **Choose how you call the library.** Use the [Python API](nemo-retriever-api-reference.md) or [CLI](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/docs/cli) from application code, or run a deployment (for example [Quickstart: NeMo Retriever Library (local)](quickstart-library-mode.md), [Deployment options](deployment-options.md), or [Quickstart: Kubernetes (Helm)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md)) and send jobs over the network.
2. **Use parallel PDF handling.** The default ingest path splits large PDFs before Ray processing; behavior and tuning are described in the [API guide — PDF pre-splitting](nemo-retriever-api-reference.md#pdf-pre-splitting-for-parallel-ingest). Set `message_client_kwargs={"api_version": "v2"}` when using the client if you need to be explicit.
3. **Tune extraction for your content.** Refer to [Multimodal extraction](multimodal-extraction.md) for formats, [text and layout](multimodal-extraction.md#text-and-layout-extraction), [tables](multimodal-extraction.md#tables), [OCR](multimodal-extraction.md#ocr-and-scanned-documents), and related subsections on that page.

Pipeline concepts and stage overview appear in [Key concepts](concepts.md).

## Build a searchable collection

After ingestion, configure **[chunking](chunking.md)** and your **[vector database](vdbs.md)** so extracted content is embedded, indexed, and ready for search. Use the **Embedding, indexing & storage** section of the site for depth:

- [Embedding NIMs and models](embedding-nims-models.md)
- [Chunking and splitting](chunking.md)
- [Vector databases](vdbs.md)
- [Multimodal embeddings (VLM)](embedding.md) when your workload needs image+text vectors

**Next:** [Semantic and hybrid retrieval](vdbs.md#semantic-and-hybrid-retrieval) when serving queries (see also [Evaluate on your data](evaluate-on-your-data.md) for reranking and quality checks).
