# Workflow: Ingest documents into a searchable VDB collection

This page covers extracting content from documents and turning that content into a searchable vector collection in one place so you can scroll and search (for example with Ctrl+F) instead of jumping across multiple short workflow stubs.

## Ingest and extract { #ingest-and-extract }

Document ingestion is the step where NeMo Retriever Library reads your files (PDFs, Office documents, images, and other [supported formats](multimodal-extraction.md#supported-file-types-and-formats)), runs extraction and optional enrichment, and returns structured content you can embed and index.

Follow these steps:

1. **Choose how you call the library.** Use the [Python API](nemo-retriever-api-reference.md) or [CLI](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/docs/cli) from application code, or run a deployment (for example [NeMo Retriever Library on GitHub](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever), [Deployment options](deployment-options.md), or [Quickstart: Kubernetes (Helm)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md)) and send jobs over the network. Runnable examples appear in [Choose how you call the library](#choose-how-you-call-the-library) below.
2. **Use parallel PDF handling.** The default ingest path splits large PDFs before Ray processing; refer to [API guide — PDF pre-splitting](nemo-retriever-api-reference.md#pdf-pre-splitting-for-parallel-ingest).
3. **Tune extraction for your content.** Refer to [Multimodal extraction](multimodal-extraction.md) for formats, [text and layout](multimodal-extraction.md#text-and-layout-extraction), [tables](multimodal-extraction.md#tables), [OCR](multimodal-extraction.md#ocr-and-scanned-documents), and related subsections on that page.

Pipeline concepts and stage overview appear in [Key concepts](concepts.md). Default chunking behavior is summarized under [Chunking](concepts.md#chunking).

`create_ingestor(...)` returns a `GraphIngestor`, which chains `.extract()`, `.embed()`, and `.vdb_upload()` into one graph. The Python example below stops after `.embed()` so you can inspect chunks first; append `.vdb_upload(vdb_op="lancedb", vdb_kwargs={...})` before `.ingest()` to write directly to LanceDB (refer to [Vector databases](vdbs.md)).

## Choose how you call the library { #choose-how-you-call-the-library }

The following examples match the [NeMo Retriever Library README](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/README.md). They assume a checkout of the [NeMo Retriever](https://github.com/NVIDIA/NeMo-Retriever) repository and the `batch` run mode with local GPU inference unless you configure remote NIMs.

### Ingest a test PDF (Python)

The [test PDF](https://github.com/NVIDIA/NeMo-Retriever/blob/main/data/multimodal_test.pdf) contains text, tables, charts, and images. The pipeline below chains `.extract()` and `.embed()` only so you can inspect embedded chunks before indexing. To upload in the same run, append `.vdb_upload(...)` before `.ingest()` (parameter details in the [Python API guide](nemo-retriever-api-reference.md)).

```python
from nemo_retriever import create_ingestor
from pathlib import Path

documents = [str(Path("data/multimodal_test.pdf"))]
ingestor = create_ingestor(run_mode="batch")

ingestor = (
    ingestor.files(documents)
    .extract(
        extract_text=True,
        extract_charts=True,
        extract_tables=True,
        extract_infographics=True,
    )
    .embed()
)

result = ingestor.ingest()  # ``pandas.DataFrame`` (``batch`` and ``inprocess``)
```

Run the above with your working directory at the repository root (so `data/multimodal_test.pdf` resolves), or adjust `documents` to the absolute path of the test PDF.

**Next:** [Semantic retrieval](vdbs.md#semantic-retrieval) when serving queries (also refer to [Evaluate on your data](evaluate-on-your-data.md) for reranking and quality checks).
