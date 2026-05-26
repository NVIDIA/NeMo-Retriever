# Workflow: Ingest documents into a searchable VDB collection

This page covers extracting content from documents and turning that content into a searchable vector collection in one place so you can scroll and search (for example with Ctrl+F) instead of jumping across multiple short workflow stubs.

## Ingest and extract

Document ingestion is the step where NeMo Retriever Library reads your files (PDFs, Office documents, images, and other [supported formats](multimodal-extraction.md#supported-file-types-and-formats)), runs extraction and optional enrichment, and returns structured content you can embed and index.

Follow these steps:

1. **Choose how you call the library.** Use the [Python API](nemo-retriever-api-reference.md) or [CLI](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/docs/cli) from application code, or run a deployment (for example [NeMo Retriever Library on GitHub](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever), [Deployment options](deployment-options.md), or [Quickstart: Kubernetes (Helm)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md)) and send jobs over the network. Runnable examples appear in [Choose how you call the library](#choose-how-you-call-the-library) below.
2. **Use parallel PDF handling.** The default ingest path splits large PDFs before Ray processing; behavior and tuning are described in the [API guide — PDF pre-splitting](nemo-retriever-api-reference.md#pdf-pre-splitting-for-parallel-ingest). Set `message_client_kwargs={"api_version": "v2"}` when using the client if you need to be explicit.
3. **Tune extraction for your content.** Refer to [Multimodal extraction](multimodal-extraction.md) for formats, [text and layout](multimodal-extraction.md#text-and-layout-extraction), [tables](multimodal-extraction.md#tables), [OCR](multimodal-extraction.md#ocr-and-scanned-documents), and related subsections on that page.

Pipeline concepts and stage overview appear in [Key concepts](concepts.md). Default chunking behavior is summarized under [Chunking](concepts.md#chunking).

`create_ingestor(...)` returns a `GraphIngestor`, which chains `.extract()` and `.embed()` but does **not** implement `.vdb_upload()` yet (the base method records the intent only and raises `NotImplementedError` if you call it). To write embeddings into LanceDB after graph ingest, use the `graph_pipeline` CLI below (it performs upload via an internal helper, not the fluent builder) or follow the storage patterns in [Vector databases](vdbs.md). The Python example stops after `.embed()` and returns a dataset you can index in a separate step.

## Choose how you call the library

The following examples match the [NeMo Retriever Library README](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/README.md). They assume a checkout of the [NeMo Retriever](https://github.com/NVIDIA/NeMo-Retriever) repository and the `batch` run mode with local GPU inference unless you configure remote NIMs.

### Ingest a test PDF (Python)

The [test PDF](https://github.com/NVIDIA/NeMo-Retriever/blob/main/data/multimodal_test.pdf) contains text, tables, charts, and images. The pipeline below chains `.extract()` and `.embed()` only—**do not** append `.vdb_upload()` on `GraphIngestor`; it is not implemented on this code path.

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

dataset = ingestor.ingest()  # ``run_mode='batch'`` → ``ray.data.Dataset``; ``inprocess`` → ``pandas.DataFrame``
chunks = dataset.take_all()  # ``take_all()`` is a Ray Dataset API; use DataFrame methods in ``inprocess``
```

Run the above with your working directory at the repository root (so `data/multimodal_test.pdf` resolves), or adjust `documents` to the absolute path of the test PDF.

### Ingest a test corpus (CLI)

`graph_pipeline` is the canonical ingestion script used throughout the [QA evaluation guide](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/evaluation/README.md#step-1-ingest-and-embed-pdfs-nemo-retriever). Point it at a **directory** of PDFs to produce a ready-to-query LanceDB table.

!!! note "Corpus size and LanceDB indexing"

    LanceDB's default IVF index needs enough chunks to train its partitions (often on the order of tens of chunks). A single small PDF can be insufficient; use a directory with enough documents for your index settings. Replace `/your-example-dir` with your corpus path.

```bash
python -m nemo_retriever.examples.graph_pipeline \
  /your-example-dir \
  --lancedb-uri lancedb
```

For build.nvidia.com hosted inference, set [`NVIDIA_API_KEY`](api-keys.md#nvidia-api-key) and pass the `--*-invoke-url` / `--embed-invoke-url` options shown in the [README remote inference section](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/README.md#ingest-a-test-corpus-cli).

**Next:** [Semantic and hybrid retrieval](vdbs.md#semantic-and-hybrid-retrieval) when serving queries (see also [Evaluate on your data](evaluate-on-your-data.md) for reranking and quality checks).
