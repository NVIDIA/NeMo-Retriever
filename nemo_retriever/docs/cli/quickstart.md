# Quick Start — `retriever` CLI

This page is the `retriever`-CLI counterpart to the legacy `nv-ingest-cli`
quickstart. Self-hosted Helm deployment docs live in the NeMo Retriever repo
under [nemo_retriever/helm](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/helm).

## Replacement for the quickstart CLI example

The original quickstart example submits a single PDF to the running service
and asks for text, tables, charts, and images:

```bash
nv-ingest-cli \
  --doc ./data/multimodal_test.pdf \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium", "extract_tables": "true", "extract_images": "true", "extract_charts": "true"}' \
  --client_host=localhost \
  --client_port=7670
```

The `retriever` equivalent runs the full pipeline locally — extraction,
embedding, and LanceDB upload — and produces the same multimodal outputs:

```bash
retriever pipeline run ./data/multimodal_test.pdf \
  --input-type pdf \
  --method pdfium \
  --extract-text --extract-tables --extract-charts \
  --store-images-uri ./processed_docs/images \
  --save-intermediate ./processed_docs
```

For a lightweight PDF-only SDK smoke workflow, the root commands provide a
smaller surface:

```bash
retriever ingest ./data/multimodal_test.pdf
retriever query "What is in this document?"
```

Route individual stages to self-hosted or hosted NIM endpoints by passing only
the URLs you want to override; omitted URLs keep the library defaults:

```bash
export NVIDIA_API_KEY=nvapi-...

retriever ingest ./data/multimodal_test.pdf \
  --page-elements-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3 \
  --ocr-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v1 \
  --ocr-version v1 \
  --graphic-elements-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-graphic-elements-v1 \
  --table-structure-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1 \
  --embed-invoke-url https://integrate.api.nvidia.com/v1/embeddings \
  --embed-model-name nvidia/llama-nemotron-embed-1b-v2

retriever query "What is in this document?" \
  --embed-invoke-url https://integrate.api.nvidia.com/v1/embeddings \
  --embed-model-name nvidia/llama-nemotron-embed-1b-v2 \
  --reranker-invoke-url https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-vl-1b-v2/reranking
```

`NVIDIA_API_KEY` is required only when those URLs point at hosted
build.nvidia.com endpoints; the root commands intentionally do not expose an
`--api-key` flag. `NGC_API_KEY` is still used separately when pulling or
running self-hosted NIM containers.

### What you get

- Extracted text, table markdown, and chart descriptions as rows in the
  LanceDB table at `./lancedb/nv-ingest.lance` (default `--lancedb-uri`).
- Per-document extraction rows as Parquet under `./processed_docs/` (from
  `--save-intermediate`).
- Extracted image assets on disk under `./processed_docs/images/` (from
  `--store-images-uri`). The stored asset URI is written to row metadata.
- Progress, timing, and stage-level logs on stderr.

### Inspect the results

```bash
ls ./processed_docs
ls ./processed_docs/images
ls ./lancedb
```

For programmatic access:

```python
import pyarrow.parquet as pq
import lancedb

df = pq.read_table("./processed_docs").to_pandas()
print(df.head())

db = lancedb.connect("./lancedb")
tbl = db.open_table("nv-ingest")
print(tbl.to_pandas().head())
```

Or query via the Retriever Python client (same workflow as the library
quickstart in `nemo_retriever/README.md`):

```python
from nemo_retriever.retriever import Retriever

retriever = Retriever(lancedb_uri="lancedb", lancedb_table="nv-ingest", top_k=5)
hits = retriever.query(
    "Given their activities, which animal is responsible for the typos?"
)
```

## Notes on running larger datasets

- Pass a directory for root batch ingestion:
  `retriever ingest ./data/pdf_corpus --run-mode batch`.
- For larger PDF batches, tune root ingest with `--pdf-extract-workers`,
  `--pdf-extract-batch-size`, `--page-elements-workers`,
  `--page-elements-batch-size`, `--ocr-workers`, `--ocr-batch-size`,
  `--embed-workers`, and `--embed-batch-size`.
- For debugging or CI, use `--run-mode inprocess` to avoid starting Ray.
