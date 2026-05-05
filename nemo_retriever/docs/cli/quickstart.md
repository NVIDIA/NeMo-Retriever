# Quick Start — `retriever` CLI

This page is the `retriever`-CLI counterpart to the CLI section of
`nv-ingest/docs/docs/extraction/quickstart-guide.md`.

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

### What you get

- Extracted text, table markdown, and chart descriptions as rows in the
  LanceDB table at `./lancedb/nv-ingest.lance` (default `--lancedb-uri`).
- Per-document extraction rows as Parquet under `./processed_docs/` (from
  `--save-intermediate`).
- Extracted images on disk under `./processed_docs/images/` (from
  `--store-images-uri`). The `content_url` column points at these paths.
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

- Pass a directory for batch ingestion:
  `retriever pipeline run ./data/pdf_corpus --input-type pdf …`.
- For faster throughput on a multi-GPU node, keep `--run-mode batch` (default,
  Ray-based) and tune `--pdf-split-batch-size`, `--embed-actors`,
  `--embed-batch-size`, `--ocr-actors`, and `--page-elements-actors`.
- For debugging or CI, use `--run-mode inprocess` to avoid starting Ray.
