# Quick Start — `retriever` CLI

This page is the `retriever`-CLI counterpart to the legacy **ingestion-service** quickstart.

Looking for local **Docker Compose** workflows? See
[`docker.md`](../../docker.md) for **unsupported developer tooling** only.

For **supported** deployment of NeMo Retriever / **NIM** containers, use the
**Helm** documentation: [nemo_retriever/helm](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/helm)
and the **NeMo Retriever Library Helm** install guides in the
[NeMo Retriever Library](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/).

## Replacement for the quickstart CLI example

The original quickstart example submits a single PDF to the running service
and asks for text, tables, charts, and images:

```bash
# Legacy one-liner that targeted localhost:7670 is superseded by:
retriever pipeline run ./data/multimodal_test.pdf \
  --input-type pdf \
  --method pdfium \
  --extract-text --extract-tables --extract-charts \
  --store-images-uri ./processed_docs/images \
  --save-intermediate ./processed_docs
```

### What you get

- Extracted text, table markdown, and chart descriptions as rows in the
  LanceDB table at `./lancedb/nemo-retriever.lance` (default `--lancedb-uri`).
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
tbl = db.open_table("nemo-retriever")
print(tbl.to_pandas().head())
```

Or query via the Retriever Python client (same workflow as the library
quickstart in `nemo_retriever/README.md`):

```python
from nemo_retriever.retriever import Retriever

retriever = Retriever(lancedb_uri="lancedb", lancedb_table="nemo-retriever", top_k=5)
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
