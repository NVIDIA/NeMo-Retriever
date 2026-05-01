# PDF Blueprint — `retriever` CLI Replacement

This page is the `retriever`-CLI counterpart to the CLI cell in
`nv-ingest/deploy/pdf-blueprint.ipynb`.

## Original blueprint cell

```bash
nv-ingest-cli \
  --doc nv-ingest/data/multimodal_test.pdf \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium", "extract_tables": "true", "extract_images": "true", "extract_charts": "true"}' \
  --client_host=host.docker.internal \
  --client_port=7670
```

This submits the blueprint's multimodal sample PDF to the running ingest
service and asks for text + tables + charts + images.

## `retriever` equivalent

```bash
retriever pipeline run nv-ingest/data/multimodal_test.pdf \
  --input-type pdf \
  --method pdfium \
  --extract-text --extract-tables --extract-charts \
  --store-images-uri ./processed_docs/images \
  --strip-base64 \
  --save-intermediate ./processed_docs
```

### What you get (end-user outcome)

- The same multimodal content (text, table markdown, chart descriptions,
  extracted images) is produced.
- Text / table / chart rows land in LanceDB at `./lancedb/nv-ingest.lance`.
- Parquet extraction rows are written under `./processed_docs/`.
- Extracted images are written under `./processed_docs/images/`, referenced by
  `content_url` in the row metadata.

### Notebook-friendly form

To keep the notebook self-contained, prefix the shell cell with `!`:

```bash
!retriever pipeline run nv-ingest/data/multimodal_test.pdf \
    --input-type pdf \
    --method pdfium \
    --extract-text --extract-tables --extract-charts \
    --store-images-uri ./processed_docs/images \
    --strip-base64 \
    --save-intermediate ./processed_docs
```

And inspect the results in the next cell:

```python
import pyarrow.parquet as pq
import lancedb

df = pq.read_table("./processed_docs").to_pandas()
print(df[["source_id", "content_type"]].value_counts())

db = lancedb.connect("./lancedb")
tbl = db.open_table("nv-ingest")
print(tbl.to_pandas().head())
```

## Migrating the blueprint `pip install` cell

The blueprint also installs `nv-ingest-client==25.9.0`. For the `retriever`
path, install `nemo-retriever` instead (see `nemo_retriever/README.md` for
current pinned versions):

```bash
pip install "nemo-retriever==26.3.0" \
    nv-ingest-client==26.3.0 nv-ingest==26.3.0 nv-ingest-api==26.3.0 \
    lancedb \
    minio \
    tritonclient
```

## Parity notes

- `client_host=host.docker.internal` / `client_port=7670` are irrelevant here:
  `retriever pipeline run` is in-process, so the blueprint no longer needs a
  running `nv-ingest-ms-runtime` container for the CLI cell.
- If you still want the blueprint to hit a live service (for example to
  exercise the REST API), replace the CLI cell with a `retriever online serve`
  container plus `retriever online stream-pdf` for per-page NDJSON output.
  Note that `retriever online submit` is currently a stub.
