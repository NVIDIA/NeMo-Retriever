# `retriever` CLI — Client-Usage Walk-through

This page is the `retriever`-CLI counterpart to
`nv-ingest/client/client_examples/examples/cli_client_usage.ipynb`.

The original notebook walks through `nv-ingest-cli` by:

1. Printing `--help`.
2. Submitting a single PDF with `extract + dedup + filter` tasks.
3. Submitting a dataset of PDFs with the same task set.

The equivalent `retriever` workflow is shown below. You can drop these cells
into a new notebook (e.g. `retriever_client_usage.ipynb`) alongside the old
one.

## 1. Help

```bash
retriever --help
retriever pipeline run --help
```

Top-level `--help` lists the subcommand tree; `pipeline run --help` shows the
ingest-specific flags you will actually use in this walk-through.

## 2. Submit a single PDF

Old notebook cell:

```bash
nv-ingest-cli \
  --doc ${SAMPLE_PDF0} \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium", "extract_text": true, "extract_images": true, "extract_tables": true, "extract_tables_method": "yolox"}' \
  --task='dedup:{"content_type": "image", "filter": true}' \
  --task='filter:{"content_type": "image", "min_size": 128, "max_aspect_ratio": 5.0, "min_aspect_ratio": 0.2, "filter": true}' \
  --client_host=${REDIS_HOST} \
  --client_port=${REDIS_PORT} \
  --output_directory=${OUTPUT_DIRECTORY_SINGLE}
```

New:

```bash
retriever pipeline run "${SAMPLE_PDF0}" \
  --input-type pdf \
  --method pdfium \
  --extract-text --extract-tables --extract-charts \
  --dedup --dedup-iou-threshold 0.45 \
  --store-images-uri "${OUTPUT_DIRECTORY_SINGLE}/images" \
  --save-intermediate "${OUTPUT_DIRECTORY_SINGLE}"
```

### Parity notes

- `extract_tables_method:"yolox"` is not a CLI selector — the pipeline picks
  its table/structure detectors automatically. Tables are still extracted.
- `dedup:{content_type:"image", filter:true}` maps to `--dedup` (with
  `--dedup-iou-threshold` for the IoU threshold).
- `filter:{content_type:"image", min_size, min/max_aspect_ratio, filter:true}`
  **has no parity.** There is no image scale/aspect-ratio filter in the
  `retriever` CLI today. If that matters, drop to the Python API or keep the
  old `nv-ingest-cli` for that example.
- `extract_images:true` is implicitly satisfied by `--store-images-uri`
  (images are extracted and persisted to the URI).

## 3. Submit a dataset of PDFs

Old notebook cell:

```bash
nv-ingest-cli \
  --dataset ${BATCH_FILE} \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium", "extract_text": true, "extract_images": true, "extract_tables": true, "extract_tables_method": "yolox"}' \
  --task='dedup:{"content_type": "image", "filter": true}' \
  --task='filter:{"content_type": "image", "min_size": 128, "max_aspect_ratio": 5.0, "min_aspect_ratio": 0.2, "filter": true}' \
  --client_host=${REDIS_HOST} \
  --client_port=${REDIS_PORT} \
  --output_directory=${OUTPUT_DIRECTORY_BATCH}
```

New — point `retriever` at a directory of PDFs instead of a dataset JSON:

```bash
# Assume $PDF_DIR is a directory holding your batch of PDFs.
retriever pipeline run "${PDF_DIR}" \
  --input-type pdf \
  --method pdfium \
  --extract-text --extract-tables --extract-charts \
  --dedup --dedup-iou-threshold 0.45 \
  --store-images-uri "${OUTPUT_DIRECTORY_BATCH}/images" \
  --save-intermediate "${OUTPUT_DIRECTORY_BATCH}"
```

### Parity notes

- The `dataset.json` (`sampled_files`) format and `gen_dataset.py` sampler
  are not reproduced. Materialize a directory (or glob) containing the files
  you want to process.
- The `--shuffle_dataset` knob is not present; set Ray block / batch sizes
  via `--pdf-split-batch-size`, `--pdf-extract-batch-size`, etc. for throughput.

## 4. Inspect results

```python
import pyarrow.parquet as pq
import lancedb

# Parquet extraction dumps written by --save-intermediate:
df = pq.read_table(OUTPUT_DIRECTORY_BATCH).to_pandas()
print(df[["source_id", "text", "content_type"]].head())

# LanceDB rows (default table name "nv-ingest"):
db = lancedb.connect("./lancedb")
tbl = db.open_table("nv-ingest")
print(tbl.to_pandas().head())
```

## Migration summary

| Old notebook cell | New `retriever` form | Parity |
|-------------------|----------------------|--------|
| `!nv-ingest-cli --help` | `!retriever --help` (plus `retriever pipeline run --help`) | Full |
| Single-file extract + dedup + filter | `retriever pipeline run <file> … --dedup …` | Partial — no image-size/aspect filter, `extract_tables_method` auto-selected |
| Dataset extract + dedup + filter | `retriever pipeline run <dir> …` | Partial — no `dataset.json` loader; use a directory |
