# `retriever` CLI — Client-Usage Walk-through

This page is the `retriever`-CLI counterpart to
`client/client_examples/examples/cli_client_usage.ipynb`.

The walk-through below covers:

1. Printing `--help`.
2. Running a single PDF with extract, dedup, and image storage.
3. Running a batch of PDFs from a directory.

You can drop these cells into a new notebook (e.g. `retriever_client_usage.ipynb`).

## 1. Help

```bash
retriever --help
retriever pipeline run --help
```

Top-level `--help` lists the subcommand tree; `pipeline run --help` shows the
ingest-specific flags you will actually use in this walk-through.

## 2. Run a single PDF

```bash
retriever pipeline run "${SAMPLE_PDF0}" \
  --input-type pdf \
  --method pdfium \
  --extract-text --extract-tables --extract-charts \
  --dedup --dedup-iou-threshold 0.45 \
  --store-images-uri "${OUTPUT_DIRECTORY_SINGLE}/images" \
  --save-intermediate "${OUTPUT_DIRECTORY_SINGLE}"
```

### Notes

- Table/structure detectors are chosen automatically by the pipeline; there is
  no CLI flag to select a specific table-extraction backend.
- `--dedup` with `--dedup-iou-threshold` removes duplicate image elements.
- There is no image scale/aspect-ratio filter in the `retriever` CLI today.
- `--store-images-uri` persists image assets produced at the configured embed
  granularity.

## 3. Run a batch of PDFs

Point `retriever` at a directory of PDFs:

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

### Notes

- Pass a directory or glob of files; there is no built-in `dataset.json` loader.
- Tune throughput with Ray batch flags such as `--pdf-split-batch-size` and
  `--pdf-extract-batch-size`.

## 4. Inspect results

```python
import pyarrow.parquet as pq
import lancedb

# Parquet extraction dumps written by --save-intermediate:
df = pq.read_table(OUTPUT_DIRECTORY_BATCH).to_pandas()
print(df[["source_id", "text", "content_type"]].head())

# LanceDB rows (default table name "nemo-retriever"):
db = lancedb.connect("./lancedb")
tbl = db.open_table("nemo-retriever")
print(tbl.to_pandas().head())
```
