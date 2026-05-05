# PDF Page-Batch Tuning with the `retriever` CLI

This page is the `retriever`-CLI counterpart to the CLI example in
`nv-ingest/docs/docs/extraction/v2-api-guide.md`.

## Background

The old V2 API introduced per-request PDF splitting via `--api_version v2 --pdf_split_page_count N`:
PDFs larger than `N` pages were split and processed in parallel inside the
service.

`retriever` has **no V1/V2 API concept** — the graph pipeline is always the
current generation. Page-batch parallelism is tuned directly on the pipeline
via `--pdf-split-batch-size` (pages per split batch).

## Replacement example

Old (V2 API with 64 pages per chunk):

```bash
nv-ingest-cli \
  --api_version v2 \
  --pdf_split_page_count 64 \
  --doc large_document.pdf \
  --task 'extract:{"document_type":"pdf", "extract_text":true}' \
  --output_directory ./results
```

New:

```bash
retriever pipeline run large_document.pdf \
  --input-type pdf \
  --method pdfium \
  --extract-text --no-extract-tables --no-extract-charts \
  --pdf-split-batch-size 64 \
  --save-intermediate ./results
```

### What `--pdf-split-batch-size` does

`--pdf-split-batch-size` controls how many pages are grouped into one batch passed
downstream to the OCR / page-elements / extraction actors. Smaller values give
more parallelism but more overhead; larger values amortize overhead but limit
concurrency — the same trade-off the V2 `--pdf_split_page_count` flag exposed.

For very wide fan-out you can also increase the number of concurrent actors:

```bash
retriever pipeline run large_document.pdf \
  --input-type pdf \
  --pdf-split-batch-size 64 \
  --page-elements-actors 4 \
  --ocr-actors 4 \
  --embed-actors 2 \
  --save-intermediate ./results
```

## Parity notes

- The pipeline always runs through the current ingest graph — there is no
  `--api_version` flag.
- `--pdf-split-batch-size` is a batching knob on the local Ray pipeline, not a
  server-side splitter. End-user outcome (parallel extraction of a large PDF)
  is preserved.
- Use `retriever pipeline run --help` to see related tuning flags
  (`--pdf-extract-batch-size`, etc.).
