# Use the Retriever Command Line Interface

This page is the `retriever`-CLI counterpart to
`nv-ingest/docs/docs/extraction/nv-ingest_cli.md` and
`nv-ingest/docs/docs/extraction/cli-reference.md`.

The original `nv-ingest-cli` docs remain valid for service-based ingestion.
The examples below show how to obtain the same end-user outcomes with the new
`retriever` Typer app (the `retriever` executable installed with the
`nemo-retriever` package).

> **Shape difference.** `nv-ingest-cli` is a single command that submits
> `--task ...` definitions to a running REST service. `retriever` is a
> multi-subcommand app. The main replacement for document ingestion is
> `retriever pipeline run INPUT_PATH`, which runs the full graph pipeline
> locally (in-process or Ray) and writes rows to LanceDB + optional Parquet.

To check the installed version:

```bash
retriever --version
```

To list the top-level subcommands:

```bash
retriever --help
```

To see the full `pipeline run` flag surface (the closest analogue to the old
single-command CLI):

```bash
retriever pipeline run --help
```

## Examples

### Example: Text / PDF file — extract with defaults

Old (`nv-ingest-cli`, returns full metadata over the service):

```bash
nv-ingest-cli \
  --doc ./data/test.pdf \
  --client_host=localhost \
  --client_port=7670
```

New (`retriever`, local pipeline; text, tables, and charts are extracted by
default):

```bash
retriever pipeline run ./data/test.pdf \
  --input-type pdf \
  --run-mode inprocess \
  --save-intermediate ./processed_docs
```

**Parity note.** Results are written to LanceDB (`./lancedb` by default) and,
with `--save-intermediate`, to a Parquet file under `./processed_docs`.
The old per-content-type `*.metadata.json` tree is not produced; use
`pyarrow.parquet.read_table` or LanceDB queries to inspect rows.

### Example: PDF with splitting only

The old `--task='split'` example submitted a split-only job to the service:

```bash
nv-ingest-cli \
  --doc ./data/test.pdf \
  --output_directory ./processed_docs \
  --task='split' \
  --client_host=localhost \
  --client_port=7670
```

With `retriever`, splitting is intrinsic to the pipeline. Control text-level
chunking via `--text-chunk`, and control PDF page-batch sizing via
`--pdf-split-batch-size`:

```bash
retriever pipeline run ./data/test.pdf \
  --input-type pdf \
  --no-extract-tables --no-extract-charts \
  --text-chunk --text-chunk-max-tokens 512 --text-chunk-overlap-tokens 64 \
  --save-intermediate ./processed_docs
```

**Parity note.** There is no "split-only, no extraction" mode. If you only
care about chunk boundaries, run the pipeline with extraction narrowed to text
and inspect the resulting chunks.

### Example: PDF with splitting and extraction

Old:

```bash
nv-ingest-cli \
  --doc ./data/test.pdf \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium"}' \
  --task='extract:{"document_type": "docx", "extract_method": "python_docx"}' \
  --task='split' \
  --client_host=localhost \
  --client_port=7670
```

New — run once per input type (PDF and docx/pptx use different `--input-type`):

```bash
retriever pipeline run ./data/test.pdf \
  --input-type pdf \
  --method pdfium \
  --text-chunk --text-chunk-max-tokens 512 \
  --save-intermediate ./processed_docs
```

```bash
retriever pipeline run ./data/test.docx \
  --input-type doc \
  --text-chunk --text-chunk-max-tokens 512 \
  --save-intermediate ./processed_docs
```

**Parity note.** `--input-type doc` matches `*.docx` and `*.pptx`
(see `src/nemo_retriever/pipeline/__main__.py::_resolve_file_patterns`). Mixed
PDF + docx in a single invocation is not supported; invoke once per type.

### Example: PDF with a custom page-batch size

Old:

```bash
nv-ingest-cli \
  --doc ./data/test.pdf \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium", "extract_text": "true"}' \
  --pdf_split_page_count 64 \
  --api_version v2 \
  --client_host=localhost \
  --client_port=7670
```

New:

```bash
retriever pipeline run ./data/test.pdf \
  --input-type pdf \
  --method pdfium \
  --extract-text --no-extract-tables --no-extract-charts \
  --pdf-split-batch-size 64 \
  --save-intermediate ./processed_docs
```

**Parity note.** There is no v1/v2 concept in `retriever`; `--pdf-split-batch-size`
is the batching knob used by the Ray pipeline.

### Example: Caption images

Old:

```bash
nv-ingest-cli \
  --doc ./data/test.pdf \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium", "extract_images": "true"}' \
  --task='caption:{"prompt": "Caption the content of this image:", "reasoning": true}' \
  --client_host=localhost \
  --client_port=7670
```

New:

```bash
retriever pipeline run ./data/test.pdf \
  --input-type pdf \
  --method pdfium \
  --caption \
  --caption-model-name nvidia/NVIDIA-Nemotron-Nano-VL-8B-V2 \
  --caption-invoke-url https://integrate.api.nvidia.com/v1/chat/completions \
  --api-key "${NVIDIA_API_KEY}" \
  --store-images-uri ./processed_docs/images \
  --save-intermediate ./processed_docs
```

**Parity gaps.**

- The `reasoning: true` option is not exposed as a CLI flag.
- A custom `prompt` is not exposed either; the caption stage uses its default
  prompt. For prompt or reasoning control, drop to the Python API
  (`nemo_retriever.ingestor.Ingestor.caption(...)`).
- If you do not set a caption endpoint / local GPU profile, the caption stage
  is skipped at runtime — matching the old behavior.

### Example: Process a directory of documents

Old (dataset file with sampled entries):

```bash
nv-ingest-cli \
  --dataset dataset.json \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium"}' \
  --client_host=localhost \
  --client_port=7670
```

New (point `retriever` at a directory; it globs files for the given
`--input-type`):

```bash
retriever pipeline run ./data/pdf_corpus \
  --input-type pdf \
  --method pdfium \
  --save-intermediate ./processed_docs
```

**Parity gap.** The `dataset.json` (`sampled_files`) schema and
`gen_dataset.py` sampler are not reproduced. Materialize a directory (or glob)
that contains the files you want.

### Example: Upload extracted images to object storage

Old: extraction + a MinIO upload configured on the service side.

```bash
nv-ingest-cli \
  --doc ./data/test.pdf \
  --output_directory ./processed_docs \
  --task='extract:{"document_type": "pdf", "extract_method": "pdfium"}' \
  --client_host=localhost \
  --client_port=7670
```

New — use the first-class `--store-images-uri` flag (local path, `s3://…`, or
any fsspec URI):

```bash
retriever pipeline run ./data/test.pdf \
  --input-type pdf \
  --method pdfium \
  --store-images-uri s3://my-bucket/images \
  --strip-base64 \
  --save-intermediate ./processed_docs
```

**Parity note.** `--strip-base64` removes inline base64 from rows after the
image has been persisted, so downstream consumers follow `content_url`
references — analogous to `--save_images_separately` on the old CLI.

## Where results live

- **LanceDB table.** `--lancedb-uri lancedb` (default). Query via
  `retriever recall vdb-recall …` or the `nemo_retriever.retriever.Retriever`
  Python class.
- **Parquet (optional).** `--save-intermediate <dir>` writes the extraction
  DataFrame for inspection.
- **Images (optional).** `--store-images-uri <uri>` writes extracted images to
  a local path or object store.

## Errors and exit codes

`retriever pipeline run` exits **0** on success, **non-zero** on Typer
validation errors (bad `--input-type`, missing `INPUT_PATH`, etc.) or pipeline
failures. Use `--debug` or `--log-file <path>` for detailed diagnostics.
