# Retriever CLI — replacement examples for the legacy ingestion-service CLI

This folder contains `retriever` command-line examples that deliver the same
end-user outcomes as the legacy **ingestion-service** CLI examples that used to
live under `docs/`, `api/`, `client/`, and `deploy/` in older repository layouts.

The historical CLI documentation is **not removed** from the ecosystem — these files sit
alongside it as a new-CLI counterpart you can link to or migrate to.

## Key shape difference

The legacy **ingestion-service** CLI was a **single command that talks to a running REST service on
`localhost:7670`** and composes work via repeated `--task extract|split|caption|embed|dedup|filter|udf`.

`retriever` is a **multi-subcommand Typer app**. Most of the old CLI examples
map to `retriever pipeline run INPUT_PATH`, which runs the graph pipeline
locally (in-process or via Ray) and writes results to LanceDB and, optionally,
to Parquet / object storage. Other subcommands cover focused tasks:

| Old intent | New subcommand |
|------------|----------------|
| Extract + embed + store a batch of documents | `retriever pipeline run` |
| Run an ad-hoc PDF extraction stage | `retriever pdf stage` |
| Run an HTML / text / audio / chart stage | `retriever html run`, `retriever txt run`, `retriever audio extract`, `retriever chart run` |
| Upload stage output to LanceDB | `retriever ingest` or `retriever pipeline run` |
| Query LanceDB + compute recall@k | `retriever recall vdb-recall` |
| Run a QA evaluation sweep | `retriever eval run` |
| Serve / submit to the online REST API | `retriever online serve` / `retriever online stream-pdf` |
| Benchmark stage throughput | `retriever benchmark {split,extract,audio-extract,page-elements,ocr,all}` |
| Benchmark orchestration | `retriever harness {run,sweep,nightly,summary,compare}` |

## Contents

| Topic | Location | Replaces example(s) in |
|-------|----------|------------------------|
| Quick start | [below](#quick-start) | Legacy service quickstart; **Helm** + [NeMo Retriever Library](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/); **Docker Compose** (unsupported): [`docker.md`](https://github.com/NVIDIA/NeMo-Retriever/blob/HEAD/nemo_retriever/docker.md) |
| CLI reference | [below](#cli-reference) | Prior `cli-reference` pages under `docs/docs/extraction/` |
| Client usage walk-through | [below](#client-usage-walk-through) | `client/client_examples/examples/cli_client_usage.ipynb` |
| PDF split tuning | [Large PDF page batches](#large-pdf-page-batches) below | `docs/docs/extraction/v2-api-guide.md` |
| Benchmarking | [`benchmarking.md`](benchmarking.md) | `docs/docs/extraction/benchmarking.md` and `tools/harness/README.md` |

<!-- --8<-- [start:quickstart] -->

## Quick start

Local **Docker Compose** workflows are **unsupported developer tooling** only — see
[`docker.md`](https://github.com/NVIDIA/NeMo-Retriever/blob/HEAD/nemo_retriever/docker.md) (GitHub `HEAD` = default branch; pin to your release tag when not on `main`).

For **supported** deployment of NeMo Retriever / **NIM** containers, use
[nemo_retriever/helm](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/helm)
and the [NeMo Retriever Library](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/)
Helm install guides.

### Ingest a PDF

```bash
retriever pipeline run ./data/multimodal_test.pdf \
  --input-type pdf \
  --method pdfium \
  --extract-text --extract-tables --extract-charts \
  --store-images-uri ./processed_docs/images \
  --save-intermediate ./processed_docs
```

For a lightweight PDF-only workflow:

```bash
retriever ingest ./data/multimodal_test.pdf
retriever query "What is in this document?"
```

Route stages to self-hosted or hosted NIM endpoints by passing only the URLs you
want to override:

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
build.nvidia.com endpoints. `NGC_API_KEY` is used separately when pulling or
running self-hosted NIM containers.

### What you get

- Extracted text, tables, and charts as rows in LanceDB at `./lancedb` (default
  table name `nv-ingest`).
- Per-document Parquet under `./processed_docs/` (`--save-intermediate`).
- Image assets under `./processed_docs/images/` (`--store-images-uri`).
- Progress and stage logs on stderr.

### Inspect the results

```bash
ls ./processed_docs
ls ./processed_docs/images
ls ./lancedb
```

```python
import pyarrow.parquet as pq
import lancedb

df = pq.read_table("./processed_docs").to_pandas()
print(df.head())

db = lancedb.connect("./lancedb")
tbl = db.open_table("nv-ingest")
print(tbl.to_pandas().head())
```

Or query via the Retriever Python client (`nemo_retriever/README.md`):

```python
from nemo_retriever.retriever import Retriever

retriever = Retriever(lancedb_uri="lancedb", lancedb_table="nv-ingest", top_k=5)
hits = retriever.query(
    "Given their activities, which animal is responsible for the typos?"
)
```

### Larger datasets

- Batch ingest: `retriever ingest ./data/pdf_corpus --run-mode batch`.
- Tune throughput with `--pdf-extract-workers`, `--pdf-extract-batch-size`,
  `--page-elements-workers`, `--page-elements-batch-size`, `--ocr-workers`,
  `--ocr-batch-size`, `--embed-workers`, and `--embed-batch-size`.
- For CI or debugging: `--run-mode inprocess` skips Ray startup.

<!-- --8<-- [end:quickstart] -->

## CLI reference

`retriever` is the Typer app installed with the `nemo-retriever` package. Document
ingestion is usually `retriever pipeline run INPUT_PATH`, which runs the graph pipeline
locally (in-process or Ray) and writes rows to LanceDB and optional Parquet.

```bash
retriever --version
retriever --help
retriever pipeline run --help
```

### Extract a PDF with defaults

```bash
retriever pipeline run ./data/test.pdf \
  --input-type pdf \
  --run-mode inprocess \
  --save-intermediate ./processed_docs
```

Results go to LanceDB (`./lancedb`, table `nv-ingest` by default) and, with
`--save-intermediate`, to Parquet under `./processed_docs`. Inspect rows with
`pyarrow.parquet` or LanceDB queries (not per-content-type `*.metadata.json` files).

### Text chunking and PDF page batches

Splitting is intrinsic to the pipeline. Control text chunks with `--text-chunk` and
page-batch sizing with `--pdf-split-batch-size`:

```bash
retriever pipeline run ./data/test.pdf \
  --input-type pdf \
  --no-extract-tables --no-extract-charts \
  --text-chunk --text-chunk-max-tokens 512 --text-chunk-overlap-tokens 64 \
  --save-intermediate ./processed_docs
```

There is no split-only mode without extraction; narrow flags to text extraction if you
only need chunk boundaries.

### PDF and Office documents

Run once per input type (`--input-type doc` matches `*.docx` and `*.pptx`):

```bash
retriever pipeline run ./data/test.pdf \
  --input-type pdf \
  --method pdfium \
  --text-chunk --text-chunk-max-tokens 512 \
  --save-intermediate ./processed_docs

retriever pipeline run ./data/test.docx \
  --input-type doc \
  --text-chunk --text-chunk-max-tokens 512 \
  --save-intermediate ./processed_docs
```

Mixed PDF and docx in one invocation is not supported.

### Large PDF page batches

```bash
retriever pipeline run ./data/test.pdf \
  --input-type pdf \
  --method pdfium \
  --extract-text --no-extract-tables --no-extract-charts \
  --pdf-split-batch-size 64 \
  --save-intermediate ./processed_docs
```

### Caption images

```bash
retriever pipeline run ./data/test.pdf \
  --input-type pdf \
  --method pdfium \
  --caption \
  --caption-model-name nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 \
  --caption-invoke-url https://integrate.api.nvidia.com/v1/chat/completions \
  --api-key "${NVIDIA_API_KEY}" \
  --store-images-uri ./processed_docs/images \
  --save-intermediate ./processed_docs
```

For hosted Omni captioning, set
`--caption-model-name nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`. Local Omni uses
`nemo_retriever[local]` and a local Hugging Face model ID. Custom caption prompts and
`reasoning` flags are not exposed on the CLI — use
`nemo_retriever.ingestor.Ingestor.caption(...)` in Python.

### Directory of documents

```bash
retriever pipeline run ./data/pdf_corpus \
  --input-type pdf \
  --method pdfium \
  --save-intermediate ./processed_docs
```

There is no `dataset.json` loader; pass a directory or glob of files.

### Store images to object storage

```bash
retriever pipeline run ./data/test.pdf \
  --input-type pdf \
  --method pdfium \
  --store-images-uri s3://my-bucket/images \
  --save-intermediate ./processed_docs
```

Image URIs are written to row metadata. Use `--store-actors` to tune object-storage
write concurrency.

### Where results live

- **LanceDB** — `--lancedb-uri lancedb` (default), table `nv-ingest`. Query via
  `retriever recall vdb-recall …` or `nemo_retriever.retriever.Retriever`.
- **Parquet** — `--save-intermediate <dir>` for the extraction DataFrame.
- **Images** — `--store-images-uri <uri>` (local path or fsspec URI). Storage follows
  `--embed-granularity` (page vs element images).

### Errors and exit codes

`retriever pipeline run` exits **0** on success and **non-zero** on validation or
pipeline failures. Use `--debug` or `--log-file <path>` for diagnostics.

## Client usage walk-through

Counterpart to `client/client_examples/examples/cli_client_usage.ipynb`. Covers help, a
single-PDF run, a batch directory run, and inspecting results. Drop these cells into a
notebook (e.g. `retriever_client_usage.ipynb`) if you prefer.

### Help

```bash
retriever --help
retriever pipeline run --help
```

Top-level `--help` lists the subcommand tree; `pipeline run --help` shows the
ingest-specific flags used below.

### Run a single PDF

```bash
retriever pipeline run "${SAMPLE_PDF0}" \
  --input-type pdf \
  --method pdfium \
  --extract-text --extract-tables --extract-charts \
  --dedup --dedup-iou-threshold 0.45 \
  --store-images-uri "${OUTPUT_DIRECTORY_SINGLE}/images" \
  --save-intermediate "${OUTPUT_DIRECTORY_SINGLE}"
```

- Table/structure detectors are chosen automatically; there is no CLI flag to pick a
  specific table-extraction backend.
- `--dedup` with `--dedup-iou-threshold` removes duplicate image elements.
- There is no image scale/aspect-ratio filter in the `retriever` CLI today.
- `--store-images-uri` persists image assets at the configured embed granularity.

### Run a batch of PDFs

```bash
# $PDF_DIR is a directory of PDFs.
retriever pipeline run "${PDF_DIR}" \
  --input-type pdf \
  --method pdfium \
  --extract-text --extract-tables --extract-charts \
  --dedup --dedup-iou-threshold 0.45 \
  --store-images-uri "${OUTPUT_DIRECTORY_BATCH}/images" \
  --save-intermediate "${OUTPUT_DIRECTORY_BATCH}"
```

- Pass a directory or glob; there is no built-in `dataset.json` loader.
- Tune throughput with `--pdf-split-batch-size`, `--pdf-extract-batch-size`, etc.

### Inspect results

```python
import pyarrow.parquet as pq
import lancedb

df = pq.read_table(OUTPUT_DIRECTORY_BATCH).to_pandas()
print(df[["source_id", "text", "content_type"]].head())

db = lancedb.connect("./lancedb")
tbl = db.open_table("nv-ingest")
print(tbl.to_pandas().head())
```

## Gaps with no retriever-CLI equivalent (kept out of this folder)

The following legacy **ingestion-service** CLI examples are **not** migrated here because the
new CLI does not yet expose an equivalent — continue to use the **ingestion-service** CLI
for these cases:

- `--task 'udf:{…}'` — user-defined functions ([NeMo Retriever Graph](../../src/nemo_retriever/graph/README.md#nemo-retriever-graph)). `retriever` does not expose UDFs.
- `--task 'filter:{content_type:"image", min_size:…, min_aspect_ratio:…, max_aspect_ratio:…}'`.
  The image scale/aspect-ratio filter stage is not reproduced in the new CLI.
- Bare service submission (legacy CLI `--doc foo.pdf` with no extract tasks
  and full content-type metadata returned by the service). `retriever online submit`
  is currently a stub — only `retriever online stream-pdf` is implemented.
- `gen_dataset.py` dataset creation with enumeration and sampling.
- `--collect_profiling_traces --zipkin_host --zipkin_port`. Use
  `--runtime-metrics-dir` / `--runtime-metrics-prefix` instead for a different
  metrics flavor.

## Conventions used in the examples

- Input paths assume you invoke `retriever` from the `nemo_retriever/`
  directory (or point at absolute paths).
- `--save-intermediate <dir>` writes the extraction DataFrame as Parquet for
  inspection. LanceDB output goes to `--lancedb-uri` (defaults to `./lancedb`).
- `--store-images-uri <uri>` stores extracted image assets to a local path or
  an fsspec URI (e.g. `s3://bucket/prefix`). Page granularity stores page
  images; element granularity stores element images.
- `--run-mode inprocess` skips Ray and is ideal for single-file demos and CI;
  `--run-mode batch` (the default) uses Ray Data for throughput.

Run `retriever pipeline run --help` for the authoritative flag list.
