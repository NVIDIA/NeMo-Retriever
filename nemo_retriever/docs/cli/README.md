# Retriever CLI — replacement examples for the legacy ingestion-service CLI

This folder contains `retriever` command-line examples that deliver the same
end-user outcomes as the legacy **ingestion-service** CLI examples that used to
live under `docs/`, `api/`, `client/`, and `deploy/` in older repository layouts.

The historical CLI documentation is **not removed** from the ecosystem — these files sit
alongside it as a new-CLI counterpart you can link to or migrate to.

## Supported vs development / experimental subcommands

For product use and published examples, treat only these top-level subcommands as
**supported**:

- **`retriever ingest`** — ingest documents into LanceDB
- **`retriever query`** — query an existing LanceDB table
- **`retriever pipeline`** — run the graph ingestion pipeline (for example `retriever pipeline run`)

Any other top-level `retriever` subcommand — including but not limited to `pdf`, `html`,
`txt`, `audio`, `chart`, `benchmark`, `harness`, `eval`, `recall`, `service`, `local`,
`compare`, `image`, and `skill-eval` — is **development and experimental**. These commands
may change or be removed without notice and **carry no compatibility, stability, or
behavior guarantees**.

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

Rows that use subcommands other than `ingest`, `query`, or `pipeline` are
[development and experimental](#supported-vs-development--experimental-subcommands).

## Contents

| Topic | Location | Replaces example(s) in |
|-------|----------|------------------------|
| Quick start | [below](#quick-start) | Legacy service quickstart; **Helm** + [NeMo Retriever Library](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/) |
| CLI reference | [below](#cli-reference) | Prior `cli-reference` pages under `docs/docs/extraction/` |
| Client usage walk-through | [below](#client-usage-walk-through) | `client/client_examples/examples/cli_client_usage.ipynb` |
| PDF pre-splitting | [API guide](../../../docs/docs/extraction/nemo-retriever-api-reference.md#pdf-pre-splitting-for-parallel-ingest); [Large PDF page batches](#large-pdf-page-batches) below | Prior extraction docs |
| Benchmarking | [`benchmarking.md`](benchmarking.md) | `docs/docs/extraction/benchmarking.md` and `nemo_retriever/harness/HANDOFF.md` |

<!-- --8<-- [start:quickstart] -->

> Only `retriever ingest`, `retriever query`, and `retriever pipeline` are supported for
> product use; see [Supported vs development / experimental subcommands](#supported-vs-development--experimental-subcommands).

## Quick start

For deployment of NeMo Retriever / **NIM** containers, use
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
  --table-structure-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1 \
  --embed-invoke-url https://integrate.api.nvidia.com/v1/embeddings \
  --embed-model-name nvidia/llama-nemotron-embed-1b-v2

retriever query "What is in this document?" \
  --embed-invoke-url https://integrate.api.nvidia.com/v1/embeddings \
  --embed-model-name nvidia/llama-nemotron-embed-1b-v2 \
  --reranker-invoke-url https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-vl-1b-v2/reranking
```

### Query result controls

`retriever query` returns compact JSON hits with `source`, `page_number`, and `text`.
By default it retrieves and returns `--top-k` rows. Use these controls when you
need a wider candidate pool or a narrower result shape:

```bash
# Retrieve 30 candidates, then return the best 10.
retriever query "where is the warranty limitation discussed?" \
  --candidate-k 30

# Keep only the first hit from each document page.
retriever query "which pages discuss operating costs?" \
  --top-k 5 \
  --candidate-k 30 \
  --page-dedup

# Search a wider pool, then keep only table rows.
retriever query "annual revenue by region" \
  --top-k 5 \
  --candidate-k 40 \
  --content-types table
```

`--top-k` is the final number of hits returned. `--candidate-k` is the wider
candidate pool retrieved before page deduplication, content-type filtering, and
final truncation. It must be greater than or equal to `--top-k`, and should
usually be larger when page deduplication or content-type filtering might
otherwise remove too many of the top retrieved rows. Page deduplication and
content-type filtering are applied after vector retrieval, preserving the
retriever's ranking order and truncating the final output to `--top-k`.
`--content-types` accepts comma-separated content types such as `text`, `table`,
`chart`, `image`, and `infographic`. `images` is accepted as an alias for
captioned image rows emitted by ingest. Hits with missing or unknown content
types are excluded while `--content-types` is active.

`NVIDIA_API_KEY` is required only when those URLs point at hosted
build.nvidia.com endpoints. `NGC_API_KEY` is used separately when pulling or
running self-hosted NIM containers.

### What you get

- Extracted text, tables, and charts as rows in LanceDB at `./lancedb` (default
  table name `nv-ingest`).
- Extraction Parquet at `./processed_docs/extraction.parquet` (`--save-intermediate`).
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

df = pq.read_table("./processed_docs/extraction.parquet").to_pandas()
print(df.head())

db = lancedb.connect("./lancedb")
tbl = db.open_table("nv-ingest")
print(tbl.to_pandas().head())
```

Or query via the Retriever Python client (`nemo_retriever/README.md`):

```python
from nemo_retriever.retriever import Retriever

retriever = Retriever(vdb_kwargs={"uri": "lancedb", "table_name": "nv-ingest"}, top_k=5)
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

`retriever` is the Typer app installed with the `nemo-retriever` package. Subcommand
support policy: [Supported vs development / experimental subcommands](#supported-vs-development--experimental-subcommands).

Document ingestion is usually `retriever pipeline run INPUT_PATH`, which runs the graph pipeline
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
`--save-intermediate`, to `./processed_docs/extraction.parquet`. Inspect rows with
`pyarrow.parquet` or LanceDB queries (not per-content-type `*.metadata.json` files).

### Text chunking and PDF page batches

Splitting is intrinsic to the pipeline. Control text chunks with `--text-chunk`. For
PDF pre-splitting and `--pdf-split-batch-size`, see
[PDF pre-splitting](../../../docs/docs/extraction/nemo-retriever-api-reference.md#pdf-pre-splitting-for-parallel-ingest)
and [Large PDF page batches](#large-pdf-page-batches):

```bash
retriever pipeline run ./data/test.pdf \
  --input-type pdf \
  --no-extract-tables --no-extract-charts \
  --text-chunk --text-chunk-max-tokens 512 --text-chunk-overlap-tokens 64 \
  --save-intermediate ./processed_docs
```

There is no split-only mode without extraction; narrow flags to text extraction if you
only need chunk boundaries.

### Nemotron OCR v2 language mode { #nemotron-ocr-v2-language-mode }

The default OCR engine for **local** extraction (Hugging Face weights, no remote
`--ocr-invoke-url`) is **Nemotron OCR v2**, which runs in **multilingual** mode
by default (`multi`).

| Flag | Values | Notes |
|------|--------|-------|
| `--ocr-lang` | `multi` (default), `english` | v2 only — English-only selector |
| `--ocr-version` | `v2` (default), `v1` | `v1` is the legacy English-only engine |

```bash
retriever pipeline run ./data/scanned.pdf \
  --input-type pdf \
  --method pdfium_hybrid \
  --ocr-lang english

retriever ingest ./data/scanned.pdf --ocr-version v1
```

Set the equivalent `ocr_lang` and `ocr_version` fields on `ExtractParams` (or the
ingest API) in Python.

Remote OCR NIM endpoints choose their own model and language behavior. Local
`--ocr-lang` and `--ocr-version` are not sent on remote requests. For hosted
examples until OCR v2 is published on build.nvidia.com, keep
`--ocr-invoke-url` pointed at `nemotron-ocr-v1` (see [Quick start](#quick-start)).

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
  --caption-model-name nvidia/nemotron-3-nano-omni-30b-a3b-reasoning \
  --caption-invoke-url https://integrate.api.nvidia.com/v1/chat/completions \
  --api-key "${NVIDIA_API_KEY}" \
  --store-images-uri ./processed_docs/images \
  --save-intermediate ./processed_docs
```

Custom caption prompts and `reasoning` flags are not exposed on the CLI — use
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
- **Parquet** — `--save-intermediate <dir>` writes `<dir>/extraction.parquet`.
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

df = pq.read_table(f"{OUTPUT_DIRECTORY_BATCH}/extraction.parquet").to_pandas()
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
- `--save-intermediate <dir>` writes the extraction DataFrame as
  `<dir>/extraction.parquet` for inspection. LanceDB output goes to `--lancedb-uri`
  (defaults to `./lancedb`).
- `--store-images-uri <uri>` stores extracted image assets to a local path or
  an fsspec URI (e.g. `s3://bucket/prefix`). Page granularity stores page
  images; element granularity stores element images.
- `--run-mode inprocess` skips Ray and is ideal for single-file demos and CI;
  `--run-mode batch` (the default) uses Ray Data for throughput.

Run `retriever pipeline run --help` for the authoritative flag list.
