# Retriever CLI

This page describes the public `retriever` command-line workflow for document
ingest and retrieval.

For product-facing examples, prefer these commands:

- `retriever ingest` - ingest supported documents and media into a Retriever index.
- `retriever query` - query a local LanceDB table written by local or batch ingest.
- `retriever query service` - query a Retriever service deployment.

`retriever pipeline run` remains available as a development and compatibility
command for legacy pipeline workflows, evaluation, intermediate artifacts, and
pipeline-specific debugging. It is not the preferred public ingest interface.

## Public ingest shape

`retriever ingest` defaults to local, in-process ingest:

```bash
retriever ingest DOCUMENTS...
```

Explicit modes are also available:

```bash
retriever ingest local DOCUMENTS...
retriever ingest batch DOCUMENTS...
retriever ingest service DOCUMENTS...
```

The root ingest CLI uses subcommands instead of a `--run-mode` flag. Choose
the command that matches where ingest runs and where results are stored.

| Command | What It Does | Writes To | Use When |
|---|---|---|---|
| `retriever ingest ...` | Local in-process ingest | local LanceDB | Default local ingest and CI/small corpus runs. |
| `retriever ingest local ...` | Local in-process ingest | local LanceDB | Same as the default, but explicit. |
| `retriever ingest batch ...` | Ray-backed batch ingest | local LanceDB | Larger or batch-tuned runs. |
| `retriever ingest service ...` | Sends documents to a Retriever service | service-configured storage | Remote service ingest. |

This separation keeps invalid flag combinations out of the parser. For example,
service ingest does not expose LanceDB target flags, Ray tuning, local endpoint
configuration, local embed backend selection, or local media controls.

<!-- --8<-- [start:quickstart] -->

> Use `retriever ingest` and `retriever query` for product-facing workflows.
> `retriever pipeline run` is development / compatibility only.

## Quick start

### Ingest a PDF locally

```bash
retriever ingest ./data/multimodal_test.pdf \
  --method pdfium \
  --extract-text --extract-tables --extract-charts \
  --use-table-structure \
  --embed-model-name nvidia/llama-nemotron-embed-1b-v2
```

Then query the default LanceDB table:

```bash
retriever query "What is in this document?" \
  --embed-model-name nvidia/llama-nemotron-embed-1b-v2
```

By default, local ingest writes to `lancedb/nemo-retriever` and `retriever query`
reads from the same table.

The plain `retriever query` examples below apply to local and batch ingest output
written to LanceDB. Use `retriever query service` to query a Retriever service.

### Ingest a larger corpus with batch mode

```bash
retriever ingest batch ./data/pdf_corpus \
  --profile fast-text \
  --pdf-extract-workers 4 \
  --embed-workers 2
```

Batch mode exposes Ray runtime and batch tuning flags such as `--ray-address`,
`--pdf-extract-workers`, `--ocr-workers`, and `--embed-workers`.

### Ingest through a Retriever service

```bash
retriever ingest service ./data/pdf_corpus \
  --service-url http://localhost:7670 \
  --service-concurrency 8
```

Use `--service-api-token` or `NEMO_RETRIEVER_API_TOKEN` when the service requires
a bearer token. Service ingest does not expose `--lancedb-uri`; the service
configures its vector database. Query the service with:

```bash
retriever query service "What is in this corpus?" \
  --service-url http://localhost:7670
```

### Route ingest to hosted or self-hosted NIM endpoints

```bash
export NVIDIA_API_KEY=nvapi-...

retriever ingest ./data/multimodal_test.pdf \
  --page-elements-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3 \
  --ocr-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v2 \
  --table-structure-invoke-url https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1 \
  --embed-invoke-url https://integrate.api.nvidia.com/v1/embeddings \
  --embed-model-name nvidia/llama-nemotron-embed-1b-v2
```

`NVIDIA_API_KEY` is required only when those URLs point at hosted
build.nvidia.com endpoints. `NGC_API_KEY` is used separately when pulling or
running self-hosted NIM containers.

For NVIDIA inference hub rerank models that expose the Cohere-style rerank
route, pass the full `/v1/rerank` URL and the model name shown in the hub
snippet:

```bash
export NGC_INFERENCE_API_KEY=...

retriever query "What is in this document?" \
  --embed-invoke-url https://integrate.api.nvidia.com/v1/embeddings \
  --embed-model-name nvidia/llama-nemotron-embed-1b-v2 \
  --reranker-invoke-url https://inference-api.nvidia.com/v1/rerank \
  --reranker-model-name nvidia/nvidia/llama-3.2-nv-rerankqa-1b-v2 \
  --reranker-api-key-env NGC_INFERENCE_API_KEY
```

### Query result controls

Both `retriever query` and `retriever query service` return compact JSON hits
with `source`, `page_number`, and `text`. Use `--candidate-k`, `--page-dedup`,
and `--content-types` to control how results are selected after vector
retrieval:

```bash
retriever query "annual revenue by region" \
  --top-k 5 \
  --candidate-k 40 \
  --content-types table
```

`--top-k` is the final number of results to return after filtering and
deduplication. `--candidate-k` is the number of raw results to retrieve from
LanceDB or the Retriever service before filtering, page deduplication, and
final truncation. If omitted, the candidate pool is the same size as
`--top-k`. Set `--candidate-k` larger than `--top-k` when page deduplication
or content-type filtering might remove too many of the nearest retrieved rows.
It must always be greater than or equal to `--top-k`.

Page deduplication and content-type filtering are applied after vector
retrieval, preserving retriever ranking order and truncating the final output to
`--top-k`. When querying a local table ingested with an explicit embedding
model, pass the same `--embed-model-name` to `retriever query`.

`--content-types` accepts comma-separated content types such as `text`, `table`,
`chart`, `image`, and `infographic`. `images` is accepted as an alias for
captioned image rows emitted by ingest. This option filters by content-type
metadata only; it does not filter by source, page, or other metadata
predicates. Hits with missing or unknown content-type metadata are excluded
while `--content-types` is active. In service mode, results must include
content-type metadata to match this filter. Default display values in the JSON
output are not used for content-type matching.

### Agentic retrieval

`--agentic` swaps the single dense pass for an LLM-driven ReAct loop: the agent
issues several retrieval sub-queries, fuses the candidates, and selects a final
ranking. It searches the same LanceDB table built by `retriever ingest`, so it is
a drop-in alternative to standard retrieval — add `--agentic` and name the chat
model the agent drives with `--agentic-llm-model` (required):

```bash
retriever query "how does the ingestion pipeline handle tables?" \
  --agentic \
  --agentic-llm-model nvidia/llama-3.3-nemotron-super-49b-v1.5

# remote agent + embedding endpoints, fewer reasoning rounds
retriever query "summarize the deployment options" \
  --agentic \
  --agentic-llm-model nvidia/llama-3.3-nemotron-super-49b-v1.5 \
  --agentic-invoke-url http://localhost:9000/v1/chat/completions \
  --embed-invoke-url http://localhost:8000/v1 \
  --agentic-react-max-steps 5
```

Unlike the dense path (which returns text-enriched hits), agentic mode returns
the agent's ranked document IDs as JSON, each annotated with the source that
produced it (`final_results`, `rrf`, or `selection_agent`). It reuses the same
`--top-k`, `--lancedb-uri`, `--table-name`, `--embed-invoke-url`, and
`--embed-model-name` options as standard retrieval.

**How it works.** Each agentic query runs `Query → ReActAgentOperator → (RRF
fusion) → SelectionAgentOperator → ranked results`:

- `ReActAgentOperator` runs the per-query ReAct loop; every `retrieve` tool call
  delegates to the standard `Retriever`, so the agent searches the same vector
  DB and embedding config as dense retrieval.
- `RRFAggregatorOperator` fuses candidates from the loop's multiple searches with
  reciprocal rank fusion.
- `SelectionAgentOperator` runs a final LLM selection pass over the fused set and
  emits the ranked document IDs.

Agentic-only knobs (apply only with `--agentic`):

- `--agentic-invoke-url` — OpenAI-compatible chat-completions endpoint for the
  agent LLM; defaults to the operators' built-in endpoint when omitted.
- `--agentic-reasoning-effort` (default `high`) — `reasoning_effort` forwarded on
  agentic LLM calls.
- `--agentic-backend-top-k` (default `20`) — candidates pulled from the vector DB
  per retrieval call.
- `--agentic-react-max-steps` (default `50`) — maximum ReAct loop iterations.
- `--agentic-text-truncation` (default `0`) — max characters of each candidate
  shown to the agent; `0` disables truncation.
- `--agentic-temperature` (default `0.0`) — sampling temperature for agentic LLM
  calls (`0.0` = greedy).

<!-- --8<-- [end:quickstart] -->

## Common ingest options

### Local and batch ingest

These options apply to `retriever ingest`, `retriever ingest local`, and
`retriever ingest batch` unless otherwise noted.

| Option | Default | Notes |
|---|---|---|
| `DOCUMENTS...` | required | Files, directories, or shell globs. Supported file families are detected automatically. |
| `--profile` | `auto` | `auto` is normal manifest-routed ingest. `fast-text` is a PDF/document text-only profile for faster fallback ingest. |
| `--lancedb-uri` | `lancedb` | LanceDB database URI. |
| `--table-name` | `nemo-retriever` | LanceDB table name. Must match query-time storage flags. |
| `--overwrite/--append` | overwrite | Overwrite the table by default; use `--append` to add rows. |
| `--method` | planner default | PDF extraction method such as `pdfium` or `nemotron_parse`. |
| `--extract-text`, `--extract-tables`, `--extract-charts` | planner default | Enable or disable extraction families. |
| `--ocr-version` | planner default | OCR engine version for local extraction. |
| `--ocr-lang` | planner default | OCR v2 language selector for local extraction. |
| `--caption` | off | Add a captioning stage. |
| `--dedup` | off | Add image deduplication before captioning and embedding. |
| `--text-chunk` | off | Enable token chunking during extraction. |
| `--store-images-uri` | unset | Store extracted images at a local path or fsspec-compatible URI. |
| `--dry-run` | off | Print the resolved ingest plan without creating an ingestor. |
| `--quiet/--no-quiet` | quiet | Suppress verbose progress output by default. |

Batch-only options include `--ray-address`, `--ray-log-to-driver`,
`--pdf-split-batch-size`, `--pdf-extract-workers`, `--ocr-workers`,
`--table-structure-workers`, `--nemotron-parse-workers`, `--embed-workers`, and
related batch-size / CPU / GPU tuning flags.

### Service ingest

`retriever ingest service` exposes only service-supported request controls.
It does not expose LanceDB target flags, Ray tuning, local endpoint URLs/API
keys, local embed backend selection, `--ocr-lang`, or local audio/video media
controls.

| Option | Default | Notes |
|---|---|---|
| `DOCUMENTS...` | required | Files, directories, or shell globs sent to the service client. |
| `--service-url` | `http://localhost:7670` | Retriever service base URL. |
| `--service-concurrency` | `8` | Maximum concurrent document uploads. |
| `--service-api-token` | env fallback | Bearer token; also reads `NEMO_RETRIEVER_API_TOKEN`. |
| `--profile` | `auto` | Same profile names as local and batch ingest where supported. |
| `--caption`, `--dedup`, `--text-chunk` | off | Service-supported ingest controls. |
| `--store-images-uri` | unset | Service-accessible image storage URI. |
| `--dry-run` | off | Print the resolved service ingest request. Tokens are redacted. |

## Examples

### Custom LanceDB location

```bash
retriever ingest ./data/multimodal_test.pdf \
  --lancedb-uri ./my-lancedb \
  --table-name my-corpus
```

```bash
retriever query "What is in this document?" \
  --lancedb-uri ./my-lancedb \
  --table-name my-corpus
```

### Fast text-only PDF fallback

```bash
retriever ingest ./data/pdf_corpus \
  --profile fast-text \
  --embed-model-name nvidia/llama-nemotron-embed-1b-v2
```

### OCR language mode

```bash
retriever ingest ./data/scanned.pdf \
  --ocr-version v2 \
  --ocr-lang english
```

For mixed-script documents, use `--ocr-lang multi` where supported by the local
OCR engine.

### Text chunking

```bash
retriever ingest ./data/test.pdf \
  --text-chunk \
  --text-chunk-max-tokens 512 \
  --text-chunk-overlap-tokens 64
```

### Captioning and image storage

```bash
retriever ingest ./data/test.pdf \
  --caption \
  --caption-invoke-url https://integrate.api.nvidia.com/v1/chat/completions \
  --api-key "${NVIDIA_API_KEY}" \
  --store-images-uri ./processed_docs/images
```

## Results and diagnostics

Local and batch ingest report the number of input files and LanceDB rows written:

```text
Ingested 20 file(s) -> 1884 row(s) in LanceDB lancedb/nemo-retriever.
```

Service ingest reports the row count returned by the service result when
available:

```text
Ingested 20 file(s) -> 1940 row(s) through retriever service http://localhost:7670.
```

Use `--dry-run` on any ingest mode to inspect the resolved request without
creating an ingestor or contacting the service.

## Development / compatibility command

Use `retriever pipeline run` only when you need pipeline-specific behavior that
is intentionally not part of the first-class ingest/query workflow, such as:

- `--save-intermediate` Parquet artifacts.
- runtime metrics and pipeline reports.
- eval, recall, harness, or BEIR/QA workflows.
- legacy compatibility while callers migrate to `retriever ingest` and
  `retriever query`.

Run `retriever pipeline run --help` for the compatibility command flag list.
