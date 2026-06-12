# retriever ingest

End-to-end ingestion of supported documents and media into a Retriever index.
The command runs extraction, optional caption/chunk/dedup behavior, embedding,
and vector-store insert in one workflow.

If flags below look stale, re-check:

```bash
<RETRIEVER_VENV>/bin/retriever ingest --help
<RETRIEVER_VENV>/bin/retriever ingest local --help
<RETRIEVER_VENV>/bin/retriever ingest batch --help
<RETRIEVER_VENV>/bin/retriever ingest service --help
```

## Modes

Use root ingest as the public CLI. Do not use `--run-mode` on this command.
Mode is command structure:

| Command | Use When | Storage |
|---|---|---|
| `retriever ingest DOCUMENTS...` | Default local/in-process ingest. Best for setup turns, CI, and small/medium corpora. | local LanceDB |
| `retriever ingest local DOCUMENTS...` | Same as the default, but explicit. | local LanceDB |
| `retriever ingest batch DOCUMENTS...` | Ray/batch ingest and batch tuning. | local LanceDB |
| `retriever ingest service DOCUMENTS...` | Send ingest work to a running retriever service. | service-owned vector DB |

Python `create_ingestor(run_mode=...)` still exists for programmatic use. The
root CLI intentionally uses subcommands so each mode shows only the options it
can honor.

## Canonical Invocations

Ingest a single file into the default table (`lancedb/nemo-retriever.lance`):

```bash
<RETRIEVER_VENV>/bin/retriever ingest data/multimodal_test.pdf
```

Ingest a directory of supported files:

```bash
<RETRIEVER_VENV>/bin/retriever ingest data/corpus/
```

Large text-only PDF fallback:

```bash
<RETRIEVER_VENV>/bin/retriever ingest data/pdfs/ --profile fast-text
```

Batch ingest with tuning:

```bash
<RETRIEVER_VENV>/bin/retriever ingest batch data/pdfs/ \
  --profile fast-text \
  --pdf-extract-workers 4 \
  --embed-workers 2
```

Optional local VLM captioning:

```bash
<RETRIEVER_VENV>/bin/retriever ingest data/pdfs/ \
  --caption \
  --caption-infographics
```

Add `--caption-invoke-url` only when a remote OpenAI-compatible VLM endpoint is
already deployed.

Ingest via glob:

```bash
<RETRIEVER_VENV>/bin/retriever ingest "data/**/*"
```

Write to a custom DB / table:

```bash
<RETRIEVER_VENV>/bin/retriever ingest data/multimodal_test.pdf \
  --lancedb-uri ./my-lancedb \
  --table-name my-corpus
```

Service ingest:

```bash
<RETRIEVER_VENV>/bin/retriever ingest service data/corpus/ \
  --service-url http://localhost:7670 \
  --service-concurrency 8
```

## Inputs

- Positional `DOCUMENTS...` is required and repeatable.
- Values may be file paths, directories, or shell globs.
- Supported input families are detected automatically from extensions:
  `pdf`, `docx`, `pptx`, `txt`, `html`, `jpg`, `jpeg`, `png`, `tiff`, `tif`,
  `bmp`, `svg`, `mp3`, `wav`, `m4a`, `mp4`, `mov`, and `mkv`.

## Outputs

Local and batch ingest write a LanceDB dataset at
`<lancedb-uri>/<table-name>.lance`. Default:
`./lancedb/nemo-retriever.lance`.

Each row includes extracted text or captions, source metadata, page information
when available, and an embedding vector.

Service ingest writes to the vector database configured by the remote service.
The client does not expose `--lancedb-uri` or `--table-name` in service mode.

## Key Flags

Graph ingest (`retriever ingest`, `local`, `batch`):

| Flag | Default | Notes |
|---|---|---|
| `--lancedb-uri` | `lancedb` | Path or URI of the LanceDB database. |
| `--table-name` | `nemo-retriever` | LanceDB table to write into. Must match `retriever query` on read. |
| `--profile` | `auto` | `fast-text` disables expensive PDF recall stages for a text-only fallback. |
| `--overwrite/--append` | overwrite | Use `--append` only when duplicates are acceptable. |
| `--caption` | `false` | Optional VLM captioning stage after extraction. |
| `--caption-invoke-url` | unset | Remote VLM endpoint. If omitted with `--caption`, local/default caption behavior is used. |
| `--caption-context-text-max-chars` | default | Include nearby extracted text in caption prompts. |
| `--caption-infographics` | default | Caption infographic crops in addition to extracted images. |
| `--text-chunk` | `false` | Enable token chunking during extraction. |
| `--dry-run` | `false` | Print the resolved request/plan JSON without creating an ingestor. |

Batch-only flags:

| Flag Family | Examples |
|---|---|
| Ray runtime | `--ray-address`, `--ray-log-to-driver` |
| PDF/extract tuning | `--pdf-split-batch-size`, `--pdf-extract-workers`, `--ocr-workers` |
| actor resources | `--page-elements-gpus-per-actor`, `--ocr-cpus-per-actor` |
| embedding tuning | `--embed-workers`, `--embed-batch-size` |

Service-only flags:

| Flag | Default | Notes |
|---|---|---|
| `--service-url` | `http://localhost:7670` | Retriever service base URL. |
| `--service-concurrency` | `8` | Maximum concurrent document uploads. |
| `--service-api-token` | env fallback | Also reads `NEMO_RETRIEVER_API_TOKEN`. |

## Pipeline Shape

The root ingest entrypoint expands inputs, builds a manifest, resolves the
selected profile into typed ingest options, and calls the canonical ingest
execution path. The manifest planner routes PDF/document, image, text, HTML,
audio, and video branches without relying on `retriever pipeline run`.

Use `retriever pipeline run` only for legacy or development behavior such as
intermediate Parquet artifacts, pipeline reports, eval, recall, or harness work.

## Common Failure Modes

- **`Clamping num_partitions from 16 to 7`** - informational, not an error.
  LanceDB IVF index needs `num_partitions < row_count`; this happens on very
  small ingests.
- **First run is slow (~60s+ before pages process)** - vLLM model load and
  CUDA-graph capture for the embedder. One-shot CLI invocations pay this cost.
- **`No existing dataset at .../nemo-retriever.lance, it will be created`** -
  expected on the first ingest into a new DB.
- **HuggingFace download on first run** - the embedder and page-element detector
  may pull weights to `~/.cache/huggingface`. They need network the first time
  and use cache afterwards.

## Related

- [[query]] - search the table this command writes.
- `retriever vector-store --help` - utilities for inspecting or moving LanceDB
  tables.
