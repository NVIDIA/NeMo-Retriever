# retriever ingest

End-to-end ingestion of supported documents and media into a Retriever index.
The command runs extraction, optional caption/chunk/dedup behavior, embedding,
and LanceDB insert in one workflow.

If flags below look stale, re-check the main help:

```bash
<RETRIEVER_VENV>/bin/retriever ingest --help
```

The main help lists available modes and points to mode-specific help when
those options are needed.

## Default usage

Use root ingest as the skill default:

```bash
<RETRIEVER_VENV>/bin/retriever ingest DOCUMENTS...
```

Do not use `--run-mode` with root ingest. Omitting a mode runs local/in-process
ingest and writes to LanceDB. Use an explicit mode only when the user asks for
one or when the CLI help makes that mode necessary.

## Canonical invocations

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

## Inputs

- Positional `DOCUMENTS...` is required and repeatable.
- Values may be file paths, directories, or shell globs.
- Supported input families are detected automatically from extensions:
  `pdf`, `docx`, `pptx`, `txt`, `html`, `jpg`, `jpeg`, `png`, `tiff`, `tif`,
  `bmp`, `svg`, `mp3`, `wav`, `m4a`, `mp4`, `mov`, and `mkv`.

## Outputs

Default ingest writes a LanceDB dataset at
`<lancedb-uri>/<table-name>.lance`. Default:
`./lancedb/nemo-retriever.lance`.

Each row includes extracted text or captions, source metadata, page information
when available, and an embedding vector.

## Key flags

| Flag | Default | Notes |
|---|---|---|
| `--lancedb-uri` | `lancedb` | Path or URI of the LanceDB database. |
| `--table-name` | `nemo-retriever` | LanceDB table to write into. |
| `--profile` | `auto` | `fast-text` disables expensive PDF extraction stages for a text-only fallback. |
| `--overwrite/--append` | overwrite | Use `--append` only when duplicates are acceptable. |
| `--caption` | `false` | Optional VLM captioning stage after extraction. |
| `--caption-invoke-url` | unset | Remote VLM endpoint. If omitted with `--caption`, local/default caption behavior is used. |
| `--caption-context-text-max-chars` | default | Include nearby extracted text in caption prompts. |
| `--caption-infographics` | default | Caption infographic crops in addition to extracted images. |
| `--text-chunk` | `false` | Enable token chunking during extraction. |
| `--dry-run` | `false` | Print the resolved request/plan JSON without creating an ingestor. |

## Pipeline shape

The root ingest entrypoint expands inputs, builds a manifest, resolves the
selected profile into typed ingest options, and calls the canonical ingest
execution path. The manifest planner routes PDF/document, image, text, HTML,
audio, and video branches without relying on `retriever pipeline run`.

Use `retriever pipeline run` only for legacy or development behavior such as
intermediate Parquet artifacts, pipeline reports, eval, recall, or harness work.

## Common failure modes

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
