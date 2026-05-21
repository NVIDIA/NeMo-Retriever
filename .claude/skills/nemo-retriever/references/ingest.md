# retriever ingest

End-to-end ingestion of documents and media into a LanceDB table — runs the
full extract → embed → vector-DB pipeline in a single command.

If flags below look stale, re-check `retriever ingest --help`.

## When to use this

- You have one or more supported files (or a directory/glob of files) and want them
  searchable via `retriever query`.
- You want the default pipeline: auto-select extraction for PDF/DOC/PPTX,
  text, HTML, image, audio, or video inputs, then embed and insert into
  LanceDB. No per-stage tuning needed.

**Use a different command when:**

- You only need a single stage (e.g. just extract text, no embeddings) →
  `retriever pdf`, `retriever chart`, `retriever image`, etc.
- You want fine-grained control over the pipeline graph → `retriever pipeline`.
- You need a long-running service rather than one-shot CLI → `retriever service`.
- You're benchmarking throughput → `retriever benchmark`.
- You're iterating on the pipeline locally and want a non-distributed runner →
  `retriever local`.

## Canonical invocations

Ingest a single file into the default table (`lancedb/nv-ingest.lance`):

```bash
retriever ingest data/multimodal_test.pdf
```

Ingest a directory of supported files:

```bash
retriever ingest data/corpus/
```

Ingest via glob:

```bash
retriever ingest "data/**/*"
```

Force a specific input family:

```bash
retriever ingest data/slides/ --input-type doc
retriever ingest data/images/ --input-type image
retriever ingest data/audio/ --input-type audio
retriever ingest data/video/ --input-type video
```

Write to a custom DB / table:

```bash
retriever ingest data/multimodal_test.pdf \
  --lancedb-uri ./my-lancedb \
  --table-name my-corpus
```

## Inputs

- **Positional `DOCUMENTS...`** — one or more file paths, directories, or
  shell globs. Required, repeatable.
- **Supported input types** — `pdf`, `doc` (`.docx`, `.pptx`), `txt`, `html`,
  `image` (`.jpg`, `.jpeg`, `.png`, `.tiff`, `.tif`, `.bmp`, `.svg`),
  `audio` (`.mp3`, `.wav`, `.m4a`), and `video` (`.mp4`, `.mov`, `.mkv`).

## Outputs

- A LanceDB dataset at `<lancedb-uri>/<table-name>.lance`. Default:
  `./lancedb/nemo-retriever.lance`.
- One row per extracted primitive (text chunk, table, chart, image region),
  each with: `text`, `source`, `page_number`, `metadata` (JSON: type, bbox, …),
  and the embedding vector.

## Key flags

| Flag | Default | Notes |
|---|---|---|
| `--lancedb-uri` | `lancedb` | Path or URI of the LanceDB database. |
| `--table-name` | `nv-ingest` | LanceDB table to write into. Must match `retriever query`'s table on read. |
| `--input-type` | `auto` | Input family to ingest. `auto` detects from file extensions and supports mixed directories. |
| `--run-mode` | `inprocess` | `inprocess` for local runs; `batch` for the SDK batch ingestor. |

## Pipeline shape

For PDF/DOC/PPTX inputs, `ingest` runs the optimized document pipeline:

1. `DocToPdfConversionActor` — non-PDF inputs → PDF (no-op for PDFs).
2. `PDFSplitActor` — split into per-page tasks.
3. `PDFExtractionActor` — extract native text/structure.
4. `PageElementDetectionActor` — detect tables, charts, images, text blocks.
5. `OCRV2Actor` — OCR text where native extraction is missing/poor.
6. `UDFOperator` — user-defined transforms (passthrough by default).
7. `_BatchEmbedActor` — embed primitives with `llama-nemotron-embed-1b-v2`.
8. `IngestVdbOperator` — insert rows into LanceDB.

For text, HTML, image, audio, video, or mixed `auto` inputs, `ingest` routes
through the same GraphIngestor extraction paths used by `retriever pipeline`.

## Common failure modes

- **`Clamping num_partitions from 16 to 7`** — informational, not an error.
  LanceDB IVF index needs `num_partitions < row_count`; happens on very small
  ingests.
- **First run is slow (~60s+ before any pages process)** — vLLM model load and
  CUDA-graph capture for the embedder. Subsequent runs in the same process
  are fast; one-shot CLI invocations always pay this cost.
- **`No existing dataset at …/nemo-retriever.lance, it will be created`** — expected
  on the first ingest into a new DB. Subsequent ingests append.
- **HuggingFace download on first run** — the embedder and page-element
  detector pull weights to `~/.cache/huggingface`. Needs network the first
  time; cached afterwards.

## Related

- [[query]] — search the table this command writes.
- `retriever vector-store --help` — utilities for inspecting/moving LanceDB
  tables.
- `retriever pipeline --help` — same end-to-end ingest but exposes per-stage
  knobs.
