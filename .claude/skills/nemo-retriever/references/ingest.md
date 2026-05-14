# retriever ingest

End-to-end ingestion of PDF documents into a LanceDB table — runs the full
extract → embed → vector-DB pipeline in a single command.

If flags below look stale, re-check `retriever ingest --help`.

## When to use this

- You have one or more PDFs (or a directory/glob of PDFs) and want them
  searchable via `retriever query`.
- You want the default pipeline: PDF split → extraction → page-element
  detection → OCRv2 → embedding → LanceDB insert. No per-stage tuning needed.

**Use a different command when:**

- You only need a single stage (e.g. just extract text, no embeddings) →
  `retriever pdf`, `retriever chart`, `retriever image`, etc.
- You want fine-grained control over the pipeline graph → `retriever pipeline`.
- You need a long-running service rather than one-shot CLI → `retriever service`.
- You're benchmarking throughput → `retriever benchmark`.
- You're iterating on the pipeline locally and want a non-distributed runner →
  `retriever local`.

## Canonical invocations

Ingest a single PDF into the default table (`lancedb/nv-ingest.lance`):

```bash
retriever ingest data/multimodal_test.pdf
```

Ingest a directory of PDFs:

```bash
retriever ingest data/pdfs/
```

Ingest via glob:

```bash
retriever ingest "data/**/*.pdf"
```

Write to a custom DB / table:

```bash
retriever ingest data/multimodal_test.pdf \
  --lancedb-uri ./my-lancedb \
  --table-name my-corpus
```

## Inputs

- **Positional `DOCUMENTS...`** — one or more of: PDF file paths, directories
  containing PDFs, or shell globs. Required, repeatable.

## Outputs

- A LanceDB dataset at `<lancedb-uri>/<table-name>.lance`. Default:
  `./lancedb/nv-ingest.lance`.
- One row per extracted primitive (text chunk, table, chart, image region),
  each with: `text`, `source`, `page_number`, `metadata` (JSON: type, bbox, …),
  and the embedding vector.

## Key flags

| Flag | Default | Notes |
|---|---|---|
| `--lancedb-uri` | `lancedb` | Path or URI of the LanceDB database. |
| `--table-name` | `nv-ingest` | LanceDB table to write into. Must match `retriever query`'s table on read. |
| `--run-mode` | `inprocess` | `inprocess` for local runs; `batch` for the SDK batch ingestor. |

## Pipeline shape

The default `ingest` runs 8 stages, in order:

1. `DocToPdfConversionActor` — non-PDF inputs → PDF (no-op for PDFs).
2. `PDFSplitActor` — split into per-page tasks.
3. `PDFExtractionActor` — extract native text/structure.
4. `PageElementDetectionActor` — detect tables, charts, images, text blocks.
5. `OCRV2Actor` — OCR text where native extraction is missing/poor.
6. `UDFOperator` — user-defined transforms (passthrough by default).
7. `_BatchEmbedActor` — embed primitives with `llama-nemotron-embed-1b-v2`.
8. `IngestVdbOperator` — insert rows into LanceDB.

## Common failure modes

- **`Clamping num_partitions from 16 to 7`** — informational, not an error.
  LanceDB IVF index needs `num_partitions < row_count`; happens on very small
  ingests.
- **First run is slow (~60s+ before any pages process)** — vLLM model load and
  CUDA-graph capture for the embedder. Subsequent runs in the same process
  are fast; one-shot CLI invocations always pay this cost.
- **`No existing dataset at …/nv-ingest.lance, it will be created`** — expected
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
