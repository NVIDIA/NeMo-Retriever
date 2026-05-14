# pipeline stages

Cross-reference for the **internal pipeline stages** (page-elements, ocr,
table-structure, graphic-elements, embed, caption, dedup, store). These are
not top-level CLI commands of their own; they're surfaced as:

1. Flag groups under [[pipeline]] (`pipeline run`).
2. Stand-alone benchmark subcommands under [[benchmark]].
3. In some cases, dedicated subcommands under other groups (e.g.
   page-elements lives under `retriever pdf stage page-elements`).

Use this page to figure out *which* command to reach for when you want to
exercise or tune a specific stage.

## Stage map

| Stage | What it does | Tuned via | Benchmarked via | Standalone CLI |
|---|---|---|---|---|
| **pdf-split** | Split PDFs into per-page tasks | `--pdf-split-batch-size` | `retriever benchmark split run` | — |
| **pdf-extract** | Native PDF text/structure extraction | `--method`, `--pdf-extract-*` | `retriever benchmark extract run` | [[pdf]] |
| **page-elements** | YOLOX text/table/chart/image detection | `--page-elements-invoke-url`, `--page-elements-actors`, `--page-elements-batch-size`, `--page-elements-{cpus,gpus}-per-actor` | `retriever benchmark page-elements run` | `retriever pdf stage page-elements` (see [[pdf]]) |
| **ocr** | OCR for sparse text regions | `--ocr-invoke-url`, `--ocr-version` (`v1`/`v2`), `--ocr-{actors,batch-size,cpus-per-actor,gpus-per-actor}` | `retriever benchmark ocr run` | — |
| **table-structure** | Structured OCR over detected tables | `--use-table-structure`, `--table-structure-invoke-url`, `--table-output-format` | — | (`nemo_retriever.table.commands` exposes `run-structure-ocr` under the table sub-app where wired) |
| **graphic-elements** | Chart parsing | `--use-graphic-elements`, `--graphic-elements-invoke-url`, `--extract-charts` | — | [[chart]] |
| **infographic** | Infographic parsing | `--extract-infographics` | — | (`retriever local stage2`) |
| **dedup** | IoU-based primitive dedup | `--dedup/--no-dedup`, `--dedup-iou-threshold` | — | — |
| **caption** | VLM caption for image primitives | `--caption/--no-caption`, `--caption-invoke-url`, `--caption-model-name`, `--caption-temperature`, `--caption-top-p`, `--caption-max-tokens` | — | — |
| **udf** | User-defined transforms (passthrough by default) | (code) | — | — |
| **embed** | Embed primitives | `--embed-invoke-url`, `--embed-model-name`, `--embed-modality`, `--embed-granularity`, `--embed-{actors,batch-size,cpus-per-actor,gpus-per-actor}`, `--local-ingest-embed-backend` | — | `retriever local stage5` |
| **audio-extract** | Chunk media + ASR | `--segment-audio`, `--audio-split-type`, `--audio-split-interval`, `--audio-match-tolerance`, audio NIM env | `retriever benchmark audio-extract run` | [[audio]] |
| **store (VDB)** | Write embeddings to LanceDB | `--store-actors`, `--lancedb-uri`, `--table-name` (set on [[ingest]] / [[vector-store]]) | — | [[vector-store]] |
| **query** | Embed query + search | (read side) | — | [[query]] / [[recall]] |

## Choosing the right entry point

- **"I want to ingest a corpus end-to-end"** → [[ingest]] (defaults) or
  [[pipeline]] (per-stage control).
- **"I only want this one stage's output"** → the *Standalone CLI* column.
- **"I want to know how fast this stage is on this machine"** → the
  *Benchmarked via* column.
- **"I want to route this stage through a NIM"** → set the matching
  `--*-invoke-url` on [[pipeline]] (and `--api-key`).
- **"I want to size Ray actors for this stage"** → tune the
  `--<stage>-actors` / `--<stage>-batch-size` /
  `--<stage>-{cpus,gpus}-per-actor` quartet on [[pipeline]].

## Stage ordering

Default order for a PDF input under [[pipeline]] / [[ingest]]:

```
pdf-split → pdf-extract → page-elements → ocr
         → (table-structure)  (graphic-elements)  (infographic)
         → dedup → (caption) → udf → embed → store
```

Audio swaps the head: `audio-extract` (chunk + ASR) replaces
pdf-split/pdf-extract/page-elements/ocr; the tail (embed, store) is the
same. Txt/html similarly replace the head with [[txt]] / [[html]].

## Related

- [[pipeline]] — the command that wires every stage above together.
- [[benchmark]] — per-stage rows/sec.
- [[local]] — non-distributed, file-per-stage version of the same flow.
