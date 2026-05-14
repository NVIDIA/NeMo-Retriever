# retriever pipeline

Graph-based end-to-end ingestion pipeline. Same outcome as [[ingest]]
(documents → LanceDB) but exposes per-stage knobs for extraction methods,
NIM endpoints, Ray actor counts, embedding model, dedup/caption, audio/video
options, and storage.

Use `retriever pipeline run --help` to see *all* flag groups — there are
many. This page covers the groups and the most-used flags within each.

## When to use this

- You need fine-grained control over a pipeline stage (e.g. swap the OCR
  model, set per-actor GPU fractions, route through a remote NIM, use a
  different embedder).
- You're tuning throughput on a Ray cluster and need actor / batch-size
  knobs.
- You want to ingest non-PDF inputs (audio, video, txt, html, image)
  through the same graph.

**Use a different command when:**

- Defaults are fine → [[ingest]] (one flag, same outcome).
- You only need a single stage's output → [[pdf]], [[chart]], [[audio]],
  [[txt]], [[html]].
- You want long-running service mode → [[service]].
- You want a non-Ray local debug runner → [[local]].
- You want throughput numbers per stage → [[benchmark]].

## Canonical invocations

Default batch ingest of a PDF directory:

```bash
retriever pipeline run data/pdfs/
```

In-process (no Ray) for quick local runs:

```bash
retriever pipeline run data/pdfs/ --run-mode inprocess
```

Ingest audio:

```bash
retriever pipeline run data/audio/ --input-type audio
```

Route through remote NIMs (no local GPU). Note: `--use-table-structure` and
`--use-graphic-elements` default to **off** — passing the matching
`--*-invoke-url` alone is not enough; the `--use-*` flag must also be set
to enable that stage.

```bash
retriever pipeline run data/pdfs/ \
  --page-elements-invoke-url http://page-elements:8000/v1/infer \
  --ocr-invoke-url http://ocr:8000/v1/infer \
  --use-table-structure \
  --table-structure-invoke-url http://table-structure:8000/v1/infer \
  --use-graphic-elements \
  --graphic-elements-invoke-url http://graphic-elements:8000/v1/infer \
  --embed-invoke-url http://embed:8000/v1/embed \
  --api-key "$NVIDIA_API_KEY"
```

Tune Ray actor counts for a busy stage:

```bash
retriever pipeline run data/pdfs/ \
  --page-elements-actors 4 --page-elements-gpus-per-actor 0.5 \
  --ocr-actors 2 --ocr-gpus-per-actor 1.0 \
  --embed-actors 1 --embed-batch-size 64
```

## Inputs

- **Positional `INPUT_PATH`** — file or directory of documents. Required.
- **`--input-type`** — `pdf` (default) / `doc` / `txt` / `html` / `image` /
  `audio`.

## Outputs

- LanceDB table populated by the `IngestVdbOperator` sink (defaults
  `lancedb/nv-ingest.lance`). See [[query]] for reading.
- If `--store-images-uri` is set, extracted images are also persisted there.

## Flag groups (from `--help`)

| Group | What it controls |
|---|---|
| **I/O and Execution** | `--run-mode` (`batch` / `inprocess` / `service`), `--input-type`, `--debug`, `--log-file`. |
| **PDF / Document Extraction** | `--method`, `--dpi`, `--extract-text/--extract-tables/--extract-charts/--extract-infographics/--extract-page-as-image`, `--use-graphic-elements`, `--use-table-structure`, `--table-output-format`. |
| **Remote NIM Endpoints** | `--api-key`, plus `--*-invoke-url` for `page-elements`, `ocr`, `graphic-elements`, `table-structure`, `embed`. `--ocr-version v1/v2`. |
| **Embedding** | `--embed-model-name`, `--embed-modality`, `--embed-granularity`, `--local-ingest-embed-backend` (`vllm`/`hf`), `--text-elements-modality`, `--structured-elements-modality`. |
| **Dedup and Caption** | `--dedup/--no-dedup`, `--dedup-iou-threshold`, `--caption/--no-caption`, `--caption-invoke-url`, `--caption-model-name`, GPU fractions, `--caption-temperature`/`--caption-top-p`/`--caption-max-tokens`. |
| **Storage and Text Chunking** | `--store-images-uri`, `--text-chunk`, `--text-chunk-max-tokens`, `--text-chunk-overlap-tokens`. |
| **Ray / Batch Tuning** | `--ray-address`, per-stage `*-actors`/`*-batch-size`/`*-cpus-per-actor`/`*-gpus-per-actor` for `page-elements`, `ocr`, `embed`, `nemotron-parse`, plus `--store-actors`, `--pdf-split-batch-size`, `--pdf-extract-*`. |
| **Audio** | `--segment-audio`, `--audio-split-type`/`--audio-split-interval`, `--audio-match-tolerance`. |
| **Video** | `--video-extract-audio`, video-specific split/sampling flags. |

## Pipeline stages (what runs end-to-end)

For a PDF input with all defaults, the graph runs roughly:

1. **PDFSplitActor** — split into per-page tasks.
2. **PDFExtractionActor** — native text/structure extraction.
3. **PageElementDetectionActor** — YOLOX detects text/table/chart/image
   regions. Tunable via `--page-elements-*` flags.
4. **OCRV2Actor** / OCRActor — OCR text where extraction is sparse. Tunable
   via `--ocr-*` flags; `--ocr-version v1` for the legacy engine.
5. **(optional) TableStructureActor** — structured-OCR on detected tables
   when `--use-table-structure` is set; route via
   `--table-structure-invoke-url`.
6. **(optional) GraphicElementsActor** — chart enrichment when
   `--use-graphic-elements`; route via `--graphic-elements-invoke-url`.
7. **(optional) CaptionActor** — VLM captioning when `--caption`.
8. **UDFOperator** — user-defined transforms (passthrough by default).
9. **EmbedActor** — embed primitives. Tunable via `--embed-*` flags.
10. **IngestVdbOperator (StoreOperator)** — write to LanceDB.

Each stage has its own `--*-invoke-url` for routing to a NIM, and (in batch
mode) `--*-actors` / `--*-batch-size` / `--*-cpus-per-actor` /
`--*-gpus-per-actor` for resource sizing.

## Common failure modes

- **Stage saturates and stalls** — bump `--<stage>-actors` and/or
  `--<stage>-batch-size`. Use [[benchmark]] to find the bottleneck stage
  first.
- **"No GPU available" with `--run-mode batch`** — set
  `--<stage>-gpus-per-actor 0` for stages you want on CPU, or pass
  `--*-invoke-url` to offload to a NIM.
- **Embedding mismatch on read** — `--embed-model-name` differs from what
  [[query]] uses. Keep ingest and query embedders aligned.
- **Output table empty** — input matched no files for `--input-type`. Check
  globs and file extensions.
- **Tables / charts not appearing in output despite `--*-invoke-url` set**
  — `--use-table-structure` / `--use-graphic-elements` default to off.
  Setting the invoke URL alone does *not* enable the stage; pass the
  `--use-*` flag too.

## Related

- [[ingest]] — defaults-only wrapper around this command.
- [[local]] — non-distributed runner for debugging stages.
- [[service]] — long-running pipeline behind an HTTP API.
- [[benchmark]] — per-stage throughput numbers.
