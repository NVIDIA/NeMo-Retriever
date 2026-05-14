# retriever pdf

Single-stage PDF extraction: scan a directory of PDFs and write per-PDF
primitives JSON sidecars (text / table / chart / image / page-image rows),
without running embedding or vector-DB stages.

If flags below look stale, re-check `retriever pdf stage page-elements --help`.

## When to use this

- You only need extraction output (primitives JSON) — no embeddings, no
  LanceDB. Useful for debugging, comparing extraction methods, or feeding a
  custom downstream pipeline.
- You want to swap extraction *methods* (pdfium, pdfium_hybrid, ocr,
  nemotron_parse, tika) without rebuilding the whole pipeline.
- You need to point at a remote YOLOX / Nemotron Parse NIM rather than the
  bundled embedded models.

**Use a different command when:**

- You want the full extract → embed → ingest flow → [[ingest]] or
  [[pipeline]].
- You want only chart enrichment over already-extracted primitives →
  [[chart]].
- You want to inspect extraction overlays visually → [[image]].
- You want to benchmark extraction throughput → [[benchmark]] (`split` /
  `extract` / `page-elements`).

## Canonical invocations

Default extraction (pdfium, text only) on a directory:

```bash
retriever pdf stage page-elements --input-dir data/pdfs/
```

Extract everything (text + tables + charts + images) via pdfium + remote
YOLOX:

```bash
retriever pdf stage page-elements \
  --input-dir data/pdfs/ \
  --method pdfium \
  --yolox-http-endpoint http://page-elements:8000/v1/infer \
  --extract-text --extract-tables --extract-charts --extract-images
```

Use NemotronParse instead of pdfium+YOLOX:

```bash
retriever pdf stage page-elements \
  --input-dir data/pdfs/ \
  --method nemotron_parse \
  --nemotron-parse-http-endpoint http://nemotron-parse:8000/v1/infer
```

Write all sidecars to a single output directory:

```bash
retriever pdf stage page-elements \
  --input-dir data/pdfs/ \
  --json-output-dir out/extractions/
```

## Inputs

- **`--input-dir DIR`** — recursively scanned for `*.pdf`. Required (or via
  `--config`).
- **`--config FILE`** — optional ingest YAML. Auto-discovered from
  `./ingest-config.yaml` then `$HOME/.ingest-config.yaml`. CLI flags override
  YAML values.

## Outputs

- One `<pdf>.pdf_extraction.json` sidecar per input PDF, written next to the
  PDF unless `--json-output-dir` is set.
- Each sidecar is a list of primitives. Per primitive: `text`,
  `source_id`/`path`, `page_number`, `metadata` (type, bbox, render info).

These sidecars are the canonical stage-1 input for the rest of the
non-distributed `local stage*` flow (`stage5` embed, `stage6` VDB upload).

## Key flags

| Flag | Default | Notes |
|---|---|---|
| `--method` | `pdfium` | `pdfium`, `pdfium_hybrid`, `ocr`, `nemotron_parse`, `tika`. |
| `--yolox-grpc-endpoint` / `--yolox-http-endpoint` | — | Required for `pdfium` family when extracting page elements. |
| `--nemotron-parse-grpc-endpoint` / `--nemotron-parse-http-endpoint` | — | Required for `method=nemotron_parse`. |
| `--extract-text/--extract-tables/--extract-charts/--extract-images/--extract-infographics/--extract-page-as-image` | text only | Toggle which primitives are written. |
| `--text-depth` | `page` | `page` or `document`. |
| `--render-mode` | `fit_to_model` | `full_dpi` (DPI-then-resize) or `fit_to_model` (≈93 DPI for US Letter). |
| `--limit` | — | Cap number of PDFs processed (debugging). |

## Method cheat-sheet

- **`pdfium`** — fast, native text + YOLOX-driven element detection. Default.
- **`pdfium_hybrid`** — pdfium text + OCR fallback per page where text
  extraction was empty/sparse.
- **`ocr`** — render each page, OCR everything. Use for scanned PDFs.
- **`nemotron_parse`** — NemotronParse end-to-end (text + tables + charts +
  layout) via a single NIM call.
- **`tika`** — Apache Tika fallback (no element detection).

## Common failure modes

- **`YOLOX endpoint is required for method='pdfium'`** — pass
  `--yolox-grpc-endpoint` or `--yolox-http-endpoint`. Without it, only
  `--extract-text` works.
- **Empty primitives for scanned PDFs with `--method pdfium`** — there's no
  embedded text. Switch to `--method ocr` or `pdfium_hybrid`.
- **No sidecars written** — `--write-json-outputs/--no-write-json-outputs`
  toggles output. Default is on; check you didn't disable it via `--config`.
- **`auth-token` errors against NGC NIMs** — set `--auth-token` or
  `NVIDIA_API_KEY` in the environment.

## Related

- [[chart]] — enrich the primitives from this stage with chart parsing.
- [[ingest]] — full pipeline that wraps this stage end-to-end.
- [[pipeline]] — graph-based pipeline exposing per-stage knobs.
- [[benchmark]] — measure throughput of this stage.
