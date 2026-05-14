# retriever chart

Chart-specific enrichment over already-extracted primitives — parses chart
images (titles, axes, series, values) and adds them as structured text to
each chart primitive. Two related subcommands:

- `retriever chart stage run` — enrich an existing primitives DataFrame.
- `retriever chart stage graphic-elements` — run the extract+detect path
  starting from PDFs, with chart extraction enabled.

If flags below look stale, re-check `retriever chart stage --help`.

## When to use this

- You already ran [[pdf]] (or another extractor) and want to add chart
  parsing on top of the primitives without re-extracting.
- You're iterating on chart parsing parameters and don't want to rerun the
  whole pipeline.

**Use a different command when:**

- You want full ingest with charts → [[ingest]] / [[pipeline]] with
  `--extract-charts`.
- You want only PDF extraction (no chart parsing) → [[pdf]].

## Canonical invocations

Enrich a primitives parquet with chart parsing:

```bash
retriever chart stage run \
  --input out/extractions.parquet \
  --output out/extractions.+chart.parquet
```

Extract from PDFs with charts enabled:

```bash
retriever chart stage graphic-elements \
  --input-dir data/pdfs/ \
  --extract-charts \
  --yolox-http-endpoint http://page-elements:8000/v1/infer
```

## Inputs

- **`run`**: `--input` parquet/jsonl/json with a `metadata` column.
- **`graphic-elements`**: `--input-dir` of PDFs (same shape as `retriever pdf
  stage page-elements`).

## Outputs

- **`run`**: enriched DataFrame at `--output` (defaults to
  `<input>.+chart<ext>`). Chart primitives gain parsed structured text in
  their `text` field.
- **`graphic-elements`**: per-PDF `*.pdf_extraction.json` sidecars including
  chart primitives.

## Key flags (`chart stage run`)

| Flag | Default | Notes |
|---|---|---|
| `--input` | — | Required. `.parquet`, `.jsonl`, or `.json` with `metadata`. |
| `--output` | `<input>.+chart<ext>` | Output path. |
| `--config` | auto-discover | YAML config (section: `chart`). |

## Key flags (`chart stage graphic-elements`)

Same as `retriever pdf stage page-elements` plus `--extract-charts` toggled
on by default. See [[pdf]] for the full flag table.

## Common failure modes

- **`KeyError: 'metadata'`** — input DataFrame is missing the `metadata`
  column. Make sure you fed it primitives JSON/parquet from
  `retriever pdf stage` or [[pipeline]].
- **No chart rows in output** — the input has no rows with
  `metadata.content_metadata.type == "structured"` and chart subtype. Run
  extraction with `--extract-charts` first.

## Related

- [[pdf]] — generate the primitives that `chart stage run` consumes.
- [[pipeline]] — wraps chart extraction into the graph pipeline.
- [[ingest]] — end-to-end including charts when enabled.
