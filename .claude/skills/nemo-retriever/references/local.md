# retriever local

Non-distributed, pandas-based runner that exposes the pipeline as discrete
numbered stages (`stage1` … `stage7`, plus `stage999` for post-mortem).
Stages are intentionally separable so you can rerun one without touching
the others.

> The top-level group is registered as a placeholder; subcommands are
> contributed by per-stage modules. Run `retriever local --help` (or the
> per-stage `--help`) to see what's currently wired up in your install.

## When to use this

- You're iterating on a single stage (e.g. tweak chunking, rerun stage5,
  re-upload stage6) without redoing extraction.
- You want to debug a specific stage with `pdb` / breakpoints — no Ray, no
  actors, deterministic ordering.
- You need the intermediate sidecar files (per-stage JSON/parquet) for
  inspection.

**Use a different command when:**

- You want full ingest in one command → [[ingest]] or [[pipeline]].
- You need parallelism on a cluster → [[pipeline]] in batch mode.
- You want a long-running endpoint → [[service]].

## Pipeline stages (mapped to files)

Stages live in `nemo_retriever/src/nemo_retriever/local/stages/`:

| Stage | File | What it does |
|---|---|---|
| `stage1` | `stage1_pdf_extraction.py` | PDF extraction (same idea as [[pdf]]). |
| `stage2` | `stage2_infographic_extraction.py` | Infographic enrichment. |
| `stage3` | `stage3_table_extractor.py` | Table structure / OCR. |
| `stage4` | `stage4_chart_extractor.py` | Chart enrichment (same idea as [[chart]]). |
| `stage5` | `stage5_text_embeddings.py` | Text embedding → `*.text_embeddings.json`. |
| `stage6` | `stage6_vdb_upload.py` | LanceDB upload (same idea as [[vector-store]]). |
| `stage7` | `stage7_vdb_query.py` | Single-query lookup against LanceDB. |
| `stage999` | `stage999_post_mortem_analysis.py` | Post-run analysis. |

Each stage's `run` reads sidecars matching a pattern (e.g.
`*.pdf_extraction.json` for stage5) and writes the next sidecar type.

## Canonical flow

```bash
# 1. extract
retriever local stage1 run --input-dir data/pdfs/

# 2. enrich (optional)
retriever local stage3 run --input-dir data/pdfs/   # tables
retriever local stage4 run --input-dir data/pdfs/   # charts

# 3. embed
retriever local stage5 run --input-dir data/pdfs/ --pattern "*.pdf_extraction.json"

# 4. upload to LanceDB
retriever local stage6 run --input-dir data/pdfs/

# 5. query
retriever local stage7 run --query "what is in chart 1?"
```

For txt/html, swap stage1 for [[txt]] / [[html]] and adjust stage5's
`--pattern`.

## Inputs / outputs

Each stage takes `--input-dir` (and stage-specific flags) and writes
sidecars next to source files. The pattern is consistent: stage N reads
stage N-1's output and writes its own type.

## Common failure modes

- **`stage5: no files matched pattern`** — `--pattern` defaults to
  `*.pdf_extraction.json`; pass `*.txt_extraction.json` /
  `*.html_extraction.json` for those inputs.
- **`stage6` overwrites a table I wanted to append to** — pass the
  stage-appropriate flag, or use [[vector-store]] which has explicit
  `--append`.
- **First `stage5` run is slow** — model load. Same trade-off as the
  one-shot CLIs; reuse the process for multiple inputs in research scripts.

## Related

- [[pdf]] / [[chart]] / [[vector-store]] — standalone equivalents of
  individual stages.
- [[pipeline]] — distributed graph version of the same flow.
