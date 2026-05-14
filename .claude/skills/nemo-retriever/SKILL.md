---
name: nemo-retriever
description: Use when the user wants to run the NeMo Retriever Library `retriever` CLI (ingest, query, pdf, image, chart, pipeline, vector-store, recall, eval, benchmark, service, etc.) from within Claude Code.
---

# nemo-retriever

Run the NeMo Retriever Library `retriever` CLI with the user's arguments.

Execute: `retriever $ARGUMENTS`

If no arguments are provided, run `retriever --help` and summarize the available subcommands.

## Subcommand references

For per-subcommand details (when to use it, canonical invocations, inputs/outputs, flags, common failure modes), read the matching file in `references/` *before* running anything non-trivial.

End-to-end / search:

- `references/ingest.md` — `retriever ingest`: docs → LanceDB (full pipeline, defaults).
- `references/query.md` — `retriever query`: text query → top-k LanceDB hits.
- `references/pipeline.md` — `retriever pipeline run`: graph-based end-to-end with per-stage knobs.
- `references/service.md` — `retriever service`: long-running ingest service + client.
- `references/local.md` — `retriever local stage{1..7}`: non-distributed per-stage runner.

Per-input-type extractors:

- `references/pdf.md` — `retriever pdf stage page-elements`: PDF → primitives JSON.
- `references/chart.md` — `retriever chart stage run` / `graphic-elements`: chart enrichment.
- `references/audio.md` — `retriever audio extract` / `discover`: chunk + ASR.
- `references/txt.md` — `retriever txt run`: plain-text chunking.
- `references/html.md` — `retriever html run`: HTML → markdown → chunks.
- `references/image.md` — `retriever image render`: detection overlay visualization.

Storage and evaluation:

- `references/vector-store.md` — `retriever vector-store stage run`: embeddings → LanceDB.
- `references/recall.md` — `retriever recall vdb-recall run`: recall@k over a query CSV.
- `references/eval.md` — `retriever eval run` / `export` / `build-page-index`: QA evaluation.
- `references/benchmark.md` — `retriever benchmark <stage> run`: per-stage rows/sec.
- `references/harness.md` — `retriever harness run` / `sweep` / `nightly` / `portal` / …: sessioned orchestration.
- `references/compare.md` — `retriever compare`: JSON / results-bundle diffs.

Cross-cutting:

- `references/pipeline-stages.md` — map of the internal pipeline stages (page-elements, ocr, table-structure, graphic-elements, embed, caption, dedup, store, …) → which CLI command exposes each.

If a subcommand isn't listed above, fall back to `retriever <subcommand> --help`.
