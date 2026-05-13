---
name: nemo-retriever
description: Use when the user wants to run the NeMo Retriever Library `retriever` CLI (ingest, query, pdf, image, chart, pipeline, vector-store, recall, eval, benchmark, service, etc.) from within Claude Code.
---

# nemo-retriever

Run the NeMo Retriever Library `retriever` CLI with the user's arguments.

Execute: `retriever $ARGUMENTS`

If no arguments are provided, run `retriever --help` and summarize the available subcommands.

## Subcommand references

For per-subcommand details (when to use it, canonical invocations, inputs/outputs, flags, common failure modes), read the matching file in `references/` *before* running anything non-trivial:

- `references/ingest.md` — `retriever ingest`: PDFs → LanceDB (full pipeline).
- `references/query.md` — `retriever query`: text query → top-k LanceDB hits.

Additional per-stage references (`pdf`, `chart`, `image`, `audio`, `txt`, `html`, `pipeline`, `vector-store`, `recall`, `eval`, `benchmark`, `service`, `local`, `compare`, `harness`) will be added as those stages stabilize. Until then, fall back to `retriever <subcommand> --help` for any subcommand not listed above.
