---
name: nemo-retriever
description: Use when the user wants to search, index, or answer questions over a folder of PDFs (or other documents) — including building a RAG / search index over PDFs, looking up information across many PDFs, or running the `retriever` CLI (ingest, query, pipeline, recall, eval, etc.).
allowed-tools: Bash Write Read
---

# nemo-retriever

The `retriever` CLI indexes a folder of PDFs into LanceDB (`retriever ingest`) and serves vector search over it (`retriever query`). For any task about searching/answering questions across a folder of PDFs, use this CLI — do not write a custom RAG.

## Install (if `retriever` is missing)

If `command -v retriever` returns nothing, follow `references/install.md` to install the NeMo Retriever Library before proceeding. It prints `RETRIEVER_VENV=<path>`; substitute that path for `<RETRIEVER_VENV>` in every example in this skill (setup, query, pitfalls, and the CLI references).

## Workflow — read the reference for the current phase, then execute

| Turn type | Read this once | Then execute |
| :--- | :--- | :--- |
| **Setup turn** (first turn — `./lancedb/nv-ingest.lance` doesn't exist) | `references/setup.md` | Build the index |
| **Query turn** (every subsequent turn — user asks a question) | `references/query.md` | One `retriever query` call, then `Write` `./output.json` |
| Anything errored or returned empty | `references/pitfalls.md` | Apply the named recovery; do not improvise |

For the full `retriever ingest` / `retriever query` CLI specs, see `references/cli/ingest.md` and `references/cli/query.md`. You do not need these for routine turns — `<RETRIEVER_VENV>/bin/retriever <subcommand> --help` is faster.

## Hard limits (apply to every turn)

- **Setup turn**: build the index in one shell command (see `references/setup.md`). STOP after the index lands.
- **Query turn**: at most **2 Bash calls** — 1 `retriever query`, +1 optional targeted text-extract per `references/query.md`. Then `Write` `./output.json` and STOP.
- **No narration between tool calls.** Tokens you emit between calls become input + cached input for every later turn — quadratic cost. Go straight from reading the summary to writing the JSON file.
- **Banned**: `TodoWrite`, Glob, Grep, `Read` of whole PDFs, re-running setup, spawning subagents, speculative "confirmation" calls.

Long query turns (5+ tool calls, 1M+ cache-read tokens) cost ~5× a disciplined turn and almost always still produce the wrong answer. **Answering partially beats timing out.**
