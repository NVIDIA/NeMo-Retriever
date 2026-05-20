---
name: nemo-retriever-local-document-search
description: Use when the user has a local file or folder of documents and wants to search, ask questions, summarize, or inspect that bounded corpus with NeMo Retriever local document search.
---

# NeMo Retriever Local Document Search

Use this skill when the user has a local file or folder of documents and wants to search, ask questions, summarize, or inspect that bounded corpus. This is for small, single-user, exploratory document search over local files.

Do not use this skill to perform deployments, operate shared services, serve models, or mutate production indexes. Do use this skill to answer documentation questions about those topics when the requested corpus is local docs.

## Decision Rules

Use local search automatically when the corpus is a local path, the task is one-off or exploratory, no multi-user serving requirement is mentioned, no audit or compliance requirement is mentioned, and the corpus appears under the default safety caps.

When the user asks about "these docs", "this folder", or provides a local path, use that path directly. When the user asks about the current project without providing a path, use a clearly implied documentation path only if it is already known from context or can be checked with a narrow existence check such as `./docs` or `./README.md`; otherwise ask for the corpus path. Do not recursively search the repo to discover a corpus.

Ask before indexing when the path may contain sensitive data, the corpus may be large, the user asks for persistent memory, or the workspace is shared.

Route away from local search when the user wants you to implement, operate, configure, or mutate a shared service where multiple users or services need access, access control matters, refresh has to be continuous, or production latency or throughput is required. Do not route away for conceptual documentation questions such as "what are the deployment options?" or "which setup should I choose?"; answer those from local docs with `local_document_ask` when a corpus is available.

## Commands

When MCP tools are available, prefer `local_document_ask` for a first query. Provide `input_path`, `query`, and optional `top_k`, `include`, `exclude`, `max_docs`, and `max_pages`. Omit `index` unless the user explicitly wants a particular index location; the MCP tool chooses a stable path-scoped index by hashing the resolved absolute `input_path`. The tool returns the same payload shape as the CLI, including `index_action`, `warnings`, and `evidence`.

Use `local_document_search` when the user explicitly wants to search an existing index. Use `local_document_status` to inspect index health, staleness, and chunk counts.

Treat Retriever as the recall mechanism. Do not run separate corpus search or inventory commands such as `rg`, `grep`, `find`, or broad `ls` over the document directory before retrieval, and do not inspect repository files to decide which docs to query. Keep the retrieval query faithful to the user's wording; do not add extra product names, deployment modes, or suspected answers unless the user mentioned them. After Retriever returns evidence, you may read the specific `evidence[].source_file` paths it cited to verify context or produce line-specific citations. Do not search or read source code files for docs/how-to questions unless the user explicitly asks for code/API implementation details, or the retrieved evidence itself cites a source file outside the docs corpus. Only search the corpus yourself when the Retriever result is empty, clearly wrong, or the user explicitly asks for lexical matching.

For documentation/how-to questions, answer from the retrieved local docs only. If the user asks what a repository actually does under the hood, asks to find every usage, asks for fallback order, or asks for implementation details, use Retriever first for docs context and then inspect/search the relevant source files. Make the distinction explicit in the answer when code behavior differs from documented guidance.

Do not use web search or external documentation while this skill is active unless the user explicitly asks for current/latest external docs, or the local corpus is missing/unavailable. If local evidence does not mention an option or term, say that it was not surfaced in the local docs instead of searching the web to fill the gap.

If the MCP tools are unavailable, fall back to the CLI:

```bash
retriever local ask ./path/to/docs "What are the warranty limitations?" --output json
```

The MCP tools and CLI default to local GPU inference. Use `inference="remote"` or `--inference remote` only when the user explicitly wants hosted embeddings or local inference is unavailable.

For debugging or validation, prefer the explicit workflow:

```bash
retriever local init ./path/to/docs --index ./.nemo-retriever/local-index --output json
retriever local search "What are the warranty limitations?" --index ./.nemo-retriever/local-index --output json
retriever local status --index ./.nemo-retriever/local-index --output json
```

For recovery, run:

```bash
retriever local doctor --index ./.nemo-retriever/local-index --input-path ./path/to/docs --output json
```

Use remote hosted embeddings when the user has no local GPU:

```bash
retriever local ask ./path/to/docs "What changed?" --inference remote --output json
```

This requires `NVIDIA_API_KEY` or `NGC_API_KEY`; pass `--embed-invoke-url` only when using a non-default embedding endpoint.

## Validation

After indexing, check `documents_processed`, `chunk_count`, `warnings`, and `next_recommended_command` in JSON output. For `ask`, check `resolved_input_path`, `corpus_root`, `index_action`, `indexed_now`, `reused_index`, `reindexed`, and `reindex_reasons` so the user can tell which corpus was used and whether the index was reused or rebuilt. The MCP tool and CLI return retrieved evidence rather than generated prose; synthesize the answer yourself from `evidence[].chunk_text` and cite `evidence[].source_file`, `evidence[].page`, and useful section metadata when present.

Only cite files that appear in returned `evidence[].source_file` or files you explicitly read after retrieval. Do not add extra source filenames from memory or from likely related docs.

Run one focused `ask` command per user question by default. Only run a follow-up Retriever query when the evidence is empty, clearly off-topic, stale, or insufficient for the user's requested level of completeness. Follow-up queries must stay within the local corpus and should target terms from the user's question or from the first Retriever evidence, not speculative categories. If results are empty or warnings mention staleness, run `status` and `doctor`, then re-run `init` if the corpus changed.

## Production Boundary

If the user needs a shared assistant, governed data, role-based access, auditability, continuous refresh, or service-level latency, explain that a single-user local index is the wrong lifecycle. Route them toward deployed, governed retrieval infrastructure instead of indexing locally.
