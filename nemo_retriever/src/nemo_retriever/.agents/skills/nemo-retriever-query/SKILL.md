---
name: nemo-retriever-query
description: Use when the user asks to search a NeMo Retriever index, run `retriever query`, retrieve evidence from LanceDB, inspect query hit schemas, answer questions from retrieved documents, or debug missing/empty query results. Do not use for creating indexes; use `nemo-retriever-ingest` instead.
---

# nemo-retriever-query

Use this skill to retrieve evidence from an existing NeMo Retriever index and
answer only from that evidence.

## Orientation

1. Verify the active public surface first: `retriever query --help`.
2. Confirm the LanceDB URI, table name, and embedding settings from the ingest
   task or project config. Do not guess if the user supplied different values.
3. If the installed CLI is absent but this is a source checkout, use
   `uv run --project nemo_retriever retriever query --help`. Retry dependency
   downloads before choosing another query validation path.
4. If neither path works, use `nemo-retriever-setup` before debugging query
   behavior.
5. Keep source citation indexing straight: `page_number` returned by Retriever
   is 1-indexed unless an external task schema says otherwise.

## References

- `references/QUERY.md`: CLI and Python query patterns, result schema, rerank
  behavior, and answer synthesis.
- `PITFALLS.md`: missing tables, empty hits, metadata shape mistakes, chart
  uncertainty, and model mismatches.

## Workflow

1. Run a focused query against the known table:

   ```bash
   retriever query "question text" --lancedb-uri ./lancedb --table-name nv-ingest --top-k 5
   ```

2. If ingest used remote embedding, include the same query embedding endpoint
   and model:

   ```bash
   retriever query "question text" \
     --lancedb-uri ./lancedb \
     --table-name nv-ingest \
     --embed-invoke-url https://integrate.api.nvidia.com/v1/embeddings \
     --embed-model-name nvidia/llama-nemotron-embed-1b-v2
   ```

3. If ingest used local HuggingFace embeddings, keep the query backend and cache
   explicit so the CLI does not fall back to the vLLM local path:

   ```bash
   retriever query "question text" \
     --lancedb-uri ./lancedb \
     --table-name nv-ingest \
     --embed-model-name nvidia/llama-nemotron-embed-1b-v2 \
     --local-query-embed-backend hf \
     --local-hf-cache-dir "$HOME/models/huggingface" \
     --local-hf-device cuda
   ```

4. Inspect ranked hits before answering. Use `_distance` as a ranking signal,
   not a calibrated score.
5. Synthesize from hit `text`, `source_id` / `path`, `pdf_basename`, and
   `page_number`. Include document and page when available.
6. If the evidence does not answer the question, say what is missing instead of
   inventing a plausible answer.

## Success Checks

- Query output is a JSON array of hits, possibly empty.
- Each answer claim is supported by one or more retrieved hit texts.
- The final answer states insufficient evidence when the retrieved text does
  not contain the requested fact.

## Evaluation Scenarios

- "Use the Retriever index to answer: what was revenue in 2024?" Use this skill.
- "The query returns no hits." Use this skill and read `PITFALLS.md`.
- "Index the PDFs first." Use `nemo-retriever-ingest`, not this skill.
