# Troubleshooting and recovery

Read this only after you hit one of the named errors below. Don't read it pre-emptively.

## If ingest fails or the index is missing

Stay on `retriever ingest`; do not switch to format-specific, stage, or pipeline
commands.

1. Read the surfaced error. If more context is needed, rerun the same ingest once
   with `--no-quiet`.
2. On a CPU-only host, verify that `NVIDIA_API_KEY` or `NGC_API_KEY` is non-empty
   without printing its value. The default hosted embedding endpoint is automatic.
3. Use `--embed-invoke-url` only when the user supplied a different endpoint.
4. If the retry fails, report the ingestion failure and its surfaced error. Do
   not bypass the index by extracting PDFs individually.

## If `retriever query` returns empty `evidence`

Read `coverage.thin_spots` to distinguish an incomplete index from a genuinely
out-of-corpus question. If the index is missing, follow the ingest recovery
above. If it exists, run the exact-term sparse query described in `SKILL.md`.
If that also returns no evidence, say the requested fact is not supported by
the indexed corpus; do not bypass retrieval with a format-specific command.

## Failure modes (expected, not errors)

- **First `ingest` takes ~60s+** — vLLM warmup. Expected.
- **First `query` is slow** — embedder cold-start. ~10–15s on an idle GPU, but **1–3 minutes under concurrent load**. Expected — wait for it; do not kill or relaunch. It is wrapped in `timeout 2000`, so let it run to that ceiling before treating it as failed.
- **Empty `evidence`** — ingest didn't run, or the question is genuinely out-of-corpus; use the recovery above.
- **`Clamping num_partitions ...`** — informational on tiny corpora, not an error.
- **Low-relevance top hit on tiny corpus** — even an unrelated query returns *something*; trust the ranking order (the `score` field is informational, not calibrated confidence).
- **Page-element-detection warnings during ingest** — non-fatal as long as the embedding step itself succeeds (and they're silenced on a successful run, since `ingest` is quiet by default).

## Unsupported file types

`retriever ingest` auto-detects supported input types from file extensions; see
the [ingest input contract](cli/ingest.md#input-contract) for the current list.
Treat unlisted extensions such as `.flac`, `.rtf`, `.eml`, `.py`, `.jsonl`, and
`.zip` as setup issues. Before ingest, inventory:

```bash
find <dir> -type f -name '*.*' | sed 's/.*\.//' | sort -u
```

If unsupported extensions appear, name them in your reply and ask the user
whether to skip or convert them.

## Query-turn cost discipline (recap)

- Use the semantic and exact-term query passes described in `SKILL.md`. Re-query
  only to resolve a distinct term, entity, or ambiguous result.
- If the required fact remains unsupported, explicitly state that the indexed
  evidence does not contain it. Do not extrapolate a plausible value or switch
  to per-file extraction.
- Don't read whole PDFs.
- Don't make speculative Read/Glob/Grep calls "to confirm". The retriever already found the relevant pages — trust the ranking.
- Don't spawn agents, write plans, or make todo lists. The workflow is the workflow.
