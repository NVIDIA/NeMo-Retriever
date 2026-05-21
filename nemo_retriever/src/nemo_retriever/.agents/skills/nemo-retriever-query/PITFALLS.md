# Query Pitfalls

## Missing Or Wrong Table

`Table ... was not found`, `[]`, or obviously irrelevant hits often mean the
query URI/table does not match ingest. Check both:

```bash
retriever query "known corpus term" --lancedb-uri ./lancedb --table-name nv-ingest --top-k 3
```

Root CLI default table is `nv-ingest`; some older graph-pipeline examples use
`nemo-retriever`. Use the table that was actually written.

Validate table existence directly when query says `Table ... was not found`:

Installed environment:

```bash
python -c "import lancedb; db=lancedb.connect('./lancedb'); print(db.table_names())"
```

Source checkout:

```bash
uv run --project nemo_retriever python -c "import lancedb; db=lancedb.connect('./lancedb'); print(db.table_names())"
```

## Metadata Shape

Current normalized hits expose `metadata` as a dict. Older docs or examples may
show a JSON string. Do not blindly pipe through `fromjson`; first inspect one
hit.

## Embedding Mismatch

If ingest used remote embedding or a non-default model, query must use the same
embedding endpoint and model. Mixed embedding spaces can look like a retrieval
failure even when the table has rows.

If ingest used the local HuggingFace backend, pass
`--local-query-embed-backend hf` plus the same cache/device settings used for
local validation. Otherwise the CLI may try the default local vLLM path and fail
before retrieval.

## Rerank Is Opt-In

Do not assume rerank is enabled. Use `--rerank` or a reranker endpoint/model
option when the user asks for reranking or when precision matters enough to pay
the extra cost.

## Chart And Image Evidence

Chart/image text can be model-generated and may misread exact numbers or
directions. For exact numeric claims, prefer corroborating text/table hits. If
only a chart/image transcription supports the answer, label it as chart-derived
or image-derived rather than making it sound verified by prose.

## Insufficient Evidence

Do not answer from general knowledge when retrieved evidence is missing. State
that the retrieved pages do not contain the requested fact and name the closest
related evidence if useful.
