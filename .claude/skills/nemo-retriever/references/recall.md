# retriever recall

Batch query + recall@k evaluation. Reads a CSV of ground-truth queries,
embeds each query, searches a LanceDB table, prints per-query hits, and
computes recall@1 / @5 / @10.

If flags below look stale, re-check `retriever recall vdb-recall run --help`.

## When to use this

- You have labelled `(query, pdf, page)` ground truth and want recall
  metrics for a retrieval setup.
- Sweeping embedding models / chunking / top-k against a fixed query set.

**Use a different command when:**

- You want a single ad-hoc lookup → [[query]].
- You want full QA quality (answer grading), not just retrieval recall →
  [[eval]].
- You want to compare two recall runs → [[compare]].

## Canonical invocations

Default recall against the project query set:

```bash
retriever recall vdb-recall run
```

Custom query CSV + custom table:

```bash
retriever recall vdb-recall run \
  --query-csv my-queries.csv \
  --top-k 10 \
  --lancedb-uri ./my-lancedb \
  --table-name my-corpus
```

Route embedding through a remote NIM:

```bash
retriever recall vdb-recall run \
  --query-csv my-queries.csv \
  --embedding-http-endpoint http://embed:8000/v1/embed
```

## Inputs

- **`--query-csv FILE`** — CSV with `query,pdf_page` or `query,pdf,page`
  columns. Default `bo767_query_gt.csv`.

## Outputs

- Per-query top-k hits printed to stdout.
- A summary line with `recall@1 / @5 / @10`.

`recall@10` always queries with `search_k = max(top_k, 10)` so the metric
remains valid even when you display fewer hits.

## Key flags

| Flag | Default | Notes |
|---|---|---|
| `--query-csv` | `bo767_query_gt.csv` | Ground-truth CSV. |
| `--top-k` | `5` | Hits shown per query (recall@10 still computed). |
| `--lancedb-uri` | `lancedb` | Must match [[ingest]] / [[vector-store]]. |
| `--table-name` | `nv-ingest` | Same. |
| `--vector-column` | `vector` | Column to search. |
| `--embedding-endpoint` / `--embedding-http-endpoint` / `--embedding-grpc-endpoint` | — | Remote query embedder. Falls back to local HF if all unset. |
| `--limit` | — | Cap queries (debug). |

## Common failure modes

- **`recall@10 = 0.0`** — query embedder doesn't match the ingest embedder
  (different model / dim). Re-ingest with the same embedder or pass the
  matching `--embedding-*-endpoint`.
- **`KeyError: 'pdf_page'`** — CSV uses `pdf,page` instead. The command
  accepts either schema, but typos in column names break both.
- **Slow first run** — local HF embedder cold-start. Reuse a single process
  or hit a warm NIM.

## Related

- [[query]] — ad-hoc retrieval against the same table.
- [[eval]] — adds answer-quality grading on top of retrieval.
- [[compare]] — diff two retrieval runs.
