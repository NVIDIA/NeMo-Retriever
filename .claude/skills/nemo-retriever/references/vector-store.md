# retriever vector-store

LanceDB upload stage: take a directory of `*.text_embeddings.json` files
(produced by the local `stage5` embedder) and load them into a LanceDB
table, optionally creating an IVF index.

If flags below look stale, re-check `retriever vector-store stage run --help`.

## When to use this

- You ran embedding offline (e.g. via [[local]] stage5 or a custom embed
  job) and now want the vectors searchable.
- You want to (re)build a LanceDB index over existing embedding sidecars.

**Use a different command when:**

- You want full ingest in one shot → [[ingest]] or [[pipeline]] (their last
  stage already does this).
- You want to *query* an existing table → [[query]] / [[recall]].

## Canonical invocations

Upload + index with defaults (overwrites the table):

```bash
retriever vector-store stage run --input-dir out/embeddings/
```

Append rather than overwrite, into a custom DB/table:

```bash
retriever vector-store stage run \
  --input-dir out/embeddings/ \
  --lancedb-uri ./my-lancedb \
  --table-name my-corpus \
  --append
```

Skip indexing (faster, but slower searches afterwards):

```bash
retriever vector-store stage run --input-dir out/embeddings/ --no-create-index
```

## Inputs

- **`--input-dir DIR`** — required. Contains `*.text_embeddings.json` files.
  `--recursive` to scan subdirectories.

## Outputs

- LanceDB table at `<lancedb-uri>/<table-name>.lance`. Defaults
  `lancedb/nv-ingest.lance` — matches [[ingest]] / [[query]] defaults.
- Each row carries `vector`, `pdf_basename`, `page_number`, `path`,
  `source_id`, and the original primitive metadata.

## Key flags

| Flag | Default | Notes |
|---|---|---|
| `--recursive` | off | Walk subdirectories of `--input-dir`. |
| `--lancedb-uri` | `lancedb` | DB path/URI. |
| `--table-name` | `nv-ingest` | Table name (must match [[query]]). |
| `--overwrite/--append` | `overwrite` | Replace or extend existing table. |
| `--create-index/--no-create-index` | `create-index` | Build vector index after upload. |
| `--index-type` | `IVF_HNSW_SQ` | LanceDB index type. |
| `--metric` | `l2` | Distance metric (must match how you'll search). |
| `--num-partitions` | `16` | IVF partitions. Clamped down for tiny tables. |
| `--num-sub-vectors` | `256` | PQ sub-vectors. |

## Common failure modes

- **`Clamping num_partitions from 16 to N`** — informational; index needs
  partitions < row count. Happens on small uploads.
- **`Table already exists`** with `--append` returning unexpected rows —
  `--append` does not dedupe. Run [[query]] / inspect the table if you
  suspect duplicates.
- **Query results look bad after upload** — metric mismatch between this
  stage's `--metric` and what [[query]] uses (`l2` everywhere by default).

## Related

- [[query]] — search the table this command writes.
- [[recall]] — batch query + recall metrics over a CSV of ground truth.
- [[pipeline]] — full ingest that uses this stage as its sink.
