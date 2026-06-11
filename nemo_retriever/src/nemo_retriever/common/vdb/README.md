# Vector DB operators and LanceDB

This package wraps **vector database backends** behind a small `VDB` interface (`adt_vdb.py`) and exposes two graph-style operators:

- **`IngestVdbOperator`** — writes embedded pipeline rows into a VDB (ingestion).
- **`RetrieveVdbOperator`** — runs similarity search given **precomputed query vectors** (retrieval).

The only built-in backend key today is **`lancedb`**, resolved by `get_vdb_op_cls()` in `factory.py` to the concrete **`LanceDB`** class in `lancedb.py`.

---

## `IngestVdbOperator` (ingestion)

### Role

`IngestVdbOperator` adapts **flat graph / DataFrame rows** (the shape produced after extract → embed in NeMo Retriever) into the **nested ingestion-pipeline record batches** expected by client VDBs, then calls **`VDB.run(records)`** once per batch.

Flow (see `operators.py` and `records.py`):

1. **`to_client_vdb_records(data)`** — converts rows to `list[list[dict]]` (one outer batch). Rows without both **text** and **embedding** are dropped.
2. Optional **sidecar metadata** — if `vdb_kwargs` contains `meta_dataframe` / `meta_source_field` / `meta_fields`, those keys are stripped for the concrete DB constructor and merged onto records via `sidecar_metadata.py`.
3. **`self._vdb.run(records)`** — delegates to the concrete backend (e.g. `LanceDB.run`).

### Ray batch pipelines (`RayDataExecutor`)

Graph ingestion with `run_mode=batch` uses **`RayDataExecutor`** (`nemo_retriever/graph/executor.py`), which walks the linear graph and, for each node, appends a Ray Data **`map_batches`** stage.

`IngestVdbOperator` declares **`REQUIRES_GLOBAL_BATCH = True`**. When the executor sees that flag on a node’s operator class it:

1. **Repartitions the dataset immediately before that stage** so the upstream `Dataset` is coalesced for this operator — by default **`ds.repartition(num_blocks=1)`**, i.e. a **single Ray Data block** holding **all rows** (the same pattern used for other global operators such as `AudioVisualFuser` and `VideoFrameTextDedup`). If the class instead defines **`GLOBAL_BATCH_GROUP_KEYS`** and **`concurrency > 1`**, the executor may repartition by those keys with multiple blocks; `IngestVdbOperator` does **not** use that path, so it always gets **one block**.
2. Sets **`batch_size=None`** for that `map_batches` call so Ray passes the **entire block** as **one pandas batch** to the operator.

Together, repartition + full batch mean **`process()`** receives **every row at once**, **`to_client_vdb_records`** builds one combined batch list, and **`VDB.run(records)`** runs **once** over the full ingest output — matching the historical “post-graph, single upload” behavior while keeping upload **inside** the graph.

**In-process** execution (`InprocessExecutor`) does not use Ray Data; it already runs each operator on the **whole** `DataFrame`, so no repartition step is needed.

### Wiring ingestion today

- **CLI** (`retriever pipeline run …`): builds `VdbUploadParams` and `GraphIngestor.vdb_upload(...)`, which appends `IngestVdbOperator` to the graph after embed/store and before webhook.
- **Direct API**:

```python
from nemo_retriever.operators.vdb import IngestVdbOperator

op = IngestVdbOperator(
    vdb_op="lancedb",
    vdb_kwargs={
        "uri": "./kb",
        "table_name": "nemo-retriever",
        "vector_dim": 2048,
    },
)
op(pandas_dataframe_of_embedded_rows)  # or list of row dicts
```

CLI-equivalent kwargs are often passed as JSON:

```bash
retriever pipeline run /data/pdfs --vdb-op lancedb \
  --vdb-kwargs-json '{"uri":"./kb","table_name":"nemo-retriever"}'
```

---

## LanceDB inside `IngestVdbOperator`

When `vdb_op="lancedb"` (or `vdb=LanceDB(...)` is passed explicitly), `_construct_vdb` instantiates **`LanceDB`** with the **clean** constructor kwargs (sidecar keys removed).

### `LanceDB.run` (ingestion path)

`LanceDB.run` (in `lancedb.py`) orchestrates:

1. **`create_index`** — connects with `lancedb.connect(self.uri)`, transforms ingestion batches into Arrow rows (`vector`, `text`, `metadata`, `source`), and **`db.create_table(...)`** with schema and `on_bad_vectors` policy.
2. **`write_to_index`** — builds the **vector index** (e.g. IVF/HNSW) and optionally an **FTS/BM25** index over the ingested `text` column when `hybrid=True`.

Common constructor arguments include:

| Parameter        | Purpose |
|-----------------|--------|
| `uri`           | LanceDB database path/URI |
| `table_name`    | Table name (default `nemo-retriever`) |
| `overwrite`     | Table create mode vs append |
| `vector_dim`    | Expected embedding dimension (default 2048) |
| `index_type` / `metric` / `num_partitions` / `num_sub_vectors` | Vector index tuning |
| `hybrid`        | Also build the LanceDB FTS/BM25 index on ingested `text` |
| `on_bad_vectors`| `drop`, `fill`, `null`, or `error` |

---

## `RetrieveVdbOperator` (retrieval)

### Role

`RetrieveVdbOperator` wraps the same concrete **`VDB`** instance but calls **`retrieval(vectors, **kwargs)`** instead of `run`. It merges per-call kwargs with the operator’s stored `vdb_kwargs` and returns **`normalize_retrieval_results(...)`** output (see `operators.py`, `records.py`).

Important: retrieval here expects **`vectors`** — a list of query embedding vectors — as the primary input. String queries are embedded elsewhere (e.g. in `Retriever`). Hybrid backends that need raw text receive aligned `query_texts` as execution-only call context.

### LanceDB inside `RetrieveVdbOperator`

For `vdb_op="lancedb"`, **`LanceDB.retrieval`**:

- Opens the table with `lancedb.connect(table_path).open_table(table_name)`.
- For dense retrieval, each query vector uses **`table.search([vector], vector_column_name=..., **search_kwargs)`**, optional **`.where(where_clause)`** (Lance / DataFusion SQL; `metadata` / `source` are stored as JSON strings), then **`.limit(top_k).refine_factor(...).nprobes(...)`**.
- For hybrid retrieval, callers pass `hybrid=True` plus `query_texts` aligned with the vectors. LanceDB uses **`table.search(query_type="hybrid", vector_column_name=..., fts_columns="text").vector(vector).text(query_text)`** before applying the same `where`, limit, refine, probe, and select handling.

Notable kwargs: `top_k`, `refine_factor`, `n_probe` / `nprobes`, `where` or `_filter`, `table_path`, `table_name`, `search_kwargs`, `hybrid`, and `query_texts`. `query_texts` is stripped from constructor kwargs and forwarded only for retrieval calls whose effective mode is hybrid.

Example of **direct** operator use (you supply vectors):

```python
from nemo_retriever.operators.vdb import RetrieveVdbOperator

op = RetrieveVdbOperator(
    vdb_op="lancedb",
    vdb_kwargs={"uri": "./kb", "table_name": "nemo-retriever"},
)
hits_per_query = op.process(
    [[0.1, 0.2, ...]],  # one query vector; dimension must match table
    top_k=5,
    where="metadata LIKE '%\"page_number\": 3%'",  # example; escape/quote for real SQL
)
```

---

## `Retriever` and `RetrieveVdbOperator`

The high-level **`Retriever`** class (`retriever.py`) uses **`RetrieveVdbOperator`** internally. Pass a flat LanceDB **`vdb_kwargs`** dict with `uri`, `table_name`, filters, etc., or the explicit nested shape `{"vdb_op": "lancedb", "vdb_kwargs": {...}}`.

It **lazy-builds** the operator:

```python
# Conceptually equivalent to:
RetrieveVdbOperator(vdb_op="lancedb", vdb_kwargs={**self.vdb_kwargs})
```

On **`query` / `queries`**, `Retriever`:

1. Embeds query text via the configured embedder (local HF or remote NIM).
2. Calls the retrieve operator’s **`process(vectors, ...)`** with merged **`vdb_kwargs`** (including per-call `where` / `_filter` for LanceDB).

Typical construction:

```python
from nemo_retriever.graph.retriever import Retriever

retriever = Retriever(
    vdb_kwargs={
        "uri": "./kb",
        "table_name": "nemo-retriever",
        "top_k": 10,
        "refine_factor": 50,
        "nprobes": 64,
    },
    embed_kwargs={
        "model_name": "nvidia/llama-nemotron-embed-1b-v2",
        "embed_model_name": "nvidia/llama-nemotron-embed-1b-v2",
    },
)
results = retriever.query("What is covered in section 2?")
```

Per-call Lance filters:

```python
retriever.query(
    "budget assumptions",
    vdb_kwargs={"where": "source LIKE '%annual_report%'", "top_k": 8},
)
```

---

## Metadata filtering

**Reference notebook:** [`examples/nemo_retriever_retriever_query_metadata_filter.ipynb`](../../../../examples/nemo_retriever_retriever_query_metadata_filter.ipynb) — runnable end-to-end demo using sidecar metadata and both filter modes below.

Two complementary mechanisms narrow `Retriever.query` results by metadata:

1. **Server-side (`where`)** — Pass a Lance / DataFusion SQL predicate in `vdb_kwargs` per call (or as a default on the `Retriever`). The predicate runs inside LanceDB on the table columns (`vector`, `text`, `metadata`, `source`) and is wired up in `LanceDB.retrieval` as a `.where(...)` clause on the vector search. **`_filter`** is accepted as an alias for `where`.
2. **Client-side** — Use **`filter_hits_by_content_metadata(hits, predicate)`** after retrieval to keep rows whose parsed `content_metadata` satisfies an arbitrary Python predicate. Useful for logic that doesn't fit SQL or for filters that depend on combined fields.

### How metadata is stored

During ingestion, each chunk's `content_metadata` is serialized as a **compact JSON string** (no spaces after `:` or `,`) in the `metadata` column of the LanceDB table. Sidecar columns supplied via `meta_dataframe` / `meta_source_field` / `meta_fields` are merged into that JSON object before upload — so sidecar keys live in the same JSON string, not in separate columns. This is why SQL filters on metadata use `LIKE` against a JSON substring rather than a real JSON operator.

### Writing `where` predicates

LanceDB evaluates `where` as DataFusion SQL. A few patterns:

```python
# Match a sidecar string field by exact value (compact JSON: "key":"value")
where = "metadata LIKE '%\"meta_a\":\"alpha\"%'"

# Match a numeric metadata field — numbers serialize without quotes
where = "metadata LIKE '%\"meta_b\":10%'"

# Combine predicates with AND / OR
where = "metadata LIKE '%\"meta_a\":\"bravo\"%' AND metadata LIKE '%\"meta_b\":10%'"

# Filter on the `source` column directly (separate from metadata JSON)
where = "source LIKE '%annual_report%'"
```

Escape single quotes in SQL strings by doubling them (`''`). Because matching is substring-based, include the JSON key (`"meta_a":` rather than just `alpha`) to avoid matching unrelated values.

### Server-side vs client-side

Use **`where`** when the predicate fits SQL and you want LanceDB to prune candidates before vector ranking — it also avoids the wasted work of materializing hits you'd discard. Use **`filter_hits_by_content_metadata`** when the predicate is easier to express in Python (e.g. combined numeric ranges, membership in a Python set, or fields that need parsing). They compose well — run a wide `top_k` with a `where` to prune broadly, then post-filter client-side for finer logic:

```python
from nemo_retriever.common.vdb.sidecar_metadata import filter_hits_by_content_metadata

hits = retriever.query(
    "budget assumptions",
    top_k=16,
    vdb_kwargs={"where": "metadata LIKE '%\"meta_a\":\"bravo\"%'"},
)
hits = filter_hits_by_content_metadata(
    hits, lambda m: m.get("meta_b", 0) >= 10
)
```

### Inspecting hit metadata

Each hit's `metadata` field is a JSON string. Use **`parse_hit_content_metadata(hit)`** to get a `dict` you can read directly (this is what `filter_hits_by_content_metadata` uses internally). Both helpers are exported from `nemo_retriever.common.vdb`.

### Hybrid retrieval

Hybrid search (`hybrid=True`) is implemented for LanceDB's precomputed-vector retrieval path. It requires `query_texts` aligned one-to-one with the query vectors so the backend can combine the dense vector query with full-text search. Filters above apply to both dense and hybrid search.

---

## End-to-end mental model

```mermaid
flowchart LR
  subgraph ingest
    G[Graph rows / DataFrame]
    IVO[IngestVdbOperator]
    R1[to_client_vdb_records]
    L1[LanceDB.run]
    G --> IVO --> R1 --> L1
  end

  subgraph retrieve
    Q[Query strings]
    E[Embed queries]
    RVO[RetrieveVdbOperator]
    L2[LanceDB.retrieval]
    Q --> E --> RVO --> L2
  end

  L1 -->[(LanceDB table on disk)]
  L2 -->[(same table)]
```

- **Ingest**: flat rows → ingestion batches → **`LanceDB.run`** → table + indexes.
- **Retrieve**: strings → vectors → **`RetrieveVdbOperator`** → **`LanceDB.retrieval`** → hit lists.

For implementation details, see `operators.py`, `lancedb.py`, `records.py`, `factory.py`, and `retriever.py`.
