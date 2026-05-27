# Custom metadata and filtering

Use this documentation to attach per-document metadata during ingestion and to narrow [LanceDB](vdbs.md) search results in [NeMo Retriever Library](overview.md). Implementation details live in the package [Vector DB operators and LanceDB](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/src/nemo_retriever/vdb#metadata-filtering) README.

## On this page { #on-this-page }

- [Attach metadata at ingestion](#attach-metadata-at-ingestion)
- [How metadata is stored](#how-metadata-is-stored)
- [Filter results at query time](#filter-results-at-query-time)
- [Writing `where` predicates](#writing-where-predicates)
- [Server-side vs client-side filters](#server-side-vs-client-side-filters)
- [Inspect hit metadata](#inspect-hit-metadata)
- [Limitations](#limitations)
- [Related content](#related-content)

## Attach metadata at ingestion { #attach-metadata-at-ingestion }

Pass a **sidecar metadata table** on `vdb_upload` so selected columns are merged into each chunk's `content_metadata` before LanceDB upload. All three parameters must be set together:

| Parameter | Purpose |
|-----------|---------|
| `meta_dataframe` | Path to CSV, JSON, or Parquet, or an in-memory `pandas.DataFrame` |
| `meta_source_field` | Column that identifies each document (must match ingest paths or basenames per `meta_join_key`) |
| `meta_fields` | Non-empty list of column names to copy into `content_metadata` |

Optional `meta_join_key` controls how rows are matched to documents: `auto` (try full path then basename), `source_id` (full path), or `source_name` (basename only).

```python
import pandas as pd
from nemo_retriever import create_ingestor

meta_df = pd.DataFrame(
    {
        "source": ["data/woods_frost.pdf", "data/multimodal_test.pdf"],
        "meta_a": ["alpha", "bravo"],
        "meta_b": [10, 20],
    }
)

ingestor = (
    create_ingestor(run_mode="batch")
    .files(["data/woods_frost.pdf", "data/multimodal_test.pdf"])
    .extract(extract_text=True, text_depth="page")
    .embed()
    .vdb_upload(
        vdb_op="lancedb",
        uri="./lancedb_data",
        table_name="nemo-retriever",
        meta_dataframe=meta_df,
        meta_source_field="source",
        meta_fields=["meta_a", "meta_b"],
    )
)
ingestor.ingest()
```

For a runnable end-to-end flow (ingest, `Retriever.query`, and both filter modes), see [nemo_retriever_retriever_query_metadata_filter.ipynb](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/nemo_retriever_retriever_query_metadata_filter.ipynb).

When you ingest through the **retriever service**, upload the sidecar with [`POST /v1/ingest/sidecar`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/service/routers/ingest.py#L1040-L1129) (multipart file; response [`SidecarUploadResponse`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/service/models/responses.py#L60-L68)), then pass the returned `sidecar_id` as `meta_dataframe_id` with `meta_source_field` and `meta_fields` in `pipeline.vdb_upload_params` on [`POST /v1/ingest`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/service/models/requests.py#L15-L32) ([`PipelineSpec`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/service/models/pipeline_spec.py#L55-L78)). Request and response shapes, form fields, and auth headers are in the service OpenAPI UI at `/docs` (or `/openapi.json`) on your retriever base URL (for example `http://localhost:7670/docs` after `retriever service start`). Do not send a raw local path as `meta_dataframe` on the service spec.

## How metadata is stored { #how-metadata-is-stored }

During ingestion, each chunk's `content_metadata` is serialized as a **compact JSON string** (no spaces after `:` or `,`) in the LanceDB `metadata` column. Sidecar columns are merged into that JSON object before upload, so custom keys live in the same string — not in separate table columns. SQL filters on custom fields therefore use `LIKE` against JSON substrings rather than a dedicated JSON operator.

The `source` column stores the document path separately from the metadata JSON.

## Filter results at query time { #filter-results-at-query-time }

Two complementary mechanisms narrow `Retriever.query` results:

1. **Server-side (`where`)** — Pass a Lance / DataFusion SQL predicate in `vdb_kwargs` per call (or as defaults on the `Retriever`). LanceDB applies it as a `.where(...)` clause on vector search. **`_filter`** is accepted as an alias for `where`.
2. **Client-side** — Use `filter_hits_by_content_metadata(hits, predicate)` after retrieval to keep rows whose parsed `content_metadata` satisfies arbitrary Python logic.

```python
from nemo_retriever.retriever import Retriever

retriever = Retriever(
    vdb="lancedb",
    vdb_kwargs={"uri": "./lancedb_data", "table_name": "nemo-retriever"},
    embedder="nvidia/llama-nemotron-embed-1b-v2",
)

hits = retriever.query(
    "budget assumptions",
    top_k=16,
    vdb_kwargs={"where": "metadata LIKE '%\"meta_a\":\"bravo\"%'"},
)
```

## Writing `where` predicates { #writing-where-predicates }

LanceDB evaluates `where` as DataFusion SQL over columns `vector`, `text`, `metadata`, and `source`:

```python
# Match a sidecar string field (compact JSON: "key":"value")
where = "metadata LIKE '%\"meta_a\":\"alpha\"%'"

# Match a numeric metadata field — numbers serialize without quotes
where = "metadata LIKE '%\"meta_b\":10%'"

# Combine predicates
where = "metadata LIKE '%\"meta_a\":\"bravo\"%' AND metadata LIKE '%\"meta_b\":10%'"

# Filter on the source column directly
where = "source LIKE '%annual_report%'"
```

Escape single quotes in SQL strings by doubling them (`''`). Because matching is substring-based, include the JSON key (`"meta_a":` rather than only `alpha`) to avoid false positives.

## Server-side vs client-side filters { #server-side-vs-client-side-filters }

Use **`where`** when the predicate fits SQL and you want LanceDB to prune candidates before vector ranking. Use **`filter_hits_by_content_metadata`** when the predicate is easier in Python (combined numeric ranges, set membership, or fields that need parsing). They compose: run a wider `top_k` with `where`, then post-filter for finer logic.

```python
from nemo_retriever.vdb import filter_hits_by_content_metadata

hits = retriever.query(
    "budget assumptions",
    top_k=16,
    vdb_kwargs={"where": "metadata LIKE '%\"meta_a\":\"bravo\"%'"},
)
hits = filter_hits_by_content_metadata(
    hits, lambda m: m.get("meta_b", 0) >= 10
)
```

## Inspect hit metadata { #inspect-hit-metadata }

Each hit's `metadata` field is a JSON string. Use **`parse_hit_content_metadata(hit)`** to obtain a `dict` (the same helper `filter_hits_by_content_metadata` uses). Both helpers are exported from `nemo_retriever.vdb`.

## Limitations { #limitations }

- **Hybrid search** — Metadata filters on the precomputed-vector retrieval path apply to **dense vector search only**. `LanceDB.retrieval` raises `NotImplementedError` when `hybrid=True`; see [Vector databases](vdbs.md#hybrid-search-lancedb).
- **Predicate shape** — `where` uses substring `LIKE` on compact JSON in `metadata`; design keys and values accordingly.
- **Sidecar updates** — Changing sidecar data requires re-ingesting affected documents so LanceDB rows pick up new metadata.

## Related content { #related-content }

- [Vector databases](vdbs.md) — LanceDB upload, retrieval, and hybrid notes
- [nemo_retriever_retriever_query_metadata_filter.ipynb](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/nemo_retriever_retriever_query_metadata_filter.ipynb) — end-to-end metadata filtering with `Retriever`
- [nemo_retriever_metadata_and_filtered_search.ipynb](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/nemo_retriever_metadata_and_filtered_search.ipynb) — graph ingest with sidecar metadata
- [Vector DB operators (source)](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/src/nemo_retriever/vdb#metadata-filtering) — canonical developer reference for this page
