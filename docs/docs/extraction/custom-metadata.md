# Custom metadata and filtering

Use this documentation to attach per-document metadata during ingestion and to narrow [LanceDB](vdbs.md) search results in [NeMo Retriever Library](overview.md). Implementation details live in the package [Vector DB operators and LanceDB](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/src/nemo_retriever/vdb#metadata-filtering) README.

## On this page { #on-this-page }

- [Attach metadata at ingestion](#attach-metadata-at-ingestion)
- [Best practices](#best-practices)
- [Filter results during retrieval](#filter-results-during-retrieval)
- [How metadata is stored](#how-metadata-is-stored)

## Attach metadata at ingestion { #attach-metadata-at-ingestion }

Pass a **sidecar metadata table** on `vdb_upload` so selected columns are merged into each chunk's `content_metadata` before LanceDB upload. All three parameters must be set together:

| Parameter | Purpose |
|-----------|---------|
| `meta_dataframe` | Path to CSV, JSON, or Parquet, or an in-memory `pandas.DataFrame` |
| `meta_source_field` | Column that identifies each document (must match ingest paths or basenames per `meta_join_key`) |
| `meta_fields` | Non-empty list of column names to copy into `content_metadata` |

Optional `meta_join_key` controls how rows are matched to documents: `auto` (try full path then basename), `source_id` (full path), or `source_name` (basename only).

For parameter details, refer to the [Python API guide](nemo-retriever-api-reference.md).

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
    create_ingestor(run_mode="service", base_url=f"http://{hostname}:7670")
        .files(["data/woods_frost.pdf", "data/multimodal_test.pdf"])
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=True,
            text_depth="page"
        )
        .embed()
        .vdb_upload(
            vdb_op="lancedb",
            vdb_kwargs={"lancedb_uri": lancedb_uri, "table_name": table_name},
            meta_dataframe=meta_df,
            meta_source_field="source",
            meta_fields=["meta_a", "meta_b"],
        )
)
results = ingestor.ingest_async().result()
```

Set `hostname`, `table_name`, and a **remote** `lancedb_uri` (for example `s3://bucket/path`) to match your deployment—the retriever service rejects local filesystem paths. The client uploads in-memory sidecar metadata to the service before ingest; do not pass a raw local file path as `meta_dataframe` on the REST spec. For local LanceDB directories, use `run_mode="batch"` instead (refer to [Vector databases](vdbs.md)). For a step-by-step walkthrough with additional fields such as category, department, and timestamp, refer to [Vector DB operators and LanceDB — Metadata filtering](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/src/nemo_retriever/vdb#metadata-filtering).

## Best practices { #best-practices }

- Plan metadata structure before ingestion.
- Test filter expressions with small datasets first.
- Consider performance implications of complex filters.
- Validate metadata during ingestion.
- Handle missing metadata fields gracefully.
- Log invalid filter expressions.

## Filter results during retrieval { #filter-results-during-retrieval }

You can use custom metadata to filter documents during retrieval operations. For **predicate pushdown**, pass a `where` SQL predicate through [`Retriever.query`](nemo-retriever-api-reference.md) (refer to [Vector databases](vdbs.md)) or chain `.where(...)` on a native LanceDB `table.search(...)` query. Application-side filtering on returned hits does not change what the database evaluates—raise `top_k` if matches might sit outside the first neighbors.

### Example filter ideas

Typical keys to filter on include `category`, `department`, `priority`, and `timestamp` (use comparable ISO-8601 strings for time ranges). Encode predicates in LanceDB SQL against your table columns (often the serialized `metadata` string), or inspect parsed hit metadata after search as in the example below.

### Example: Use a Filter Expression in Search

After ingestion is complete and documents are uploaded to LanceDB with metadata, you can narrow results in the database with a **`where`** clause, or in Python on the returned hits.

**Native LanceDB (SQL pushdown):** connect, embed the query yourself (same model as ingestion), then chain `.where("<LanceDB SQL predicate>")` on `table.search(...)` so filtering happens before the `limit`. Exact SQL depends on how `metadata` is stored; refer to [LanceDB metadata filtering](https://docs.lancedb.com/search/filtering#filtering-with-sql).

```python
import lancedb

# pseudocode — replace YOUR_VECTOR and YOUR_PREDICATE with real values.
db = lancedb.connect("./lancedb_data")
table = db.open_table("nemo_retriever_collection")
# table.search(YOUR_VECTOR, vector_column_name="vector").where(YOUR_PREDICATE).limit(10).to_list()
```

**`Retriever.query` + `where`:** LanceDB applies the predicate before ranking. For post-filter logic in Python, use a wider `top_k` first.

```python
from nemo_retriever.graph.retriever import Retriever

retriever = Retriever(
    vdb_kwargs={"uri": "./lancedb_data", "table_name": "nemo_retriever_collection"},
    embed_kwargs={
        "model_name": "nvidia/llama-nemotron-embed-1b-v2",
        "embed_model_name": "nvidia/llama-nemotron-embed-1b-v2",
    },
)

hits = retriever.query(
    "this is expensive",
    top_k=16,
    vdb_kwargs={"where": "metadata LIKE '%\"department\":\"Engineering\"%'"},
)
```

For a runnable end-to-end flow (ingest, `Retriever.query`, and both filter modes), refer to [nemo_retriever_retriever_query_metadata_filter.ipynb](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/nemo_retriever_retriever_query_metadata_filter.ipynb).

When you ingest through the **retriever service**, upload the sidecar with [`POST /v1/ingest/sidecar`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/service/routers/ingest.py#L1040-L1129) (multipart file; response [`SidecarUploadResponse`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/service/models/responses.py#L60-L68)), then pass the returned `sidecar_id` as `meta_dataframe_id` with `meta_source_field` and `meta_fields` in `pipeline.vdb_upload_params` on [`POST /v1/ingest`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/service/models/requests.py#L15-L32) ([`PipelineSpec`](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/service/models/pipeline_spec.py#L55-L78)). Request and response shapes, form fields, and auth headers are in the service OpenAPI UI at `/docs` (or `/openapi.json`) on your retriever base URL (for example `http://localhost:7670/docs` after `retriever service start`). Do not send a raw local path as `meta_dataframe` on the service spec.

## How metadata is stored { #how-metadata-is-stored }

- [Vector databases](vdbs.md) — canonical LanceDB upload and retrieval guide
- [nemo_retriever_retriever_query_metadata_filter.ipynb](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/nemo_retriever_retriever_query_metadata_filter.ipynb) — runnable notebook for sidecar metadata at ingest and filtered `Retriever.query`
