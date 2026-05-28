# Use Custom Metadata to Filter Search Results

You can upload custom metadata for documents during ingestion. 
By uploading custom metadata you can attach additional information to documents, 
and use it for filtering results during retrieval operations. 
For example, you can add author metadata to your documents, and filter by author when you retrieve results. 
To create filters at query time, use predicates supported by [LanceDB SQL](https://lancedb.github.io/lancedb/sql/) against your table schema (custom fields are serialized into the `metadata` column with your ingested chunks). For a worked example, see the repository notebook linked at the end of this page.

Use this documentation to use custom metadata to filter search results when you work with [NeMo Retriever Library](overview.md).


## Limitations

The following are limitation when you use custom metadata:

- Metadata fields must be consistent across documents in the same collection.
- Complex filter expressions may impact retrieval performance.
- If you update your custom metadata, you must ingest your documents again to use the new metadata.



## Add Custom Metadata During Ingestion

You can add custom metadata during the document ingestion process. 
You can specify metadata for each file, 
and you can specify different metadata for different documents in the same ingestion batch.


### Metadata Structure

You specify custom metadata as a dataframe or a file (json, csv, or parquet). 

The following example contains metadata fields for category, department, and timestamp. 
You can create whatever metadata is helpful for your scenario.

```python
import pandas as pd

meta_df = pd.DataFrame(
    {
        "source": ["data/woods_frost.pdf", "data/multimodal_test.pdf"],
        "category": ["Alpha", "Bravo"],
        "department": ["Language", "Engineering"],
        "timestamp": ["2025-05-01T00:00:00", "2025-05-02T00:00:00"]
    }
)

# Convert the dataframe to a csv file, 
# to demonstrate how to ingest a metadata file in a later step.

file_path = "./meta_file.csv"
meta_df.to_csv(file_path)
```


### Example: Add Custom Metadata During Ingestion

The following example adds custom metadata during ingestion. 
For more information about `create_ingestor` and run modes, refer to [Use the Python API](nemo-retriever-api-reference.md).
For more information about the `vdb_upload` method, refer to [Upload Data](vdbs.md).

```python
from nemo_retriever import create_ingestor

# Service-backed pipeline: point `base_url` at your running retriever service.
# For local graph execution instead, see [Use the Python API](nemo-retriever-api-reference.md).

hostname = "localhost"
table_name = "nemo_retriever_collection"
lancedb_uri = "./lancedb_data"

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
            uri=lancedb_uri,
            table_name=table_name,
        )
)
results = ingestor.ingest_async().result()
```

Merge values from `meta_df` (or `file_path`) into each document's `content_metadata` before `vdb_upload`, or follow the step-by-step pattern in [metadata_and_filtered_search.ipynb](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/metadata_and_filtered_search.ipynb), so category, department, and timestamp are present on the chunks LanceDB indexes.

## Best Practices

The following are the best practices when you work with custom metadata:

- Plan metadata structure before ingestion.
- Test filter expressions with small datasets first.
- Consider performance implications of complex filters.
- Validate metadata during ingestion.
- Handle missing metadata fields gracefully.
- Log invalid filter expressions.



## Use Custom Metadata to Filter Results During Retrieval

You can use custom metadata to filter documents during retrieval operations.
For **predicate pushdown**, pass a `where` SQL predicate through [`Retriever.query`](nemo-retriever-api-reference.md) (see [Vector databases](vdbs.md)) or chain `.where(...)` on a native LanceDB `table.search(...)` query. Application-side filtering on returned hits does not change what the database evaluates—raise `top_k` if matches might sit outside the first neighbors.


### Example filter ideas

Typical keys to filter on include `category`, `department`, `priority`, and `timestamp` (use comparable ISO-8601 strings for time ranges). Encode predicates in LanceDB SQL against your table columns (often the serialized `metadata` string), or inspect parsed hit metadata after search as in the example below.

### Example: Use a Filter Expression in Search

After ingestion is complete, and documents are uploaded to LanceDB with metadata,
you can narrow results in the database with a **`where`** clause, or in Python on the returned hits.

**Native LanceDB (SQL pushdown):** connect, embed the query yourself (same model as ingestion), then chain `.where("<LanceDB SQL predicate>")` on `table.search(...)` so filtering happens before the `limit`. Exact SQL depends on how `metadata` is stored; see [LanceDB SQL](https://lancedb.github.io/lancedb/sql/).

```python
import lancedb

# Pseudocode sketch — replace YOUR_VECTOR and YOUR_PREDICATE with real values.
db = lancedb.connect("./lancedb_data")
table = db.open_table("nemo_retriever_collection")
# table.search(YOUR_VECTOR, vector_column_name="vector").where(YOUR_PREDICATE).limit(10).to_list()
```

**`Retriever.query` + `where`:** LanceDB applies the predicate before ranking. For post-filter logic in Python, use a wider `top_k` first.

```python
from nemo_retriever.retriever import Retriever

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



## Related Content

- [Vector databases](vdbs.md) — canonical LanceDB upload and retrieval guide
- [metadata_and_filtered_search.ipynb](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/metadata_and_filtered_search.ipynb) — CLI and graph ingest with sidecar metadata
