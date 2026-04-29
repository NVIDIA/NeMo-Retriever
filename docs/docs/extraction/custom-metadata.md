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
For more information about the `Ingestor` class, refer to [Use the Python API](nemo-retriever-api-reference.md).
For more information about the `vdb_upload` method, refer to [Upload Data](vdbs.md).

```python
from nv_ingest_client.client import Ingestor

hostname = "localhost"
table_name = "nemo_retriever_collection"
lancedb_uri = "./lancedb_data"

ingestor = (
    Ingestor(message_client_hostname=hostname)
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
            hybrid=False,
            dense_dim=2048,
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
Express filters with [LanceDB SQL](https://lancedb.github.io/lancedb/sql/) (for example predicates on the `metadata` JSON your pipeline stored). The helper below performs vector search; add table-native filtering or post-filter in application code as needed for your schema.


### Example filter ideas

Typical keys to filter on include `category`, `department`, `priority`, and `timestamp` (use comparable ISO-8601 strings for time ranges). Apply them with [LanceDB SQL](https://lancedb.github.io/lancedb/sql/) on the stored `metadata` column, or by inspecting `hit["entity"]["content_metadata"]` after vector search as in the example below.

### Example: Use a Filter Expression in Search

After ingestion is complete, and documents are uploaded to LanceDB with metadata,
you can read `content_metadata` from each hit (or push predicates into LanceDB SQL) to narrow results—for example by department.

The following example runs vector search with [lancedb_retrieval](https://github.com/NVIDIA/NeMo-Retriever/blob/main/client/src/nv_ingest_client/util/vdb/lancedb.py); filter hits in Python or extend the query with LanceDB-native filters for your deployment.

```python
from nv_ingest_client.util.vdb.lancedb import lancedb_retrieval

hostname = "localhost"
table_name = "nemo_retriever_collection"
lancedb_uri = "./lancedb_data"
top_k = 5
model_name = "nvidia/llama-3.2-nv-embedqa-1b-v2"

queries = ["this is expensive"]
q_results = []
for que in queries:
    batch = lancedb_retrieval(
        [que],
        table_path=lancedb_uri,
        table_name=table_name,
        embedding_endpoint=f"http://{hostname}:8012/v1",
        top_k=top_k,
        model_name=model_name,
    )
    # Example post-filter: keep hits whose content_metadata lists Engineering
    filtered = [
        hit
        for hit in batch[0]
        if hit.get("entity", {})
        .get("content_metadata", {})
        .get("department")
        == "Engineering"
    ]
    q_results.append(filtered)

print(f"{q_results}")
```



## Related Content

- For a notebook that uses the CLI to add custom metadata and filter query results, refer to [metadata_and_filtered_search.ipynb
](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/metadata_and_filtered_search.ipynb).
