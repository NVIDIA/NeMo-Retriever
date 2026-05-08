# NeMo Retriever Agent MCP

The NeMo Retriever agent MCP server exposes a local NeMo Retriever installation to MCP-capable agents. Agents can create collections, ingest local files from allowed roots, query the vector database, and receive normalized evidence for their own reasoning loop.

The server returns evidence only and does not generate final answers.

## Start The Server

```bash
retriever agent-mcp start \
  --data-root .nemo-retriever-mcp \
  --allowed-root /path/to/local/docs \
  --host 127.0.0.1 \
  --port 8099
```

Collections persist under `--data-root`. The default collection is named `default` and is created lazily when first needed.

## Tools

- `list_collections`
- `create_collection`
- `describe_collection`
- `delete_collection`
- `start_ingestion`
- `get_ingestion_status`
- `ingest_local_paths`
- `query_collection`
- `rerank_results`

## Evidence Shape

Query and rerank tools return normalized evidence objects that agents can inspect, cite, and reason over.

```json
{
  "text": "The quarterly review starts at 00:03:12 and covers revenue growth by region.",
  "score": 0.87,
  "source_path": "/path/to/local/docs/review.mp4",
  "media_type": "video",
  "content_type": "transcript",
  "locator": {
    "start_time": 192.4,
    "end_time": 246.8
  },
  "artifacts": [
    {
      "type": "thumbnail",
      "path": ".nemo-retriever-mcp/collections/default/artifacts/review-frame-192.jpg"
    }
  ],
  "metadata": {
    "collection": "default",
    "chunk_id": "review-mp4-transcript-0007",
    "speaker": "Presenter"
  }
}
```

PDF and image evidence may populate page and bounding-box locator fields. Audio and video evidence may populate timestamps or frame information. Text and HTML evidence may include chunk metadata such as section, heading, or character offsets.

## Safety

Version 1 accepts local paths only. The server resolves every path before validation, requires at least one configured `--allowed-root`, and rejects remote URLs and cloud URIs.
