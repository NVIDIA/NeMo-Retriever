# NeMo Retriever Agent MCP Design

## Summary

Build a dedicated MCP server that exposes a working NeMo Retriever installation
to agents. The server lets an agent start with no vector database, create or use
a collection, ingest local multimedia files into that collection, query the
resulting evidence, and optionally rerank retrieved hits. Version 1 runs
NeMo Retriever in-process through the Python API, while keeping the MCP tool
contract neutral enough to support a future remote NeMo Retriever service
backend.

The MCP server is an evidence engine, not an answer-generation system. It
returns retrieved chunks, scores, source locators, artifacts, and metadata so
the calling agent model can reason over the evidence itself.

## Goals

- Expose NeMo Retriever extraction, embedding, VDB upload, retrieval, and
  reranking to MCP-capable agents.
- Let agents create, list, describe, query, and delete VDB-backed collections.
- Start from an empty environment and populate collections entirely through MCP
  usage.
- Support local-path ingestion for PDFs, documents, images, text, HTML, audio,
  and video where the installed NeMo Retriever runtime supports those formats.
- Persist collections by default, with an explicit temporary collection option.
- Use a job-based ingestion model internally, with a blocking convenience tool
  for small or simple workflows.
- Keep the agent-facing MCP contract backend-neutral so a future remote service
  backend can replace the in-process backend without changing tool semantics.

## Non-Goals

- Fetching remote URLs, S3/GCS paths, or other network resources in v1.
- Exposing arbitrary shell execution through MCP.
- Building an agent orchestration loop or answer-generation tool.
- Designing a distributed VDB scaling model beyond multiple named collections.
- Guaranteeing GPU-backed full multimedia ingestion in normal unit test runs.

## Architecture

The MCP server is separate from the existing benchmark harness portal MCP. The
harness MCP remains focused on benchmark runs and schedules; this server is a
product-facing retrieval surface for agent workflows.

Core components:

- **MCP server**: FastMCP-based process that registers collection, ingestion,
  retrieval, and reranking tools.
- **Collection registry**: persistent SQLite store under a configured MCP data
  root. It tracks collection metadata, backend configuration, job state, source
  summaries, and queryability.
- **Backend interface**: a stable internal interface implemented first by
  `InProcessRetrieverBackend`. A later `ServiceRetrieverBackend` can map the
  same operations to a deployed NeMo Retriever service.
- **VDB storage**: v1 collections use LanceDB directories and tables under the
  configured data root. `default` is a normal named collection, created lazily.
- **Artifact storage**: optional per-collection storage for extracted images,
  rendered pages, video frames, thumbnails, or other non-text evidence
  artifacts.
- **Job manager**: ingestion runs as jobs internally. Blocking tools submit a
  job and wait for terminal state or timeout.
- **Safety boundary**: local paths only, constrained to configured allowed
  roots. The server resolves symlinks before validation.

High-level flow:

```text
agent
  -> create_collection("default")
  -> ingest_local_paths("default", ["/allowed/docs"], wait=true)
  -> InProcessRetrieverBackend
  -> GraphIngestor extraction/embed/store pipeline
  -> LanceDB collection
  -> query_collection("default", "question")
  -> Retriever query/rerank path
  -> normalized multimedia evidence
```

## MCP Tool Surface

The initial tools are intentionally small and evidence-focused:

- `list_collections()`: list persistent and temporary collections.
- `create_collection(name?, temporary=false, config?)`: create a named
  collection, defaulting to `default` when `name` is omitted.
- `describe_collection(name="default")`: return collection metadata, VDB
  location, embedding config, source summaries, and recent jobs.
- `delete_collection(name, delete_data=false)`: remove registry metadata and,
  only when explicitly requested, delete collection data under the configured
  collection root.
- `start_ingestion(collection="default", paths, options?)`: validate local
  paths, expand directories, create a job, and begin ingestion asynchronously.
- `get_ingestion_status(job_id)`: return job status, counts, warnings, errors,
  and queryability.
- `ingest_local_paths(collection="default", paths, options?, wait=true,
  timeout_s?)`: convenience wrapper around `start_ingestion` plus polling.
- `query_collection(collection="default", query, top_k=10, filters?,
  hybrid=false, rerank=false)`: retrieve normalized evidence hits.
- `rerank_results(query, hits, top_n?)`: rerank provided evidence hits without
  running a new VDB query.

The v1 surface does not include `answer_question`. The caller can use the
returned evidence in its own reasoning loop.

## Collection And Storage Model

Collections are modality-neutral workspaces. Each collection has:

- name and optional aliases, including the conventional `default`;
- persistent or temporary lifetime;
- backend type and backend-specific configuration;
- VDB location, table name, and embedding model/configuration;
- artifact root;
- creation and update timestamps;
- ingestion jobs and source summaries;
- queryable state.

Recommended on-disk layout:

```text
<nemo-mcp-root>/
  registry.sqlite
  collections/
    default/
      lancedb/
      artifacts/
    quarterly-training/
      lancedb/
      artifacts/
  tmp/
    <temporary-collection-id>/
      lancedb/
      artifacts/
```

SQLite is preferred over loose JSON because the server needs durable job state,
collection listings, source summaries, and status transitions. LanceDB remains
responsible for vector storage.

## Ingestion Design

The backend accepts local files and directories only. It resolves every input
path, checks it against configured allowed roots, expands directories and globs,
and rejects unsupported paths before recording the job.

Ingestion is internally job-based:

1. Validate paths and collection.
2. Record a queued job.
3. Group files by media type.
4. Run modality-specific NeMo Retriever pipelines.
5. Embed extracted text representations.
6. Optionally store artifacts.
7. Upload embedded rows to the collection VDB.
8. Update collection metadata and mark the collection queryable when at least
   one upload succeeds.

The recommended v1 execution shape is grouped sub-jobs by modality rather than
a single opaque mixed-media run. Grouping makes status clearer and allows
different options per media type:

- PDFs, documents, and images use `GraphIngestor.extract(...)` or
  `extract_image_files(...)` where appropriate.
- Text uses `GraphIngestor.extract_txt(...)`.
- HTML uses `GraphIngestor.extract_html(...)`.
- Audio uses `GraphIngestor.extract_audio(...)`.
- Video uses `GraphIngestor.extract_video(...)`.

The job state model should include `queued`, `running`, `complete`, `failed`,
and `partial`. Partial jobs are valid when some files ingest successfully and
others fail or are skipped.

Job status should expose source counts, accepted and skipped files, per-file
errors, modality summaries, row count when available, artifact count when
available, warnings, and whether the target collection is queryable.

## Query And Rerank Design

`query_collection` resolves the collection, constructs a `Retriever` with the
collection's VDB settings and embedding configuration, runs semantic or hybrid
retrieval, optionally reranks, and returns normalized evidence.

Key rules:

- Query embedding must use the same model and endpoint family used for
  collection ingestion.
- Hybrid search is available only when the collection was indexed for hybrid
  retrieval.
- Reranking can be requested inline with `query_collection(..., rerank=true)`
  or separately through `rerank_results`.
- Filters are supported conservatively. v1 may pass simple metadata filters to
  backends that support them, but must not promise a cross-backend filter
  language before that contract exists.
- Raw metadata remains available as an escape hatch for advanced agents.

The normalized evidence shape is modality-neutral and sparse:

```json
{
  "text": "...",
  "score": 0.82,
  "source_path": "/allowed/docs/demo.mp4",
  "media_type": "video",
  "content_type": "transcript",
  "locator": {
    "page_number": null,
    "timestamp_start_s": 42.0,
    "timestamp_end_s": 55.5,
    "frame_index": 18,
    "bbox_xyxy_norm": [0.1, 0.2, 0.8, 0.6],
    "chunk_id": "optional-backend-id"
  },
  "artifacts": {
    "stored_image_uri": "optional-uri",
    "thumbnail_uri": "optional-uri"
  },
  "metadata": {}
}
```

PDFs and images tend to populate `page_number` and bounding boxes. Audio and
video tend to populate timestamps, frame indices, transcript segment metadata,
and extracted frame artifacts. Text and HTML may populate chunk IDs, headings,
or offsets when available.

## Backend Abstraction

The MCP tools call a backend interface rather than importing pipeline details
directly. A representative interface:

```text
RetrieverBackend
  create_collection(...)
  list_collections(...)
  describe_collection(...)
  delete_collection(...)
  start_ingestion(...)
  get_job_status(...)
  query_collection(...)
  rerank_results(...)
```

`InProcessRetrieverBackend` uses local NeMo Retriever Python APIs and local
LanceDB. This is the v1 implementation.

`ServiceRetrieverBackend` can later route the same operations to a running
NeMo Retriever service. That path may require the service API to become more
collection-aware than it is today, because the current service is oriented
around configured LanceDB locations and tables. The backend abstraction lets the
MCP contract lead while the service catches up.

The agent-facing tool contract should describe retrieval capabilities and
collection intent. LanceDB, local paths, in-process execution, Ray, and NIM
details are backend facts, not MCP semantics.

## Error Handling And Safety

The MCP server returns structured errors with:

- `code`;
- `message`;
- `retryable`;
- optional `details`.

Expected error codes include:

- `PATH_OUTSIDE_ALLOWED_ROOT`
- `PATH_NOT_FOUND`
- `UNSUPPORTED_MEDIA_TYPE`
- `COLLECTION_NOT_FOUND`
- `COLLECTION_NOT_QUERYABLE`
- `INGEST_JOB_FAILED`
- `HYBRID_NOT_AVAILABLE`
- `EMBEDDING_CONFIG_MISMATCH`
- `VDB_NOT_FOUND`
- `BACKEND_ERROR`

Safety constraints:

- All input paths must resolve under configured allowed roots.
- Symlinks are resolved before validation.
- URLs and cloud URIs are rejected in v1.
- The MCP server does not execute shell commands for ingest/query.
- Delete operations cannot remove data outside the configured MCP data root.
- `delete_collection(..., delete_data=true)` is the only data removal path.
- Resource guardrails should include max files, max total bytes, timeout, and
  optional per-modality enablement.
- Partial success is reported as `partial`; successful rows remain queryable.
- Temporary collections are listable and explicitly cleanable. Automatic
  cleanup can be a later policy.

## Testing And Validation

Unit tests should cover:

- collection registry CRUD and persistence;
- allowed-root path validation, including symlinks;
- collection naming and lazy `default` creation;
- temporary versus persistent layout;
- job state transitions;
- evidence-hit normalization across representative PDF, image, text, audio,
  and video metadata;
- structured error responses.

Backend tests should use fake backend objects so MCP tool behavior can be
tested without GPUs, Ray, NIM endpoints, or full LanceDB fixtures.

Integration smoke tests should cover:

- create collection;
- ingest a small local fixture through the real in-process backend when the
  environment supports it;
- query the collection;
- rerank retrieved hits when a reranker is configured;
- list and describe the collection;
- delete registry metadata and, in a separate explicit test, collection data.

Normal CI should not require GPU-backed full multimedia extraction. Optional
GPU/NIM smoke tests can run in suitable environments.

## Success Criteria

- An MCP-capable agent can start with no VDB, create or lazily use `default`,
  ingest local allowed files, wait for completion, query the populated
  collection, and receive normalized multimedia evidence.
- Collections persist across MCP server restarts unless marked temporary.
- The tool contract remains evidence-focused and does not include answer
  generation.
- The implementation can later route through a remote NeMo Retriever service
  without changing the MCP tool names or core semantics.
- Unsupported files and per-file failures are visible to the agent without
  hiding successful ingested evidence.
