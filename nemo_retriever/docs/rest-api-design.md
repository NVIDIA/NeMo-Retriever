# NeMo Retriever REST API Specification

Base URL: `http://{host}:{port}` (default `http://0.0.0.0:7670`)

All request/response bodies are JSON unless otherwise noted (multipart for file uploads).
Errors follow a consistent shape: `{"detail": "<message>"}` with an appropriate HTTP status code.

---

## A. System

### `GET /health`

Health check for liveness/readiness probes.

**Response** `200`
```json
{"status": "ok"}
```

---

### `GET /version`

Return build version and metadata.

**Response** `200`
```json
{
  "version": "0.5.0",
  "git_sha": "abc1234",
  "build_date": "2026-03-31",
  "full_version": "0.5.0+abc1234"
}
```

**Maps to:** `nemo_retriever.version.get_version_info()`

---

### `GET /config`

Return the active ingest configuration (merged from YAML + defaults).

**Response** `200`
```json
{
  "source": "local",
  "path": "/home/user/ingest-config.yaml",
  "config": {
    "pdf": { "method": "pdfium", "dpi": 200 },
    "embedding": { "model_name": "nvidia/llama-nemotron-embed-1b-v2" },
    "..."
  }
}
```

**Maps to:** `nemo_retriever.ingest_config.load_ingest_config_file()`

---

## B. Documents

Unified document store for all supported file types. Replaces the separate file/pdf/video stores.

**Supported extensions:** `.pdf`, `.docx`, `.pptx`, `.txt`, `.html`, `.htm`, `.csv`, `.json`, `.jsonl`, `.md`, `.rst`, `.xml`, `.yaml`, `.yml`, `.xlsx`, `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`, `.svg`, `.mp3`, `.wav`, `.mp4`

### `POST /api/v1/documents`

Upload one or more documents.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `files` | file(s) | yes | One or more files to upload |
| `purpose` | string | no | Tag for the upload (default: `"ingest"`) |

**Response** `201`
```json
{
  "documents": [
    {
      "id": "doc-a1b2c3d4e5f6",
      "filename": "report.pdf",
      "content_type": "application/pdf",
      "bytes": 1048576,
      "purpose": "ingest",
      "num_pages": 42,
      "created_at": 1711843200,
      "status": "uploaded"
    }
  ]
}
```

**Maps to:** disk persistence + in-memory metadata store; `num_pages` populated via `pypdfium2` for PDFs.

---

### `GET /api/v1/documents`

List uploaded documents.

**Query parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `purpose` | string | — | Filter by purpose tag |
| `content_type` | string | — | Filter by MIME type (e.g. `application/pdf`) |
| `limit` | int | 100 | Max results |
| `offset` | int | 0 | Pagination offset |
| `order` | string | `desc` | Sort by `created_at` (`asc` or `desc`) |

**Response** `200`
```json
{
  "documents": [ "..." ],
  "total": 150,
  "limit": 100,
  "offset": 0
}
```

---

### `GET /api/v1/documents/{document_id}`

Retrieve document metadata.

**Response** `200` — Document object (same shape as in upload response).

**Response** `404` — `{"detail": "Document not found."}`

---

### `GET /api/v1/documents/{document_id}/content`

Download the raw file content.

**Response** `200` — binary stream with `Content-Disposition: attachment` and appropriate `Content-Type`.

---

### `DELETE /api/v1/documents/{document_id}`

Delete a document and its content from disk.

**Response** `200`
```json
{"id": "doc-a1b2c3d4e5f6", "deleted": true}
```

---

## C. Ingestion Pipeline

The core API: run the configurable multi-stage pipeline on documents. This is an async job-based model since pipelines can be long-running.

### Pipeline stages (in order)

1. **extract** — PDF text/structure extraction, page element detection, OCR
2. **dedup** — Duplicate image removal (content hash + bbox IoU)
3. **caption** — VLM-based image captioning
4. **split** — Text chunking by token count
5. **embed** — Text/image embedding via local model or NIM endpoint
6. **vdb_upload** — Upload embeddings to LanceDB

### `POST /api/v1/ingest`

Submit documents for ingestion through the pipeline.

**Request:** `application/json`
```json
{
  "document_ids": ["doc-a1b2c3d4e5f6"],
  "files_inline": false,

  "extract": {
    "enabled": true,
    "extract_text": true,
    "extract_images": true,
    "extract_tables": true,
    "extract_charts": true,
    "extract_infographics": true,
    "method": "pdfium",
    "dpi": 200,
    "use_table_structure": false,
    "table_output_format": "pseudo_markdown",
    "use_graphic_elements": false,
    "inference_batch_size": 8
  },

  "dedup": {
    "enabled": false,
    "content_hash": true,
    "bbox_iou": true,
    "iou_threshold": 0.45
  },

  "caption": {
    "enabled": false,
    "model_name": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
    "prompt": "Caption the content of this image:",
    "temperature": 1.0,
    "batch_size": 8
  },

  "split": {
    "enabled": false,
    "max_tokens": 1024,
    "overlap_tokens": 0,
    "tokenizer_model_id": null
  },

  "embed": {
    "enabled": true,
    "model_name": "nvidia/llama-nemotron-embed-1b-v2",
    "embedding_endpoint": null,
    "api_key": null,
    "embed_modality": "text",
    "embed_granularity": "element",
    "inference_batch_size": 32
  },

  "vdb_upload": {
    "enabled": true,
    "lancedb_uri": "lancedb",
    "table_name": "nv-ingest",
    "overwrite": true,
    "create_index": true,
    "hybrid": false
  },

  "save_to_disk": {
    "enabled": false,
    "output_directory": null
  }
}
```

All stage blocks are optional. Omitted stages use server defaults. Set `"enabled": false` to explicitly skip a stage.

**Alternative:** Submit files inline via multipart instead of referencing `document_ids`:

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `files` | file(s) | yes | Documents to ingest |
| `config` | string (JSON) | no | Pipeline configuration (same JSON shape as above minus `document_ids`) |

**Response** `202`
```json
{
  "job_id": "job-x9y8z7w6",
  "status": "queued",
  "num_documents": 3,
  "created_at": 1711843200
}
```

**Maps to:** `InProcessIngestor` — builds the task chain from `extract()`, `dedup()`, `caption()`, `split()`, `embed()`, `vdb_upload()`, then runs `ingest()`. Each stage maps to its corresponding `Params` class (`ExtractParams`, `DedupParams`, `CaptionParams`, `TextChunkParams`, `EmbedParams`, `VdbUploadParams`).

---

### `POST /api/v1/ingest/sync`

Synchronous variant — blocks until the pipeline completes and returns results directly. Intended for small document sets or testing.

**Request:** Same as `POST /api/v1/ingest` (both JSON and multipart variants).

**Response** `200`
```json
{
  "num_documents": 1,
  "elapsed_seconds": 12.34,
  "documents": [
    {
      "filename": "report.pdf",
      "num_pages": 5,
      "pages": [
        {
          "page_number": 1,
          "text": "Introduction to...",
          "tables": [
            {
              "bbox_xyxy_norm": [0.1, 0.2, 0.8, 0.9],
              "label_name": "table",
              "score": 0.97,
              "text": "| Col A | Col B |..."
            }
          ],
          "charts": [],
          "infographics": [],
          "elapsed_seconds": 2.1
        }
      ],
      "num_embeddings": 12,
      "vdb_uploaded": true
    }
  ]
}
```

**Maps to:** Same as async but calls `run_pipeline_tasks_on_df()` directly and returns results.

---

### `POST /api/v1/ingest/stream`

Streaming variant -- processes pages individually through the pipeline graph and streams each result back as NDJSON the moment it completes. PDF/DOCX/PPTX documents are pre-split into individual pages so results arrive progressively. VDB upload runs as a batch at the end.

**Request:** `multipart/form-data` (same as `POST /api/v1/ingest`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `files` | file(s) | yes | Documents to ingest |
| `config` | string (JSON) | no | Pipeline configuration (same shape as async variant) |

**Response** `200` -- `application/x-ndjson`

Each line is a JSON object. Regular result lines contain page data:
```json
{"page_number": 1, "_filename": "report.pdf", "text": "Introduction to...", "tables": [], "charts": [], "infographics": [], "_elapsed_seconds": 1.23}
```

Error lines have `"_error": true`:
```json
{"_error": true, "filename": "report.pdf", "page_number": 3, "detail": "OCR failed: timeout"}
```

The final line has `"_summary": true` with aggregate statistics:
```json
{"_summary": true, "num_documents": 1, "num_pages": 5, "elapsed_seconds": 12.34, "vdb_uploaded": true}
```

**Maps to:** `pdf_to_pages_df()` for pre-splitting, then `Graph.execute()` per page via `_build_serve_graph()`. VDB upload via `upload_embeddings_to_lancedb_inprocess()` on the accumulated DataFrame after all pages complete.

---

### `GET /api/v1/jobs/{job_id}`

Poll the status of an async ingestion job.

**Response** `200`
```json
{
  "job_id": "job-x9y8z7w6",
  "status": "running",
  "progress": {
    "documents_total": 3,
    "documents_completed": 1,
    "current_stage": "embed",
    "elapsed_seconds": 45.2
  },
  "created_at": 1711843200
}
```

Status values: `queued`, `running`, `completed`, `failed`, `cancelled`.

---

### `GET /api/v1/jobs/{job_id}/results`

Retrieve the results of a completed job.

**Response** `200` — Same shape as the sync ingest response body.

**Response** `409` — `{"detail": "Job not yet completed."}` if still running.

---

### `DELETE /api/v1/jobs/{job_id}`

Cancel a running job or delete a completed job's results.

**Response** `200`
```json
{"job_id": "job-x9y8z7w6", "cancelled": true}
```

---

### `GET /api/v1/jobs`

List all jobs.

**Query parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `status` | string | — | Filter by status |
| `limit` | int | 100 | Max results |
| `offset` | int | 0 | Pagination offset |

**Response** `200`
```json
{
  "jobs": [ "..." ],
  "total": 5,
  "limit": 100,
  "offset": 0
}
```

---

## D. Extraction (Standalone)

Standalone extraction endpoints for quick document processing without the full pipeline.

### `POST /api/v1/extract`

Extract text and structure from an uploaded document or file.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | conditional | Document to extract (or use `document_id`) |
| `document_id` | string | conditional | Reference to previously uploaded document |
| `extract_text` | bool | no | Default: `true` |
| `extract_tables` | bool | no | Default: `true` |
| `extract_charts` | bool | no | Default: `true` |
| `extract_infographics` | bool | no | Default: `true` |
| `method` | string | no | `"pdfium"` (default) or `"nemotron_parse"` |
| `dpi` | int | no | Render DPI for detection (default: `200`) |

**Response** `200`
```json
{
  "filename": "report.pdf",
  "method": "pdfium",
  "num_pages": 5,
  "elapsed_seconds": 3.21,
  "pages": [
    {
      "page_number": 1,
      "text": "Page text content...",
      "tables": [
        {
          "bbox_xyxy_norm": [0.1, 0.2, 0.8, 0.9],
          "label_name": "table",
          "score": 0.97,
          "text": "| Col A | Col B |..."
        }
      ],
      "charts": [],
      "infographics": [],
      "elapsed_seconds": 0.64
    }
  ]
}
```

**Maps to:** `pdf_extraction()`, `pdf_path_to_pages_df()` from `nemo_retriever.pdf`; for text/HTML: `TxtSplitActor`, `HtmlSplitActor`.

---

### `POST /api/v1/extract/stream`

Stream extraction results page-by-page as NDJSON. Useful for large PDFs where you want progressive results.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | yes | PDF file to stream |

**Response** `200` — `application/x-ndjson`

Each line:
```json
{"page": 1, "text": "Page 1 text...", "tables": [], "charts": [], "infographics": []}
```

**Maps to:** `_pdf_page_text_stream()` (enhanced to include structured content).

---

## E. Embeddings

### `POST /api/v1/embeddings`

Generate embeddings for text inputs. OpenAI-compatible.

**Request:** `application/json`
```json
{
  "input": ["text to embed", "another text"],
  "model": "nvidia/llama-nemotron-embed-1b-v2",
  "input_type": "passage",
  "encoding_format": "float",
  "dimensions": null
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input` | string or string[] | yes | Text(s) to embed |
| `model` | string | yes | Model ID |
| `input_type` | string | no | `"passage"` (default) or `"query"` |
| `encoding_format` | string | no | `"float"` (default) or `"base64"` |
| `dimensions` | int | no | Target dimensionality (model-dependent) |

**Response** `200`
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.0123, -0.0456, "..."],
      "index": 0
    }
  ],
  "model": "nvidia/llama-nemotron-embed-1b-v2",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

**Maps to:** `TextEmbedActor`, `embed_text_1b_v2()` from `nemo_retriever.text_embed`.

---

## F. Vector Store

Manage LanceDB tables and indexes.

### `GET /api/v1/vector-stores`

List available LanceDB tables.

**Response** `200`
```json
{
  "stores": [
    {
      "name": "nv-ingest",
      "uri": "lancedb",
      "num_rows": 1234,
      "has_index": true,
      "has_fts_index": false
    }
  ]
}
```

**Maps to:** `lancedb.connect()`, `db.table_names()`, `table.count_rows()`.

---

### `POST /api/v1/vector-stores`

Create a new LanceDB table with index configuration.

**Request:** `application/json`
```json
{
  "name": "my-collection",
  "uri": "lancedb",
  "index_type": "IVF_HNSW_SQ",
  "metric": "l2",
  "num_partitions": 16,
  "hybrid": false,
  "fts_language": "English"
}
```

**Response** `201`
```json
{
  "name": "my-collection",
  "uri": "lancedb",
  "created": true
}
```

**Maps to:** `lancedb_utils.py` — table creation and index building.

---

### `GET /api/v1/vector-stores/{name}`

Get details about a specific table.

**Response** `200`
```json
{
  "name": "nv-ingest",
  "uri": "lancedb",
  "num_rows": 1234,
  "has_index": true,
  "has_fts_index": false,
  "schema": {
    "columns": ["text", "vector", "metadata", "source", "page_number"]
  }
}
```

---

### `DELETE /api/v1/vector-stores/{name}`

Drop a LanceDB table.

**Response** `200`
```json
{"name": "nv-ingest", "deleted": true}
```

---

## G. Retrieval

### `POST /api/v1/retrieve`

Query the vector store and return ranked results with optional reranking.

**Request:** `application/json`
```json
{
  "query": "What is machine learning?",
  "queries": null,
  "top_k": 10,
  "lancedb_uri": null,
  "lancedb_table": null,
  "hybrid": false,
  "rerank": false,
  "reranker_model": "nvidia/llama-nemotron-rerank-1b-v2",
  "reranker_endpoint": null
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | conditional | Single query string (use `query` or `queries`) |
| `queries` | string[] | conditional | Multiple query strings for batch retrieval |
| `top_k` | int | no | Number of results per query (default: `10`) |
| `lancedb_uri` | string | no | Override the server's LanceDB URI |
| `lancedb_table` | string | no | Override the server's LanceDB table |
| `hybrid` | bool | no | Use hybrid search — vector + BM25 (default: `false`) |
| `rerank` | bool | no | Apply cross-encoder reranking (default: `false`) |
| `reranker_model` | string | no | Reranker model name |
| `reranker_endpoint` | string | no | Remote reranker endpoint URL |

**Response** `200`
```json
{
  "results": [
    {
      "query": "What is machine learning?",
      "hits": [
        {
          "text": "Machine learning is a subset of...",
          "source": "report.pdf",
          "page_number": 3,
          "score": 0.92,
          "_rerank_score": 0.87,
          "metadata": {}
        }
      ]
    }
  ]
}
```

**Maps to:** `Retriever.query()` / `Retriever.queries()` from `nemo_retriever.retriever`. Reranking via `nemo_retriever.rerank.rerank_hits()`.

---

## H. Models

### `GET /api/v1/models`

List available models from the HuggingFace model registry.

**Response** `200`
```json
{
  "object": "list",
  "data": [
    {
      "id": "nvidia/llama-nemotron-embed-1b-v2",
      "object": "model",
      "created": 1700000000,
      "owned_by": "nvidia",
      "url": "https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2"
    }
  ]
}
```

**Maps to:** `nemo_retriever.utils.hf_model_registry.HF_MODEL_REGISTRY`

---

### `GET /api/v1/models/{model_id}`

Get details about a specific model.

**Response** `200` — Single model object.

**Response** `404` — `{"detail": "Model not found."}`

---

## I. Output Rendering

### `POST /api/v1/render/markdown`

Convert ingestion/extraction results into a markdown document.

**Request:** `application/json`
```json
{
  "job_id": "job-x9y8z7w6",
  "by_page": false
}
```

Or inline results:
```json
{
  "results": [ { "page_number": 1, "text": "...", "tables": [] } ],
  "by_page": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `job_id` | string | conditional | Reference a completed job's results |
| `results` | object | conditional | Inline result records to render |
| `by_page` | bool | no | If `true`, return per-page markdown (default: `false`) |

**Response** `200` (when `by_page` is `false`)
```json
{
  "markdown": "# Extracted Content\n\n## Page 1\n\nIntroduction to..."
}
```

**Response** `200` (when `by_page` is `true`)
```json
{
  "pages": {
    "1": "## Page 1\n\nIntroduction to...",
    "2": "## Page 2\n\nResults show..."
  }
}
```

**Maps to:** `nemo_retriever.io.markdown.to_markdown()`, `to_markdown_by_page()`

---

## Endpoint Summary

| Method | Path | Description | Category |
|--------|------|-------------|----------|
| `GET` | `/health` | Health check | System |
| `GET` | `/version` | Build version info | System |
| `GET` | `/config` | Active ingest config | System |
| `POST` | `/api/v1/documents` | Upload documents | Documents |
| `GET` | `/api/v1/documents` | List documents | Documents |
| `GET` | `/api/v1/documents/{id}` | Get document metadata | Documents |
| `GET` | `/api/v1/documents/{id}/content` | Download document | Documents |
| `DELETE` | `/api/v1/documents/{id}` | Delete document | Documents |
| `POST` | `/api/v1/ingest` | Async pipeline ingestion | Ingestion |
| `POST` | `/api/v1/ingest/stream` | Streaming page-by-page ingestion (NDJSON) | Ingestion |
| `POST` | `/api/v1/ingest/sync` | Sync pipeline ingestion | Ingestion |
| `GET` | `/api/v1/jobs` | List jobs | Ingestion |
| `GET` | `/api/v1/jobs/{id}` | Get job status | Ingestion |
| `GET` | `/api/v1/jobs/{id}/results` | Get job results | Ingestion |
| `DELETE` | `/api/v1/jobs/{id}` | Cancel/delete job | Ingestion |
| `POST` | `/api/v1/extract` | Standalone extraction | Extraction |
| `POST` | `/api/v1/extract/stream` | Streaming extraction | Extraction |
| `POST` | `/api/v1/embeddings` | Generate embeddings | Embeddings |
| `GET` | `/api/v1/vector-stores` | List vector stores | Vector Store |
| `POST` | `/api/v1/vector-stores` | Create vector store | Vector Store |
| `GET` | `/api/v1/vector-stores/{name}` | Get vector store info | Vector Store |
| `DELETE` | `/api/v1/vector-stores/{name}` | Delete vector store | Vector Store |
| `POST` | `/api/v1/retrieve` | Query with retrieval | Retrieval |
| `GET` | `/api/v1/models` | List models | Models |
| `GET` | `/api/v1/models/{id}` | Get model info | Models |
| `POST` | `/api/v1/render/markdown` | Render results as markdown | Rendering |

---

## Internal Module Mapping

| API Area | Primary Modules |
|----------|-----------------|
| System | `version.py`, `ingest_config.py` |
| Documents | `adapters/service/app.py` (new document store) |
| Ingestion | `ingest_modes/inprocess.py` (`InProcessIngestor`, `run_pipeline_tasks_on_df`, `get_pipeline_tasks`) |
| Extraction | `pdf/extract.py` (`pdf_extraction`), `pdf/split.py` (`pdf_path_to_pages_df`), `txt/`, `html/` |
| Embeddings | `text_embed/` (`TextEmbedActor`, `embed_text_1b_v2`) |
| Vector Store | `ingest_modes/lancedb_utils.py`, `vector_store/lancedb_store.py` |
| Retrieval | `retriever.py` (`Retriever`), `rerank/rerank.py` |
| Models | `utils/hf_model_registry.py` |
| Rendering | `io/markdown.py` (`to_markdown`, `to_markdown_by_page`) |

---

## Configuration Params to Pydantic Model Mapping

| API Stage Config | Pydantic Model | Location |
|-----------------|----------------|----------|
| `extract` | `ExtractParams` | `params/models.py` |
| `dedup` | `DedupParams` | `params/models.py` |
| `caption` | `CaptionParams` | `params/models.py` |
| `split` | `TextChunkParams` | `params/models.py` |
| `embed` | `EmbedParams` | `params/models.py` |
| `vdb_upload` | `VdbUploadParams` + `LanceDbParams` | `params/models.py` |

---

## Authentication

API key authentication is passed through to downstream NIM services. The REST API itself does not enforce authentication but accepts:

- `Authorization: Bearer <token>` header — forwarded as `api_key` to NIM endpoints
- Per-request `api_key` fields in embed/extract configs — override the header

---

## Error Responses

All errors follow:

```json
{
  "detail": "Human-readable error message."
}
```

| Status | Meaning |
|--------|---------|
| `400` | Bad request (invalid params, missing fields) |
| `404` | Resource not found |
| `409` | Conflict (e.g. job not completed) |
| `413` | Payload too large |
| `422` | Validation error (Pydantic) |
| `500` | Internal server error |
| `503` | Backend unavailable (model not loaded, service down) |
