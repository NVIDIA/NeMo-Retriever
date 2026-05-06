# Service Module Architecture

This document describes the internal architecture of the `nemo_retriever.service` module — the FastAPI-based service that provides HTTP-driven document ingestion, vector search, and reranking.

---

## Table of Contents

- [High-Level Overview](#high-level-overview)
- [Directory Layout](#directory-layout)
- [Request Lifecycle](#request-lifecycle)
  - [Ingest Flow](#ingest-flow)
  - [Query Flow](#query-flow)
  - [Rerank Flow](#rerank-flow)
- [Core Subsystems](#core-subsystems)
  - [Application Factory & Lifespan](#application-factory--lifespan)
  - [Configuration](#configuration)
  - [Processing Pool](#processing-pool)
  - [Database Layer](#database-layer)
  - [Spool (Durability)](#spool-durability)
  - [Event Bus & SSE Streaming](#event-bus--sse-streaming)
  - [Authentication](#authentication)
  - [Metrics](#metrics)
  - [Failure Classification](#failure-classification)
  - [Event Log (Provenance)](#event-log-provenance)
- [Design Decisions](#design-decisions)
- [Client Library](#client-library)
- [Developer Guide: Extending the Service](#developer-guide-extending-the-service)
  - [Adding a New Router](#adding-a-new-router)
  - [Adding a New NIM Endpoint](#adding-a-new-nim-endpoint)
  - [Adding a New Pydantic Model](#adding-a-new-pydantic-model)
  - [Adding a New Configuration Section](#adding-a-new-configuration-section)
  - [Adding a Capability Flag](#adding-a-capability-flag)
  - [Adding a Pipeline Stage](#adding-a-pipeline-stage)
  - [Adding a New SSE Event Type](#adding-a-new-sse-event-type)
  - [Recording a Provenance Event](#recording-a-provenance-event)

---

## High-Level Overview

```
                 ┌──────────────────────────────────────────────────┐
                 │              FastAPI Application                  │
                 │                                                  │
   HTTP ────────►│  Routers ─────► Processing Pool ──► Workers      │
   requests      │    │                  │                          │
                 │    │            Batch Buffer                     │
                 │    │                  │                          │
                 │    ▼                  ▼                          │
                 │  SQLite DB     Worker Processes                  │
                 │    │           (operator chains)                 │
                 │    │                  │                          │
                 │    │                  ▼                          │
                 │    │            NIM Endpoints                    │
                 │    │          (OCR, embed, etc.)                 │
                 │    │                  │                          │
                 │    │                  ▼                          │
                 │    │            LanceDB Write                   │
                 │    │          (append embeddings)                │
                 │    │                  │                          │
                 │    ▼                  ▼                          │
                 │  Event Bus ◄──── Results ──────► SQLite DB       │
                 │    │                                             │
                 │    ▼                                             │
                 │  SSE Stream ──────────────────────► Clients      │
                 │                                                  │
                 │  /v1/query  ──► Embed NIM ──► LanceDB Search    │
                 └──────────────────────────────────────────────────┘
```

The service operates as a single-process asyncio application (the "main process") that manages a pool of worker subprocesses via `ProcessPoolExecutor`. Each worker builds its own operator chain at startup and processes document pages in batches. Results are written directly to SQLite by workers and published as SSE events by the main process.

After the embedding stage, each worker appends the computed vectors to a shared **LanceDB** table (`nv-ingest` at the configured URI). This ensures embeddings are immediately searchable by the `/v1/query` endpoint without a separate ETL step.

Two orthogonal features — **vector search** (`/v1/query`) and **reranking** (`/v1/rerank`) — run synchronously in the main process via `asyncio.to_thread`, delegating to external NIM endpoints. They do not use the processing pool. The query endpoint reads from the same LanceDB table that the ingest workers write to.

A cross-cutting **event log** (provenance subsystem) records every significant event — errors, recoveries, cancellations, and successful page completions — into an append-only `event_log` SQLite table. This data is exposed via `GET /v1/events` and is designed to drive a future auditing and provenance UI.

---

## Directory Layout

```
service/
├── ARCHITECTURE.md          # This file
├── __init__.py
├── app.py                   # FastAPI application factory + lifespan
├── auth.py                  # Bearer-token authentication middleware
├── cli.py                   # Typer CLI: `retriever service start/ingest`
├── client.py                # Async client for ingesting documents
├── config.py                # Pydantic configuration models + YAML loader
├── event_bus.py             # In-memory pub/sub for SSE streaming
├── event_logger.py          # record_event() helper for provenance logging
├── failure_types.py         # FailureType + EventCategory enums + classifier
├── metrics.py               # Prometheus instrumentation + gauge refresh
├── retriever-service.yaml   # Bundled default configuration
├── spool.py                 # Durable page spool (write-ahead)
├── db/
│   ├── __init__.py
│   ├── engine.py            # Thread-local SQLite connections + DDL
│   └── repository.py        # CRUD operations for all tables
├── models/
│   ├── __init__.py
│   ├── document.py          # Document ORM model + ProcessingStatus enum
│   ├── event_log.py         # EventRecord Pydantic model + severity/outcome enums
│   ├── job.py               # Job ORM model
│   ├── metrics.py           # ProcessingMetric ORM model
│   ├── page_processing_log.py  # Per-page audit log model
│   ├── page_result.py       # PageResult ORM model
│   ├── requests.py          # Pydantic request models (API input)
│   └── responses.py         # Pydantic response models (API output)
├── processing/
│   ├── __init__.py
│   └── pool.py              # ProcessPoolExecutor + batch buffer
└── routers/
    ├── __init__.py
    ├── events.py            # GET /v1/events (provenance event log)
    ├── ingest.py            # POST /v1/ingest, job management, batch upload
    ├── internal.py          # Internal/admin endpoints
    ├── metrics.py           # GET /v1/ingest_metrics
    ├── query.py             # POST /v1/query (vector search)
    ├── rerank.py            # POST /v1/rerank (cross-encoder reranking)
    ├── stream.py            # SSE streaming endpoints
    └── system.py            # GET /v1/health, GET /v1/capabilities
```

---

## Request Lifecycle

### Ingest Flow

This is the primary flow — uploading documents for extraction and embedding.

```
Client                     Service                    Worker Process
  │                          │                              │
  │  POST /v1/ingest         │                              │
  │  (multipart: file+meta)  │                              │
  │─────────────────────────►│                              │
  │                          │  1. Parse multipart          │
  │                          │  2. SHA-256 content hash     │
  │                          │  3. Spool to disk (fsync)    │
  │                          │  4. Insert document row      │
  │                          │  5. Enqueue into BatchBuffer │
  │  202 Accepted ◄──────────│                              │
  │                          │                              │
  │                          │  When batch full OR timeout: │
  │                          │  Submit to ProcessPoolExecutor
  │                          │─────────────────────────────►│
  │                          │                              │  6. Read bytes from spool
  │                          │                              │  7. Build DataFrame
  │                          │                              │  8. Run operator chain
  │                          │                              │     (PDF parse → OCR →
  │                          │                              │      table detect →
  │                          │                              │      embed →
  │                          │                              │      LanceDB write → ...)
  │                          │                              │  9. Write results to SQLite
  │                          │  BatchWorkerResult ◄─────────│
  │                          │                              │
  │                          │  10. Publish SSE events      │
  │  SSE: page_complete ◄────│                              │
  │  SSE: document_complete ◄│                              │
  │  SSE: job_complete ◄─────│                              │
```

**Key design points:**

- **Accept-before-process**: The endpoint returns 202 immediately after spooling to disk. Processing is fully asynchronous.
- **Batch dispatch**: Pages accumulate in a `_BatchBuffer` (default 32) and flush either when full or after a timeout (default 2s). This ensures NIM endpoints receive efficient batches.
- **Worker isolation**: Each worker is a separate OS process with its own operator chain. No shared mutable state, no C-library thread safety concerns.
- **Dual-write**: Workers write results to SQLite directly. The main process receives a lightweight `BatchWorkerResult` for SSE event publishing only.

### Query Flow

Vector search — synchronous, does not use the processing pool.

```
Client                     Service
  │                          │
  │  POST /v1/query          │
  │  {"query": "...",        │
  │   "top_k": 10}           │
  │─────────────────────────►│
  │                          │  1. Validate config (embed endpoint)
  │                          │  2. Call embedding NIM (asyncio.to_thread)
  │                          │  3. Search LanceDB (asyncio.to_thread)
  │                          │  4. Build QueryResponse
  │  200 QueryResponse ◄─────│
```

The query router accepts single or batch queries (`str | list[str]`), with optional hybrid search (vector + BM25 via LanceDB's `RRFReranker`).

### Rerank Flow

Cross-encoder reranking — also synchronous, delegates to the reranker NIM.

```
Client                     Service
  │                          │
  │  POST /v1/rerank         │
  │  {"query": "...",        │
  │   "passages": [...]}     │
  │─────────────────────────►│
  │                          │  1. Validate config (rerank endpoint)
  │                          │  2. Extract text from passages
  │                          │  3. Call reranker NIM (asyncio.to_thread)
  │                          │  4. Sort by score, apply top_n
  │                          │  5. Build RerankResponse
  │  200 RerankResponse ◄────│
```

The rerank endpoint accepts passages as arbitrary dicts (must contain `text`). All extra keys survive the round-trip, so clients can send full hit dicts from `/v1/query` and get them back with `rerank_score` added.

---

## Core Subsystems

### Application Factory & Lifespan

**File:** `app.py`

`create_app(config)` builds the FastAPI application:

1. Configures root logging (console + rotating file)
2. Applies resource limits (CPU affinity, memory rlimit, CUDA devices)
3. Attaches `_RequestIdMiddleware` (assigns a UUID to `request.state.request_id` on every HTTP request for provenance correlation)
4. Conditionally attaches `BearerAuthMiddleware`
5. Registers all routers under `/v1`
6. Sets up Prometheus instrumentation

The `_lifespan` async context manager handles startup and shutdown:

- **Startup**: initializes SQLite, creates `EventBus`, optionally creates `SpoolStore`, starts `ProcessingPool`, recovers spooled pages, starts metrics refresh loop.
- **Shutdown**: cancels metrics task, drains pool (with configurable timeout), shuts down pool, closes DB.

### Configuration

**File:** `config.py`

All configuration is Pydantic-validated with `extra="forbid"` to catch typos. The top-level `ServiceConfig` composes:

| Section | Class | Purpose |
|---------|-------|---------|
| `server` | `ServerConfig` | Host/port bindings |
| `logging` | `LoggingConfig` | Level, file, format |
| `database` | `DatabaseConfig` | SQLite path |
| `processing` | `ProcessingConfig` | Workers, batch size, timeout |
| `nim_endpoints` | `NimEndpointsConfig` | NIM URLs (comma-separated for multi-NIM) |
| `resources` | `ResourceLimitsConfig` | Memory, CPU, GPU constraints |
| `auth` | `AuthConfig` | Bearer token + bypass paths |
| `drain` | `DrainConfig` | Graceful shutdown timeout |
| `spool` | `SpoolConfig` | Durable page spool settings |
| `event_bus` | `EventBusConfig` | SSE overflow policy + queue sizing |
| `vector_store` | `VectorStoreConfig` | LanceDB URI, table, top_k, embedding model |
| `reranker` | `RerankerConfig` | Reranker model name, default top_n |

**Config precedence** (highest to lowest):
1. CLI flags (`--port`, `--embed-url`, etc.)
2. `./retriever-service.yaml` in the working directory
3. Bundled default `retriever-service.yaml` in the package

`load_config()` merges YAML and CLI overrides using dotted-key notation (e.g., `server.port`).

### Processing Pool

**File:** `processing/pool.py`

The pool has three main components:

**`ProcessingPool`** — main-process coordinator:
- Owns the `ProcessPoolExecutor` (spawned, not forked, for C-library safety)
- Distributes comma-separated NIM URLs round-robin across workers
- Manages drain/cancel state
- Publishes SSE events when workers complete

**`_BatchBuffer`** — thread-safe page accumulator:
- Accepts individual pages from the ingest router
- Auto-flushes when reaching `batch_size` or after `timeout_s`
- Enforces a hard cap (`num_workers * batch_size`) to bound memory
- Returns `False` when full, causing the ingest router to 503

**Worker processes** — isolated operator chains:
- Each worker calls `_worker_initializer` once, which builds the full `nv-ingest` operator chain
- `_run_pipeline_batch` receives page descriptors, builds a DataFrame, runs the chain, and writes results to SQLite
- Provenance columns (`_page_document_id`, etc.) track which output rows belong to which input page through content-explosion stages

**LanceDB write stage** (`LanceDBWriteOperator`):
- Appended to the operator chain after the embedding stage when `embed_invoke_url` is configured
- Each batch's embedding rows are appended (not overwritten) to the shared LanceDB table
- The table is created lazily on the first write; subsequent batches open and append
- Multiple worker processes can write concurrently (LanceDB uses file-level locking)
- The operator uses `build_lancedb_rows()` from `vector_store/lancedb_utils.py` for consistent row format across all ingest paths

**Multi-NIM load balancing:** When a NIM URL field contains commas (e.g., `ocr_invoke_url: "http://nim1:8000,http://nim2:8000"`), workers are assigned URLs round-robin so each worker talks to exactly one endpoint per NIM type.

### Database Layer

**Files:** `db/engine.py`, `db/repository.py`

**`DatabaseEngine`** manages thread-local SQLite connections:
- WAL journal mode for concurrent multi-process writes
- 60-second busy timeout to handle write contention
- Application-level retry with exponential backoff via `execute_with_retry`
- Additive schema migrations (`_safe_add_column`) so upgrades don't require dropping the DB

**Schema** (6 tables):

| Table | Purpose |
|-------|---------|
| `jobs` | One row per uploaded file. Tracks total pages, completion count. |
| `documents` | One row per page. Links to parent job, stores spool path. |
| `page_results` | Pipeline output rows (JSON content), keyed by document. |
| `processing_metrics` | Per-model detection/invocation counts per document. |
| `page_processing_log` | Audit trail: timing, failure type, error messages. |
| `event_log` | Full provenance event stream (errors, recoveries, lifecycle). |

**`Repository`** provides typed CRUD operations. All write methods wrap with `execute_with_retry` to transparently handle `OperationalError: database is locked`.

### Spool (Durability)

**File:** `spool.py`

The spool closes the gap between "page accepted (202)" and "page dispatched to a worker". Without it, an OOM kill or pod restart loses every accepted-but-unprocessed page.

**Write path:**
```
1. Accept multipart upload
2. SpoolStore.write(sha256, bytes)     # atomic: write → fsync → rename
3. INSERT document row (spool_path=...)
4. Return 202
```

**Recovery path (startup):**
```
1. Query documents WHERE status IN ('queued', 'processing') AND spool_path IS NOT NULL
2. For each: read spool file → pool.try_submit()
```

**Cleanup:**
A background asyncio task periodically queries terminal documents (complete/failed/cancelled), unlinks their spool files, and clears `spool_path`.

**Layout:** `<root>/<sha[0:2]>/<sha[2:4]>/<sha>.bin` — sharded so no directory exceeds ~256 files.

### Event Bus & SSE Streaming

**File:** `event_bus.py`

The `EventBus` is a per-key fan-out pub/sub system. Keys are `document_id` or `job_id` strings. Events published under a key are delivered to every subscription for that key.

**Backpressure model** (two layered defenses):

1. **Per-subscription event-type filter** — subscribers declare which event types they care about. Filtered events never touch the queue.

2. **Priority-drop load shedding** — when the queue fills past a watermark (75%), low-priority events (`page_complete`) are silently dropped. Terminal events (`job_complete`, `document_complete`) are always preserved. If a terminal event hits a full queue, the subscription is drained and a sentinel is injected.

**Overflow policies** (configurable):

| Policy | Behavior |
|--------|----------|
| `drop_low_priority` (default) | Synchronous, never blocks. Sheds low-priority events. |
| `backpressure` | Async `await Queue.put()` with timeout. Back-pressures the worker pool. |
| `block` | Same as backpressure but waits forever. Zero loss, unbounded latency. |

**Replay buffer:** Every event gets a monotonic `seq` number and is stored in a per-key ring buffer (default 1024 events). Reconnecting clients send `Last-Event-ID` and replay missed events.

### Authentication

**File:** `auth.py`

`BearerAuthMiddleware` is a Starlette middleware that:
- Compares the incoming `Authorization` header against the configured token using `hmac.compare_digest` (constant-time)
- Bypasses configured paths (default: `/v1/health`, `/docs`, `/openapi.json`, `/redoc`) so Kubernetes probes and API docs work without tokens
- Is only attached when `auth.api_token` is non-empty — zero overhead when auth is disabled

### Metrics

**File:** `metrics.py`

Two layers of Prometheus metrics at `GET /metrics`:

1. **HTTP-level** (via `prometheus-fastapi-instrumentator`): request totals, latency histograms, in-progress counts by route and status code.

2. **Service-specific gauges** (refreshed every 15s by a background task):
   - `nemo_retriever_pool_workers` — configured worker count
   - `nemo_retriever_pool_capacity_pages` — free batch buffer slots
   - `nemo_retriever_pool_buffered_pages` — pages waiting for dispatch
   - `nemo_retriever_pool_in_flight_batches` — batches currently executing
   - `nemo_retriever_pool_draining` — 1 during graceful shutdown
   - `nemo_retriever_jobs{status=...}` — job counts by status

### Failure Classification

**File:** `failure_types.py`

The `FailureType` enum classifies per-page failures into actionable categories:

| Type | Meaning | Retryable? |
|------|---------|------------|
| `pdf_parse` | Corrupt/password-protected PDF | No |
| `nim_timeout` | NIM endpoint timed out | Yes |
| `nim_5xx` | NIM returned server error | Yes |
| `nim_4xx` | NIM returned client error | No (check input) |
| `oom` | Out of memory | Maybe (reduce batch) |
| `internal` | Pipeline bug | No (file an issue) |
| `cancelled` | Cancelled before processing | N/A |
| `unknown` | Unrecognized exception | Depends |

`categorize_exception(exc)` inspects the exception type and message string to map real exceptions to these categories.

The broader `EventCategory` enum is a superset of `FailureType` used by the event log. It adds categories for every NIM endpoint type (`page_elements`, `ocr`, `table_structure`, `graphic_elements`, `embed`, `rerank`), infrastructure categories (`lancedb`, `spool`, `auth`, `validation`, `dedup`, `html`), and includes a `from_failure_type()` class method for backward-compatible mapping.

### Event Log (Provenance)

**Files:** `event_logger.py`, `models/event_log.py`, `routers/events.py`

The event log is an append-only provenance subsystem that captures every significant event across all service endpoints. It is designed to eventually drive an auditing and provenance UI.

**Data model** (`EventRecord`, a Pydantic `BaseModel` mirroring the `event_log` table):

| Column | Type | Purpose |
|--------|------|---------|
| `id` | TEXT PK | UUID, auto-generated |
| `timestamp` | TEXT | ISO-8601 UTC instant when the event occurred |
| `job_id` | TEXT FK | Links to the parent job (nullable) |
| `document_id` | TEXT FK | Links to the document/page (nullable) |
| `source_file` | TEXT | Human-readable filename |
| `page_number` | INTEGER | 1-based input page number (nullable) |
| `category` | TEXT | `EventCategory` value (e.g., `embed`, `lancedb`, `spool`, `dedup`) |
| `severity` | TEXT | `info`, `warning`, or `error` |
| `outcome` | TEXT | `failed`, `recovered`, or `in_progress` |
| `stage` | TEXT | Pipeline stage or endpoint step name |
| `summary` | TEXT | One-line human-readable description |
| `detail` | TEXT | Full error message or extended context |
| `stack_trace` | TEXT | Python traceback string |
| `endpoint` | TEXT | Originating path (e.g., `/v1/query`, `pipeline`) |
| `request_id` | TEXT | Correlates events from the same HTTP request |
| `extra_json` | TEXT | Extensible JSON blob for future metadata |
| `created_at` | TEXT | Row insertion timestamp |

**Key enums:**
- `EventSeverity` — `info` (lifecycle markers), `warning` (transient/recoverable), `error` (permanent failures)
- `EventOutcome` — `failed` (not retried), `recovered` (transient error, system continued), `in_progress` (normal lifecycle event)

**Instrumentation:** All instrumentation flows through a single entry-point, `record_event()` in `event_logger.py`. This function constructs an `EventRecord`, persists it via `Repository.insert_event()`, and returns the record. Call sites include:

- **Processing pool** (`pool.py`) — spool-file-missing, no-page-bytes, batch pipeline exception (with full stack trace), provenance column lost, page-produced-no-output, per-page success, page cancellation
- **Query router** (`query.py`) — embedding NIM failure, LanceDB search failure
- **Rerank router** (`rerank.py`) — validation errors, reranker NIM failure
- **Ingest router** (`ingest.py`) — server draining/busy (503), dedup detection

**Request ID correlation:** The `_RequestIdMiddleware` in `app.py` assigns a UUID to `request.state.request_id` on every HTTP request. Routers pass this to `record_event()`, allowing the UI to group all events from a single API call. Worker processes generate their own batch-level correlation IDs.

**REST API** (`routers/events.py`):

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/v1/events` | Paginated listing with filters (job_id, document_id, category, severity, outcome, since, until) |
| GET | `/v1/events/summary` | Aggregated counts by category (for dashboard widgets) |
| GET | `/v1/events/{event_id}` | Single event detail |
| PATCH | `/v1/events/{event_id}` | Update an event's outcome (e.g. mark as `recovered` / acknowledged) |
| DELETE | `/v1/events/{event_id}` | Delete a single event |
| POST | `/v1/events/acknowledge` | Bulk-acknowledge events (set outcome on many IDs at once) |
| POST | `/v1/events/delete` | Bulk-delete events by IDs or by filter (category, severity, outcome) |

The full CRUD surface allows programmatic triage: an error-reporting CLI or AI agent can fetch events, acknowledge resolved issues, and purge stale entries without manual DB access.

**Request/response models for mutations:**

- `EventUpdateRequest` — body for `PATCH`, contains a single `outcome` field.
- `EventBulkAcknowledgeRequest` — body for `POST /acknowledge`, accepts `event_ids` (list) and `outcome` (default `recovered`).
- `EventBulkDeleteRequest` — body for `POST /delete`, accepts `event_ids` (list) or filter fields (`category`, `severity`, `outcome`). At least one filter is required to prevent accidental blanket deletes.
- `EventMutationResponse` — returned by all mutation endpoints, reports `affected` (count) and `action` (`updated` or `deleted`).

**Design for extensibility:** The `extra_json` column is the primary extension point. Future provenance metadata (operator timings, retry counts, GPU memory snapshots, downstream lineage links) can be added without schema migrations. The `EventCategory` enum can be extended with new values as new pipeline stages or endpoints are added.

---

## Design Decisions

### Why `ProcessPoolExecutor` instead of threads?

The document processing pipeline uses C-extension libraries (pypdfium2, image processing) that are not thread-safe and hold the GIL during CPU-intensive operations. Separate processes eliminate both problems:
- Each process has its own address space — no C-library thread safety issues
- True CPU parallelism for image processing and model inference
- Clean isolation: a segfault in one worker doesn't crash the service

### Why batch buffering?

NIM endpoints and GPU models are most efficient with larger batches. The `_BatchBuffer` accumulates pages and dispatches them as a single DataFrame, so downstream models see 32 samples instead of 1. The configurable timeout ensures latency stays bounded for small jobs.

### Why SQLite instead of Postgres?

The service targets single-node deployments (one pod with a PV). SQLite's WAL mode handles concurrent writes from multiple worker processes with zero operational overhead. The busy-timeout + retry pattern makes write contention transparent. For multi-node deployments, the DB layer could be swapped via the `Repository` abstraction.

### Why spool-then-accept?

The 202 response is a contract: "your work will not be lost." Without the spool, that contract is broken by any crash between accept and processing. The fsync + atomic rename pattern guarantees the page bytes survive any failure mode short of disk corruption.

### Why drop-on-overflow instead of blocking?

The default `drop_low_priority` policy ensures the worker callback thread (which publishes SSE events) never blocks. A blocked worker thread back-pressures the `ProcessPoolExecutor`, which cascades to 503s on the ingest endpoint. The dropped events (`page_complete`) are reconstructible from REST endpoints, so the client loses observability but not correctness.

### Why `asyncio.to_thread` for query/rerank?

The embedding NIM call and LanceDB search are blocking I/O operations. Running them in `asyncio.to_thread` keeps the event loop responsive for SSE streaming and health checks, without the complexity of a worker process for what is a simple RPC + query.

### Why a separate `event_log` table alongside `page_processing_log`?

`page_processing_log` records one row per page at terminal state (complete or failed). It is a concise audit trail optimized for per-page status queries. `event_log` is a broader provenance stream: it captures transient errors that recovered, lifecycle markers, query/rerank endpoint failures, dedup detections, server-busy rejections, and per-stage timing — none of which belong in the page log. The two tables serve different audiences: `page_processing_log` answers "did this page succeed?", while `event_log` answers "what happened to this job across all subsystems and why?"

---

## Client Library

**File:** `client.py`

`RetrieverServiceClient` is a full-featured async client that:

1. **Uploads whole documents** to `POST /v1/ingest/job` — the server splits PDFs internally
2. **Uploads concurrently** (configurable concurrency, default 8)
3. **Opens SSE in parallel** with uploads — no events are missed
4. **Handles back-pressure** — retries on 503 with exponential backoff, respects `Retry-After` header
5. **Reconnects on transport failure** — TCP resets (common on Kubernetes NodePort) trigger automatic reconnection with `Last-Event-ID`
6. **Rich live display** — compact progress bars, active job table, detection metrics
7. **Streaming generator** — `aingest_documents_stream()` yields per-document results when each job completes (full results fetched from `GET /v1/ingest/job/{job_id}/results`)

The CLI command `retriever service ingest` wraps this client:

```bash
retriever service ingest doc1.pdf doc2.pdf \
    --server-url http://localhost:7670 \
    --concurrency 16 \
    --api-token $TOKEN
```

---

## Developer Guide: Extending the Service

### Adding a New Router

1. Create `routers/my_feature.py`:

```python
from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, Request

from nemo_retriever.service.config import ServiceConfig

logger = logging.getLogger(__name__)
router = APIRouter(tags=["my_feature"])


@router.post("/my_feature", summary="Do something useful")
async def my_feature(request: Request) -> dict:
    config: ServiceConfig = request.app.state.config
    # Access shared state via request.app.state:
    #   config, repository, processing_pool, event_bus, spool_store
    return {"status": "ok"}
```

2. Register in `app.py`:

```python
from nemo_retriever.service.routers import ..., my_feature

app.include_router(my_feature.router, prefix="/v1")
```

**Convention:** All NIM-calling routers use `asyncio.to_thread` for blocking operations, catch specific exception types, and return descriptive `HTTPException` details with 400/503 status codes.

### Adding a New NIM Endpoint

1. Add the URL field to `NimEndpointsConfig` in `config.py`:

```python
class NimEndpointsConfig(BaseModel):
    # ... existing fields ...
    my_nim_invoke_url: str | None = None
```

2. Add a CLI flag in `cli.py` if desired:

```python
my_nim_url: Optional[str] = typer.Option(None, "--my-nim-url", help="...")
# Then in the overrides dict:
if my_nim_url is not None:
    overrides["nim_endpoints.my_nim_invoke_url"] = my_nim_url
```

3. Add the NIM URL field to the `_NIM_URL_FIELDS` tuple in `processing/pool.py` if it should be round-robin distributed across workers:

```python
_NIM_URL_FIELDS = (
    # ... existing ...
    "my_nim_invoke_url",
)
```

### Adding a New Pydantic Model

**Request models** go in `models/requests.py`:

```python
class MyRequest(BaseModel):
    """JSON body for ``POST /v1/my_feature``."""
    field: str = Field(..., description="Required field.")
    optional_field: int | None = Field(default=None, ge=1, le=100)
```

**Response models** go in `models/responses.py`:

```python
class MyResponse(BaseModel):
    """Response for ``POST /v1/my_feature``."""
    result: str
    count: int = Field(description="How many items were processed.")
```

**DB/ORM models** go in their own file under `models/` (e.g., `models/my_entity.py`) and follow the `to_row()` / `from_row()` pattern used by `Document` and `Job`.

### Adding a New Configuration Section

1. Define the Pydantic model in `config.py`:

```python
class MyFeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    threshold: float = 0.5
```

2. Add it to `ServiceConfig`:

```python
class ServiceConfig(BaseModel):
    # ... existing fields ...
    my_feature: MyFeatureConfig = Field(default_factory=MyFeatureConfig)
```

3. Access it from any router via `request.app.state.config.my_feature`.

4. Override from YAML:

```yaml
my_feature:
  enabled: true
  threshold: 0.8
```

### Adding a Capability Flag

The capabilities endpoint lets clients introspect what the server supports.

1. Add the field to `CapabilitiesResponse` in `models/responses.py`:

```python
class CapabilitiesResponse(BaseModel):
    # ... existing fields ...
    my_feature: CapabilityFlag
```

2. Wire it in `routers/system.py`:

```python
return CapabilitiesResponse(
    # ... existing ...
    my_feature=_flag_for(nim.my_nim_invoke_url),
)
```

### Adding a Pipeline Stage

Pipeline stages (operators) live outside this module in `nemo_retriever.graph`, but they are wired into the service via the processing pool.

1. Implement the operator (subclass of `AbstractOperator` with CPU/GPU variants)
2. Register it in the graph builder (`nemo_retriever.graph.ingestor_runtime.build_graph`)
3. If it requires a NIM endpoint, add the URL field per the [Adding a New NIM Endpoint](#adding-a-new-nim-endpoint) guide
4. The pool's `_build_operator_chain` will automatically pick it up via `build_graph` → `resolve_graph` → `_linearize`

### Adding a New SSE Event Type

1. Publish the event from the pool's result handler in `processing/pool.py`:

```python
self._publish_event(
    doc_id,
    {
        "event": "my_event",
        "document_id": doc_id,
        "job_id": job_id,
        "custom_field": value,
    },
)
```

2. If the event is low-priority (safe to drop under load), add it to `_DEFAULT_PRIORITY_DROP` in `event_bus.py`:

```python
_DEFAULT_PRIORITY_DROP: frozenset[str] = frozenset({
    "page_complete",
    "my_event",  # safe to drop — client can reconstruct from REST
})
```

3. Handle the event in `client.py`'s `_handle_sse_event` if the CLI client needs to react to it.

**Important:** Terminal events (`job_complete`, `document_complete`) must NOT be added to the priority-drop set — they are needed for correctness.

### Recording a Provenance Event

All provenance events flow through `record_event()` in `event_logger.py`. To instrument a new error or lifecycle path:

1. Import the helper and the enums:

```python
from nemo_retriever.service.event_logger import record_event
from nemo_retriever.service.failure_types import EventCategory
from nemo_retriever.service.models.event_log import EventOutcome, EventSeverity
```

2. Call `record_event()` at the instrumentation site:

```python
record_event(
    repo,                                      # Repository instance
    category=EventCategory.MY_CATEGORY.value,  # from EventCategory enum
    severity=EventSeverity.ERROR,
    outcome=EventOutcome.FAILED,
    summary="Short description for list views",
    detail=str(exc),                           # full error message
    stack_trace=traceback.format_exc(),         # Python traceback (or "")
    stage="my_stage",                          # pipeline stage or step name
    endpoint="/v1/my_endpoint",                # or "pipeline" for workers
    job_id=job_id,                             # optional correlation keys
    document_id=document_id,
    source_file=filename,
    page_number=page_num,
    request_id=req_id,                         # from request.state.request_id
    extra={"retry_count": 3},                  # arbitrary JSON-safe dict
)
```

3. If you need a new category, add it to `EventCategory` in `failure_types.py`. If the new category maps to a `FailureType`, update `from_failure_type()` as well.

**Guidelines:**
- Use `EventSeverity.ERROR` for permanent failures, `WARNING` for transient/recoverable issues, `INFO` for normal lifecycle events (page completed, dedup detected).
- Use `EventOutcome.RECOVERED` when the system handled the error and continued (e.g., dedup, retry success). Use `FAILED` when the operation did not succeed. Use `IN_PROGRESS` for informational lifecycle markers.
- Always pass `request_id` from routers (via `request.state.request_id`). Worker processes can generate a batch-level ID if needed.
- The `extra` dict is serialized to `extra_json`. Use it for structured metadata that doesn't warrant a schema migration (operator timings, model versions, GPU memory, etc.).

---

## API Endpoint Summary

| Method | Path | Router | Purpose |
|--------|------|--------|---------|
| POST | `/v1/ingest` | `ingest` | Upload a single page for processing |
| POST | `/v1/ingest/batch` | `ingest` | Upload multiple pages in one request |
| POST | `/v1/ingest/job` | `ingest` | Upload a full PDF for server-side splitting |
| GET | `/v1/ingest/status/{doc_id}` | `ingest` | Document status + results |
| GET | `/v1/ingest/job/{job_id}` | `ingest` | Job status |
| GET | `/v1/ingest/job/{job_id}/results` | `ingest` | Full reassembled job results |
| GET | `/v1/ingest/jobs` | `ingest` | Paginated job listing |
| GET | `/v1/ingest/jobs/summary` | `ingest` | Aggregate job counts by status |
| POST | `/v1/ingest/job/{job_id}/cancel` | `ingest` | Cancel a job |
| POST | `/v1/ingest/stream/{doc_id}` | `stream` | SSE stream for one document |
| POST | `/v1/ingest/stream/jobs` | `stream` | SSE stream for multiple jobs |
| POST | `/v1/query` | `query` | Embed queries + search LanceDB |
| POST | `/v1/rerank` | `rerank` | Rerank passages via NIM |
| GET | `/v1/events` | `events` | Paginated provenance event listing |
| GET | `/v1/events/summary` | `events` | Aggregated event counts by category |
| GET | `/v1/events/{event_id}` | `events` | Single provenance event detail |
| PATCH | `/v1/events/{event_id}` | `events` | Update event outcome (acknowledge/resolve) |
| DELETE | `/v1/events/{event_id}` | `events` | Delete a single provenance event |
| POST | `/v1/events/acknowledge` | `events` | Bulk-acknowledge events |
| POST | `/v1/events/delete` | `events` | Bulk-delete events by IDs or filters |
| GET | `/v1/health` | `system` | Liveness probe + pool snapshot |
| GET | `/v1/capabilities` | `system` | Discover enabled NIM endpoints + features |
| GET | `/v1/ingest_metrics` | `metrics` | Per-file/per-page processing metrics |
| GET | `/metrics` | (prometheus) | Prometheus scrape endpoint |
