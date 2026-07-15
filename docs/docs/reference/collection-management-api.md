# Collection management Python API

`RetrieverServiceClient` is the supported boundary for long-lived agentic
applications. The client talks to the NeMo Retriever service; applications do
not open LanceDB, choose table names, or reproduce ingestion stages.

## End-to-end workflow

```python
import time
from nemo_retriever import RetrieverServiceClient

client = RetrieverServiceClient(
    base_url="http://nemo-retriever:7670",  # Helm service DNS
    api_token="...",
    scope="workspace-123",
)

collection = client.create_collection("research-session")
job = client.submit_documents(
    collection.name,
    ["report.pdf"],
    idempotency_key="agent-request-42",
)

# Submission means the job and uploads were accepted. It does not mean that
# extraction, OCR, splitting, captioning, embedding, and indexing are done.
while True:
    job = client.get_job(job.job_id)
    if job.status in {"completed", "failed", "partial_success"}:
        break
    time.sleep(2)

hits = client.query(
    "What are the major findings?", collection_name=collection.name, top_k=10,
)
documents = client.list_documents(collection.name)
client.delete_document(collection.name, documents.items[0].document_id)
client.delete_collection(collection.name)
```

For local Docker Compose deployments, use the published gateway address, such
as `http://localhost:7670`. In Helm, use the gateway Kubernetes Service DNS from
the calling pod. Authentication, tracing, retryable upload handling, collection
routing, and result normalization remain server/SDK responsibilities.

## Sync and async methods

Every lifecycle method has a native async equivalent prefixed with `a`:
`create_collection`/`acreate_collection`, `submit_documents`/`asubmit_documents`,
`get_job`/`aget_job`, `list_documents`/`alist_documents`, and
`query`/`aquery`. Use async methods inside an event loop.

Collection methods include create, get, list, update, and delete. Document
methods include get, list, delete, and atomic replace. Job methods expose the
aggregate and paginated per-file status. List operations use bounded `limit`
values and opaque continuation tokens; callers must not interpret tokens.

## Append, idempotency, and replacement

Normal submission appends documents without changing existing documents. An
idempotency key replay with the same request returns the original job; reuse
with a different request returns `RetrieverServiceConflictError` (HTTP 409).

`replace_document()` submits one replacement file. NeMo Retriever builds the
new version first, then uses a single LanceDB merge transaction to insert its
chunks and remove obsolete chunks for that document. Failed processing never
removes the prior version, and queries never intentionally expose mixed
versions.

## Errors, scopes, expiration, and compatibility

The SDK raises `RetrieverServiceNotFoundError`,
`RetrieverServiceConflictError`, `RetrieverServiceValidationError`, or the base
`RetrieverServiceError`. Resources are isolated by `scope`; cross-scope reads
return 404. `expires_at` can be set at collection creation or update time for an
operator cleanup process. Deletion is retryable and `if_exists=True` makes
repeated deletion safe.

Legacy fixed-table ingestion and query remain available when
`collection_name` is omitted. A collection-aware request may not specify a raw
table name, storage URI, or other physical LanceDB location.

## Future AIQ adapter

An AIQ knowledge-layer `adapter.py` should construct one client from its service
URL, token, and workspace scope, then call this SDK directly. The expected
structural contract is `nemo_retriever.service.aiq_contract.AIQCompatibleClient`.
The adapter should orchestrate calls and translate configuration only; NeMo
Retriever owns processing status, stable chunk/document identity, normalized
scores, citation provenance, retries, idempotency, and lifecycle truth.

Collection query hits provide stable `chunk_id` and `document_id`, non-null
`text`, normalized `score` in `[0, 1]`, filename, one-based page number when
known, content type, source/source ID, stored image URI, bounding box, and
metadata. This contract is identical whether the service is reached through a
Helm Service or Docker Compose networking.
