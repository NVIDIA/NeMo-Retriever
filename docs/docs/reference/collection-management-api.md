# Collection management Python API

`RetrieverServiceClient` is the supported boundary for long-lived agentic
applications. The client talks to the NeMo Retriever service; applications do
not open LanceDB, choose table names, or reproduce ingestion stages.

## End-to-end workflow

```python
import time
from nemo_retriever import RetrieverServiceClient

client = RetrieverServiceClient(
    base_url="http://nemo-retriever:7670",  # Published service endpoint
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
as `http://localhost:7670`. For other deployments, use the published gateway
endpoint reachable by the calling application. Authentication, tracing,
retryable upload handling, collection routing, and result normalization remain
server/SDK responsibilities.

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
idempotency key replay with the same request returns the original job. The SDK
then safely replays every manifest entry, including after the client loses a
response before, during, or after upload. Each file has a deterministic
`manifest_entry_id` derived from its position, filename, and SHA-256. The
service returns the original acceptance for entries it already accepted,
without consuming capacity or starting duplicate processing. Reusing the key
or an entry ID with different content returns
`RetrieverServiceConflictError` (HTTP 409).

Job document status separates `attempt_id` (one processing attempt) from
`document_id` (the stable collection identity). Append creates a new stable
document ID. Replacement creates a new attempt but retains the target document
ID. Collection document APIs show only indexed materializations; pending,
processing, and failed attempts remain visible through job APIs.

`replace_document()` submits one replacement file. NeMo Retriever records a
pending-version recovery marker, uses a single LanceDB merge transaction to
insert the new chunks and remove obsolete chunks for that document, and then
finalizes the catalog. The VectorDB reconciler inspects stored chunk versions
after a crash and either finalizes the new version or preserves the old one.
Failed processing never removes the prior version, and queries never expose
mixed versions.

## Errors, scopes, expiration, and compatibility

The SDK raises `RetrieverServiceNotFoundError`,
`RetrieverServiceConflictError`, `RetrieverServiceValidationError`, or the base
`RetrieverServiceError`. Resources are isolated by `scope`; cross-scope reads
return 404. `expires_at` can be set at collection creation or update time for an
operator cleanup process. Deletion is retryable and `if_exists=True` makes
repeated deletion safe. Delete results report `existed`, `deleted`, `status`,
and `cleanup_pending`; synchronous completion returns HTTP 200 and a retryable
pending cleanup may return HTTP 202.

Production deployments map bearer tokens to allowed workspace scopes. A valid
token requesting an unauthorized scope receives 404 so resource existence is
not disclosed; missing or invalid credentials receive 401. Configure either a
single token bound to `default_scope`, or mount a Secret-backed JSON file:

```json
{"tokens":[{"token":"<secret>","scopes":["workspace-123"]}]}
```

Set `allow_unscoped_dev` only for an explicitly auth-disabled development
deployment. The gateway records the authorized scope on the request. Pod-only
callback routes and VectorDB calls require the separate internal credential;
an external bearer token is never used to authorize those internal routes or
forwarded to VectorDB.

`expires_at` must be timezone-aware RFC3339 and is normalized to UTC. Expired
collections enter the same retryable deletion state machine as explicit
deletion. The local VectorDB reconciler runs every 60 seconds by default,
applies exponential retry capped at one hour, and resumes replacement,
document deletion, collection deletion, and expiration cleanup after a crash.
Run one VectorDB replica while this reconciler is enabled; durable distributed
coordination remains separate infrastructure work. An interval of zero is
reserved for deployments where an external reconciler owns cleanup.

When a collection request uses the existing StoreOperator, the service ignores
client path selection and injects a prefix beneath the operator-configured
artifact root using scope, collection, stable document ID, and version. A
client destination outside that root returns 422. Document and collection
deletion recursively remove only these server-owned prefixes; arbitrary or
legacy external URIs are never deleted. fsspec credentials belong in a mounted
storage-options Secret, never request metadata or a ConfigMap.

Legacy fixed-table ingestion and query remain available when
`collection_name` is omitted, but only against the operator-configured table.
No service request may specify a raw table name, storage URI, or physical
LanceDB location. `/document` is the canonical ingestion route and `/whole` is
supported; collection-aware `/page` returns 422 before work is registered.

Continuation tokens are versioned keyset cursors rather than offsets.
Collection cursors advance by collection name; document cursors advance by
`(created_at, document_id)`. Tokens are bound to their resource type, scope,
and collection and return 422 when reused in another context. This keeps pages
stable while resources are inserted or deleted.

VectorDB health and metrics expose only aggregate catalog schema health,
active/deleting/expired counts, pending cleanup count and oldest age,
reconciliation successes/failures, and open-table cache size. Physical table
names and tenant identifiers are never emitted as public values or labels.

## Docker Compose operations

The local Compose example lives at
`nemo_retriever/dev/compose/collection-management.compose.yaml`. Copy the
development secret examples to protected files, set `NRL_EMBED_ENDPOINT`, and
override the three `*_FILE` environment variables before starting it. The
same SDK workflow targets `http://localhost:7670`.

## Application integration and query-result contract

External applications should construct a `RetrieverServiceClient` from the
service URL, token, and workspace scope, then call the SDK directly. Applications
should orchestrate calls and translate their own configuration only; NeMo
Retriever owns processing status, stable chunk/document identity, normalized
scores, citation provenance, retries, idempotency, and lifecycle truth. Clients
must not open LanceDB directly or reproduce the ingestion pipeline.

Collection query hits provide stable `chunk_id` and `document_id`, non-null
`text`, normalized `score` in `[0, 1]`, filename, one-based page number when
known, content type, source/source ID, stored image URI, bounding box, and
metadata. This contract is identical regardless of the network path used to
reach the service.
