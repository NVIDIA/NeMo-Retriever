# Collection management API

The Retriever service gateway exposes a supported public REST contract for
scoped collection and document catalog operations at `/v1/collections`.
`RetrieverServiceClient` is the supported Python SDK for the same contract.
It wraps those endpoints plus the related ingest and query REST APIs.

Applications do not open LanceDB, choose table names, or reproduce ingestion
stages.

## Support boundary { #support-boundary }

The following surfaces are supported for collection-managed applications:

- **Public REST.** Call the published Retriever service gateway (default port
  7670) with the eight `/v1/collections` operations listed in
  [REST reference](#rest-reference). Use this path from `curl`, a non-Python
  client, or an external orchestrator.
- **Python SDK.** Construct a `RetrieverServiceClient` with the service URL,
  token, and workspace scope. Python applications should prefer the SDK for
  retries, upload handling, and typed models. The SDK is not the only
  supported boundary.
- **Related REST, outside `/v1/collections`.** Ingest and replace use
  `POST /v1/ingest/job` with `collection_name`, then multipart upload to
  `POST /v1/ingest/job/{job_id}/document`. Retrieval uses `POST /v1/query`
  with `collection_name`. Those operations are supported REST APIs. They are
  not members of the `/v1/collections` path family.

The following surfaces are not a public collection contract:

- Internal VectorDB write routes such as `/internal/vectordb/write`.
- Direct calls to the VectorDB pod, including its private `/openapi.json`.
- Raw LanceDB table names, storage URIs, or physical table locations.

The gateway `/openapi.json` publishes the eight collection paths. The
gateway forwards those routes without request models. Generated `/docs` can
omit `requestBody` schemas for `POST /v1/collections` and
`PATCH /v1/collections/{collection_name}`. Use the request fields on this
page. They match the service Pydantic models.

## REST reference { #rest-reference }

Send collection requests to the published gateway address, such as
`http://localhost:7670` for Docker Compose. For other deployments, use the
gateway endpoint that is reachable by the calling application. Collection
management is not available on realtime or batch worker pods.

When service authentication is enabled, send `Authorization: Bearer <token>`.
Send `X-NRL-Scope` with a workspace scope that the token is allowed to use.
Missing or invalid credentials and a valid token that requests an
unauthorized scope both return HTTP 401. After authorization, a resource
owned by another scope returns HTTP 404. Do not send
`X-NRL-Internal-Token`; that header is for service-internal hops.

Collection names and document IDs must match
`^[A-Za-z0-9][A-Za-z0-9._-]*$` and are 1 through 128 characters.

The following table lists the eight catalog operations.

| Method | Path | Purpose | Success status |
| --- | --- | --- | --- |
| `GET` | `/v1/collections` | List collections in the authorized scope | 200 |
| `POST` | `/v1/collections` | Create a collection | 201 |
| `GET` | `/v1/collections/{collection_name}` | Get one collection | 200 |
| `PATCH` | `/v1/collections/{collection_name}` | Update mutable collection fields | 200 |
| `DELETE` | `/v1/collections/{collection_name}` | Delete a collection and its VectorDB-owned data | 200 or 202 |
| `GET` | `/v1/collections/{collection_name}/documents` | List committed documents | 200 |
| `GET` | `/v1/collections/{collection_name}/documents/{document_id}` | Get one committed document | 200 |
| `DELETE` | `/v1/collections/{collection_name}/documents/{document_id}` | Delete one document and its collection chunks | 200 or 202 |

List operations accept `limit` (1 through 200, default 100) and an opaque
`continuation_token`. Callers must not interpret tokens. Tokens are bound to
resource type, scope, and collection, and they return HTTP 422 when reused in
another context.

Duplicate collection names in the same scope return HTTP 409. Missing
resources return HTTP 404 unless `if_exists=true` on delete. Invalid names or
continuation tokens return HTTP 422. VectorDB unavailable at the gateway
returns HTTP 502. Collection routes return HTTP 404 when VectorDB is disabled
in the service configuration.

The examples below use `http://localhost:7670`. Replace the token and scope
with values from your deployment. When authentication is disabled for local
development, omit the `Authorization` header.

### Create a collection { #create-a-collection }

`POST /v1/collections` accepts the following JSON fields.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `name` | string | yes | Logical collection name. |
| `description` | string or null | no | Optional description, at most 4096 characters. |
| `metadata` | object | no | Caller-defined metadata. Defaults to `{}`. |
| `expires_at` | string or null | no | Timezone-aware RFC3339 timestamp, normalized to UTC. |

The response is a collection object with `name`, `scope`, `status`
(`active` or `deleting`), `description`, `metadata`, `created_at`,
`updated_at`, and `expires_at`.

```bash
curl -sS -X POST http://localhost:7670/v1/collections \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -H 'X-NRL-Scope: workspace-123' \
  -d '{"name": "research-session", "description": "Agent workspace"}'
```

### List and get collections { #list-and-get-collections }

The following example lists collections in the authorized scope.

```bash
curl -sS 'http://localhost:7670/v1/collections?limit=100' \
  -H 'Authorization: Bearer <token>' \
  -H 'X-NRL-Scope: workspace-123'
```

A list response contains `items` and `next_token`. Follow `next_token` until
it is `null`.

The following example gets one collection by name.

```bash
curl -sS http://localhost:7670/v1/collections/research-session \
  -H 'Authorization: Bearer <token>' \
  -H 'X-NRL-Scope: workspace-123'
```

### Update a collection { #update-a-collection }

`PATCH /v1/collections/{collection_name}` accepts any subset of
`description`, `metadata`, and `expires_at`. Omitted fields stay unchanged,
except that omitting `expires_at` refreshes the existing expiration window
when one is set. Sending `"expires_at": null` disables expiration. Sending a
new timestamp establishes a new window.

```bash
curl -sS -X PATCH http://localhost:7670/v1/collections/research-session \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -H 'X-NRL-Scope: workspace-123' \
  -d '{"description": "Updated agent workspace"}'
```

### Delete a collection { #delete-a-collection }

`DELETE /v1/collections/{collection_name}` accepts `if_exists` (default
`false`). When `if_exists=true`, a missing collection returns HTTP 200 with
`existed` false instead of HTTP 404. The result reports `existed`, `deleted`,
`status`, and `cleanup_pending`. Synchronous completion returns HTTP 200. A
retryable pending cleanup returns HTTP 202.

```bash
curl -sS -X DELETE 'http://localhost:7670/v1/collections/research-session?if_exists=true' \
  -H 'Authorization: Bearer <token>' \
  -H 'X-NRL-Scope: workspace-123'
```

Collection deletion removes collection and document catalog entries,
chunk and vector rows, and the backend-owned physical collection table. It
does not delete extracted artifacts from S3, NFS, or the local filesystem.

### List, get, and delete documents { #list-get-and-delete-documents }

Document catalog APIs return committed indexed materializations only.
Pending, processing, and failed attempts remain visible through job APIs.

A document object includes `document_id`, `collection_name`, `scope`,
`filename`, `content_sha256`, `document_version`, `status`, `chunk_count`,
`job_id`, `created_at`, `updated_at`, and `error`.

The following examples list, get, and delete a committed document.

```bash
curl -sS 'http://localhost:7670/v1/collections/research-session/documents?limit=100' \
  -H 'Authorization: Bearer <token>' \
  -H 'X-NRL-Scope: workspace-123'
```

```bash
curl -sS http://localhost:7670/v1/collections/research-session/documents/document-1 \
  -H 'Authorization: Bearer <token>' \
  -H 'X-NRL-Scope: workspace-123'
```

```bash
curl -sS -X DELETE \
  'http://localhost:7670/v1/collections/research-session/documents/document-1?if_exists=true' \
  -H 'Authorization: Bearer <token>' \
  -H 'X-NRL-Scope: workspace-123'
```

Document deletion uses the same `if_exists`, HTTP 200 or 202, and
`cleanup_pending` contract as collection deletion.

### Ingest, replace, and query { #ingest-replace-and-query }

To ingest into a collection, create a job with `collection_name` and
`operation` `append` (the default), then upload files. Refer to
[Job](../extraction/concepts.md#job) for the two-step ingest workflow.

```bash
curl -sS -X POST http://localhost:7670/v1/ingest/job \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -H 'X-NRL-Scope: workspace-123' \
  -d '{"expected_documents": 1, "collection_name": "research-session"}'
```

To replace one stable document, set `operation` to `replace`,
`expected_documents` to 1, and `target_document_id` to the existing document
ID, then upload one file.

To search a collection, send `POST /v1/query` with `collection_name`. Do not
send LanceDB table names. Refer to [Workflow: Agentic retrieval](../extraction/workflow-agentic-retrieval.md)
for the query envelope and agentic flag.

```bash
curl -sS -X POST http://localhost:7670/v1/query \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -H 'X-NRL-Scope: workspace-123' \
  -d '{"query": "What are the major findings?", "top_k": 10, "collection_name": "research-session"}'
```

## Python SDK workflow { #python-sdk-workflow }

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
server and SDK responsibilities.

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

Before the first physical append, the VectorDB records a pending-version
recovery marker and writes deterministic chunk IDs with an idempotent merge.
After an interrupted write, reconciliation either finalizes committed chunks
or removes an empty marker, so retrying the same document version does not
duplicate chunks.
Pending initial appends remain hidden from document reads and collection
queries until reconciliation commits them.

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
pending cleanup returns HTTP 202.

Production deployments map bearer tokens to allowed workspace scopes. Missing
or invalid credentials and valid tokens requesting an unauthorized scope
receive the same 401 response, preventing callers from distinguishing token
validity. After you are authorized for a scope, looking up a resource owned by another
scope returns 404 so its existence is not disclosed. Configure either a single
token bound to `default_scope`, or mount a Secret-backed JSON file:

```json
{"tokens":[{"token":"<secret>","scopes":["workspace-123"]}]}
```

Set `allow_unscoped_dev` only for an explicitly auth-disabled development
deployment. The gateway records the authorized scope on the request. Pod-only
callback routes and VectorDB calls require the separate internal credential;
an external bearer token is never used to authorize those internal routes or
forwarded to VectorDB.

`expires_at` must be timezone-aware RFC3339 and is normalized to UTC. For an
expiring collection, successful append and replace indexing activity refreshes
the expiration while preserving the configured window between `updated_at` and
`expires_at`. Collection metadata updates do the same when they omit
`expires_at`; supplying `expires_at` establishes a new window and setting it to
null disables expiration. Writes that do not commit vector data, including
empty writes, do not refresh collection activity. During recovery from an
interrupted write, NRL records activity refresh as durable recovery work. If
the collection update fails, reconciliation retains the marker and retries the
refresh. After the refresh succeeds, it clears the marker without refreshing
the collection again, so retries do not extend the expiration more than once.
Expired collections enter the
same retryable deletion state machine as explicit deletion. The local VectorDB
reconciler runs every 60 seconds by default, applies exponential retry capped at
one hour, and resumes replacement, document deletion, collection deletion, and
expiration cleanup after a crash.
Run one VectorDB replica while this reconciler is enabled; durable distributed
coordination remains separate infrastructure work. An interval of zero is
reserved for deployments where an external reconciler owns cleanup.

`StoreOperator` artifact persistence remains an independent pipeline and storage
concern. Collection deletion removes collection and document catalog entries,
chunk/vector rows, and the backend-owned physical collection table; it does not
delete extracted artifacts from S3, NFS, or the local filesystem. Configure
artifact retention and garbage collection at the storage/operator boundary,
where the corresponding credentials and ownership policy already live.

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

The default development stack lives at
`nemo_retriever/dev/compose/service-mode.compose.yaml` and runs the Retriever
and VectorDB as separate services. Set `NRL_API_TOKEN` to opt into a public
bearer credential and `NRL_INTERNAL_VDB_TOKEN` to protect the private service
hop; leaving them unset preserves the existing unauthenticated development
behavior. Runtime tokens must not be committed. Production deployments can
continue to use the service's Secret-backed multi-scope token-file support.
The same SDK workflow targets `http://localhost:7670`.

## Application integration and query-result contract

Python applications should construct a `RetrieverServiceClient` from the
service URL, token, and workspace scope, then call the SDK. REST clients should
call the gateway paths in [REST reference](#rest-reference). Applications
should orchestrate calls and translate their own configuration only; NeMo
Retriever owns processing status, stable chunk/document identity, retrieval
ordering, citation provenance, retries, idempotency, and lifecycle truth. Clients
must not open LanceDB directly or reproduce the ingestion pipeline.

Collection query hits provide stable `chunk_id` and `document_id`, non-null
`text`, a finite native `distance`, filename, a one-based page number when
known, content type, source/source ID, stored image URI, bounding box, and
metadata. Collection queries use dense vector retrieval in this release;
lower distances are more similar and list order is authoritative. NRL does
not reinterpret distance as a normalized similarity or confidence. Consumers
that require a bounded score must translate the complete result set at their
own adapter boundary. `page_number` is `null` for non-paginated content or
invalid/unknown page provenance. Audio segments, video frames, and timestamps
keep their existing modality-specific metadata rather than being converted
into document pages. This contract is identical regardless of the network
path used to reach the service.

For `format=evidence`, each evidence item's `score` is the same native dense
vector distance, not a normalized confidence or probability. Lower is better,
and values are not comparable across queries.
