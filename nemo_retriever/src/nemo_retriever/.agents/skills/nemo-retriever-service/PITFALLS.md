# Service Pitfalls

## Service Versus SDK Run Modes

`retriever service start` runs the FastAPI ingestion service.
`Retriever(run_mode="service")` means HTTP query embedding in the Python
retriever. Do not conflate them.

## Auth Token Mismatch

When `auth.api_token` is configured, every non-bypassed request needs
`Authorization: Bearer <token>`. The CLI can read `NEMO_RETRIEVER_API_TOKEN`.
Health and docs paths are bypassed by default, so a successful health check does
not prove ingest requests are authenticated correctly.

## Endpoint Overrides Are Server-Owned

Do not let client payloads set NIM endpoint URLs or API keys. Use YAML, CLI
overrides, environment variables, or Helm values. Request-level overrides are
policy-gated and endpoint/api-key keys are denied.

## Legacy Routes

Use `GET /v1/ingest/job/{job_id}/events` for SSE. The old
`GET /v1/ingest/events` route should be treated as stale.

## Service Ingest CLI Drift

Some current builds expose `retriever service ingest` options but call the
client with stale keyword arguments, producing:

```text
TypeError: RetrieverServiceClient.ingest_documents() got an unexpected keyword argument 'use_sse'
```

Do not stop there. Use the HTTP job API directly or the current Python client
signature: `ingest_documents(files=..., show_progress=True, pipeline_spec=...)`.

## Default Service Extraction Is PDF

The document upload route defaults to `extraction_mode='pdf'`. Uploading a TXT
file without a pipeline override fails with `Input file type(s) do not match
extraction_mode='pdf'`. For service smoke tests, use a PDF fixture or provide a
valid `pipeline.extraction_mode` and the dependencies needed by that mode.

## Page-Elements 401 Can Be Non-Fatal

A text-only PDF smoke upload can complete even if page-elements detection logs
an HTTP 401 inside `page_elements_v3.error`, as long as text extraction produced
rows and the job status is `completed`. Treat the embedded stage error as a
capability/config warning, not automatically as failed service ingestion.

## Helm Replica Limit

The Helm chart currently uses SQLite on a single ReadWriteOnce PVC, which caps
the service at one replica until a shared database backend is introduced.

## ffmpeg For Audio And Video

Audio/video extraction requires `ffmpeg` and `ffprobe`. The service image can
install them at startup with `service.installFfmpeg=true`, but that requires
network egress, writable root filesystem, and a security policy allowing the
scoped sudo path. Locked-down clusters should use a custom image with ffmpeg
already installed.
