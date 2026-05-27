# Service Reference

## Contents

- [Local Commands](#local-commands)
- [Service Config](#service-config)
- [HTTP Surface](#http-surface)
- [Helm Deployment](#helm-deployment)

## Local Commands

Start a local service:

```bash
retriever service start --config ./retriever-service.yaml --port 7670
```

From this source checkout:

```bash
uv run --project nemo_retriever retriever service start --host 127.0.0.1 --port 7670
```

Submit files to a running service:

```bash
BASE_URL=http://localhost:7670
retriever service ingest ./data/file.pdf --server-url "$BASE_URL"
```

If the service ingest CLI raises `TypeError: RetrieverServiceClient.ingest_documents()
got an unexpected keyword argument 'use_sse'`, drive the public HTTP API
directly. If auth is enabled, uncomment the `AUTH` line.

```bash
BASE_URL=http://localhost:7670
AUTH=()
# AUTH=(-H "Authorization: Bearer $NEMO_RETRIEVER_API_TOKEN")

curl -sS -X POST "$BASE_URL/v1/ingest/job" \
  "${AUTH[@]}" \
  -H 'Content-Type: application/json' \
  -d '{"expected_documents":1,"label":"smoke"}'

curl -sS -X POST "$BASE_URL/v1/ingest/job/<job_id>/document" \
  "${AUTH[@]}" \
  -F file=@./data/file.pdf \
  -F metadata='{"filename":"file.pdf"}'

curl -sS "$BASE_URL/v1/ingest/job/<job_id>?include_documents=true" \
  "${AUTH[@]}"
```

The service CLI supports:

- `--nim-api-key` for NIM endpoints, overriding YAML / `NVIDIA_API_KEY`.
- `--api-token` for service bearer-token auth, also read from
  `NEMO_RETRIEVER_API_TOKEN`.
- `--gpu-devices` to override service resource config.
- `--server-url` and `--api-token` on client ingest. The current client path may
  reject `--sse/--no-sse` or `--poll-interval`; use the HTTP job API above if
  that happens.

## Service Config

The bundled default is `nemo_retriever.service/retriever-service.yaml`.
Discovery order:

1. `retriever service start --config /path/to/retriever-service.yaml`
2. `./retriever-service.yaml`
3. bundled package default

Important config sections:

- `server.host` / `server.port`
- `nim_endpoints.*_invoke_url` and `nim_endpoints.api_key`
- `pipeline.realtime_workers` / `pipeline.batch_workers`
- `auth.api_token`
- `pipeline_overrides.mode` and sink allow lists

Client-supplied endpoint URLs and API keys are trust-sensitive. The policy layer
denies those through request overrides; configure them server-side.

For a cheap PDF text-only smoke upload, use allowed per-request extraction
overrides to disable expensive table/chart/image extraction:

```bash
curl -sS -X POST "$BASE_URL/v1/ingest/job/<job_id>/document" \
  "${AUTH[@]}" \
  -F file=@./data/file.pdf \
  -F metadata='{"filename":"file.pdf","pipeline":{"extraction_mode":"pdf","extract_params":{"method":"pdfium","extract_tables":false,"extract_charts":false,"extract_images":false,"extract_page_as_image":false},"stage_order":[]}}'
```

Do not include `use_page_elements` in request overrides unless the service
operator widened the allow list; the default policy rejects that key.

## HTTP Surface

Common public endpoints:

- `GET /v1/health`
- `POST /v1/ingest/job`
- `POST /v1/ingest/job/{job_id}/document`
- `GET /v1/ingest/job/{job_id}`
- `GET /v1/ingest/job/{job_id}/events`
- `GET /v1/ingest/pipeline-config`
- `GET /v1/ingest/metrics`
- `POST /v1/query` when the vectordb route is configured

The legacy firehose `GET /v1/ingest/events` is removed. Use the per-job events
route.

## Helm Deployment

The chart at `nemo_retriever/helm` deploys the service and optionally NIM
Operator resources. For external NIM endpoints:

```bash
helm install retriever ./nemo_retriever/helm \
  --set nims.enabled=false \
  --set serviceConfig.nimEndpoints.pageElementsInvokeUrl=http://page-elements.svc:8000/v1/infer \
  --set serviceConfig.nimEndpoints.tableStructureInvokeUrl=http://table-structure.svc:8000/v1/infer \
  --set serviceConfig.nimEndpoints.ocrInvokeUrl=http://ocr.svc:8000/v1/infer \
  --set serviceConfig.nimEndpoints.embedInvokeUrl=http://embed.svc:8000/v1/embeddings
```

For NGC image pulls or build.nvidia.com endpoints, configure the relevant
`NGC_API_KEY` / `NVIDIA_API_KEY` secrets through the chart values.
