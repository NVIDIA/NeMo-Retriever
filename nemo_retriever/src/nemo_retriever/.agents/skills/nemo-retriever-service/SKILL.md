---
name: nemo-retriever-service
description: Use when the user asks to run, deploy, configure, operate, or call the NeMo Retriever service, including `retriever service start`, `retriever service ingest`, FastAPI `/v1` endpoints, service YAML, Helm chart deployment, auth tokens, NIM endpoint wiring, or service-mode troubleshooting.
---

# nemo-retriever-service

Use this skill for NeMo Retriever service operation. Do not use it for a simple
local one-shot ingest unless the user specifically wants a long-running service.

## Orientation

1. Verify the installed service commands: `retriever service --help`,
   `retriever service start --help`, and `retriever service ingest --help`.
2. Decide local service versus Kubernetes Helm:
   - Local: `retriever service start` plus HTTP health checks.
   - Kubernetes: `nemo_retriever/helm` chart and NIM endpoint/secret wiring.
3. If the installed CLI is absent but this is a source checkout, use
   `uv run --project nemo_retriever retriever service ...`. Retry dependency
   downloads before choosing another service validation path.
4. If neither path works, use `nemo-retriever-setup` before debugging service
   behavior.

## References

- `references/SERVICE.md`: service commands, key endpoints, YAML settings, Helm
  deployment choices, and auth behavior.
- `PITFALLS.md`: endpoint policy, missing NIMs, token mismatch, SQLite replica
  limits, ffmpeg runtime install, and stale service routes.

## Workflow

1. Locate or create a service config. Discovery order is explicit `--config`,
   `./retriever-service.yaml`, then the bundled package default.
2. Start locally when appropriate:

   ```bash
   retriever service start --config ./retriever-service.yaml --host 0.0.0.0 --port 7670
   ```

3. Verify health before submitting work:

   ```bash
   curl http://localhost:7670/v1/health
   ```

4. Submit files through the CLI client:

   ```bash
   retriever service ingest ./data/file.pdf --server-url http://localhost:7670
   ```

5. If service ingest CLI raises `TypeError: ... unexpected keyword argument
   'use_sse'`, use the HTTP job API directly: `POST /v1/ingest/job`, then
   `POST /v1/ingest/job/{job_id}/document`, then poll
   `GET /v1/ingest/job/{job_id}?include_documents=true`.
6. For Kubernetes, use Helm and decide whether NIMs are operator-managed or
   external URLs supplied through `serviceConfig.nimEndpoints.*`.

## Success Checks

- `/v1/health` responds.
- The CLI client or HTTP job API can create a job, accept a document, and report
  completion or useful job status.
- If auth is enabled, requests include the same bearer token configured by
  `--api-token`, YAML `auth.api_token`, or `NEMO_RETRIEVER_API_TOKEN`.

## Evaluation Scenarios

- "Start a Retriever service for document ingestion." Use this skill.
- "Deploy Retriever with external NIM endpoints in Kubernetes." Use this skill.
- "Run a one-shot local ingest into LanceDB." Use `nemo-retriever-ingest`.
