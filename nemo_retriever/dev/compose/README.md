# NeMo Retriever Development Compose Helpers

These Compose files are local development helpers. They are not supported
production deployment artifacts and do not aim to expose every Helm option.
Use the [Helm chart](../../helm/README.md) for production and release
deployments.

Run commands from the repository root.

## Service Mode

The development service-mode stack builds the current checkout and starts a
standalone NeMo Retriever service plus its LanceDB VectorDB. Inference stays
external to the stack: the defaults call NVIDIA-hosted NIM endpoints, so no
local GPU or Kubernetes cluster is required.

> **Data handling:** The default endpoints send document and query data to
> NVIDIA-hosted services. Confirm that this is appropriate for your data and
> environment before starting the stack. To keep inference local, use the
> endpoint overrides described below.

Prerequisites:

- Docker Engine with Docker Compose 2.23.1 or newer. The stack uses Compose
  inline config content and environment interpolation.
- A `NVIDIA_API_KEY` authorized for the hosted NIMs when using the default
  endpoints. Self-hosted endpoints that do not require authentication can use
  an empty key.

Export the hosted-inference key and start the stack:

```bash
export NVIDIA_API_KEY=nvapi-...
docker compose -f nemo_retriever/dev/compose/service-mode.compose.yaml up --build -d
```

The first build can take a while. Once both containers are healthy, open
`http://localhost:7670/docs` or check the service directly:

```bash
curl -fsSL http://localhost:7670/v1/health
docker compose -f nemo_retriever/dev/compose/service-mode.compose.yaml ps
```

The default endpoint and model settings can be overridden individually:

```bash
export NVIDIA_API_KEY=
export NIM_PAGE_ELEMENTS_URL=http://host.docker.internal:8001/v1/infer
export NIM_TABLE_STRUCTURE_URL=http://host.docker.internal:8002/v1/infer
export NIM_OCR_URL=http://host.docker.internal:8003/v1/infer
export NIM_EMBED_URL=http://host.docker.internal:8004/v1/embeddings
export NIM_EMBED_MODEL=nvidia/llama-nemotron-embed-vl-1b-v2
docker compose -f nemo_retriever/dev/compose/service-mode.compose.yaml up --build -d
```

`host.docker.internal` is mapped to the Docker host by the Compose file. URLs
for services on another machine or Docker network can be supplied directly.
The embed URL and model are shared by the retriever and VectorDB so ingestion
and query embedding remain consistent.

Other useful development overrides are:

- `NEMO_RETRIEVER_IMAGE`: image name/tag to build or run. To use an existing
  published image, set this variable and add `--no-build` to `docker compose
  up`.
- `RETRIEVER_HTTP_PORT`: host port for the service; defaults to `7670`.
- `INSTALL_FFMPEG=true`: install `ffmpeg` and `ffprobe` when the service
  starts for audio/video development. This requires package-repository access.

Inspect logs or stop the stack while retaining its named volumes:

```bash
docker compose -f nemo_retriever/dev/compose/service-mode.compose.yaml logs -f
docker compose -f nemo_retriever/dev/compose/service-mode.compose.yaml down
```

LanceDB data and service logs persist across restarts. Remove them explicitly
when a clean development environment is needed:

```bash
docker compose -f nemo_retriever/dev/compose/service-mode.compose.yaml down -v
```

This stack intentionally omits NIM lifecycle management, split topology,
autoscaling, ingress, local GPU models, OpenTelemetry, Zipkin, and optional
answer/audio/caption services. Use Helm when those deployment behaviors need
to be exercised.

## Local Judge

The judge helper starts an OpenAI-compatible Nemotron NIM for `retriever skill-eval` runs that use a local judge endpoint.

Set `NGC_API_KEY` or `NIM_NGC_API_KEY` before starting this helper.

```bash
echo "${NGC_API_KEY}" | docker login nvcr.io --username '$oauthtoken' --password-stdin
docker compose -f nemo_retriever/dev/compose/judge.compose.yaml up -d judge
```

Then point `judge.api_base` at `http://localhost:8000/v1` in your skill-eval config.

## Neo4j

The Neo4j helper starts a local database for tabular/graph development.

```bash
docker compose -f nemo_retriever/dev/compose/neo4j.compose.yaml up -d neo4j
```

Set `NEO4J_PASSWORD` in your environment or `.env` file before starting this helper. `NEO4J_USERNAME` defaults to `neo4j`.
