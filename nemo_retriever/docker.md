# Docker Service Image

This page covers the standalone Docker image for the NeMo Retriever service. For production-scale service and NIM deployment, use the Helm chart in [`helm/README.md`](helm/README.md) and the published NeMo Retriever Library install procedures.

## Build The Service Image

Run from the repository root:

```bash
docker build \
  -f Dockerfile \
  --target service \
  -t nemo-retriever-service:dev \
  .
```

For a release-tagged image whose OpenAPI document should report a specific version, pass matching build arguments:

```bash
docker build \
  -f Dockerfile \
  --target service \
  --build-arg RETRIEVER_VERSION=26.08.1 \
  --build-arg RETRIEVER_RELEASE_TYPE=release \
  -t nemo-retriever-service:26.08.1 \
  .
```

The `service` target installs `nemo_retriever[service]`, copies the packaged `retriever-service.yaml`, and starts the service with:

```bash
retriever service start --config /etc/nemo-retriever/retriever-service.yaml
```

## Run The Service Container

For remote NVIDIA-hosted NIM endpoints, pass the API key into the container and publish the service port:

```bash
docker run --rm \
  -p 7670:7670 \
  -e NVIDIA_API_KEY="${NVIDIA_API_KEY}" \
  nemo-retriever-service:dev
```

Open `http://localhost:7670/docs` for the OpenAPI UI, or check health with:

```bash
curl -fsSL http://localhost:7670/v1/health
```

## Configure The Service

The image reads `/etc/nemo-retriever/retriever-service.yaml` by default. To run with a custom config, mount it and pass the path through the service CLI:

```bash
docker run --rm \
  -p 7670:7670 \
  -e NVIDIA_API_KEY="${NVIDIA_API_KEY}" \
  -v "$PWD/my-retriever-service.yaml:/etc/nemo-retriever/retriever-service.yaml:ro" \
  nemo-retriever-service:dev \
  retriever service start --config /etc/nemo-retriever/retriever-service.yaml
```

Use Kubernetes Secrets, Helm values, or container environment variables for credentials. Do not bake API keys into derived images.

## Run A Local VectorDB With The Service

For a local development deployment, start the service with `--launch-vectordb`. It starts a VectorDB child process on `127.0.0.1:7671`, waits for `/v1/health`, and terminates the child when the service exits. If the child does not exit promptly, the service forcefully stops it.

The VectorDB child uses `nim_endpoints.embed_invoke_url` when configured. If that endpoint is unset, it uses local Hugging Face embedding when both `local_models.enabled` and `local_models.embed.enabled` are `true`.

The child inherits credentials from the service environment. Set `NVIDIA_API_KEY` or `NGC_API_KEY` for a remote embedding endpoint. Set `NRL_INTERNAL_VDB_TOKEN` or `NRL_INTERNAL_VDB_TOKEN_FILE` for the VectorDB internal credential. Do not place credentials in the service YAML.

For a fully local deployment, use a CUDA-capable host and install the service and local extras:

```bash
pip install "nemo-retriever[service,local]"
```

Install the `multimedia` extra when you ingest audio or video. From a source checkout, start the included local configuration with:

```bash
scripts/launch_local_service_with_vectordb.sh \
  nemo_retriever/examples/retriever-service.local.yaml
```

The example leaves all NIM endpoints unset and uses local Hugging Face models for service ingestion and VectorDB query embeddings. The launcher validates `/v1/health` on both components, then keeps the deployment running until you stop it with `Ctrl+C`.

To use remote embedding instead, configure `nim_endpoints.embed_invoke_url` and run:

```bash
retriever service start --config my-retriever-service.yaml --launch-vectordb
```

The launcher is limited to loopback VectorDB URLs. Omit the flag for an existing VectorDB. Helm deployments continue to run VectorDB in a separate pod.

If VectorDB exits during startup or does not become ready, read the VectorDB output in the terminal that started the service. Verify the VectorDB configuration, embedding model setup and credentials, writable LanceDB directory, and that port `7671` is available.



## Audio And Video

The service image omits `ffmpeg` and `ffprobe` by default. For audio or video extraction on a development machine with package-repository network access, set `INSTALL_FFMPEG=true`:

```bash
docker run --rm \
  -p 7670:7670 \
  -e NVIDIA_API_KEY="${NVIDIA_API_KEY}" \
  -e INSTALL_FFMPEG=true \
  nemo-retriever-service:dev
```

For restricted or air-gapped environments, build a derived image that includes `ffmpeg` and `ffprobe`, then set the Helm `service.image.*` values or run that derived image directly.
