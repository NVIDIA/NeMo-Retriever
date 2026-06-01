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
