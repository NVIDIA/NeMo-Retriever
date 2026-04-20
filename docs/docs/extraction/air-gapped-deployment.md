# Air-Gapped Deployment (End-to-End)

This guide consolidates what you need to run NeMo Retriever Library in a secured, network-isolated environment (for example 26.3.0 on Kubernetes or Docker Compose). It focuses on NeMo Retriever–specific images, Helm assets, and configuration so you can plug them into your broader air-gapped platform (private container registry, model artifact storage, [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/), and [NIM Operator](https://docs.nvidia.com/nim-operator/latest/install.html)).

!!! note "Source of truth for versions"

    Image repositories and default tags change between releases. Always verify pins against the `release/26.3.0` (or your exact stack) branches of:

    - [NeMo Retriever `docker-compose.yaml`](https://github.com/NVIDIA/NeMo-Retriever/blob/release/26.3.0/docker-compose.yaml) for self-hosted Compose (check out the **Git tag or branch that matches 26.3.0** in your environment; `main` moves forward)
    - [`helm/values.yaml`](https://github.com/NVIDIA/nv-ingest/blob/release/26.3.0/helm/values.yaml) and [`helm/README.md`](https://github.com/NVIDIA/nv-ingest/blob/release/26.3.0/helm/README.md) for Kubernetes / Helm

## End-to-end workflow

Use a staging machine (or bastion) that can reach the public internet and NGC, then promote artifacts into the disconnected site.

1. **Inventory** — Decide deployment mode (Compose vs Helm), optional profiles / NIMs (audio, Nemotron Parse, VLM, Milvus retrieval, reranker), and list every image and chart you must mirror (sections below).
2. **Mirror container images** — Pull from upstream registries, retag to your private registry (optional but recommended), record digests for reproducibility, and push to the registry reachable from the air-gapped environment.
3. **Stage non-image assets** — Helm chart `.tgz` packages from NGC (or vendor them internally), Python wheels for clients (`nv-ingest-client`), and any operator bundles your cluster policy requires.
4. **Configure pulls and runtime** — Point all `image.repository` values (and Compose image env vars) at the private registry; use `imagePullSecrets`; set `imagePullPolicy: IfNotPresent` (or `Never` only if every node is preloaded). Ensure **no** workload still references `integrate.api.nvidia.com`, `ai.api.nvidia.com`, or other hosted NVIDIA APIs unless you intentionally proxy them.
5. **NIM models and caches** — For Kubernetes, follow [NIM Operator: Air-gapped environments](https://docs.nvidia.com/nim-operator/latest/air-gap.html) to preload models into NIMCache / private artifact storage so NIM pods never need outbound registry access at runtime.
6. **Validate** — From a jump host inside the enclave, run image pull tests, `helm template` with offline values, then smoke-test ingest health (`/v1/health/ready`) and a minimal extract job.

## What NeMo Retriever / NV-Ingest needs (container images)

### Core pipeline (typical default)

These services are commonly enabled for document extraction with self-hosted NIMs (names match Docker Compose services where applicable).

| Role | Default image (verify tag in repo) | Notes |
|------|--------------------------------------|--------|
| Ingest runtime | `nvcr.io/nvidia/nemo-microservices/nv-ingest:26.3.0` | Main API / Ray service |
| Page elements NIM | `nvcr.io/nim/nvidia/nemotron-page-elements-v3` | Default tag in Compose: `1.8.0` |
| Graphic elements NIM | `nvcr.io/nim/nvidia/nemotron-graphic-elements-v1` | Default tag: `1.8.0` |
| Table structure NIM | `nvcr.io/nim/nvidia/nemotron-table-structure-v1` | Default tag: `1.8.0` |
| OCR NIM | `nvcr.io/nim/nvidia/nemotron-ocr-v1` | Default tag: `1.3.0` |
| Embedding NIM | `nvcr.io/nim/nvidia/llama-nemotron-embed-1b-v2` | Default tag: `1.13.0` |
| Redis | `redis/redis-stack` | Message broker / stack |
| OpenTelemetry Collector | `otel/opentelemetry-collector-contrib` | e.g. `0.140.0` in Helm values |
| Zipkin | `openzipkin/zipkin` | e.g. `3.5.0` in Helm values |

### Optional profiles / features

Add these images only if you enable the matching Compose profiles or Helm NIM toggles.

| Feature | Image | When needed |
|---------|--------|-------------|
| Nemotron Parse | `nvcr.io/nim/nvidia/nemotron-parse` | Advanced PDF parsing profile |
| VLM captioning | `nvcr.io/nim/nvidia/nemotron-nano-12b-v2-vl` | `vlm` profile / `nemotron_nano_12b_v2_vl` in Helm |
| Audio (ASR) | `nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us` | `audio` profile |
| Reranking NIM | `nvcr.io/nim/nvidia/llama-nemotron-rerank-1b-v2` | Reranker profile / `rerankqa` in Helm |
| Retrieval stack (Milvus path) | `milvusdb/milvus`, `minio/minio`, `quay.io/coreos/etcd`, `zilliz/attu` | Compose `retrieval` profile; Helm deploys Milvus/MinIO via subchart defaults |
| Observability extras | `prom/prometheus`, `grafana/grafana` | Only if you enable monitoring containers in Compose |

### Kubernetes-only dependencies (Helm chart defaults)

When you install from the nv-ingest Helm chart, also plan to mirror subchart images your values enable, for example:

- Milvus (`milvusdb/milvus` and bundled etcd/minio images from chart values)
- Redis (`redis` with chart tag, for example `8.2.3` in default `values.yaml`)

Again, take the exact repository and tag from your pinned `values.yaml` for 26.3.0.

## Helm charts and packaging artifacts

From a connected environment, download and version-control the chart archive you install, for example (see [NV-Ingest Helm README](https://github.com/NVIDIA/nv-ingest/blob/release/26.3.0/helm/README.md)):

- `nv-ingest-26.3.0.tgz` from NGC Helm (`helm pull` with NGC credentials)

If your process forbids live `helm install` from URLs, use `helm pull` on the staging host, copy the `.tgz` and any dependent charts into the enclave, then `helm upgrade --install` using the local file path.

## Pinning versions and digests

- Tags — Align Compose `*_TAG` / Helm `image.tag` fields with the same release line you qualified (for example `26.3.0` for the ingest image, NIM tags from `docker-compose.yaml` / `nimOperator` in `values.yaml`).
- Digests — After `docker pull nvcr.io/...:tag`, run `docker inspect --format='{{index .RepoDigests 0}}' image:tag` (or use crane / skopeo; see tooling below) and record `repository@sha256:...`. Prefer deploying with digests in highly regulated environments; keep a mapping table from digest → human-readable tag for operations.

## Mirroring images into a private registry

Typical pattern on the staging host:

1. `docker login nvcr.io` (NGC key as password, username `$oauthtoken`) and log in to other upstream registries you use (`docker.io`, `quay.io`, etc.).
2. For each image: `docker pull upstream/image:tag`
3. `docker tag upstream/image:tag <PRIVATE_REGISTRY>/nv-ingest-mirror/upstream-image:tag`
4. `docker push <PRIVATE_REGISTRY>/nv-ingest-mirror/upstream-image:tag`

For large fleets, prefer [skopeo](https://github.com/containers/skopeo) or [crane](https://github.com/google/go-containerregistry/blob/main/cmd/crane/README.md) for copy/sync between registries without loading into a local Docker daemon.

Transfer tarballs instead when the enclave has no registry yet: `docker save -o nv-ingest-bundle.tar` multiple images, move media, then `docker load` / `ctr -n k8s.io images import` on nodes (cluster-specific).

## Pointing deployments at the private registry (avoid runtime pulls)

### Docker Compose

- Override each `*_IMAGE` / `*_TAG` environment variable (see [`docker-compose.yaml`](https://github.com/NVIDIA/NeMo-Retriever/blob/release/26.3.0/docker-compose.yaml)) so every `image:` resolves to your mirror.
- Keep hosted API endpoints disabled: use in-stack URLs for NIMs (defaults in the compose file already prefer `http://…` service names over `https://integrate.api.nvidia.com`).
- Provide `.env` or config alongside the compose file in the enclave; never rely on pulling new images at `up` time without registry access.

For a short Compose-oriented procedure, see [Air-Gapped Deployment (Docker Compose)](quickstart-guide.md#air-gapped-deployment-docker-compose) in the self-hosted quickstart; this page is the complete checklist.

### Helm (Kubernetes)

1. Set the main ingest image, for example:

   ```text
   image.repository=<PRIVATE_REGISTRY>/nvidia/nemo-microservices/nv-ingest
   image.tag=26.3.0
   ```

2. Under `nimOperator` in `values.yaml`, override `image.repository` / `image.tag` for each enabled NIM (page elements, graphic elements, table structure, OCR, embed, optional parse/VLM/audio/rerank) to your mirrored paths.

3. Configure `imagePullSecrets` so kubelet can authenticate to your private registry (the chart defaults assume NGC-style secrets; replace with secrets that reference your mirror’s credentials).

4. Set `imagePullPolicy: IfNotPresent` (default in many paths) once images are pre-pulled or pulled through the mirror; use `Never` only if you fully preload images on every node and understand scheduling failure modes.

5. Review `envVars` for any URL that still points to the public internet (for example hosted VLM or Parse endpoints). For fully offline captioning, deploy the Nemotron Nano VL NIM in-cluster and set `VLM_CAPTION_ENDPOINT` to the in-cluster HTTP/gRPC endpoint (see comments in `values.yaml` for the expected pattern).

6. Install [GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/) and [NIM Operator](https://docs.nvidia.com/nim-operator/latest/), configured for offline registries and cached models ([NIM Operator air-gap guide](https://docs.nvidia.com/nim-operator/latest/air-gap.html)).

## Recommended tooling references

| Tool | Use case |
|------|-----------|
| [skopeo](https://github.com/containers/skopeo) | Copy/sign/inspect images between registries without Docker |
| [crane](https://github.com/google/go-containerregistry/blob/main/cmd/crane/README.md) | Bulk copy, tag, digest resolution |
| [Helm](https://helm.sh/docs/) | Package, template, and install charts from local `.tgz` |
| [Harbor](https://goharbor.io/) or [Docker Registry](https://distribution.github.io/distribution/) | Private registry to host mirrored images |
| [NIM Operator — Air-gapped environments](https://docs.nvidia.com/nim-operator/latest/air-gap.html) | NIM-specific mirroring, secrets, and model cache behavior |

## Related documentation

- [Deploy With Helm for NeMo Retriever Library](helm.md)
- [Deploy (Self-Hosted) Quickstart](quickstart-guide.md) — includes Compose air-gap summary
- [Environment Variables](environment-config.md) — runtime tuning and endpoints
- [Generate Your NGC Keys](ngc-api-key.md) — staging-time pulls from `nvcr.io`

## Broader dependencies (outside this doc’s scope)

Plan separately for GPU drivers / GPU Operator, ingress / TLS, storage classes for Milvus and NIM PVCs, enterprise image scanning, and internal PyPI / wheel mirrors for Python clients. NeMo Retriever’s ingest container is built so common tokenizer assets are present at build time; if you enable features that need extra Hugging Face access, you must **pre-stage** those artifacts per your security policy (see [Environment Variables](environment-config.md) for tokens such as `HF_ACCESS_TOKEN`).
