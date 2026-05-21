# Deployment options

Use this page to compare how you run NeMo Retriever — including when to use [NVIDIA-hosted NIMs](https://build.nvidia.com/) versus self-hosting on your own infrastructure.

## Compare deployment options

Use the sections below to pick documentation and deployment options that match your goal.

### I want to run locally or embed the library

1. [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
2. [Use the Python API](nemo-retriever-api-reference.md) or [Use the CLI](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/docs/cli) — install and run the [`nemo_retriever`](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever) package in your environment

### I want a Kubernetes / Helm deployment

1. [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
2. **NeMo Retriever Helm chart (supported):** [Deploy (Helm chart)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md) — sources in [`nemo_retriever/helm`](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/helm) on GitHub
3. **Published Library Helm charts (supported):** cluster install and upgrade procedures are covered in the [NeMo Retriever Library](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/) — use alongside the NeMo Retriever chart README for your release
4. [Environment variables](environment-config.md) and [Troubleshoot](troubleshoot.md) as needed

**Default NIMs in the published NeMo Retriever Library Helm chart** (26.03): `page_elements`, `table_structure`, `ocr`, and `vlm_embed` (`llama-nemotron-embed-vl-1b-v2:1.12.0`). **Nemotron Parse**, **Nemotron 3 Nano Omni**, and the **VL reranker** are optional and disabled by default—enable them only when needed. See [Pre-Requisites & Support Matrix — Default Helm NIMs](prerequisites-support-matrix.md#default-helm-nims).

**Docker Compose (unsupported, developer-only):** [Docker Compose for local development](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/docker.md) — **not** a substitute for Helm or the published Library charts.

For audio and video extraction in Kubernetes, set `service.installFfmpeg=true`
so the service container installs `ffmpeg` and `ffprobe` at startup. This
runtime install requires package-repository network egress, a writable root
filesystem, and security policy that allows the image's scoped sudo use. If
your cluster blocks startup package installation (for example air-gapped
environments), use a custom service image that already contains `ffmpeg` and
`ffprobe`, then set `service.image.repository` and `service.image.tag`.

### I want examples and notebooks

1. [Jupyter Notebooks](notebooks.md)
2. [Integrate with LangChain, LlamaIndex, Haystack](integrations-langchain-llamaindex-haystack.md)

### I need API details and keys

1. [Get your API key](api-keys.md)
2. [API reference — PDF pre-splitting](nemo-retriever-api-reference.md#pdf-pre-splitting-for-parallel-ingest) if applicable

### I am tuning performance or cost

1. [Evaluation and performance](evaluate-on-your-data.md)
2. [Throughput is dataset-dependent](multimodal-extraction.md#extraction-limitations-and-quality)
3. [Evaluate on your data](evaluate-on-your-data.md)

## When to use NVIDIA-hosted NIMs

[NVIDIA-hosted NIMs](https://build.nvidia.com/) run inference on NVIDIA-managed infrastructure. You call models with API keys (refer to [Get your API key](api-keys.md)) without operating GPU nodes yourself.

Consider hosted NIMs when:

- You want the fastest path to try models and iterate without installing drivers, containers, or the [NIM Operator](https://docs.nvidia.com/nim-operator/latest/index.html) on your own clusters.
- Latency to NVIDIA endpoints works for your region and use case.
- Your compliance and data policies allow document or query content in the hosted service (confirm with your security review).

**Also refer to:** [NVIDIA NIM catalog](https://build.nvidia.com/)

## When to self-host NIMs

Self-hosted NIMs run on your GPUs or air-gapped hardware, typically with Kubernetes and the [NIM Operator](https://docs.nvidia.com/nim-operator/latest/index.html).

Consider self-hosting when:

- You need an air gap, strict data residency, or customer data must not leave your network.
- You run at large scale where dedicated capacity can cost less than hosted API usage.
- You must meet latency or locality requirements that hosted regions cannot satisfy.

**GPU sharing.** The NIM Operator supports time-slicing and MIG so multiple NIM workloads can share GPUs. A NIM used with NeMo Retriever Library does not always need a full dedicated GPU when the operator and GPU profile are set correctly. For scheduling and GPU partitioning, refer to the [NIM Operator documentation](https://docs.nvidia.com/nim-operator/latest/index.html).

## Air-gapped and disconnected deployment { #air-gapped-deployment }

The **default document extraction pipeline** (core Helm NIMs: page elements, table structure, OCR, and VL embed) supports disconnected operation when you mirror container images and model artifacts into a private registry and configure the [NIM Operator for air-gapped environments](https://docs.nvidia.com/nim-operator/latest/air-gap.html).

Use a staging host with internet access to pull from NGC and upstream registries, retag images to your private registry, stage Helm chart `.tgz` archives, then install inside the enclave with registry overrides. Step-by-step procedures, image inventory for the 26.05 chart topology, and Helm value patterns are in the [NeMo Retriever Helm chart — Air-gapped deployment](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#air-gapped-deployment).

!!! warning "Audio and video extraction"

    [Audio and video](audio-video.md) workflows require **`ffmpeg` and `ffprobe` on `PATH`** in the service container. The bundled service image omits them by default. Do **not** rely on `service.installFfmpeg=true` in an air gap — that setting installs the Ubuntu `ffmpeg` package at container startup and needs outbound access to package repositories. On a connected staging host, build a custom service image with ffmpeg/ffprobe already installed, mirror it to your private registry, and set `service.image.repository` / `service.image.tag`. The default pipeline without audio/video does not need this step.

For offline image captioning, deploy the in-cluster [Nemotron 3 Nano Omni](prerequisites-support-matrix.md#image-captioning-2605) NIM and point your pipeline caption endpoint at the in-cluster HTTP URL instead of `integrate.api.nvidia.com` or other hosted APIs.

**Related**

- [Deploy (Helm chart)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md) ([`nemo_retriever/helm`](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/helm) on GitHub) — [air-gapped deployment](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#air-gapped-deployment)
- [NeMo Retriever Library — prerequisites / deployment](https://docs.nvidia.com/nemo/retriever/latest/extraction/overview/) (supported **Helm** handoff)
- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md) — [air-gapped notes](prerequisites-support-matrix.md#air-gapped-deployment)
- [Audio and video](audio-video.md)
- **Docker Compose (unsupported):** [docker.md](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/docker.md) — local developer tooling only
