# Deployment options

Use this page to compare how you run NeMo Retriever — including when to use [NVIDIA-hosted NIMs](https://build.nvidia.com/) versus self-hosting on your own infrastructure.

## Compare deployment options

Use the sections below to pick documentation and deployment options that match your goal.

### I want to run locally or embed the library

1. [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
2. [Use the Python API](nemo-retriever-api-reference.md) or [Use the CLI](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/docs/cli) — install and run the [`nemo_retriever`](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever) package in your environment

### I want a standalone Docker service container

Build and run the NeMo Retriever service image with the [Docker service image guide](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/docker.md). Use this for local service-container validation; use Helm for multi-service Kubernetes deployments.

### I want a Kubernetes / Helm deployment

1. [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
2. **NeMo Retriever Helm chart (supported):** [Deploy (Helm chart)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md) — sources in [`nemo_retriever/helm`](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/helm) on GitHub
3. **Published Library Helm charts (supported):** cluster install and upgrade procedures are covered in [About getting started](getting-started-about.md) — use alongside the NeMo Retriever chart README for your release
4. [Environment variables](environment-config.md) and [Troubleshoot](troubleshoot.md) as needed

**Core NIMs for the default extraction pipeline:** `page_elements`, `table_structure`, `ocr`, and `vlm_embed` (`llama-nemotron-embed-vl-1b-v2:1.12.0`). These four are auto-wired into the retriever service. **Nemotron Parse**, **Nemotron 3 Nano Omni**, the **VL reranker**, and **Parakeet ASR** are optional and not auto-wired. For a minimal GPU footprint, disable optional keys you do not need (refer to [Recommended minimal install](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#recommended-minimal-install-2605)). Refer to [Pre-Requisites & Support Matrix — Default NIMs](prerequisites-support-matrix.md#default-helm-nims) and [Default NVCF endpoints](prerequisites-support-matrix.md#default-nvcf-endpoints).

For audio and video extraction in Kubernetes, set `service.installFfmpeg=true` so the service container installs `ffmpeg` and `ffprobe` at startup. This runtime install requires package-repository network egress, a writable root filesystem, and security policy that allows the image's scoped sudo use. If your cluster blocks startup package installation, use a custom service image that already contains `ffmpeg` and `ffprobe`, then set `service.image.repository` and `service.image.tag`. For Parakeet ASR chart values, OpenShift-specific Helm configuration, and air-gapped alternatives, refer to [Audio and video (Parakeet ASR)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#audio-video-parakeet) and [OpenShift deployment](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/openshift.md) in the Helm chart directory.

### I want examples and notebooks

1. [Jupyter Notebooks](starter-kits.md)

### I need API details and keys

1. [Get your API key](api-keys.md)
2. [API reference — PDF pre-splitting](nemo-retriever-api-reference.md#pdf-pre-splitting-for-parallel-ingest) if applicable

### I am tuning performance or cost

1. [Evaluation and performance](evaluate-on-your-data.md)
2. [Throughput is dataset-dependent](multimodal-extraction.md#extraction-limitations-and-quality)
3. [Evaluate on your data](evaluate-on-your-data.md)

## When to use NVIDIA-hosted NIMs { #when-to-use-nvidia-hosted-nims }

[NVIDIA-hosted NIMs](https://build.nvidia.com/) run inference on NVIDIA-managed infrastructure. You call models with API keys (refer to [Get your API key](api-keys.md)) without operating GPU nodes yourself.

Consider hosted NIMs when:

- You want the fastest path to try models and iterate without installing drivers, containers, or the [NIM Operator](https://docs.nvidia.com/nim-operator/latest/index.html) on your own clusters.
- Latency to NVIDIA endpoints works for your region and use case.
- Your compliance and data policies allow document or query content in the hosted service (confirm with your security review).

**Also refer to:** [NVIDIA NIM catalog](https://build.nvidia.com/)

## When to self-host NIMs { #when-to-self-host-nims }

Self-hosted NIMs run on your GPUs or air-gapped hardware, typically with Kubernetes and the [NIM Operator](https://docs.nvidia.com/nim-operator/latest/index.html).

Consider self-hosting when:

- You need an air gap, strict data residency, or customer data must not leave your network.
- You run at large scale where dedicated capacity can cost less than hosted API usage.
- You must meet latency or locality requirements that hosted regions cannot satisfy.

**GPU sharing.** The NIM Operator supports time-slicing and MIG so multiple NIM workloads can share GPUs. A NIM used with NeMo Retriever Library does not always need a full dedicated GPU when the operator and GPU profile are set correctly. For scheduling and GPU partitioning, refer to the [NIM Operator documentation](https://docs.nvidia.com/nim-operator/latest/index.html).

## Air-gapped and disconnected deployment { #air-gapped-deployment }

The **default document extraction pipeline** (page elements, table structure, OCR, and VL embed) runs disconnected when you mirror images and models into a private registry and configure the [NIM Operator for air-gapped environments](https://docs.nvidia.com/nim-operator/latest/air-gap.html).

On a staging host with internet access, pull from NGC, retag to your private registry, stage chart archives, then install in the enclave with registry overrides. Procedures, the chart image inventory, and Helm value patterns are in [Helm — Air-gapped deployment](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#air-gapped-deployment).

!!! warning "Audio and video extraction"

    [Audio and video](audio-video.md) need **`ffmpeg` and `ffprobe` on `PATH`**. The bundled image omits them. Do **not** use `service.installFfmpeg=true` in an air gap (startup install needs package-repo egress). Build a custom service image on a connected staging host, mirror it, and set `service.image.repository` / `service.image.tag`. Skip this step if you do not use audio/video.

For offline image captioning, deploy the in-cluster [Nemotron 3 Nano Omni](prerequisites-support-matrix.md#image-captioning) NIM and point your pipeline caption endpoint at the in-cluster HTTP URL instead of `integrate.api.nvidia.com` or other hosted APIs.

**Related**

- [Deploy (Helm chart)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md) ([`nemo_retriever/helm`](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/helm) on GitHub) — [air-gapped deployment](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#air-gapped-deployment)
- [About getting started](getting-started-about.md) (prerequisites through first deployment)
- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
- [Audio and video](audio-video.md)
