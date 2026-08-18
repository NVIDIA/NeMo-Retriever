# Pre-Requisites & Support Matrix

Before you begin using [NeMo Retriever Library](overview.md), confirm your software stack, deployment hardware, Kubernetes persistent storage and GPU scheduling if you use Helm, and advanced features you plan to enable (audio and video, Nemotron Parse, VLM image captioning, reranking) against the guidance on this page.

**Platform summary:** Supported **local GPU inference** requires **Linux** and CUDA 13. For **remote NIM inference**, the base Python package also installs on **Windows x64** and **macOS Apple Silicon (arm64)**; local GPU inference is not supported on those platforms. **macOS Intel (x86_64) is not supported** — `pip`/`uv` installs fail because Ray no longer publishes Intel Mac wheels.

> **Note — NVIDIA AI Enterprise (NVAIE) support**
>
> The NeMo Retriever Library, including its container image and Helm chart artifacts, is not supported under NVIDIA AI Enterprise (NVAIE), even though some NIM microservices and models it uses may be individually covered by NVAIE. For more information, refer to [NVIDIA AI Enterprise (NVAIE) support](overview.md#nvidia-ai-enterprise-nvaie-support).

## Software Requirements { #software-requirements }

- Linux operating systems (Ubuntu 22.04 or later recommended) for supported local GPU inference. For remote NIM inference, the base package can also be installed on Windows x64 and macOS Apple Silicon (arm64); local GPU inference is not supported on those platforms. macOS Intel (x86_64) is not supported: package installation fails because Ray `>=2.56.1` has no Intel Mac wheels (including in-process library mode).
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (local GPU inference only; NVIDIA Driver >= `580`, CUDA >= `13.0`)
- [Python](https://www.python.org/downloads/) `3.12` — required to install and run the NeMo Retriever Library Python API, CLI, and related packages from PyPI (for example `pip` or `uv`). Older Python versions will fail dependency resolution without a clear error.
- [UV Python package and environment manager](https://docs.astral.sh/uv/getting-started/installation/) (optional; recommended for creating isolated environments)
- For audio and video, `ffmpeg` and `ffprobe` must be on `PATH` (for example
  `sudo apt-get install -y --no-install-recommends ffmpeg` on Debian/Ubuntu).
  `ffmpeg-python` and `nemo-retriever[multimedia]` do not install these binaries.
  For container and Kubernetes guidance, refer to [Audio and video](audio-video.md).
- For PDF extraction with `method="nemotron_parse"`, install the Nemotron Parse
  client dependencies with `uv pip install "nemo-retriever[nemotron-parse]"` (pulls
  `open-clip-torch`, which provides the `open_clip` module required by the Nemotron Parse
  NIM client). The base `nemo-retriever` install and `[local]` extra do not include this
  package. You can use the equivalent `pip install` command if you do not use UV.

> **Note**
>
> When you use UV, create the environment with Python 3.12 — for example, `uv venv --python 3.12`. This matches the `requires-python` metadata in the library packages.

## Kubernetes Helm Storage Requirements { #kubernetes-helm-storage-requirements }

The production Helm chart requires a working persistent-volume provisioning and binding strategy. A default install creates **seven** PersistentVolumeClaims: three chart-managed service claims and four NIM Operator NIMCache claims for the core NIMs.

Before you run `helm install`, confirm that the cluster can bind those claims. Use one of the following strategies:

- A default StorageClass backed by a working provisioner.
- Explicit `storageClass` values for every default claim, each backed by a working provisioner or matching persistent volumes.
- Compatible static persistent volumes, or pre-created claims where the chart supports `existingClaim`.

Run the following preflight commands:

```bash
kubectl get storageclass
kubectl get pv
```

If the cluster has no StorageClass and no compatible `Available` persistent volumes, stop and add a binding strategy before you install. `helm install` can report `STATUS: deployed` while every claim remains `Pending`, which leaves the retriever service, VectorDB, and core NIM workloads unschedulable.

For claim names, Helm value paths, and example `--set` flags, refer to [Persistent storage prerequisite](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#persistent-storage-prerequisite) in the Helm chart README. For Helm success with Pending claims, refer to [Helm install succeeds but PersistentVolumeClaims stay Pending](troubleshoot.md#helm-pending-pvcs).

## Kubernetes Helm GPU scheduling { #kubernetes-helm-gpu-scheduling }

The [model hardware requirements](#model-hardware-requirements) **Total GPUs** row for Core Features is combined GPU memory co-residency. The four default NIMs together use about 4.8 GiB and can co-reside on one A10G or better GPU.

The default Helm chart does not pack those NIMs onto one Kubernetes GPU request. It creates four independent `NIMService` workloads (`page_elements`, `table_structure`, `ocr`, and `vlm_embed`). Each replica requests `nvidia.com/gpu: 1` through `nimOperator.nimServiceGpuLimit` (default `1`).

On a conventional Kubernetes GPU cluster without MIG, time-slicing, or another sharing mechanism, the scheduler treats those as four exclusive GPU claims. Those claims can land on different eligible nodes. Four one-GPU nodes provide enough conventional capacity. A cluster that has only one allocatable GPU schedules only one of the four core NIM pods. The other pods stay `Pending`, and the documented core workflow is incomplete.

Choose one of the following strategies before you run `helm install`:

- Provide **four allocatable GPU slots across eligible nodes** for the default self-hosted Helm topology. Each optional NIM you enable adds its Helm GPU request. Most optional NIMs request `nvidia.com/gpu: 1`. The default `answer_llm` Super-49B NIMService requests `nvidia.com/gpu: 2`.
- Configure GPU sharing so the cluster advertises at least four `nvidia.com/gpu` slots. Time-slicing is the documented sharing path, including on GPUs that do not support Multi-Instance GPU (MIG).

Time-slicing creates logical GPU slots. It does not pin the four independently scheduled NIM pods onto one physical GPU. The scheduler can spread them across GPUs or nodes. The one-physical-GPU recipe requires both of the following: the target node has one physical GPU and advertises at least four replicas, and all four NIMServices are pinned to that node with `nimOperator.<key>.nodeSelector`. On a multi-GPU node, `nodeSelector` constrains node placement. It does not ensure all four pods receive logical replicas from the same physical GPU. The default `answer_llm` Super-49B NIMService is outside that one-physical-GPU recipe. It requires two physical GPUs unless you override it with a separately validated model and profile. Time-slice replicas cannot satisfy that tensor-parallel request.

The chart still renders `nvidia.com/gpu: 1` per NIMService unless a per-NIM `resources` block overrides it. Sharing is cluster configuration through the GPU Operator, not a Helm value. Applying `devicePlugin.config.default: "any"` is a cluster-administrator change that oversubscribes every eligible GPU Operator node and can affect unrelated GPU workloads. Time-slicing does not isolate GPU memory. Combined VRAM of the scheduled NIMs must still fit on the physical GPU. MIG is an advanced GPU Operator configuration outside this chart. The chart does not set a MIG strategy, MIG profile, or MIG resource requests.

For the time-slicing ConfigMap, ClusterPolicy patch, node placement, and preflight commands, refer to [GPU scheduling prerequisite](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#gpu-scheduling-prerequisite) in the Helm chart README. For pods that stay `Pending` on `nvidia.com/gpu`, refer to [Core NIM pods stay Pending for GPU](troubleshoot.md#helm-pending-gpus).

## Hardware Requirements { #hardware-requirements }

The full ingestion pipeline is designed to consume significant CPU and memory resources to achieve maximal parallelism. 
Resource usage scales up to the limits of your deployed system.

For per-feature GPU memory, disk, and co-residency rules, refer to [Model hardware requirements](#model-hardware-requirements) below.


### Recommended Production Deployment Specifications

- **System Memory**: At least 256 GB RAM
- **CPU Cores**: At least 32 CPU cores
- **GPU**: NVIDIA GPU with at least 24 GB VRAM (for example, A100, H100, L40S, or equivalent)

> **Note**
>
> Using less powerful systems or lower resource limits is still viable, but performance will suffer.

### Resource Consumption Notes

- Batch mode sizes unspecified actor pools from the CPU and GPU resources that Ray reports as available at pipeline submission.
- Explicit batch worker counts and direct Ray node overrides must fit the available resource budget. The library rejects infeasible plans before submission.
- Memory usage can reach up to the full system capacity for large document processing
- CPU utilization scales with the number of concurrent processing tasks
- GPU is required for inference using HuggingFace models or NIMs
- GPU is NOT required for build.nvidia.com hosted inference

### Scaling Considerations

For production deployments processing large volumes of documents, consider:
- Higher memory configurations for processing large PDF files or image collections
- Additional CPU cores for improved parallel processing
- Multiple GPUs for distributed processing workloads

### Environment Requirements

Ensure your deployment environment meets these specifications before running the full pipeline. Resource-constrained environments may experience performance degradation.

## Core and Advanced Pipeline Features { #core-and-advanced-pipeline-features }

The NeMo Retriever Library extraction core pipeline features have a combined GPU memory footprint that fits on a single A10G or better GPU (about 4.8 GiB). That figure is model capacity. It is not a Kubernetes scheduling guarantee.

The default Helm chart creates four independent NIMService workloads. Each requests `nvidia.com/gpu: 1`. On a conventional Kubernetes GPU cluster without MIG, time-slicing, or another sharing mechanism, you need **four allocatable GPU slots across eligible nodes**. To run all four core NIMs on one physical GPU, configure GPU sharing on a single-GPU target node before you install, and pin the four NIMServices to that node. Refer to [Kubernetes Helm GPU scheduling](#kubernetes-helm-gpu-scheduling).

Optional advanced features (audio and video transcription, Nemotron Parse, Omni image captioning, and the VL reranker) are **not** part of that core footprint. Audio, video, Nemotron Parse, and Omni captioning each need **one or more additional dedicated GPUs** beyond the GPU running the four core NIMs. The VL reranker can share the core GPU when it has at least 80 GB VRAM. Capacity requirements are listed in the **Additional Dedicated GPUs** rows of the [model hardware requirements](#model-hardware-requirements) table below. On a conventional exclusive-GPU Helm cluster, each optional NIMService adds its Helm GPU request. Most optional NIMs request one GPU slot. The default `answer_llm` Super-49B NIMService requests two.

<a id="optional-helm-nims-not-auto-wired-by-default"></a>

### Default NIMs { #default-helm-nims }

> **Important — NVAIE support applies to individual NIMs only**
>
> A NIM or model listed in the default and optional NIM rows in the table below might be supported under NVIDIA AI Enterprise (NVAIE) as an individual product. That support does **not** cover its use through NeMo Retriever Library or extend to the library, its container image, its Helm chart, or the end-to-end extraction workflow.

The production Helm chart reconciles NIM microservices through `nimOperator.<key>.enabled`. Four core NIMs are **enabled by default** and auto-wired into the retriever service; optional NIMs reconcile only when you opt in. For chart keys, image overrides, and enablement, refer to the [NeMo Retriever Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#nim-operator-sub-stack) and [Recommended minimal install](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#recommended-minimal-install-2608).

| Helm flag | NIM | Default image (`repository:tag`) | Role | Enabled by default |
|-----------|-----|----------------------------------|------|--------------------|
| `page_elements` | [nemotron-page-elements-v3](https://build.nvidia.com/nvidia/nemotron-page-elements-v3) | `nvcr.io/nim/nvidia/nemotron-object-detection:2.0.1` | Page layout and element detection | Yes |
| `table_structure` | [nemotron-table-structure-v1](https://build.nvidia.com/nvidia/nemotron-table-structure-v1) | `nvcr.io/nim/nvidia/nemotron-object-detection:2.0.1` | Table structure extraction | Yes |
| `ocr` | [nemotron-ocr-v2](https://build.nvidia.com/nvidia/nemotron-ocr-v2) | `nvcr.io/nim/nvidia/nemotron-ocr-v2:2.0.1` | Image OCR | Yes |
| `vlm_embed` | [llama-nemotron-embed-vl-1b-v2](https://build.nvidia.com/nvidia/llama-nemotron-embed-vl-1b-v2) | `nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:2.3.0` | Multimodal (VL) embedding | Yes |
| `rerankqa` | [llama-nemotron-rerank-vl-1b-v2](https://build.nvidia.com/nvidia/llama-nemotron-rerank-vl-1b-v2) | `nvcr.io/nim/nvidia/llama-nemotron-rerank-vl-1b-v2:2.3.0` | Reranking for improved retrieval accuracy | No |
| `nemotron_parse` | [nemotron-parse](https://build.nvidia.com/nvidia/nemotron-parse) | `nvcr.io/nim/nvidia/nemotron-parse-v1.2:1.7.0-variant` | Optional PDF `method="nemotron_parse"` (default PDF extraction uses **pdfium**). You can select Parse v2.0 with an image override. | No |
| `nemotron_3_nano_omni_30b_a3b_reasoning` | [nemotron-3-nano-omni-30b-a3b-reasoning](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning) | `nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:1.7.0-variant` | Image captioning when you enable the caption stage | No |
| `audio` | [parakeet-1-1b-ctc-en-us](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/index.html) | `nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us:1.5.0` | [Audio and video](audio-video.md) transcription | No |
| `answer_llm` | [llama-3.3-nemotron-super-49b-v1.5](https://build.nvidia.com/nvidia/llama-3.3-nemotron-super-49b-v1.5) | `nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1.5:2.0.5` | Optional `/v1/answer` generation LLM (not part of the default extraction pipeline) | No |

The `page_elements` and `table_structure` services share the combined `nemotron-object-detection:2.0.1` image and select distinct models. For air-gapped, mirrored, or allowlisted deployments, pull that image once. Do not treat the older standalone `nemotron-page-elements-v3` or `nemotron-table-structure-v1` container images as the current Helm defaults.

### Configure query reranking with Helm { #configure-query-reranking-with-helm }

The optional `nimOperator.rerankqa` NIM is not auto-wired into the retriever service. To use service query reranking, enable the NIM and configure its in-cluster ranking endpoint. You can also configure the model ID. Add the following values to your Helm values file:

```yaml
nimOperator:
  rerankqa:
    enabled: true

serviceConfig:
  nimEndpoints:
    rerankInvokeUrl: http://llama-nemotron-rerank-vl-1b-v2:8000/v1/ranking
    rerankModelName: nvidia/llama-nemotron-rerank-vl-1b-v2
```

The chart renders these values as `nim_endpoints.rerank_invoke_url` and `nim_endpoints.rerank_model_name` in the retriever service configuration. After deployment, send `rerank: true` in a `/v1/query` request to use the configured reranker.

Setting `nimOperator.rerankqa.enabled=true` without `serviceConfig.nimEndpoints.rerankInvokeUrl` deploys the NIM but does not enable query reranking.

<a id="nemotron-ocr-v2-language-mode"></a>

### Default NVCF endpoints { #default-nvcf-endpoints }

When you call [NVIDIA-hosted NIMs](deployment-options.md#when-to-use-nvidia-hosted-nims) from the Python library or CLI, these are the default remote endpoints the library uses when you do not set invoke URLs. Self-hosted Helm NIMs use in-cluster service URLs instead (refer to the [Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#nim-operator-sub-stack)).

| NIM | Default hosted endpoint | Notes |
|-----|-------------------------|-------|
| nemotron-page-elements-v3 | `https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3` | Core layout detection |
| nemotron-table-structure-v1 | `https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1` | Core table structure |
| nemotron-ocr-v2 | `https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v2` | Chart default OCR SKU; library CPU actors default to this URL when no OCR invoke URL is set. **Local OCR language selectors (`--ocr-lang`, API `ocr_lang`) are not sent on remote requests** — hosted OCR v2 uses its own language behavior |
| llama-nemotron-embed-vl-1b-v2 | `https://integrate.api.nvidia.com/v1/embeddings` with model ID `nvidia/llama-nemotron-embed-vl-1b-v2` | Core multimodal embedding |
| llama-nemotron-rerank-vl-1b-v2 | `https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-vl-1b-v2/reranking` | Optional VL reranker |
| nemotron-parse | `https://integrate.api.nvidia.com/v1/chat/completions` with model ID `nvidia/nemotron-parse` | Optional `method="nemotron_parse"`. Hosted Build, self-hosted Parse v1.2, and self-hosted Parse v2.0 use different request contracts. Refer to [Nemotron Parse: hosted Build endpoint vs self-hosted NIM](#nemotron-parse-hosted-vs-self-hosted) |
| nemotron-3-nano-omni-30b-a3b-reasoning | `https://integrate.api.nvidia.com/v1/chat/completions` with model ID `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | Optional image captioning |
| llama-3.3-nemotron-super-49b-v1.5 | `https://integrate.api.nvidia.com/v1/chat/completions` with model ID `nvidia/llama-3.3-nemotron-super-49b-v1.5` | Optional `/v1/answer` (Helm `answer_llm`) and OpenAI-compatible agentic RAG endpoint mode; not part of the default extraction pipeline. Agentic query/harness runs default to local in-process vLLM instead. Helm auto-wires to the in-cluster NIM when `nimOperator.answer_llm` is enabled. Refer to [Workflow: Agentic retrieval](workflow-agentic-retrieval.md). |
| parakeet-1-1b-ctc-en-us | `grpc.nvcf.nvidia.com:443` (function ID from [build.nvidia.com](https://build.nvidia.com/)) | Optional ASR; refer to [Parakeet hosted inference](audio-video.md#parakeet-hosted-inference-build-nvidia) |

<a id="nemotron-parse-hosted-vs-self-hosted"></a>

!!! note "Nemotron Parse: hosted Build and self-hosted NIM contracts"

    NVIDIA Build and self-hosted Nemotron Parse use distinct request contracts:

    - **Hosted Build** (`https://integrate.api.nvidia.com/v1/chat/completions`) resolves to model ID `nvidia/nemotron-parse` and uses an image-only tool-call contract.
    - **Self-hosted Parse v1.2** uses `nvidia/nemotron-parse-v1.2` and the tagged text-prompt contract. The Helm chart defaults to `nvcr.io/nim/nvidia/nemotron-parse-v1.2:1.7.0-variant`.
    - **Self-hosted Parse v2.0** uses `nvidia/nemotron-parse-v2.0` and its v2.0 contract. It is an optional Helm image override, not the chart default.

    Local Hugging Face inference defaults to `nvidia/NVIDIA-Nemotron-Parse-v1.2`. Set `nemotron_parse_model="nvidia/NVIDIA-Nemotron-Parse-2.0"` to use Parse 2.0 locally.

    To use hosted Build, set `nemotron_parse_invoke_url` to the Build chat-completions URL and set `method="nemotron_parse"`. You can normally omit `nemotron_parse_model` so the library selects the model automatically. If you set `nemotron_parse_model` explicitly, it must match the endpoint contract. Mixed Build and self-hosted endpoint lists require an explicit model.

    When the chart manages the Parse NIM, it wires the model that matches the selected v1.2 or v2.0 image. To select Parse v2.0, enable Parse and set `nimOperator.nemotron_parse.image.repository=nvcr.io/nim/nvidia/nemotron-parse-v2.0` and `nimOperator.nemotron_parse.image.tag=2.0.8-variant`. For a direct external v2.0 endpoint, set `nemotron_parse_model="nvidia/nemotron-parse-v2.0"` explicitly with its invoke URL.

    For model/endpoint mismatch symptoms, refer to [Nemotron Parse model and endpoint mismatch](troubleshoot.md#nemotron-parse-model-endpoint-mismatch).

For local Hugging Face OCR language mode (`multi` vs `english`), Helm OCR image overrides, and local model install, refer to [OCR and scanned documents](multimodal-extraction.md#ocr-and-scanned-documents), [OCR NIM configuration](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#ocr-nim-configuration), and [CLI — OCR language mode](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/docs/cli/README.md#ocr-language-mode).

### Image captioning { #image-captioning }

Use **`nemotron_3_nano_omni_30b_a3b_reasoning`** when you enable the caption stage (hosted model ID `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`). The Helm key is in the [Default NIMs](#default-helm-nims) table above.

Optional features in the table above require GPU capacity **beyond the four default NIMs**. Audio and video transcription, Nemotron Parse, and Omni image captioning each need a **dedicated additional GPU** (or two, for Omni on L40S) separate from the core pipeline GPU. The VL reranker can share the core GPU only when that GPU has at least 80 GB of VRAM. Otherwise, treat the reranker as a standalone workload. Each optional feature also needs extra disk space and feature-specific system dependencies. On a conventional exclusive-GPU Helm cluster, each optional NIMService also adds its Helm GPU request. Most optional NIMs request one GPU slot. The default `answer_llm` Super-49B NIMService requests two.

For published NIM model IDs and deployment-specific constraints, use the product support matrices linked under [Related Topics](#related-topics) below.

The Omni rows in the following table describe self-hosted NIM deployments. Direct local Hugging Face inference uses vLLM and also needs GPU memory for the KV cache and runtime allocations. For the local Omni BF16 profile, NeMo Retriever Library uses `gpu_memory_utilization=0.95` by default. On an 80 GB GPU, dedicate the GPU to local captioning. A checkpoint's approximately 62 GiB weight size alone does not establish local vLLM capacity.

## Model Hardware Requirements { #model-hardware-requirements }

NeMo Retriever Library supports the following GPU hardware given system constraints in the table.

**Total GPUs** for Core Features is combined VRAM co-residency for the four default NIMs (one A10G or better). It is not the number of exclusive Kubernetes GPU requests the default Helm chart makes. Without GPU sharing, plan for four allocatable GPU slots across eligible nodes. Refer to [Kubernetes Helm GPU scheduling](#kubernetes-helm-gpu-scheduling).

**Additional Dedicated GPUs** counts VRAM required **in addition to** that core co-residency. For example, a library-mode or GPU-shared deployment that runs the core pipeline on one H100 and self-hosted Parakeet ASR needs **two GPUs total**. On a conventional exclusive-GPU Helm cluster, the same optional ASR NIM adds a fifth allocatable GPU slot. Enabling the default `answer_llm` Super-49B profile adds two physical GPUs, not one. Time-slice replicas cannot satisfy that tensor-parallel request.

- **HF model weights** — approximate Hugging Face checkpoint footprint (files such as `model*.safetensors`, `weights.pth`, or other published weight bundles in the model repository). Values are rounded from the current public file listing and can change when the repository is updated.
- **NIM disk space** — approximate container and on-disk model cache for self-hosted NIM microservices (not the same as HF download size). For Nemotron 3 Nano Omni captioning, refer to the [NVIDIA NIM for Vision Language Models support matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html#nemotron-3-nano-omni-30b-a3b-reasoning).

Model repositories and NIM references are linked in [Core and Advanced Pipeline Features](#core-and-advanced-pipeline-features) above.

**B200, H200 NVL, and audio/video extraction:** The [audio and video](audio-video.md) transcription path (self-hosted Parakeet ASR through `nimOperator.audio`) is **not supported on B200**, other Blackwell GPUs, or **H200 NVL**. Core PDF and multimodal extraction on those GPUs is unchanged. Refer to footnote ⁴ below.

| Feature | HF Model Weights | GPU Option | [RTX Pro 6000](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/) | [B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) | [H200 NVL](https://www.nvidia.com/en-us/data-center/h200/) | [H100](https://www.nvidia.com/en-us/data-center/h100/) | [A100 80GB](https://www.nvidia.com/en-us/data-center/a100/) | A100 40GB | [A10G](https://aws.amazon.com/ec2/instance-types/g5/) | L40S | [RTX PRO 4500 Blackwell](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-4500/) |
|---------|------------------|------------|--------|--------|--------|--------|--------|--------|--------|--------|------------------------|
| GPU | — | Memory | 96GB | 180GB | 141GB | 80GB | 80GB | 40GB | 24GB | 48GB | 32GB GDDR7 (GB203) |
| Core Features | ~4.8 GiB combined: embed VL 1b ~3.1 GiB; page-elements ~0.41 GiB; table-structure ~0.81 GiB; OCR ~0.51 GiB | Total GPUs⁵ | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| Core Features | — | Total Disk Space | ~150GB | ~150GB | ~150GB | ~150GB | ~150GB | ~150GB | ~150GB | ~150GB | ~150GB |
| Audio/video extraction (parakeet-1-1b-ctc-en-us) | ~4.0 GiB (`model.safetensors`; the repo also ships `parakeet-ctc-1.1b.nemo` of similar size—use one format to avoid roughly doubling disk use) | Additional Dedicated GPUs | Not supported⁴ | Not supported⁴ | Not supported⁴ | 1¹ | 1¹ | 1¹ | 1¹ | 1¹ | Not supported⁴ |
| | — | Additional Disk Space | Not supported⁴ | Not supported⁴ | Not supported⁴ | ~37GB¹ | ~37GB¹ | ~37GB¹ | ~37GB¹ | ~37GB¹ | Not supported⁴ |
| nemotron-parse | ~3.5 GiB | Additional Dedicated GPUs | Not supported | 1 | Not supported | 1 | 1 | 1 | 1 | 1 | 1 |
| nemotron-parse | — | Additional Disk Space | Not supported | ~16GB | Not supported | ~16GB | ~16GB | ~16GB | ~16GB | ~16GB | ~16GB |
| Omni caption (nemotron-3-nano-omni-30b-a3b-reasoning) | ~62 GiB (BF16); ~33 GiB (FP8); ~21 GiB (NVFP4) | Additional Dedicated GPUs | 1 | 1 | 1 | 1 | 1 | Not supported | Not supported | 2 | Not supported³ |
| Omni caption (nemotron-3-nano-omni-30b-a3b-reasoning) | — | Additional Disk Space (HF) | ~21–62GB | ~21–62GB | ~21–62GB | ~21–62GB | ~21–62GB | Not supported | Not supported | ~21–62GB | Not supported³ |
| Omni caption (nemotron-3-nano-omni-30b-a3b-reasoning) | — | Additional Disk Space (NIM) | ~80GB | ~80GB | ~80GB | ~80GB | ~80GB | Not supported | Not supported | ~80GB | Not supported³ |
| Reranker | ~3.1 GiB (llama-nemotron-rerank-vl-1b-v2) | With Core Pipeline | Yes | Yes | Yes | Yes | Yes | No* | No* | No* | No* |
| Reranker | — | Standalone (recall only) | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

¹ On other supported GPUs, Parakeet ASR (`parakeet-1-1b-ctc-en-us:1.5.0`) may require a runtime TensorRT engine build (no prebuilt profile in the chart image).

⁴ Self-hosted [audio/video extraction](audio-video.md) through Parakeet ASR (`parakeet-1-1b-ctc-en-us:1.5.0`, `nimOperator.audio`) is **not supported** on **B200**, other **Blackwell** GPUs (compute capability 12.0), including RTX PRO 6000 Blackwell and RTX PRO 4500 Blackwell, or **H200 NVL**. Core PDF and multimodal extraction on those GPUs is unchanged. Video workflows that depend on Parakeet for speech transcription are affected the same way. `NIMService` for `nimOperator.audio` may stay not Ready or enter `CrashLoopBackOff` while building the Riva/TensorRT engine (for example ONNX Runtime IR version, cuDNN visibility, or FP8 tactic errors). Use a supported dedicated GPU (for example H100 or A100), [hosted Parakeet on build.nvidia.com](audio-video.md#parakeet-hosted-inference-build-nvidia), or set `nimOperator.audio.enabled=false`.

³ Opt-in Omni captioning uses the [nemotron-3-nano-omni-30b-a3b-reasoning](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-nano-omni-30b-a3b-reasoning) NIM (`nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:1.7.0-variant`). BF16 requires at least 80 GB total GPU memory for this NIM deployment; refer to the [VLM NIM support matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html#nemotron-3-nano-omni-30b-a3b-reasoning). L40S requires two GPUs. A100 40GB, A10G, and RTX PRO 4500 are below the minimum.

⁵ **Total GPUs** is combined VRAM co-residency for the four core models. The default Helm chart still requests `nvidia.com/gpu: 1` per NIMService. Without GPU sharing, plan for four allocatable GPU slots across eligible nodes. Refer to [Kubernetes Helm GPU scheduling](#kubernetes-helm-gpu-scheduling).

\* GPUs with less than 80GB VRAM cannot run the reranker concurrently with the core pipeline. 
To perform recall testing with the reranker on these GPUs, shut down the core pipeline NIM microservices 
and run only the embedder, reranker, and your vector database.

## Related Topics { #related-topics }

- [Troubleshooting](troubleshoot.md)
- [Release Notes](releasenotes.md)
- [Deployment options](deployment-options.md) (local Python, hosted NIMs, and Kubernetes)
- [Deploy with Helm](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md)
- [Persistent storage prerequisite](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#persistent-storage-prerequisite)
- [GPU scheduling prerequisite](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#gpu-scheduling-prerequisite)
- [NVIDIA NIM for Object Detection (support matrix)](https://docs.nvidia.com/nim/ingestion/object-detection/latest/support-matrix.html)
- [NVIDIA NIM for Image OCR (support matrix)](https://docs.nvidia.com/nim/ingestion/image-ocr/latest/support-matrix.html)
- [NVIDIA NIM for Vision Language Models (support matrix)](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html)
- [NVIDIA Speech NIM Microservices (support matrix)](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/index.html)
