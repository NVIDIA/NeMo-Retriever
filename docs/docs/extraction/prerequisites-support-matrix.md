# Pre-Requisites & Support Matrix

Before you begin using [NeMo Retriever Library](overview.md), confirm your software stack, deployment hardware, and—if you use them—advanced features (audio and video, Nemotron Parse, VLM image captioning, reranking) against the guidance in this page.

!!! note "NVIDIA AI Enterprise (NVAIE) support"

    The NeMo Retriever Library, including its container image and Helm chart artifacts, is not supported under NVIDIA AI Enterprise (NVAIE), even though some NIM microservices and models it uses may be individually covered by NVAIE. For more information, refer to [NVIDIA AI Enterprise (NVAIE) support](overview.md#nvidia-ai-enterprise-nvaie-support).

## Software Requirements { #software-requirements }

- Linux operating systems (Ubuntu 22.04 or later recommended)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (NVIDIA Driver >= `580`, CUDA >= `13.0`)
- [Python](https://www.python.org/downloads/) `3.12` — required to install and run the NeMo Retriever Library Python API, CLI, and related packages from PyPI (for example `pip` or `uv`). Older Python versions will fail dependency resolution without a clear error.
- [UV Python package and environment manager](https://docs.astral.sh/uv/getting-started/installation/) (optional; recommended for creating isolated environments)
- For audio and video, `ffmpeg` and `ffprobe` must be on `PATH` (for example
  `sudo apt-get install -y --no-install-recommends ffmpeg` on Debian/Ubuntu).
  `ffmpeg-python` and `nemo-retriever[multimedia]` do not install these binaries.
  For container and Kubernetes guidance, refer to [Audio and video](audio-video.md).
- For PDF extraction with `extract_method="nemotron_parse"`, install the Nemotron Parse
  client dependencies with `pip install "nemo-retriever[nemotron-parse]"` (pulls
  `open-clip-torch`, which provides the `open_clip` module required by the Nemotron Parse
  NIM client). The base `nemo-retriever` install and `[local]` extra do not include this
  package.

!!! note

    When you use UV, create the environment with Python 3.12 — for example, `uv venv --python 3.12`. This matches the `requires-python` metadata in the library packages.

## Hardware Requirements { #hardware-requirements }

The full ingestion pipeline is designed to consume significant CPU and memory resources to achieve maximal parallelism. 
Resource usage scales up to the limits of your deployed system.

For per-feature GPU memory, disk, and co-residency rules, refer to [Model hardware requirements](#model-hardware-requirements) below.


### Recommended Production Deployment Specifications

- **System Memory**: At least 256 GB RAM
- **CPU Cores**: At least 32 CPU cores
- **GPU**: NVIDIA GPU with at least 24 GB VRAM (for example, A100, H100, L40S, or equivalent)

!!! note

    Using less powerful systems or lower resource limits is still viable, but performance will suffer.

### Resource Consumption Notes

- The pipeline performs runtime allocation of parallel resources based on system configuration
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

The NeMo Retriever Library extraction core pipeline features run on a single A10G or better GPU.

Optional advanced features—audio and video transcription, Nemotron Parse, Omni image captioning, and the VL reranker—are **not** part of that core footprint. Audio, video, Nemotron Parse, and Omni captioning each need **one or more additional dedicated GPUs** beyond the GPU running the four core NIMs; the VL reranker can share the core GPU when it has at least 80 GB VRAM. Capacity requirements are listed in the **Additional Dedicated GPUs** rows of the [model hardware requirements](#model-hardware-requirements) table below.

<a id="optional-helm-nims-not-auto-wired-by-default"></a>

### Default NIMs { #default-helm-nims }

!!! important "NVAIE support applies to individual NIMs only"

    A NIM or model listed in the default and optional NIM rows in the table below might be supported under NVIDIA AI Enterprise (NVAIE) as an individual product. That support does **not** cover its use through NeMo Retriever Library or extend to the library, its container image, its Helm chart, or the end-to-end extraction workflow.

The production Helm chart reconciles NIM microservices through `nimOperator.<key>.enabled`. Four core NIMs are **enabled by default** and auto-wired into the retriever service; optional NIMs reconcile only when you opt in. For chart keys, image overrides, and enablement, refer to the [NeMo Retriever Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#nim-operator-sub-stack) and [Recommended minimal install](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#recommended-minimal-install-2605).

| Helm flag | NIM | Default image (`repository:tag`) | Role | Enabled by default |
|-----------|-----|----------------------------------|------|--------------------|
| `page_elements` | [nemotron-page-elements-v3](https://build.nvidia.com/nvidia/nemotron-page-elements-v3) | `nvcr.io/nim/nvidia/nemotron-page-elements-v3:1.8.0` | Page layout and element detection | Yes |
| `table_structure` | [nemotron-table-structure-v1](https://build.nvidia.com/nvidia/nemotron-table-structure-v1) | `nvcr.io/nim/nvidia/nemotron-table-structure-v1:1.8.0` | Table structure extraction | Yes |
| `ocr` | [nemotron-ocr-v2](https://build.nvidia.com/nvidia/nemotron-ocr-v2) | `nvcr.io/nim/nvidia/nemotron-ocr-v2:1.4.0` | Image OCR | Yes |
| `vlm_embed` | [llama-nemotron-embed-vl-1b-v2](https://build.nvidia.com/nvidia/llama-nemotron-embed-vl-1b-v2) | `nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:1.12.0` | Multimodal (VL) embedding | Yes |
| `rerankqa` | [llama-nemotron-rerank-vl-1b-v2](https://build.nvidia.com/nvidia/llama-nemotron-rerank-vl-1b-v2) | `nvcr.io/nim/nvidia/llama-nemotron-rerank-vl-1b-v2:1.11.0` | Reranking for improved retrieval accuracy | No |
| `nemotron_parse` | [nemotron-parse](https://build.nvidia.com/nvidia/nemotron-parse) | `nvcr.io/nim/nvidia/nemotron-parse-v1.2:1.7.0-variant` | Optional PDF `extract_method="nemotron_parse"` (default PDF extraction uses **pdfium**) | No |
| `nemotron_3_nano_omni_30b_a3b_reasoning` | [nemotron-3-nano-omni-30b-a3b-reasoning](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning) | `nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:1.7.0-variant` | Image captioning when you enable the caption stage | No |
| `audio` | [parakeet-1-1b-ctc-en-us](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/index.html) | `nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us:1.5.0` | [Audio and video](audio-video.md) transcription | No |
| `answer_llm` | [llama-3.3-nemotron-super-49b-v1.5](https://build.nvidia.com/nvidia/llama-3.3-nemotron-super-49b-v1.5) | `nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1.5:2.0.5` | Optional `/v1/answer` generation LLM (not part of the default extraction pipeline) | No |

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
| nemotron-parse | `https://integrate.api.nvidia.com/v1/chat/completions` with model ID `nvidia/nemotron-parse` | Optional `extract_method="nemotron_parse"` |
| nemotron-3-nano-omni-30b-a3b-reasoning | `https://integrate.api.nvidia.com/v1/chat/completions` with model ID `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | Optional image captioning |
| llama-3.3-nemotron-super-49b-v1.5 | `https://integrate.api.nvidia.com/v1/chat/completions` with model ID `nvidia/llama-3.3-nemotron-super-49b-v1.5` | Optional `/v1/answer` (Helm `answer_llm`) and agentic RAG; not part of the default extraction pipeline. Helm auto-wires to the in-cluster NIM when `nimOperator.answer_llm` is enabled |
| parakeet-1-1b-ctc-en-us | `grpc.nvcf.nvidia.com:443` (function ID from [build.nvidia.com](https://build.nvidia.com/)) | Optional ASR; refer to [Parakeet hosted inference](audio-video.md#parakeet-hosted-inference-build-nvidia) |

For local Hugging Face OCR language mode (`multi` vs `english`), Helm OCR image overrides, and local model install, refer to [OCR and scanned documents](multimodal-extraction.md#ocr-and-scanned-documents), [OCR NIM configuration](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#ocr-nim-configuration), and [CLI — OCR language mode](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/docs/cli/README.md#ocr-language-mode).

### Image captioning { #image-captioning }

Use **`nemotron_3_nano_omni_30b_a3b_reasoning`** when you enable the caption stage (hosted model ID `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`). The Helm key is in the [Default NIMs](#default-helm-nims) table above.

Optional features in the table above require GPU capacity **beyond the four default NIMs**. Audio and video transcription, Nemotron Parse, and Omni image captioning each need a **dedicated additional GPU** (or two, for Omni on L40S) separate from the core pipeline GPU. The VL reranker can share the core GPU only when that GPU has at least 80 GB of VRAM. Otherwise, treat the reranker as a standalone workload. Each optional feature also needs extra disk space and feature-specific system dependencies.

For published NIM model IDs and deployment-specific constraints, use the product support matrices linked under [Related Topics](#related-topics) below.

## Model Hardware Requirements { #model-hardware-requirements }

NeMo Retriever Library supports the following GPU hardware given system constraints in the table.

**Additional Dedicated GPUs** counts GPUs required **in addition to** the one GPU reserved for the [core pipeline](#core-and-advanced-pipeline-features) (the four default NIMs). For example, a deployment that runs the core pipeline on one H100 and self-hosted Parakeet ASR needs **two GPUs total**: one for the core pipeline and one additional.

- **HF model weights** — approximate Hugging Face checkpoint footprint (files such as `model*.safetensors`, `weights.pth`, or other published weight bundles in the model repository). Values are rounded from the current public file listing and can change when the repository is updated.
- **NIM disk space** — approximate container and on-disk model cache for self-hosted NIM microservices (not the same as HF download size). For Nemotron 3 Nano Omni captioning, refer to the [NVIDIA NIM for Vision Language Models support matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html#nemotron-3-nano-omni-30b-a3b-reasoning).

Model repositories and NIM references are linked in [Core and Advanced Pipeline Features](#core-and-advanced-pipeline-features) above.

**B200, H200 NVL, and audio/video extraction:** The [audio and video](audio-video.md) transcription path (self-hosted Parakeet ASR through `nimOperator.audio`) is **not supported on B200**, other Blackwell GPUs, or **H200 NVL**. Core PDF and multimodal extraction on those GPUs is unchanged. Refer to footnote ⁴ below.

| Feature | HF Model Weights | GPU Option | [RTX Pro 6000](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/) | [B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) | [H200 NVL](https://www.nvidia.com/en-us/data-center/h200/) | [H100](https://www.nvidia.com/en-us/data-center/h100/) | [A100 80GB](https://www.nvidia.com/en-us/data-center/a100/) | A100 40GB | [A10G](https://aws.amazon.com/ec2/instance-types/g5/) | L40S | [RTX PRO 4500 Blackwell](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-4500/) |
|---------|------------------|------------|--------|--------|--------|--------|--------|--------|--------|--------|------------------------|
| GPU | — | Memory | 96GB | 180GB | 141GB | 80GB | 80GB | 40GB | 24GB | 48GB | 32GB GDDR7 (GB203) |
| Core Features | ~4.8 GiB combined: embed VL 1b ~3.1 GiB; page-elements ~0.41 GiB; table-structure ~0.81 GiB; OCR ~0.51 GiB | Total GPUs | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
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

³ Opt-in Omni captioning uses the [nemotron-3-nano-omni-30b-a3b-reasoning](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-nano-omni-30b-a3b-reasoning) NIM (`nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:1.7.0-variant`). BF16 requires at least 80 GB total GPU memory; refer to the [VLM NIM support matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html#nemotron-3-nano-omni-30b-a3b-reasoning). L40S requires two GPUs. A100 40GB, A10G, and RTX PRO 4500 are below the minimum.

\* GPUs with less than 80GB VRAM cannot run the reranker concurrently with the core pipeline. 
To perform recall testing with the reranker on these GPUs, shut down the core pipeline NIM microservices 
and run only the embedder, reranker, and your vector database.

## Related Topics { #related-topics }

- [Troubleshooting](troubleshoot.md)
- [Release Notes](releasenotes.md)
- [Deployment options](deployment-options.md) (local Python, hosted NIMs, and Kubernetes)
- [Deploy with Helm](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md)
- [NVIDIA NIM for Object Detection (support matrix)](https://docs.nvidia.com/nim/ingestion/object-detection/latest/support-matrix.html)
- [NVIDIA NIM for Image OCR (support matrix)](https://docs.nvidia.com/nim/ingestion/image-ocr/latest/support-matrix.html)
- [NVIDIA NIM for Vision Language Models (support matrix)](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html)
- [NVIDIA Speech NIM Microservices (support matrix)](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/index.html)
