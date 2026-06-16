# Pre-Requisites & Support Matrix

Before you begin using [NeMo Retriever Library](overview.md), confirm your software stack, deployment hardware, and—if you use them—advanced features (audio and video, Nemotron Parse, VLM image captioning, reranking) against the guidance in this page.

## Software Requirements

- Linux operating systems (Ubuntu 22.04 or later recommended)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (NVIDIA Driver >= `580`, CUDA >= `13.0`)
- [Python](https://www.python.org/downloads/) `3.12` — required to install and run the NeMo Retriever Library Python API, CLI, and related packages from PyPI (for example `pip` or `uv`). Older Python versions will fail dependency resolution without a clear error.
- [UV Python package and environment manager](https://docs.astral.sh/uv/getting-started/installation/) (optional; recommended for creating isolated environments)
- For audio and video, `ffmpeg` and `ffprobe` must be on `PATH` (for example
  `sudo apt-get install -y --no-install-recommends ffmpeg` on Debian/Ubuntu).
  `ffmpeg-python` and `nemo-retriever[multimedia]` do not install these binaries.
  On Helm with package-repo access, set `service.installFfmpeg=true`. For
  air-gapped clusters, see [Air-gapped and disconnected deployment](deployment-options.md#air-gapped-deployment).
- For PDF extraction with `extract_method="nemotron_parse"`, install the Nemotron Parse
  client dependencies with `pip install "nemo-retriever[nemotron-parse]"` (pulls
  `open-clip-torch`, which provides the `open_clip` module required by the Nemotron Parse
  NIM client). The base `nemo-retriever` install and `[local]` extra do not include this
  package.

!!! note

    When you use UV, create the environment with Python 3.12 — for example, `uv venv --python 3.12`. This matches the `requires-python` metadata in the library packages.

## Hardware Requirements

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

## Core and Advanced Pipeline Features

The NeMo Retriever Library extraction core pipeline features run on a single A10G or better GPU.

### Default Helm NIMs

The production Helm chart enables these NIM microservices **by default** (for example via `nimOperator.*.enabled=true`):

| Helm flag | NIM | Role |
|-----------|-----|------|
| `page_elements` | [nemotron-page-elements-v3](https://huggingface.co/nvidia/nemotron-page-elements-v3) | Page layout and element detection |
| `table_structure` | [nemotron-table-structure-v1](https://huggingface.co/nvidia/nemotron-table-structure-v1) | Table structure extraction |
| `ocr` | [nemotron-ocr-v1](https://huggingface.co/nvidia/nemotron-ocr-v1) | Image OCR |
| `vlm_embed` | [llama-nemotron-embed-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2) | Multimodal (VL) embedding |

### OCR artifacts (Helm vs local Hugging Face) { #nemotron-ocr-v2-language-mode }

!!! note

    **Helm / NIM:** The production chart deploys **Nemotron OCR v1** under `nimOperator.ocr` (`nvcr.io/nim/nvidia/nemotron-ocr-v1:1.3.0`). For image defaults and upgrade notes, see [OCR NIM configuration](https://github.com/NVIDIA/NeMo-Retriever/blob/26.05/nemo_retriever/helm/README.md#ocr-nim-configuration) in the Helm chart README.

    **Local Hugging Face inference:** When you deploy locally with HuggingFace model weights (for example `pip install "nemo-retriever[local]"` and GPU inference without remote OCR NIM URLs), the default OCR engine is **Nemotron OCR v2**, which runs in **multilingual** mode by default. For CLI flags and API parameters, see [Nemotron OCR v2 — language mode](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/docs/cli/README.md#nemotron-ocr-v2-language-mode). Remote OCR NIM endpoints use their own model and language behavior; local OCR language selectors are not sent on remote requests.

Default OCR NIM container for release Helm deployments:

- **Image:** `nvcr.io/nim/nvidia/nemotron-ocr-v1:1.3.0`

Default VL embedder container and model for release deployments:

- **Image:** `nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:1.12.0`
- **Model ID:** `nvidia/llama-nemotron-embed-vl-1b-v2`

### Optional Helm NIMs (not auto-wired) { #optional-helm-nims-not-auto-wired-by-default }

These NIM microservices are **optional** for the default extraction pipeline. The retriever service does **not** call them until you enable the matching pipeline stage (reranker, Nemotron Parse, caption, or audio). For **26.05 production**, disable keys you do not need (see [Recommended minimal install (26.05)](https://github.com/NVIDIA/NeMo-Retriever/blob/26.05/nemo_retriever/helm/README.md#recommended-minimal-install-2605)). Set `nimOperator.<key>.enabled=true` when you want that NIM reconciled. Chart keys are in the [NeMo Retriever Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/26.05/nemo_retriever/helm/README.md#nim-operator-sub-stack).

| Helm flag | NIM | Role |
|-----------|-----|------|
| `rerankqa` | [llama-nemotron-rerank-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2) | Reranking for improved retrieval accuracy |
| `nemotron_parse` | [nemotron-parse](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2) | Optional PDF `extract_method="nemotron_parse"` (default PDF extraction uses **pdfium**) |
| `nemotron_3_nano_omni_30b_a3b_reasoning` | [nemotron-3-nano-omni-30b-a3b-reasoning](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) | Supported image captioning for 26.05 when you enable the caption stage |
| `audio` | [parakeet-1-1b-ctc-en-us](https://huggingface.co/nvidia/parakeet-ctc-1.1b) | [Audio and video](audio-video.md) transcription |

### Image captioning (26.05) { #image-captioning-2605 }

For 26.05, use **`nemotron_3_nano_omni_30b_a3b_reasoning`** when you enable the caption stage (hosted model ID `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`). The Helm key is in the [optional NIMs](#optional-helm-nims-not-auto-wired-by-default) table above.

!!! important "PDF chart regions are not captioned by Omni"

    When **nemotron-page-elements-v3** classifies a PDF region as **chart**, that region is processed through layout detection and OCR—not the Omni caption stage. Enabling the caption NIM and the `caption` pipeline stage does **not** send chart-labeled figures to `/v1/chat/completions`.

    The caption stage covers:

    - Unstructured content in the `images` column (standalone image files and page-element regions **not** classified as table, chart, or infographic)
    - Optional infographic regions when you set `caption_infographics=True` on `CaptionParams` (the VLM caption is stored in `caption`, separate from OCR `text`)

    To validate caption traffic during ingest, inspect metadata such as `page_elements_v3_counts_by_label`. If the figure is labeled `chart`, expect no Omni chat-completions requests for that region even when captioning is enabled.

Optional features listed in the table above require additional GPU support, disk space, and feature-specific system dependencies beyond the four default NIMs.

For published NIM model IDs and deployment-specific constraints, use the product support matrices linked under [Related Topics](#related-topics) below.

## Model Hardware Requirements

NeMo Retriever Library supports the following GPU hardware given system constraints in the table.

- **HF model weights** — approximate Hugging Face checkpoint footprint (files such as `model*.safetensors`, `weights.pth`, or other published weight bundles in the model repository). Values are rounded from the current public file listing and can change when the repository is updated.
- **NIM disk space** — approximate container and on-disk model cache for self-hosted NIM microservices (not the same as HF download size). For Nemotron 3 Nano Omni captioning, see the [NVIDIA NIM for Vision Language Models support matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html#nemotron-3-nano-omni-30b-a3b-reasoning).

Model repositories and NIM references are linked in [Core and Advanced Pipeline Features](#core-and-advanced-pipeline-features) above.

**B200 and audio/video extraction (26.05):** The [audio and video](audio-video.md) transcription path (self-hosted Parakeet ASR via `nimOperator.audio`) is **not supported on B200** or other Blackwell GPUs. Core PDF and multimodal extraction on B200 is unchanged. See footnote ⁴ below.

| Feature | HF Model Weights | GPU Option | [RTX Pro 6000](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/) | [B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) | [H200 NVL](https://www.nvidia.com/en-us/data-center/h200/) | [H100](https://www.nvidia.com/en-us/data-center/h100/) | [A100 80GB](https://www.nvidia.com/en-us/data-center/a100/) | A100 40GB | [A10G](https://aws.amazon.com/ec2/instance-types/g5/) | L40S | [RTX PRO 4500 Blackwell](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-4500/) |
|---------|------------------|------------|--------|--------|--------|--------|--------|--------|--------|--------|------------------------|
| GPU | — | Memory | 96GB | 180GB | 141GB | 80GB | 80GB | 40GB | 24GB | 48GB | 32GB GDDR7 (GB203) |
| Core Features | ~4.8 GiB combined: embed VL 1b ~3.1 GiB; page-elements ~0.41 GiB; table-structure ~0.81 GiB; OCR ~0.51 GiB | Total GPUs | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| Core Features | — | Total Disk Space | ~150GB | ~150GB | ~150GB | ~150GB | ~150GB | ~150GB | ~150GB | ~150GB | ~150GB |
| Audio/video extraction (parakeet-1-1b-ctc-en-us) | ~4.0 GiB (`model.safetensors`; the repo also ships `parakeet-ctc-1.1b.nemo` of similar size—use one format to avoid roughly doubling disk use) | Additional Dedicated GPUs | Not supported⁴ | Not supported⁴ | 1¹ | 1¹ | 1¹ | 1¹ | 1¹ | 1¹ | Not supported⁴ |
| | — | Additional Disk Space | Not supported⁴ | Not supported⁴ | ~37GB¹ | ~37GB¹ | ~37GB¹ | ~37GB¹ | ~37GB¹ | ~37GB¹ | Not supported⁴ |
| nemotron-parse | ~3.5 GiB | Additional Dedicated GPUs | Not supported | 1 | Not supported | 1 | 1 | 1 | 1 | 1 | 1 |
| nemotron-parse | — | Additional Disk Space | Not supported | ~16GB | Not supported | ~16GB | ~16GB | ~16GB | ~16GB | ~16GB | ~16GB |
| Omni caption (nemotron-3-nano-omni-30b-a3b-reasoning) | ~62 GiB (BF16); ~33 GiB (FP8); ~21 GiB (NVFP4) | Additional Dedicated GPUs | 1 | 1 | 1 | 1 | 1 | Not supported | Not supported | 2 | Not supported³ |
| Omni caption (nemotron-3-nano-omni-30b-a3b-reasoning) | — | Additional Disk Space (HF) | ~21–62GB | ~21–62GB | ~21–62GB | ~21–62GB | ~21–62GB | Not supported | Not supported | ~21–62GB | Not supported³ |
| Omni caption (nemotron-3-nano-omni-30b-a3b-reasoning) | — | Additional Disk Space (NIM) | ~80GB | ~80GB | ~80GB | ~80GB | ~80GB | Not supported | Not supported | ~80GB | Not supported³ |
| Reranker | ~3.1 GiB (llama-nemotron-rerank-vl-1b-v2) | With Core Pipeline | Yes | Yes | Yes | Yes | Yes | No* | No* | No* | No* |
| Reranker | — | Standalone (recall only) | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

¹ On other supported GPUs, Parakeet ASR (`parakeet-1-1b-ctc-en-us:1.5.0`) may require a runtime TensorRT engine build (no prebuilt profile in the chart image).

⁴ On **B200** and other **Blackwell** GPUs (compute capability 12.0), including RTX PRO 6000 Blackwell and RTX PRO 4500 Blackwell, self-hosted [audio/video extraction](audio-video.md) via Parakeet ASR (`parakeet-1-1b-ctc-en-us:1.5.0`, `nimOperator.audio`) is **not supported**. Core PDF and multimodal extraction on Blackwell is unchanged. Video workflows that depend on Parakeet for speech transcription are affected the same way. `NIMService` for `nimOperator.audio` may stay not Ready or enter `CrashLoopBackOff` while building the Riva/TensorRT engine (for example ONNX Runtime IR version, cuDNN visibility, or FP8 tactic errors). Use a non-Blackwell dedicated GPU, [hosted Parakeet on build.nvidia.com](audio-video.md#parakeet-hosted-inference-build-nvidia), or set `nimOperator.audio.enabled=false`.

³ Opt-in Omni captioning uses the [nemotron-3-nano-omni-30b-a3b-reasoning](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-nano-omni-30b-a3b-reasoning) NIM (`nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:1.7.0-variant`). BF16 requires at least 80 GB total GPU memory; see the [VLM NIM support matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html#nemotron-3-nano-omni-30b-a3b-reasoning). L40S requires two GPUs. A100 40GB, A10G, and RTX PRO 4500 are below the minimum.

\* GPUs with less than 80GB VRAM cannot run the reranker concurrently with the core pipeline. 
To perform recall testing with the reranker on these GPUs, shut down the core pipeline NIM microservices 
and run only the embedder, reranker, and your vector database.

## Related Topics

- [Troubleshooting](troubleshoot.md)
- [Release Notes](releasenotes.md)
- [Deployment options](deployment-options.md) (local Python, hosted NIMs, and Kubernetes)
- [Deploy with Helm](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md)
- [NVIDIA NIM for Object Detection (support matrix)](https://docs.nvidia.com/nim/ingestion/object-detection/latest/support-matrix.html)
- [NVIDIA NIM for Image OCR (support matrix)](https://docs.nvidia.com/nim/ingestion/image-ocr/latest/support-matrix.html)
- [NVIDIA NIM for Vision Language Models (support matrix)](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html)
- [NVIDIA Speech NIM Microservices (support matrix)](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/index.html)
