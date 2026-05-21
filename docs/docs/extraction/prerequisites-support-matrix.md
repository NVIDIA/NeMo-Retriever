# Pre-Requisites & Support Matrix

Before you begin using [NeMo Retriever Library](overview.md), confirm your software stack, deployment hardware, and—if you use them—advanced features (audio and video, Nemotron Parse, VLM image captioning, reranking) against the guidance in this page.

## Software Requirements

- Linux operating systems (Ubuntu 22.04 or later recommended)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (NVIDIA Driver >= `535`, CUDA >= `12.2`)
- [Python](https://www.python.org/downloads/) `3.12` — required to install and run the NeMo Retriever Library Python API, CLI, and related packages from PyPI (for example `pip` or `uv`). Older Python versions will fail dependency resolution without a clear error.
- [UV Python package and environment manager](https://docs.astral.sh/uv/getting-started/installation/) (optional; recommended for creating isolated environments)
- For audio and video extraction, the `ffmpeg` and `ffprobe` command-line
  binaries must be installed and available on `PATH`. On Debian/Ubuntu systems,
  install them with root privileges, for example
  `sudo apt-get update && sudo apt-get install -y --no-install-recommends ffmpeg`.
  Python packages such as `ffmpeg-python` or `nemo-retriever[multimedia]` do not
  provide these system binaries. For Helm deployments, set
  `service.installFfmpeg=true`.

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
| `ocr` | [nemotron-ocr-v2](https://huggingface.co/nvidia/nemotron-ocr-v2) | Image OCR |
| `vlm_embed` | [llama-nemotron-embed-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2) | Multimodal (VL) embedding |

Default VL embedder container and model for release deployments:

- **Image:** `nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:1.12.0`
- **Model ID:** `nvidia/llama-nemotron-embed-vl-1b-v2`

### Optional Helm NIMs (disabled by default)

Enable these only when your workload needs them — the same pattern as the **VL reranker** (not deployed unless you turn on the reranker flags):

- [llama-nemotron-rerank-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2) [NIM](https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest/overview.html) — reranking for improved retrieval accuracy
- [nemotron-parse](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2) [NIM](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-parse) — optional PDF `extract_method="nemotron_parse"` (default PDF extraction uses **pdfium**)

Advanced features (for example, audio and video, Nemotron Parse, VLM image captioning, reranking) require additional GPU support, disk space, and feature-specific system dependencies.
This includes the following:

- [parakeet-1-1b-ctc-en-us](https://huggingface.co/nvidia/parakeet-ctc-1.1b) [NIM](https://docs.nvidia.com/nim/speech/latest/index.html) — transcript extraction from [audio and video](audio-video.md)
- [nemotron-parse](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.2) [NIM](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-parse) — alternate PDF extraction method when you set `extract_method="nemotron_parse"` (default PDF extraction uses **pdfium**)
- [nemotron-3-nano-omni-30b-a3b-reasoning](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) [NIM](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-nano-omni-30b-a3b-reasoning) — optional image captioning when you enable the caption stage
- [llama-nemotron-rerank-vl-1b-v2](https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2) [NIM](https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest/overview.html) — reranking for improved retrieval accuracy

For published NIM model IDs and deployment-specific constraints, use the product support matrices linked under [Related Topics](#related-topics) below.

## Model Hardware Requirements

NeMo Retriever Library supports the following GPU hardware given system constraints in the table.

- **HF model weights** — approximate Hugging Face checkpoint footprint (files such as `model*.safetensors`, `weights.pth`, or other published weight bundles in the model repository). Values are rounded from the current public file listing and can change when the repository is updated.
- **NIM disk space** — approximate container and on-disk model cache for self-hosted NIM microservices (not the same as HF download size). For Nemotron 3 Nano Omni captioning, see the [NVIDIA NIM for Vision Language Models support matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html#nemotron-3-nano-omni-30b-a3b-reasoning).

Model repositories and NIM references are linked in [Core and Advanced Pipeline Features](#core-and-advanced-pipeline-features) above.

| Feature | HF Model Weights | GPU Option | [RTX Pro 6000](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/) | [B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) | [H200 NVL](https://www.nvidia.com/en-us/data-center/h200/) | [H100](https://www.nvidia.com/en-us/data-center/h100/) | [A100 80GB](https://www.nvidia.com/en-us/data-center/a100/) | A100 40GB | [A10G](https://aws.amazon.com/ec2/instance-types/g5/) | L40S | [RTX PRO 4500 Blackwell](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-4500/) |
|---------|------------------|------------|--------|--------|--------|--------|--------|--------|--------|--------|------------------------|
| GPU | — | Memory | 96GB | 180GB | 141GB | 80GB | 80GB | 40GB | 24GB | 48GB | 32GB GDDR7 (GB203) |
| Core Features | ~4.8 GiB combined: embed VL 1b ~3.1 GiB; page-elements ~0.41 GiB; table-structure ~0.81 GiB; OCR ~0.51 GiB | Total GPUs | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| Core Features | — | Total Disk Space | ~150GB | ~150GB | ~150GB | ~150GB | ~150GB | ~150GB | ~150GB | ~150GB | ~150GB |
| Audio (parakeet-1-1b-ctc-en-us) | ~4.0 GiB (`model.safetensors`; the repo also ships `parakeet-ctc-1.1b.nemo` of similar size—use one format to avoid roughly doubling disk use) | Additional Dedicated GPUs | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1¹ |
| Audio (parakeet-1-1b-ctc-en-us) | — | Additional Disk Space | ~37GB | ~37GB | ~37GB | ~37GB | ~37GB | ~37GB | ~37GB | ~37GB | ~37GB¹ |
| nemotron-parse | ~3.5 GiB | Additional Dedicated GPUs | Not supported | Not supported | Not supported | 1 | 1 | 1 | 1 | 1 | Not supported² |
| nemotron-parse | — | Additional Disk Space | Not supported | Not supported | Not supported | ~16GB | ~16GB | ~16GB | ~16GB | ~16GB | Not supported² |
| Omni caption (nemotron-3-nano-omni-30b-a3b-reasoning) | ~62 GiB (BF16); ~33 GiB (FP8); ~21 GiB (NVFP4) | Additional Dedicated GPUs | 1 | 1 | 1 | 1 | 1 | Not supported | Not supported | 2 | Not supported³ |
| Omni caption (nemotron-3-nano-omni-30b-a3b-reasoning) | — | Additional Disk Space (HF) | ~21–62GB | ~21–62GB | ~21–62GB | ~21–62GB | ~21–62GB | Not supported | Not supported | ~21–62GB | Not supported³ |
| Omni caption (nemotron-3-nano-omni-30b-a3b-reasoning) | — | Additional Disk Space (NIM) | ~80GB | ~80GB | ~80GB | ~80GB | ~80GB | Not supported | Not supported | ~80GB | Not supported³ |
| Reranker | ~3.1 GiB (llama-nemotron-rerank-vl-1b-v2) | With Core Pipeline | Yes | Yes | Yes | Yes | Yes | No* | No* | No* | No* |
| Reranker | — | Standalone (recall only) | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

¹ Audio runs but requires runtime engine build — no pre-defined model profile.

² Nemotron Parse fails to start on 32GB.

³ Opt-in Omni captioning uses the [nemotron-3-nano-omni-30b-a3b-reasoning](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-nano-omni-30b-a3b-reasoning) NIM (`nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:latest`). BF16 requires at least 80 GB total GPU memory; see the [VLM NIM support matrix](https://docs.nvidia.com/nim/vision-language-models/latest/support-matrix.html#nemotron-3-nano-omni-30b-a3b-reasoning). L40S requires two GPUs. A100 40GB, A10G, and RTX PRO 4500 are below the minimum.

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
