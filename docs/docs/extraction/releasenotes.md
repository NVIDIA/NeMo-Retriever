# Release Notes for NeMo Retriever Library

This documentation contains the release notes for [NeMo Retriever Library](overview.md).

## 26.03 Release Notes (26.3.0)

NVIDIA® NeMo Retriever Library version 26.03 adds broader hardware and software support along with many pipeline, evaluation, and deployment enhancements.

To upgrade the Helm charts for this release, refer to the [NeMo Retriever Library Helm Charts](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md).

Highlights for the 26.03 release include:

- Legacy ingestion repository consolidated under NeMo-Retriever  
- NeMo Retriever Extraction pipeline renamed to NeMo Retriever Library  
- NeMo Retriever Library now supports two deployment options:  
  - A new no-container, pip-installable in-process library for development (available on PyPI)  
  - Existing production-ready Helm chart with NIMs  
- Added documentation notes on Air-gapped deployment support  
- Added documentation notes on OpenShift support  
- Added support for RTX4500 Pro Blackwell SKU  
- Added support for [llama-nemotron-embed-vl-1b-v2](https://build.nvidia.com/nvidia/llama-nemotron-embed-vl-1b-v2) in text and text+image modes; default Helm VL embedder is `nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:1.12.0` (replaces deprecated `llama-3.2-nemoretriever-1b-vlm-embed-v1` and default `embedqa` NIMs)  
- Default Helm NIMs: `page_elements`, `table_structure`, `ocr`, and `vlm_embed` — **Nemotron Parse**, **Nemotron 3 Nano Omni**, and the **VL reranker** are optional and disabled by default (enable only when needed)  
- **Caption model:** `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` (`nemotron-nano-12b-v2-vl`) is no longer documented for Helm or sizing. Migrate to optional [Nemotron 3 Nano Omni](https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning) (`nemotron-3-nano-omni-30b-a3b-reasoning`) when you enable the caption stage. GPU, disk, and co-residency requirements are in the **Omni caption** rows of the [Pre-Requisites hardware table](prerequisites-support-matrix.md#model-hardware-requirements) (replacing the former Nano 12B VL rows, including the prior 32 GB load limitation).  
- New extract methods `pdfium_hybrid` and `ocr` target scanned PDFs to improve text and layout extraction from image-based pages  
- VLM-based image caption enhancements:  
  - Infographics can be captioned  
  - Reasoning mode is configurable  
- Enabled hybrid search with Lancedb  
- Added retrieval_bench subfolder with generalizable agentic retrieval pipeline  
- The project now uses UV as the primary environment and package manager instead of Conda, resulting in faster installs and simpler dependency handling  
- Default TTL for long-running pipeline job state increased from 1–2 hours to 48 hours so long-running jobs (for example, VLM captioning) do not expire before completion  
- NeMo Retriever Library currently does not support image captioning via VLM; this feature will be added in the next release
- Documentation: multimodal extraction is covered on one page with an in-page table of contents and redirects from the former per-topic URLs

## Release Notes for Previous Versions

| [26.1.2](https://docs.nvidia.com/nemo/retriever/26.1.2/extraction/releasenotes/)
| [26.1.1](https://docs.nvidia.com/nemo/retriever/26.1.1/extraction/releasenotes/)
| [25.9.0](https://docs.nvidia.com/nemo/retriever/25.9.0/extraction/releasenotes/) 
| [25.6.3](https://docs.nvidia.com/nemo/retriever/25.6.3/extraction/releasenotes/) 
| [25.6.2](https://docs.nvidia.com/nemo/retriever/25.6.2/extraction/releasenotes/) 
| [25.4.2](https://docs.nvidia.com/nemo/retriever/25.4.2/extraction/releasenotes/) 
| [25.3.0](https://docs.nvidia.com/nemo/retriever/25.3.0/extraction/releasenotes/) 
| [24.12.1](https://docs.nvidia.com/nemo/retriever/25.3.0/extraction/releasenotes/#release-24121) 
| [24.12.0](https://docs.nvidia.com/nemo/retriever/25.3.0/extraction/releasenotes/#release-2412) 

## Related Topics

- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
- [Deployment options](deployment-options.md)
- [Deploy with Helm](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md)
