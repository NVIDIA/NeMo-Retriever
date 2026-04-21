# Release Notes for NeMo Retriever Library

This documentation contains the release notes for [NeMo Retriever Library](overview.md).

!!! note

    This documentation describes NeMo Retriever Library.   

## 26.03 Release Notes (26.3.0)

NVIDIA® NeMo Retriever Library version 26.03 adds broader hardware and software support along with many pipeline, evaluation, and deployment enhancements.

To upgrade the Helm charts for this release, refer to the [NeMo Retriever Library Helm Charts](https://github.com/NVIDIA/NeMo-Retriever/blob/26.3.0/helm/README.md).

Highlights for the 26.03 release include:

- Legacy ingestion repository consolidated under NeMo-Retriever  
- NeMo Retriever Extraction pipeline renamed to NeMo Retriever Library  
- NeMo Retriever Library now supports two deployment options:  
  - A new no-container, pip-installable in-process library for development (available on PyPI)  
  - Existing production-ready Helm chart with NIMs  
- Added documentation notes on Air-gapped deployment support  
- Added documentation notes on OpenShift support  
- Added support for RTX4500 Pro Blackwell SKU  
- Added support for llama-nemotron-embed-vl-v2 in text and text+image modes  
- New extract methods `pdfium_hybrid` and `ocr` target scanned PDFs to improve text and layout extraction from image-based pages  
- VLM-based image caption enhancements:  
  - Infographics can be captioned  
  - Reasoning mode is configurable  
- Enabled hybrid search with Lancedb  
- Added retrieval_bench subfolder with generalizable agentic retrieval pipeline  
- The project now uses UV as the primary environment and package manager instead of Conda, resulting in faster installs and simpler dependency handling  
- Default Redis TTL increased from 1–2 hours to 48 hours so long-running jobs (e.g., VLM captioning) don’t expire before completion  
- NeMo Retriever Library currently does not support image captioning via VLM; this feature will be added in the next release

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

- [Prerequisites](prerequisites.md)
- [Deploy Without Containers (Library Mode)](quickstart-library-mode.md)
- [Deploy With Docker Compose (Self-Hosted)](quickstart-guide.md)
- [Deploy with Helm](https://github.com/NVIDIA/NeMo-Retriever/blob/main/helm/README.md)
