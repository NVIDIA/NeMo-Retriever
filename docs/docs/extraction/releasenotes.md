# Release Notes for NeMo Retriever Library

This documentation contains the release notes for [NeMo Retriever Library](overview.md).

## 26.05 Release Notes (26.5.0)

NVIDIA® NeMo Retriever Library version **26.05** (PyPI **26.5.0** at GA) continues the 26.05 release line on the [`26.05`](https://github.com/NVIDIA/NeMo-Retriever/tree/26.05) branch. Pre-release builds are tagged **`26.05-RC1`**, **`26.05-RC2`**, and so on; install and deploy using the RC tag that matches your build.

To upgrade the Helm charts for this release, refer to the [NeMo Retriever Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/26.05/nemo_retriever/helm/README.md) and pin chart version **`26.05-RC1`** (or the RC you are validating).

Highlights for the 26.05 release line include everything in [26.03](#2603-release-notes-2630) plus changes on `main` merged into the `26.05` branch. See the [Git compare view](https://github.com/NVIDIA/NeMo-Retriever/compare/26.03...26.05) for the full commit list.

**Migration note:** Direct `Retriever(...)` construction uses grouped configuration dictionaries. Replace flat `lancedb_uri=`, `lancedb_table=`, `embedder=`, `embedding_endpoint=`, and `reranker=` arguments with `vdb_kwargs={...}`, `embed_kwargs={...}`, and `rerank=...`. Helper APIs that document their own flat kwargs keep their own compatibility layer.

**Install (RC1 example):**

```bash
uv pip install nemo-retriever==26.05-RC1
```

Use your organization's Artifactory or PyPI index URL when installing published wheels from CI (see the Perform Release workflow summary for the exact index).

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
- Added support for llama-nemotron-embed-vl-v2 in text and text+image modes  
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
- Container images built from this repository no longer install `ffmpeg` and
  `ffprobe` by default. Audio and video extraction require these binaries on
  `PATH`; for Helm deployments set `service.installFfmpeg=true`, or install
  system FFmpeg manually in non-container environments.

## Release Notes for Previous Versions

| [26.03](https://docs.nvidia.com/nemo/retriever/26.03/extraction/releasenotes/)
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
