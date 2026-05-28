# Release Notes for NeMo Retriever Library

This documentation contains the release notes for [NeMo Retriever Library](overview.md).

## 26.05 Release Notes (26.5.0)

NVIDIA® NeMo Retriever Library version 26.05 builds on the 26.03 foundation with a graph-based ingest architecture, expanded multimodal and tabular capabilities, production-oriented service deployment, and documentation aligned to a Helm-first supported path.

To upgrade the Helm charts for this release, refer to the [NeMo Retriever Library Helm Charts](https://github.com/NVIDIA/NeMo-Retriever/blob/26.05/nemo_retriever/helm/README.md).

Highlights for the 26.05 release include:

- Legacy `nv-ingest` code paths removed; `graph_pipeline` and the graph stage registry are the canonical ingestion path  
- Manifest-based ingest routing replaces input-type routing; `retriever ingest` is input-aware for PDF, image, audio, video, text, HTML, DOCX/PPTX, SVG, and related types  
- Root CLI adds `retriever ingest` and `retriever query` with NIM URL flags, batch tuning, and LanceDB overwrite/append controls, plus `retriever pipeline` for graph execution  
- For product use, only `retriever ingest`, `retriever query`, and `retriever pipeline` (for example `retriever pipeline run`) are supported; other top-level subcommands—including `pdf`, `html`, `eval`, `benchmark`, `harness`, `online`, `compare`, `image`, and `skill-eval`—are development and experimental  
- Retriever Service v2 adds a scalable multi-pod architecture with gateway, process isolation, and VectorDB integration  
- Nemotron OCR v2 is the default OCR engine, with CLI language selectors and unified OCR actors  
- Nemotron Parse is available as an alternate PDF extraction method (v1.2 HTTP interface; optional Helm NIM; local inference via vLLM where configured)  
- VLM image captioning via vLLM (including Omni caption model profiles) addresses the capability deferred in 26.03  
- vLLM-backed text and vision-language embedders, multimodal VL reranker, and torch 2.11 for local GPU installs  
- Video retrieval pipeline with frame extraction, OCR, audio-visual fusion, and text deduplication  
- Text-to-SQL agent graph and tabular tooling for structured data retrieval, including tabular data ingestion  
- Live RAG SDK with `Retriever.retrieve()`, `Retriever.answer()`, and optional batch operator graphs via LiteLLM (`[llm]` extra)  
- Vector database operators integrated directly in the pipeline; custom metadata support; LanceDB hybrid search guidance updated  
- LanceDB is documented as the first-party vector path for new deployments; Milvus/MinIO guidance removed from the primary extraction doc set  
- BEIR-centric evaluation overhaul and `retriever skill-eval` benchmark CLI for the NeMo Retriever skill (experimental)  
- OpenTelemetry basic support for pipeline and service observability  
- Optional install extras (`[local]`, `[multimedia]`, `[llm]`, `[tabular]`, `[nemotron-parse]`, `[service]`, and others), including slim remote/NIM-only installs on Mac and Windows  
- Long-audio Parakeet chunking with time-aligned segments; punctuation-based audio segmenting; ASR batch/streaming improvements  
- `allow_no_gpu` option to skip GPU requirement during ingest for CPU-only experimentation  
- Helm chart refresh under `nemo_retriever/helm/` with GA VL embedder defaults and optional Nemotron Parse and Omni caption NIMs  
- Expanded air-gapped deployment guidance in [deployment options](deployment-options.md) and the Helm chart README  
- `nemo_retriever` requires Python 3.12  
- Breaking changes and migration:  
  - Text splitting for graph and library ingest moved into `.extract(split_config=...)` instead of standalone `.split()` on the graph ingest path (the service ingestor API may still expose `.split()` separately)  
  - Direct `Retriever(...)` construction uses `vdb_kwargs`, `embed_kwargs`, and `rerank` instead of flat `lancedb_uri`, `lancedb_table`, `embedder`, `embedding_endpoint`, `local_query_embed_backend`, and `reranker` arguments  
  - For Helm audio and video extraction, set `service.installFfmpeg=true` when images no longer bundle `ffmpeg` and `ffprobe` by default  
- Documentation aligned to a Helm-first supported path; [Docker Compose for local development](https://github.com/NVIDIA/NeMo-Retriever/blob/26.05/nemo_retriever/docker.md) documented as unsupported developer tooling (not a production NIM deployment path)  
- Documentation consolidates extraction concepts, ingest workflow, embeddings, audio/video guides, prerequisites and support matrix, and UDF/custom stages in the [graph README](https://github.com/NVIDIA/NeMo-Retriever/tree/26.05/nemo_retriever/src/nemo_retriever/graph#nemo-retriever-graph)  
- See the [26.03…26.05 compare view](https://github.com/NVIDIA/NeMo-Retriever/compare/26.03...26.05) on GitHub for the full commit list since 26.03  

## 26.03 Release Notes (26.3.0)

NVIDIA® NeMo Retriever Library version 26.03 adds broader hardware and software support along with many pipeline, evaluation, and deployment enhancements.

To upgrade the Helm charts for this release, refer to the [NeMo Retriever Library Helm Charts](https://github.com/NVIDIA/NeMo-Retriever/blob/26.3.0/nemo_retriever/helm/README.md).

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
- **LanceDB is now the default vector database backend** for extraction and indexing; Milvus remained supported in 26.03. For upload, hybrid search, and infrastructure options, see [Vector databases](vdbs.md).  
- Enabled hybrid search with LanceDB (BM25 full-text search combined with dense vectors and reciprocal rank fusion)  
- Added `retrieval_bench` with a generalizable agentic retrieval pipeline  
- The project now uses UV as the primary environment and package manager instead of Conda, resulting in faster installs and simpler dependency handling  
- Default TTL for long-running pipeline job state increased from 1–2 hours to 48 hours so long-running jobs (for example, VLM captioning) do not expire before completion  
- NeMo Retriever Library currently does not support image captioning via VLM; this feature was added in 26.05  
- Documentation: multimodal extraction is covered on one page with an in-page table of contents and redirects from the former per-topic URLs  
- Container images built from this repository no longer install `ffmpeg` and `ffprobe` by default. Audio and video extraction require these binaries on `PATH`; for Helm deployments set `service.installFfmpeg=true`, or install system FFmpeg manually in non-container environments.

## Release Notes for Previous Versions

| [26.03](https://docs.nvidia.com/nemo/retriever/26.3.0/extraction/releasenotes/)
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
- [NeMo Retriever Library Helm Charts](https://github.com/NVIDIA/NeMo-Retriever/blob/26.05/nemo_retriever/helm/README.md)
