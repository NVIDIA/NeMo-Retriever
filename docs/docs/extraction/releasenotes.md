# Release Notes for NeMo Retriever Library

This documentation contains the release notes for [NeMo Retriever Library](overview.md).

## 26.08 Release Notes (26.8.0) { #release-2608 }

NVIDIA® NeMo Retriever Library version 26.08 builds on the 26.05 foundation with a graph-based ingest architecture, expanded multimodal and tabular capabilities, production-oriented service deployment, and documentation aligned to a Helm-first supported path.

To upgrade the Helm charts for this release, refer to the [NeMo Retriever Library Helm Charts](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md).

Highlights for the 26.08 release include:

### Upgrade notes { #upgrade-notes }

- Text splitting for graph and library ingest moved into `.extract(split_config=...)` instead of standalone `.split()` on the graph ingest path (the service ingestor API may still expose `.split()` separately)
- Direct `Retriever(...)` construction uses `vdb_kwargs`, `embed_kwargs`, and `rerank` instead of flat `lancedb_uri`, `lancedb_table`, `embedder`, `embedding_endpoint`, `local_query_embed_backend`, and `reranker` arguments
- For Helm audio and video extraction, set `service.installFfmpeg: true` in `values.yaml` (or pass `--set service.installFfmpeg=true`) when images no longer bundle `ffmpeg` and `ffprobe` by default
- Changing a Helm NIM image repository or tag on an existing release cannot patch `NIMCache` `spec.source.ngc.modelPuller`. Delete the `NIMCache` and its PVC, then upgrade. The affected NIM is unavailable while the operator re-caches weights. Refer to [Changing a NIM image repository or tag](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#changing-nim-image-repository-or-tag).
- `nemo_retriever` requires Python 3.12

### Pipeline and ingestion { #pipeline-and-ingestion }

- Legacy `nv-ingest` and compatibility pipeline CLI code paths removed; `retriever ingest` and the graph stage registry are the canonical ingestion paths
- Manifest-based ingest routing replaces input-type routing; `retriever ingest` is input-aware for PDF, image, audio, video, text, HTML, DOCX/PPTX, SVG, and related types
- `allow_no_gpu` option to skip GPU requirement during ingest for CPU-only experimentation

### CLI { #cli }

- Root CLI adds first-class `retriever ingest` and `retriever query` commands with NIM URL flags, batch tuning, and LanceDB overwrite/append controls
- `retriever ingest` and `retriever query` replace the retired compatibility pipeline command. Other top-level subcommands—including `eval`, `benchmark`, `harness`, and `skill-eval`—are development and experimental

### Retriever Service and deployment { #retriever-service-and-deployment }

- Retriever Service v2 adds a scalable multi-pod architecture with gateway, process isolation, and VectorDB integration
- OpenTelemetry basic support for pipeline and service observability
- Expanded air-gapped deployment guidance in [deployment options](deployment-options.md) and the Helm chart README

### Models, OCR, and captioning { #models-ocr-and-captioning }

- Nemotron OCR v2 is the default OCR engine for HuggingFace and for Helm NIM deployments, with CLI language selectors and unified OCR actors.
- Nemotron Parse is available as an alternate PDF extraction method (self-hosted Helm NIM defaults to the v2.0 tagged text-prompt interface; local Hugging Face inference remains on v1.2 where configured)
- VLM image captioning via vLLM (including Omni caption model profiles) addresses the capability deferred in 26.03
- vLLM-backed text and vision-language embedders, multimodal VL reranker, and torch 2.11 for local GPU installs

### Multimodal extraction { #multimodal-extraction }

- Video retrieval pipeline with frame extraction, OCR, audio-visual fusion, and text deduplication
- Long-audio Parakeet chunking with time-aligned segments; punctuation-based audio segmenting; ASR batch/streaming improvements
- Fixed an issue that could cause local Hugging Face batch audio extraction to hang in interactive terminals when FFmpeg inherited the parent process's standard input.

### Retrieval and RAG { #retrieval-and-rag }

- Live RAG SDK with `Retriever.retrieve()`,  reference answer generation `Retriever.answer()`, and optional batch operator graphs via LiteLLM (`[llm]` extra)

### Vector database { #vector-database }

- Vector database operators integrated directly in the pipeline; custom metadata support; LanceDB hybrid search guidance updated
- LanceDB is documented as the first-party vector path for new deployments; Milvus/MinIO guidance removed from the primary extraction doc set

### Evaluation { #evaluation }

- BEIR-centric evaluation overhaul and `retriever skill-eval` benchmark CLI for the NeMo Retriever skill (experimental)


- Text-to-SQL agent graph and tabular tooling for structured data retrieval, including tabular data ingestion

### Packaging and platform { #packaging-and-platform }

- Optional install extras (`[local]`, `[multimedia]`, `[llm]`, `[tabular]`, `[nemotron-parse]`, `[service]`, and others), including slim remote/NIM-only installs on Mac and Windows

### Helm chart { #helm-chart }

- Helm chart refresh under `nemo_retriever/helm/` with VL embedder defaults and optional Nemotron Parse and Omni caption NIMs
- Chart defaults pin the 26.08-RC1 service image and the release-approved optional NIM tags (Omni `2.0.4-variant`, Nemotron Parse `v2.0:2.0.8-variant`, Answer LLM Super-49B `v1:1.10.1`)

### Documentation { #documentation }

- Documentation aligned to a Helm-first supported path for NIM and service deployment
- Documentation consolidates extraction concepts, ingest workflow, embeddings, audio/video guides, prerequisites and support matrix, and UDF/custom stages in the [graph README](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/src/nemo_retriever/graph#nemo-retriever-graph)

## Release Notes for Previous Versions { #previous-versions }

- [26.05](https://docs.nvidia.com/nemo/retriever/26.5.0/extraction/releasenotes-nv-ingest/)
- [26.03](https://docs.nvidia.com/nemo/retriever/26.3.0/extraction/releasenotes-nv-ingest/)
- [26.1.2](https://archive.docs.nvidia.com/nemo/retriever/26.1.2/extraction/releasenotes-nv-ingest/)
- [26.1.1](https://archive.docs.nvidia.com/nemo/retriever/26.1.1/extraction/releasenotes-nv-ingest/)
- [25.9.0](https://archive.docs.nvidia.com/nemo/retriever/25.9.0/extraction/releasenotes-nv-ingest/)
- [25.6.3](https://archive.docs.nvidia.com/nemo/retriever/25.6.3/extraction/releasenotes-nv-ingest/)
- [25.6.2](https://archive.docs.nvidia.com/nemo/retriever/25.6.2/extraction/releasenotes-nv-ingest/)
- [25.4.2](https://archive.docs.nvidia.com/nemo/retriever/25.4.2/extraction/releasenotes-nv-ingest/)
- [25.3.0](https://archive.docs.nvidia.com/nemo/retriever/25.3.0/extraction/releasenotes-nv-ingest/)

Release notes for 24.12.1 and 24.12.0 are on the [25.3.0 archived release notes](https://archive.docs.nvidia.com/nemo/retriever/25.3.0/extraction/releasenotes-nv-ingest/).

## Related Topics { #related-topics }

- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
- [Deployment options](deployment-options.md)
- [NeMo Retriever Library Helm Charts](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md)
