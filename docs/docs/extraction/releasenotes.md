# Release Notes for NeMo Retriever Library

This documentation contains the release notes for [NeMo Retriever Library](overview.md).

## 26.05 Release Notes (26.5.0)

NVIDIA® NeMo Retriever Library version 26.05 builds on the 26.03 foundation with a graph-based ingest architecture, expanded multimodal and tabular capabilities, production-oriented service deployment, and documentation aligned to a Helm-first supported path.

To upgrade the Helm charts for this release, refer to the [NeMo Retriever Library Helm Charts](https://github.com/NVIDIA/NeMo-Retriever/blob/26.05/nemo_retriever/helm/README.md).

Highlights for the 26.05 release include:

### Upgrade notes

- Text splitting for graph and library ingest moved into `.extract(split_config=...)` instead of standalone `.split()` on the graph ingest path (the service ingestor API may still expose `.split()` separately)  
- Direct `Retriever(...)` construction uses `vdb_kwargs`, `embed_kwargs`, and `rerank` instead of flat `lancedb_uri`, `lancedb_table`, `embedder`, `embedding_endpoint`, `local_query_embed_backend`, and `reranker` arguments  
- For Helm audio and video extraction, set `service.installFfmpeg: true` in `values.yaml` (or pass `--set service.installFfmpeg=true`) when images no longer bundle `ffmpeg` and `ffprobe` by default  
- `nemo_retriever` requires Python 3.12  

### Pipeline and ingestion

- Legacy `nv-ingest` code paths removed; `graph_pipeline` and the graph stage registry are the canonical ingestion path  
- Manifest-based ingest routing replaces input-type routing; `retriever ingest` is input-aware for PDF, image, audio, video, text, HTML, DOCX/PPTX, SVG, and related types  
- `allow_no_gpu` option to skip GPU requirement during ingest for CPU-only experimentation  

### CLI

- Root CLI adds `retriever ingest` and `retriever query` with NIM URL flags, batch tuning, and LanceDB overwrite/append controls, plus `retriever pipeline` for graph execution  
- For product use, only `retriever ingest`, `retriever query`, and `retriever pipeline` (for example `retriever pipeline run`) are supported; other top-level subcommands—including `pdf`, `html`, `eval`, `benchmark`, `harness`, `online`, `compare`, `image`, and `skill-eval`—are development and experimental  

### Retriever Service and deployment

- Retriever Service v2 adds a scalable multi-pod architecture with gateway, process isolation, and VectorDB integration  
- OpenTelemetry basic support for pipeline and service observability  
- Expanded air-gapped deployment guidance in [deployment options](deployment-options.md) and the Helm chart README  

### Models, OCR, and captioning

- Nemotron OCR v2 is the default OCR engine for HuggingFace, with CLI language selectors and unified OCR actors. For Helm NIM deployments, Nemotron OCR v1 is the default.  
- Nemotron Parse is available as an alternate PDF extraction method (v1.2 HTTP interface; optional Helm NIM; local inference via vLLM where configured)  
- VLM image captioning via vLLM (including Omni caption model profiles) addresses the capability deferred in 26.03  
- vLLM-backed text and vision-language embedders, multimodal VL reranker, and torch 2.11 for local GPU installs  

### Multimodal extraction

- Video retrieval pipeline with frame extraction, OCR, audio-visual fusion, and text deduplication  
- Long-audio Parakeet chunking with time-aligned segments; punctuation-based audio segmenting; ASR batch/streaming improvements  

### Retrieval and RAG

- Live RAG SDK with `Retriever.retrieve()`,  reference answer generation `Retriever.answer()`, and optional batch operator graphs via LiteLLM (`[llm]` extra)  

### Vector database

- Vector database operators integrated directly in the pipeline; custom metadata support; LanceDB hybrid search guidance updated  
- LanceDB is documented as the first-party vector path for new deployments; Milvus/MinIO guidance removed from the primary extraction doc set  

### Evaluation

- BEIR-centric evaluation overhaul and `retriever skill-eval` benchmark CLI for the NeMo Retriever skill (experimental)  


- Text-to-SQL agent graph and tabular tooling for structured data retrieval, including tabular data ingestion  

### Packaging and platform

- Optional install extras (`[local]`, `[multimedia]`, `[llm]`, `[tabular]`, `[nemotron-parse]`, `[service]`, and others), including slim remote/NIM-only installs on Mac and Windows  

### Helm chart

- Helm chart refresh under `nemo_retriever/helm/` with GA VL embedder defaults and optional Nemotron Parse and Omni caption NIMs  

### Documentation

- Documentation aligned to a Helm-first supported path for NIM and service deployment
- Documentation consolidates extraction concepts, ingest workflow, embeddings, audio/video guides, prerequisites and support matrix, and UDF/custom stages in the [graph README](https://github.com/NVIDIA/NeMo-Retriever/tree/26.05/nemo_retriever/src/nemo_retriever/graph#nemo-retriever-graph)  

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
