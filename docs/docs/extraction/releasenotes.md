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
- Self-hosted Parakeet on Helm requires both `nimOperator.audio.enabled=true` and `serviceConfig.nimEndpoints.audioGrpcEndpoint=audio:50051`; enabling the audio NIM alone does not wire the service ASR endpoint  
- Changing a Helm NIM image repository or tag on an existing release cannot patch `NIMCache` `spec.source.ngc.modelPuller`. Delete the `NIMCache` and its PVC, then upgrade. The affected NIM is unavailable while the operator re-caches weights. Refer to [Changing a NIM image repository or tag](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#changing-nim-image-repository-or-tag).
- `nemo_retriever` requires Python 3.12  

### Pipeline and ingestion { #pipeline-and-ingestion }

- Legacy `nv-ingest` and compatibility pipeline CLI code paths removed; `retriever ingest` and the graph stage registry are the canonical ingestion paths
- Manifest-based ingest routing replaces input-type routing; `retriever ingest` is input-aware for PDF, image, audio, video, text, HTML, DOCX/PPTX, SVG, and related types  
- `allow_no_gpu` option to skip GPU requirement during ingest for CPU-only experimentation  

### CLI { #cli }

- Root CLI adds first-class `retriever ingest` and `retriever query` commands with NIM URL flags, batch tuning, and LanceDB overwrite/append controls
- `retriever query --agentic` runs an LLM-driven ReAct retrieval loop over the same LanceDB table as one-pass retrieval. Local CLI and NRB benchmark runs default to in-process vLLM (`nemotron-8b`). Remote OpenAI-compatible NIM or NVIDIA-hosted endpoints use `--agentic-invoke-url`. Refer to [Workflow: Agentic retrieval](workflow-agentic-retrieval.md).
- `retriever ingest` and `retriever query` replace the retired compatibility pipeline command. Other top-level subcommands—including `eval`, `benchmark`, and `skill-eval`—are development and experimental

### Retriever Service and deployment { #retriever-service-and-deployment }

- Retriever Service v2 adds a scalable multi-pod architecture with gateway, process isolation, and VectorDB integration
- Retriever Service exposes agentic retrieval on `POST /v1/query` with `agentic=true` and an `agentic_query` MCP tool when `agentic.enabled` is true. Service mode requires a remote OpenAI-compatible LLM endpoint. Refer to [Workflow: Agentic retrieval](workflow-agentic-retrieval.md).
- OpenTelemetry basic support for pipeline and service observability  
- Expanded air-gapped deployment guidance in [deployment options](deployment-options.md) and the Helm chart README  
- Helm maps `serviceConfig.nimEndpoints.rerankInvokeUrl` / `rerankModelName` into `nim_endpoints.rerank_invoke_url` / `rerank_model_name`, and auto-wires those fields when `nimOperator.rerankqa.enabled=true`, so `/v1/query` with `rerank=true` works in split topology
- Fixed Retriever Service OpenAPI `info.version` reporting a stale `26.5.0` value. The service now reports the package version, and Helm sets `RETRIEVER_SERVICE_VERSION` from the running service image tag so `/openapi.json` matches the deployed release.

### Models, OCR, and captioning { #models-ocr-and-captioning }

- Nemotron OCR v2 is the default OCR engine for HuggingFace, with CLI language selectors and unified OCR actors. For Helm NIM deployments, Nemotron OCR v1 is the default.  
- Nemotron Parse is available as an alternate PDF extraction method (v1.2 HTTP interface; optional Helm NIM; local inference via vLLM where configured)  
- VLM image captioning via vLLM (including Omni caption model profiles) addresses the capability deferred in 26.03  
- vLLM-backed text and vision-language embedders, multimodal VL reranker, and torch 2.11 for local GPU installs  

### Multimodal extraction { #multimodal-extraction }

- Video retrieval pipeline with frame extraction, OCR, audio-visual fusion, and text deduplication  
- Long-audio Parakeet chunking with time-aligned segments; punctuation-based audio segmenting; ASR batch/streaming improvements  
- Fixed an issue that could cause local Hugging Face batch audio extraction to hang in interactive terminals when FFmpeg inherited the parent process's standard input.

### Retrieval and RAG { #retrieval-and-rag }

- Live RAG SDK with `Retriever.retrieve()`,  reference answer generation `Retriever.answer()`, and optional batch operator graphs via LiteLLM (`[llm]` extra)
- Agentic retrieval: an LLM agent issues multiple searches, fuses candidates, and returns a document-level ranking. The CLI, Python query workflow, REST, and MCP surfaces share this path. Refer to [Agentic retrieval (concept)](agentic-retrieval-concept.md) and [Workflow: Agentic retrieval](workflow-agentic-retrieval.md).

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

### Documentation { #documentation }

- Documentation aligned to a Helm-first supported path for NIM and service deployment
- Published [Agentic retrieval (concept)](agentic-retrieval-concept.md) and [Workflow: Agentic retrieval](workflow-agentic-retrieval.md) for CLI, service, REST, and MCP usage
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
