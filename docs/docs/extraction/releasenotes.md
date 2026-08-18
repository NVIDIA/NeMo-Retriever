# Release Notes for NeMo Retriever Library

This documentation contains the release notes for [NeMo Retriever Library](overview.md).

## 26.08 Release Notes (26.8.0) { #release-2608 }

NVIDIA® NeMo Retriever Library version 26.08 builds on the 26.05 foundation with a graph-based ingest architecture, expanded multimodal and tabular capabilities, production-oriented service deployment, and documentation aligned to a Helm-first supported path.

To upgrade the Helm charts for this release, refer to the [NeMo Retriever Library Helm Charts](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md).

Highlights for the 26.08 release include:

### Upgrade notes { #upgrade-notes }

- Nemotron OCR v2 is now the default OCR engine for local Hugging Face, hosted CPU actors, and Helm NIM deployments (26.05 kept Helm on OCR v1)
- Helm replaces separate page-elements and table-structure NIMs with the combined `nemotron-object-detection:2.0.1` image and Nemotron OCR v2 `2.0.1`. Development Compose uses the same combined object-detection image and OCR v2, but still defaults to `2.0.0` tags unless you override `NIM_*_TAG`
- Default VLM image captioning is Nemotron 3 Nano Omni for local and hosted paths; chart-classified PDF regions remain on the layout and OCR path
- Helm default VL embed and VL rerank NIM images bump to `2.3.0` (26.05 used `1.12.0` / `1.11.0`). Development Compose still defaults to `1.12.0` / `1.11.0` unless you override `NIM_EMBED_TAG` / `NIM_RERANK_TAG`
- Hosted Nemotron Parse and self-hosted Nemotron Parse use distinct HTTP contracts—select the matching client path for your endpoint
- macOS Intel (x86_64) is no longer supported for package installs; use Apple Silicon (arm64) macOS, Windows x64, or Linux. Refer to [Packaging and platform](#packaging-and-platform)
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
- Documented Markdown, JSON, and shell text inputs, plus inline text ingestion support
- Service-mode TXT and HTML chunking
- `return_failures` supported across in-process and batch ingest modes
- Tabular ingestion and embedding improvements, including table-type handling
- PDF render parameters forwarded through ingestion graphs

### CLI { #cli }

- Root CLI adds first-class `retriever ingest` and `retriever query` commands with NIM URL flags, batch tuning, and LanceDB overwrite/append controls
- `retriever query --agentic` runs an LLM-driven ReAct retrieval loop over the same LanceDB table as one-pass retrieval. Local CLI and harness runs default to in-process vLLM (`nemotron-8b`). Remote OpenAI-compatible NIM or NVIDIA-hosted endpoints use `--agentic-invoke-url`. Refer to [Workflow: Agentic retrieval](workflow-agentic-retrieval.md).
- `retriever ingest` and `retriever query` replace the retired compatibility pipeline command. Other top-level subcommands—including `eval`, `benchmark`, `harness`, and `skill-eval`—are development and experimental

### Retriever Service and deployment { #retriever-service-and-deployment }

- Retriever Service v2 adds a scalable multi-pod architecture with gateway, process isolation, and VectorDB integration
- Retriever Service exposes agentic retrieval on `POST /v1/query` with `agentic=true` and an `agentic_query` MCP tool when `agentic.enabled` is true. Service mode requires a remote OpenAI-compatible LLM endpoint. Refer to [Workflow: Agentic retrieval](workflow-agentic-retrieval.md).
- OpenTelemetry basic support for pipeline and service observability  
- Expanded air-gapped deployment guidance in [deployment options](deployment-options.md) and the Helm chart README  
- Helm maps `serviceConfig.nimEndpoints.rerankInvokeUrl` / `rerankModelName` into `nim_endpoints.rerank_invoke_url` / `rerank_model_name`, and auto-wires those fields when `nimOperator.rerankqa.enabled=true`, so `/v1/query` with `rerank=true` works in split topology
- Fixed Retriever Service OpenAPI `info.version` reporting a stale `26.5.0` value. The service now reports the package version, and Helm sets `RETRIEVER_SERVICE_VERSION` from the running service image tag so `/openapi.json` matches the deployed release.
- Gateway worker pull scheduling replaces push routing
- Development Docker Compose deployment for local service stacks
- Zipkin tracing parity alongside OpenTelemetry
- Helm maximum upload size configuration and OpenShift deployment follow-ups
- Secret-backed Helm authentication for public and internal service tokens (inline tokens are gated for insecure development only)

### Models, OCR, and captioning { #models-ocr-and-captioning }

- OCR v2 is unified across library, hosted, and Helm defaults; hosted OCR uses its own language behavior
- Local OCR crop batching across page rows for throughput
- Nemotron 3 Nano Omni is the canonical caption model (opt-in on Helm; larger GPU footprint than Nano caption profiles)
- Nemotron Parse endpoint wiring in service extraction workers, with documented hosted versus self-hosted contract selection
- Embedder model router for additional Llama Nemotron embedding checkpoints (including 1B, 3B, 8B, local, fine-tuned, and ModelOpt)
- Helm extraction NIMs (OCR and object detection) enable performance mode by default; the VL embed NIM does not

### Agentic retrieval and query { #agentic-retrieval-and-query }

- Agentic retrieval as a first-class retrieval mode through the CLI (`retriever query --agentic`), SDK helpers, and service HTTP / MCP endpoints
- Local in-process vLLM agent LLMs by default for agentic query paths; optional OpenAI-compatible remote endpoints
- Optional Helm NIM for the agentic / answer LLM (`llama-3.3-nemotron-super-49b-v1.5`); agentic remains opt-in (`serviceConfig.agentic.enabled`)
- Configurable auto-retrieval on the service query path; evidence and coverage output formats on `/v1/query`
- Service-mode `Retriever.answer` support and FastMCP integration for local and remote agents
- MCP query-method selection and rerank tools

### Retrieval and RAG { #retrieval-and-rag }

- Live RAG SDK with `Retriever.retrieve()`, reference answer generation `Retriever.answer()`, and optional batch operator graphs through LiteLLM (`[llm]` extra)
- Agentic retrieval: an LLM agent issues multiple searches, fuses candidates, and returns a document-level ranking. The CLI, Python query workflow, REST, and MCP surfaces share this path. Refer to [Agentic retrieval (concept)](agentic-retrieval-concept.md) and [Workflow: Agentic retrieval](workflow-agentic-retrieval.md).

### Vector database and retrieval { #vector-database-and-retrieval }

- True LanceDB hybrid retrieval
- LanceDB retrieval-mode autodetection and persisted embedding identity for automatic local queries
- Dense image-only VDB records retained where applicable
- Scope-isolated collection and document lifecycle APIs (`/v1/collections`) for create, ingest, replace, query, and cleanup without exposing LanceDB table names


### Packaging and platform { #packaging-and-platform }

- Public nightlies published to PyPI while keeping local install extras stable
- Ray raised to `>=2.56.1` for CVE remediation (26.05 used `>=2.49.0`). Ray no longer publishes wheels for macOS Intel (x86_64), so `pip`/`uv` installs fail on Intel Macs (including in-process library mode). Apple Silicon (arm64) macOS remains supported for slim remote/NIM-only installs, alongside Windows x64.
- Optional install extras (`[local]`, `[multimedia]`, `[llm]`, `[tabular]`, `[nemotron-parse]`, `[service]`, and others), including slim remote/NIM-only installs on Mac and Windows  

### Helm chart { #helm-chart }

- Helm chart refresh under `nemo_retriever/helm/` with VL embedder defaults and optional Nemotron Parse and Omni caption NIMs  

### Documentation { #documentation }

- Documentation aligned to a Helm-first supported path for NIM and service deployment
- Published [Agentic retrieval (concept)](agentic-retrieval-concept.md) and [Workflow: Agentic retrieval](workflow-agentic-retrieval.md) for CLI, service, REST, and MCP usage
- Documentation consolidates extraction concepts, ingest workflow, embeddings, audio/video guides, prerequisites and support matrix, and UDF/custom stages in the [graph README](https://github.com/NVIDIA/NeMo-Retriever/tree/main/nemo_retriever/src/nemo_retriever/graph#nemo-retriever-graph)  

## Release Notes for Previous Versions { #previous-versions }

- [26.05](https://docs.nvidia.com/nemo/retriever/26.5.0/extraction/releasenotes-nv-ingest/) (GA on docs.nvidia.com at time of 26.08 RC)
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
