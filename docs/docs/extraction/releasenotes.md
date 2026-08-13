# Release Notes for NeMo Retriever Library

This documentation contains the release notes for [NeMo Retriever Library](overview.md).

## 26.08 Release Notes (26.8.0) { #release-2608 }

NVIDIA® NeMo Retriever Library version 26.08 builds on the 26.05 foundation with a graph-based ingest architecture, expanded multimodal and tabular capabilities, production-oriented service deployment, and documentation aligned to a Helm-first supported path.

To upgrade the Helm charts for this release, refer to the [NeMo Retriever Library Helm Charts](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md).

Highlights for the 26.08 release include:

### Upgrade notes { #upgrade-notes }

- Nemotron OCR v2 is now the default OCR engine for local Hugging Face, hosted CPU actors, and Helm NIM deployments (26.05 kept Helm on OCR v1)
- Helm and development Compose replace separate page-elements and table-structure NIMs with the combined `nemotron-object-detection:2.0.1` image, and use Nemotron OCR v2 `2.0.1` on the matching release train
- Default VLM image captioning is Nemotron 3 Nano Omni for local and hosted paths; chart-classified PDF regions remain on the layout and OCR path
- Default VL embed and VL rerank NIM images bump to `2.3.0` (26.05 used `1.12.0` / `1.11.0`)
- Hosted Nemotron Parse and self-hosted Nemotron Parse use distinct HTTP contracts—select the matching client path for your endpoint
- macOS Intel (x86_64) is no longer supported for package installs; use Apple Silicon (arm64) macOS, Windows x64, or Linux. Refer to [Packaging and platform](#packaging-and-platform)

### Models, OCR, and captioning { #models-ocr-and-captioning }

- OCR v2 is unified across library, hosted, and Helm defaults; hosted OCR uses its own language behavior
- Local OCR crop batching across page rows for throughput
- Nemotron 3 Nano Omni is the canonical caption model (opt-in on Helm; larger GPU footprint than Nano caption profiles)
- Nemotron Parse endpoint wiring in service extraction workers, with documented hosted versus self-hosted contract selection
- Embedder model router for additional Llama Nemotron embedding checkpoints (including 1B, 3B, 8B, local, fine-tuned, and ModelOpt)
- Helm extraction NIMs (OCR and object detection) enable performance mode by default; the VL embed NIM does not

### Agentic retrieval and query { #agentic-retrieval-and-query }

- Agentic retrieval as a first-class retrieval mode via CLI (`retriever query --agentic`), SDK helpers, and service HTTP / MCP endpoints
- Local in-process vLLM agent LLMs by default for agentic query paths; optional OpenAI-compatible remote endpoints
- Optional Helm NIM for the agentic / answer LLM (`llama-3.3-nemotron-super-49b-v1.5`); agentic remains opt-in (`serviceConfig.agentic.enabled`)
- Configurable auto-retrieval on the service query path; evidence and coverage output formats on `/v1/query`
- Service-mode `Retriever.answer` support and FastMCP integration for local and remote agents
- MCP query-method selection and rerank tools

### Pipeline and ingestion { #pipeline-and-ingestion }

- Documented Markdown, JSON, and shell text inputs, plus inline text ingestion support
- Service-mode TXT and HTML chunking
- `return_failures` supported across in-process and batch ingest modes
- Tabular ingestion and embedding improvements, including table-type handling
- PDF render parameters forwarded through ingestion graphs

### Vector database and retrieval { #vector-database-and-retrieval }

- True LanceDB hybrid retrieval
- LanceDB retrieval-mode autodetection and persisted embedding identity for automatic local queries
- Dense image-only VDB records retained where applicable
- Scope-isolated collection and document lifecycle APIs (`/v1/collections`) for create, ingest, replace, query, and cleanup without exposing LanceDB table names

### Retriever Service and deployment { #retriever-service-and-deployment }

- Gateway worker pull scheduling replaces push routing
- Development Docker Compose deployment for local service stacks
- Zipkin tracing parity alongside OpenTelemetry
- Helm maximum upload size configuration and OpenShift deployment follow-ups
- Secret-backed Helm authentication for public and internal service tokens (inline tokens are gated for insecure development only)

### Packaging and platform { #packaging-and-platform }

- Public nightlies published to PyPI while keeping local install extras stable
- Ray raised to `>=2.56.1` for CVE remediation (26.05 used `>=2.49.0`). Ray no longer publishes wheels for macOS Intel (x86_64), so `pip`/`uv` installs fail on Intel Macs (including in-process library mode). Apple Silicon (arm64) macOS remains supported for slim remote/NIM-only installs, alongside Windows x64.

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
