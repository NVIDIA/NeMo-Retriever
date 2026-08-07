# Release Notes for NeMo Retriever Library

This documentation contains the release notes for [NeMo Retriever Library](overview.md).

## 26.08 Release Notes (26.8.0)

> **26.08 RC readiness:** GitHub Pages builds from `main` and is the QA review surface for the 26.08 release candidate. docs.nvidia.com remains on the published 26.05 / 26.5.0 snapshot until a separate publish workflow run after GA.

NVIDIA® NeMo Retriever Library version 26.08 builds on the 26.05 foundation with an OCR and object-detection stack refresh, first-class agentic retrieval, deeper service and query surfaces, and broader text ingest support.

To upgrade the Helm charts for this release, refer to the [NeMo Retriever Library Helm Charts](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md).

Highlights for the 26.08 release include:

### Upgrade notes

- Nemotron OCR v2 is now the default OCR engine for local Hugging Face, hosted CPU actors, and Helm NIM deployments (26.05 kept Helm on OCR v1)
- Helm and development Compose use the combined `nemotron-object-detection:2.0.0` image for page-elements and table-structure, with the Nemotron OCR v2 NIM on the matching release train
- Default VLM image captioning is Nemotron 3 Nano Omni for local and hosted paths; chart-classified PDF regions remain on the layout and OCR path
- Default VL embed and VL rerank NIM images bump to `2.3.0`
- Retired compatibility and pipeline CLI paths are fully removed; use `retriever ingest` and `retriever query` only
- Hosted Nemotron Parse and self-hosted Nemotron Parse use distinct HTTP contracts— select the matching client path for your endpoint
- macOS Intel (x86_64) is no longer supported for package installs; use Apple Silicon (arm64) macOS, Windows x64, or Linux. Refer to [Packaging and platform](#packaging-and-platform)

### Models, OCR, and captioning

- OCR v2 is unified across library, hosted, and Helm defaults, with local language selectors (`--ocr-lang` / API `ocr_lang`); hosted OCR uses its own language behavior
- Local OCR crop batching across page rows for throughput
- Nemotron 3 Nano Omni is the canonical caption model (opt-in on Helm; larger GPU footprint than Nano caption profiles)
- Nemotron Parse endpoint wiring in service extraction workers, with documented hosted versus self-hosted contract selection

### Agentic retrieval and query

- Agentic retrieval as a first-class retrieval mode via CLI (`retriever query --agentic`), SDK helpers, and service HTTP / MCP endpoints
- Local in-process vLLM agent LLMs by default for agentic query paths; optional OpenAI-compatible remote endpoints
- Configurable auto-retrieval on the service query path; evidence and coverage output formats on `/v1/query`
- Service-mode `Retriever.answer` support and FastMCP integration for local and remote agents

### Pipeline and ingestion

- Documented Markdown, JSON, and shell text inputs, plus inline text ingestion support
- Service-mode TXT and HTML chunking
- `return_failures` supported across in-process and batch ingest modes
- Tabular ingestion and embedding improvements, including table-type handling

### CLI

- CLI consolidated around `retriever ingest` and `retriever query` with improved help discoverability
- Remaining top-level subcommands (`eval`, `benchmark`, `harness`, `skill-eval`, and related) remain development and experimental

### Vector database and retrieval

- True LanceDB hybrid retrieval
- LanceDB retrieval-mode autodetection and persisted embedding identity for automatic local queries
- Dense image-only VDB records retained where applicable

### Retriever Service and deployment

- Gateway worker pull scheduling replaces push routing
- Development Docker Compose deployment for local service stacks
- Zipkin tracing parity alongside OpenTelemetry
- Helm maximum upload size configuration and OpenShift deployment follow-ups
- Optional agentic service configuration (`serviceConfig.agentic.*`)

### Packaging and platform

- Public nightlies published to PyPI while keeping local install extras stable
- Torch 2.11.0 documented for local GPU (`[local]`) installs
- Ray raised to `>=2.56.1` for CVE remediation. Ray no longer publishes wheels for macOS Intel (x86_64), so `pip`/`uv` installs fail on Intel Macs (including in-process library mode). Apple Silicon (arm64) macOS remains supported for slim remote/NIM-only installs, alongside Windows x64.

### Documentation

- RC readiness corrections for install paths, Live RAG public imports, remote NIM platform support, and statement-style headings
- Nemotron Parse hosted versus self-hosted contract documentation
- Agentic retrieval examples restored in the library README

## Release Notes for Previous Versions

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

## Related Topics

- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
- [Deployment options](deployment-options.md)
- [NeMo Retriever Library Helm Charts](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md)
