# Release Notes for NeMo Retriever Library

This documentation contains the release notes for [NeMo Retriever Library](overview.md).

## 26.05 Release Notes (26.5.0)

NVIDIA® NeMo Retriever Library version **26.05** (PyPI **26.5.0** at GA) continues the 26.05 release line on the [`26.05`](https://github.com/NVIDIA/NeMo-Retriever/tree/26.05) branch. Pre-release builds are tagged **`26.05-RC1`**, **`26.05-RC2`**, and so on; install and deploy using the RC tag that matches your build.

This release builds on the [26.03 release](#2603-release-notes-2630) below. It replaces the legacy batch ingest stack with a graph-based pipeline, ships Retriever Service v2 for production, delivers VLM image captioning deferred from 26.03, and aligns customer-facing documentation with a Helm-first supported path.

To upgrade the Helm charts for this release, refer to the [NeMo Retriever Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/26.05/nemo_retriever/helm/README.md) and pin chart version **`26.05-RC1`** (or the RC you are validating).

For every commit merged between the two release branches, see the [26.03…26.05 compare view](https://github.com/NVIDIA/NeMo-Retriever/compare/26.03...26.05) on GitHub (302 commits on the `26.05` line since `26.03`).

### Install

```bash
uv pip install nemo-retriever==26.05-RC1
```

Use your organization's Artifactory or PyPI index URL when installing published wheels from CI (see the Perform Release workflow summary for the exact index).

### Breaking changes and migration

- **Legacy `nv-ingest` paths removed** — the graph pipeline and `nemo_retriever` package are the supported ingestion surface; old `nv-ingest` tree and batch-only examples were deleted.
- **Chunking API** — text splitting moved from standalone `.split()` into `.extract(split_config=...)`. Update scripts that called `.split()` directly.
- **`Retriever(...)` constructor** — grouped configuration dictionaries replace flat kwargs. Replace `lancedb_uri=`, `lancedb_table=`, `embedder=`, `embedding_endpoint=`, `local_query_embed_backend=`, and `reranker=` with `vdb_kwargs={...}`, `embed_kwargs={...}`, and `rerank=...`. For example, `local_query_embed_backend="hf"` maps to `embed_kwargs={"local_ingest_embed_backend": "hf"}`. Helper APIs that document their own flat kwargs keep their own compatibility layer.
- **Vector database documentation** — extraction docs now describe LanceDB as the first-party vector path. Milvus/MinIO deployment guidance was removed from the primary extraction doc set (the legacy Python client still accepts `vdb_op="milvus"` for compatibility in some code paths).
- **Container FFmpeg** — images no longer bundle `ffmpeg`/`ffprobe` by default. Audio and video extraction require these binaries on `PATH`; for Helm set `service.installFfmpeg=true`, or install system FFmpeg manually.
- **Experimental CLI** — `retriever` subcommands other than `ingest`, `query`, and `pipeline` are marked experimental (NVBugs 6199005, 6198526).
- **Python version** — `nemo_retriever` requires Python 3.12.

### Pipeline and ingestion architecture

- **Graph-based ingest pipeline** — `graph_pipeline`, the graph stage registry, and Pipeline Graph Registry primitives are the canonical ingestion path; performance regressions vs the old batch path were addressed.
- **Manifest-based routing** — input-type routing replaced with manifest branches; ingest plans resolve file types through manifest configuration.
- **Input-aware `retriever ingest`** — CLI and library routing for PDF, image, audio, video, text, HTML, DOCX/PPTX (LibreOffice required), SVG, and related types; unsupported extensions raise clear errors.
- **Detection mode** — HTML and plain-text inputs are honored correctly in detection mode.
- **`.extract()` fixes** — unknown kwargs are no longer silently dropped; `extract_page_as_image=True` renders page images; remote OCR and page-elements logic aligned with local behavior.
- **Table structure** — OCR detections joined with table-structure output to restore markdown cells; OCR removed from table-structure actor path where redundant.
- **Image pipeline** — image deduplication stage, auto-routing for image files in `.extract()`, content transform reloads images from disk, autoscaling store-sink actors for image storage.
- **Webhook operator** — posts intermediate pipeline results back to a webserver layer.
- **Remote NIM errors** — stage failures from remote NIMs surface clearly to callers.
- **Async HTTP pools** — NIM interactions use pooled async HTTP for throughput.
- **Actor auth probes** — revoked API keys detected at actor startup.
- **Ray** — batch de/serialization lag fixed; `HF_HUB_OFFLINE=1` set in Ray workers to avoid Hub rate limits.
- **`allow_no_gpu`** — optional flag to run ingest on CPU when no GPU is available (for experimentation).

### CLI and developer experience

- **Root CLI** — `retriever ingest` and `retriever query` with NIM URL flags (`--*-invoke-url`), batch tuning, and LanceDB overwrite/append controls.
- **`retriever pipeline`** — pipeline subcommand for graph execution and configuration.
- **`retriever skill-eval`** — benchmark CLI for the `/nemo-retriever` skill.
- **OCR language selectors** — CLI flags for Nemotron OCR v2 language selection.
- **Quieter ingest** — reduced default verbosity for `retriever ingest`.
- **UV lockfiles** — pre-commit hook regenerates and verifies `uv.lock` files; lockfiles committed to the repo.

### Retriever Service and deployment

- **Retriever Service v2** — multi-pod architecture with gateway, process isolation, and VectorDB integration.
- **Service ingestor client API** — updated client for submitting and tracking ingestion jobs.
- **Service mode defaults** — ingestion jobs return results by default; service-mode parameter validation added.
- **PDF pre-split** — service-only `pdf_split_config` for large PDF handling (documented; NVBugs 6218013).
- **Runtime FFmpeg** — `service.installFfmpeg` installs `ffmpeg`/`ffprobe` at container startup when audio/video extraction is required.
- **OpenTelemetry** — basic OTEL instrumentation for pipeline and service observability.
- **Air-gapped deployment** — expanded guidance in [deployment options](deployment-options.md) and the [Helm chart README](https://github.com/NVIDIA/NeMo-Retriever/blob/26.05/nemo_retriever/helm/README.md) (image inventory, mirroring, packaging, private-registry overrides; NVBugs 6195103).
- **Kubernetes** — Helm uses cluster default `storageClass` instead of forcing `local-path`.
- **NIM Operator** — GPU resource requests fixed for NIM subcharts.

### Models, OCR, PDF, and captioning

- **Nemotron OCR v2** — default OCR engine; unified OCR actors; nightly packaging and language selectors; consumers pinned through RC validation.
- **Nemotron Parse** — alternate PDF extraction method with v1.2 HTTP interface; local inference via vLLM; optional Helm NIM; tuning param types fixed.
- **VLM image captioning** — delivered via vLLM (addresses 26.03 deferral); Omni caption model profiles (`nemotron_3_nano_omni_30b_a3b_reasoning`); CaptionGPUActor exclusive GPU access.
- **Charts and infographics** — captioning and chart extraction aligned with Helm NIM topology (NVBugs 6195023, 6195296).
- **Page elements** — skip model download when only `pdfium` text extraction is requested; page-elements model name updated to `nvidia/nemotron-page-elements-v3` in docs.
- **vLLM inference stack** — vLLM-backed text and vision-language embedders, multimodal VL reranker, Nemotron Parse local inference; torch bumped to 2.11 (from 2.10 on the 26.05 line) for local GPU installs.
- **Embedder throughput** — bf16 loading and single-call tokenization; default embed batch size 32; local embedder cached across repeated `query()` calls.
- **Normalization** — `normalization=False` works with vLLM-backed text embeddings.

### Multimodal: video, audio, and images

- **Video retrieval pipeline** — frame extraction, OCR, audio-visual fusion, text deduplication; video ASR demuxing fixes.
- **Audio** — long-audio Parakeet chunking with time-aligned segments; punctuation-based segmenting; `segment_audio` documented; batched audio extraction performance improvements; ASR batch/streaming mode with auto-selection.
- **Multimedia extra** — audio extraction documented as requiring the `[multimedia]` install extra.
- **Image captioning and storage** — simplified image storage; `.store()` persists extracted images and text to configured storage URIs (relative URIs resolved to absolute paths).

### Retrieval, RAG, and query

- **Live RAG SDK** — `Retriever.retrieve()`, `Retriever.answer()`, and fluent batch operator graphs via LiteLLM (`[llm]` extra); optional LLM judge and reference scoring.
- **Reranking** — rerank support in library and service paths; `--reranker-invoke-url` on `graph_pipeline.py`; reranker endpoint docstrings updated; remote rerank posts to NIMs fixed.
- **Query and recall** — local query-embedding regressions fixed for recall evaluation; pre/post-processing updates for improved recall.
- **Metadata filtering** — notebook and docs for LanceDB `where` filters; cross-links to RAG Blueprint Elasticsearch, Pinecone, and Teradata VDB guides.

### Vector database and storage

- **VDB in pipeline** — vector-database operators integrated directly in the graph pipeline (not only post-hoc upload).
- **Custom metadata** — metadata fields supported through the NRL ingest path.
- **LanceDB** — hybrid search guidance updated; variable-length vector upload crash fixed; `source_id` restored in LanceDB schema; deprecation warnings addressed; overwrite/append CLI controls.
- **Milvus** — dropped from primary extraction documentation; LanceDB documented as the supported first-party path for new deployments.
- **Partner VDBs** — ADT VDB operator cleanup; custom `VDB` subclass pattern documented for third-party backends.

### Evaluation and benchmarks

- **BEIR-centric evaluation** — non-audio recall eval switched to BEIR; legacy eval code removed; earnings, FinanceBench, Bo767, and Bo10k migrated.
- **`retriever skill-eval`** — skill benchmark CLI and harness integration.
- **Harness** — graph runs emit structured metrics; audio recall eval; Vidore v3; BRIGHT agentic retrieval intro; Spider2 and BIRD benchmarks; in-memory QA eval pipeline; queries parser and comparison tooling.
- **AbstractOperators** — agentic retrieval patterns in `retrieval_bench`.
- **Pipeline eval default** — default pipeline evaluation mode set to `none`; audio recall mode renamed.

### Text-to-SQL and tabular

- **Text-to-SQL agent** — agent graph for structured data retrieval; response formatting cleanup.
- **Tabular ingestion** — tabular data ingestion path; dict-shaped metadata filters in tabular semantic search; tabular dev tools reorganized under `tabular/`.
- **Embedder params in SQL generation** — embedder configuration passed through text-to-SQL generation.

### Packaging, platform support, and dependencies

- **Optional extras** — `[local]`, `[multimedia]`, `[llm]`, `[tabular]`, `[nemotron-parse]`, `[service]`, and others for minimal vs full installs.
- **Slim remote install** — Mac and Windows support for remote/NIM-only inference without local GPU stacks.
- **PR install smoke** — Windows and macOS install validation in CI.
- **Stable HF PyPI dispatch** — Hugging Face PyPI release workflow for model artifacts.
- **Perform Release** — PyPI and Helm publish workflows for the `26.05` release line (RC tags, wheel layout fixes, idempotent Helm publish).
- **Security** — dependency bumps for CVE fixes (cryptography, requests, pillow, pypdf, aiohttp, litellm pin, and others).
- **Requires Python 3.12** — `.python-version` added; `requires-python` tightened.

### Helm chart and NIM stack

- **Chart refresh** — service ingestor client, NIM stack updates, rerank VL version alignment.
- **Core vs optional NIMs** — four core NIMs documented vs optional Parse and Omni caption NIMs (NVBugs 6204537 and related).
- **GA VL embedder** — `llama-nemotron-embed-vl-v2` defaults updated for GA.
- **Helm paths** — chart moved under `nemo_retriever/helm/`; old top-level Helm tree removed.
- **B200** — nemotron-parse guidance for B200 in extraction docs.

### Documentation

- **Helm-first deployment** — supported production path through Helm and NIM Operator; [Docker Compose for local development](https://github.com/NVIDIA/NeMo-Retriever/blob/26.05/nemo_retriever/docker.md) documented as **unsupported** developer tooling.
- **Unified prerequisites & support matrix** — hardware, NIM pins, HF weight sizes, and caption guidance consolidated.
- **Extraction doc refresh** — concepts, ingest workflow, embeddings, audio/video, VDBs, custom metadata, multimodal extraction, and deployment options updated for the graph pipeline.
- **UDF and custom stages** — duplicate user-defined stages page removed; guidance consolidated in the [graph README](https://github.com/NVIDIA/NeMo-Retriever/tree/26.05/nemo_retriever/src/nemo_retriever/graph#nemo-retriever-graph).
- **Strict MkDocs** — strict build, TOC cleanup, autorefs doctest fixes, GitHub Pages deploy for NRL docs only.
- **Sphinx API docs** — Python API reference for `nemo_retriever`.
- **Token splitting** — default Llama tokenizer documented for token-based chunking.
- **Telemetry page removed** — outdated telemetry doc dropped; Omni caption NIM documented instead.
- **CHANGELOG removed** — release notes in this file replace the repo-root `CHANGELOG.md`.

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
- [Deploy with Helm](https://github.com/NVIDIA/NeMo-Retriever/blob/26.05/nemo_retriever/helm/README.md)
