# Release Notes for NeMo Retriever Library

This documentation contains the release notes for [NeMo Retriever Library](overview.md).

!!! note

    NVIDIA Ingest (nv-ingest) has been renamed to the NeMo Retriever Library.   

## 26.03 Release Notes (26.1.3)

NVIDIA® NeMo Retriever Library version 26.03 adds broader hardware and software support along with many pipeline, evaluation, and deployment enhancements.

To upgrade the Helm charts for this release, refer to the (NeMo Retriever Library Helm Charts](https://github.com/NVIDIA/NeMo-Retriever/blob/release/26.3.0/helm/README.md).

Highlights for the 26.03 release include:

- Product and package naming
  - NV‑Ingest is now called NVIDIA NeMo Retriever, and the extraction pipeline is referred to as NeMo Retriever Library so naming is consistent across docs and tooling.
  - The Python package was renamed from `retriever` to `nemo_retriever`, and the published PyPI wheel is now `nemo-retriever`, so imports and installations use the new name.
  - Documentation and Helm charts now use NeMo Retriever Library and Nemotron branding in place of the older NV‑Ingest naming.

- Image, audio, and multimodal support
  - You can now ingest image files (PNG, JPEG, BMP, TIFF, SVG) directly; the pipeline automatically routes them in `.extract()` for both in‑process and batch runs so images are handled like other document types.
  - The pipeline can extract and process audio (for example, through RIVA ASR) so audio can be transcribed and used in the same ingestion flow as documents.
  - Multimodal embedding now supports both images and text (multimodal and text_image modalities), so documents that combine figures and text share a single embedding space for retrieval.
  - You can select a vision‑language embedder for retrieval so image and text content are embedded together, improving multimodal recall.
  - Text elements can use the text_image modality in the multimodal embedder, which can improve how mixed text‑and‑image chunks are represented.
  - When text is empty in a multimodal flow, the pipeline falls back to image‑only embeddings so retrieval still works for image‑only chunks.

- Extraction, OCR, PDFs, and Markdown
  - The retriever pipeline can use Nemotron Parse for extraction, and the batch pipeline documentation now covers how to configure Nemotron Parse.
  - OCR now accepts the model name `"pipeline"` and uses it as the default instead of `'scene_text_ensemble'`, aligning with the standard pipeline behavior.
  - New extract methods, `pdfium_hybrid` and `ocr`, improve extraction from scanned PDFs, yielding better text and layout from image‑based pages.
  - Helper functions for rendering content as Markdown were added, and the table‑structure stage now supports Markdown tables so batch and in‑process pipelines produce correct Markdown output.
  - Infographics can be captioned with a configurable VLM, and documentation describes how to use VLM captioning on infographics and how to include surrounding page text for better‑grounded captions.
  - PDF pages are rendered as JPEG instead of PNG to reduce memory use and speed up processing while preserving suitable quality for downstream stages.
  - PDF rendering now uses the target scale directly, which reduces extra work and memory, and the PDFium integration replaces `rev_byteorder` with an in‑place OpenCV channel swap so image byte order is correct.

- Vector store and retrieval
  - LanceDB is now the default vector database backend instead of Milvus, so new deployments use LanceDB unless you override it.
  - LanceDB supports hybrid search (dense vectors plus full‑text search), and harness defaults were updated to use hybrid search where appropriate.
  - A `source_id` field was added to output columns and restored in LanceDB output so each result can be traced to its source document or chunk.
  - LanceDB index creation now happens after datasets are materialized, and fixes ensure LanceDB indexing, recall, and the `source_id` column behave correctly after pipeline runs, including a “slim” build path fix and output recall corrections.

- Chunking, embedding, and retrieval behavior
  - In addition to chunk‑level embedding, you can embed at page level so each page is represented as a single vector, which can simplify some retrieval setups.
  - The retriever adds a `.split()` method for chunking text by token count, giving you more control over chunk sizes for embedding and retrieval.
  - Pre‑ and post‑processing for retrieval were updated to improve recall so results better match user queries.
  - The embed stage now respects the configured pipeline endpoint instead of ignoring it.
  - Reranker behavior was corrected for in‑process mode and for network‑call reranking, including removal of an incorrect assertion in the rerank path and a fix to the score field so relevance scores are reported properly.

- Nemotron Parse and extraction defaults
  - The `nemotron_parse` extraction method was updated for better compatibility and behavior with the current Nemotron Parse service.
  - Extract options such as `extract_text` and `extract_images` now default to true, aligning with `extract_tables` and `extract_charts` so extraction behavior is more consistent.
  - The in‑process `.extract()` path correctly handles `.txt` files and automatically routes image files for both in‑process and batch runs.
  - The `extract_primitives_from_pdf_*` functions now accept an authentication keyword argument that matches the extractor schemas so auth configuration works consistently.

- Layout, graphics, and tables
  - The graphic‑elements stage was updated for model and configuration compatibility so chart and graphic detection continues to work as expected.
  - Layout handling removed padding from page elements so bounding boxes line up more closely with content.
  - Images overlapping table regions are deduplicated based on bounding boxes so tables and figures are not double‑counted.

- retrieval_bench and evaluation
  - A new `retrieval_bench` package supports retrieval pipeline evaluation (for example, ViDoRe V3 and BRIGHT leaderboards) so you can measure and compare retrieval configurations.
  - Dense retrieval evaluation supports pluggable backends (for example, different embedders) so you can compare models and settings.
  - An agentic retrieval mode lets an LLM agent iteratively refine retrieval so you can evaluate agent‑based pipelines.
  - Dataset recall adapters were added for earnings, FinanceBench, and the JP20 dataset so those benchmarks can be run in the same framework.
  - The `jp20_recall` evaluator accepts a `hybrid` parameter so you can evaluate dense and hybrid search.
  - Recall‑related helpers and detection‑summary logic were consolidated so evaluation and recall logic live in a single, clearer place.

- Harness features and tests
  - The test harness can deploy and run tests using Helm as well as Docker Compose, and it includes a service manager that can install or upgrade stacks, choose versions, and use Compose override files.
  - Harness runs now attach metadata to result artifacts so you can see configuration, version, and environment when you review results.
  - A CLI and artifact flow were added so nightly and ad hoc retriever harness runs can be launched and stored in a standard way.
  - New or updated harness test cases cover page_elements, graphic_elements, table_structure, and OCR so Nemotron model behavior is regression‑tested.
  - The harness was updated to work with the current batch pipeline, and it now runs tests for the Hugging Face PyPI packages (page‑elements, graphic‑elements, table‑structure).
  - The harness and `retrieval_bench` dependency groups were merged and pip install instructions were updated.
  - The retriever nightly refresh and harness are wired together so nightly retriever builds are exercised automatically.

- Deployment, Helm, and Compose
  - The Helm chart documentation now includes guidance for air‑gapped or offline deployments, a Helm override for OpenShift, and override files that align Helm deployments with existing Docker Compose setups for hardware such as NVIDIA A10G, L40S, and RTX PRO 4500.
  - Helm values and documentation were updated for Nemotron branding, and Helm resource limits for Ray environments and ingest pods now align more closely with Compose defaults.
  - Docker Compose and Helm were updated to support the NVIDIA RTX PRO 4500 (Blackwell), including a Helm override that tunes object‑detection warmup batch size for this SKU.
  - Docker‑related changes include added port forwards in `docker-compose`, updated override files for more GPU and configuration profiles, explicit warmup batch size for object‑detection NIMs on NVIDIA A10G, and a default of disabled dynamic scaling in Compose so behavior is more predictable unless you turn it on.
  - Support and documentation were added for configuring MIG slices for NIM models to support multi‑instance GPU deployments.
  - The test harness now supports Helm‑based deployments and can use Docker Compose override files for custom test configurations.

- Environment, NIMs, and models
  - The project now uses UV as the primary environment and package manager instead of Conda for faster installation and simpler dependency handling.
  - When you use remote hosted NIMs, you can pass `NVIDIA_API_KEY` from the environment so authentication works without hardcoded secrets.
  - You can run NeMo Retriever Library with locally loaded Hugging Face models on your GPU for embedding and OCR instead of relying on NIM endpoints.
  - The Hugging Face model registry and documentation now include `nvidia/llama-nemotron-embed-1b-v2` and `llama-nemotron-embed-vl-1b-v2` for dense and multimodal retrieval.
  - Nemotron OCR builds were updated for the appropriate NVIDIA GPU architectures, and nightly builds of NVIDIA Nemotron Hugging Face models are published so prebuilt artifacts are available.
  - NIM image and configuration versions were bumped for OCR and related services so the stack uses supported NIM versions, and NIM resource and limit guidance (“NIM‑trition”) is documented.
  - KV‑cache reuse was disabled for VLMs in configurations where it caused incorrect behavior so outputs remain consistent.

- Performance, caching, and heuristics
  - Default Redis TTL was increased to 48 hours so longer‑running jobs, including VLM captioning, are less likely to expire before completion.
  - Redis gained an instrumented connection pool to improve throughput and enable better metrics.
  - Ray DAG balancing heuristics and an extended default system resources heuristic improve batch jobs’ ability to use cluster resources evenly and pick better defaults across a wider range of machines.
  - Hugging Face cache directories are now resolved correctly in batch runs so cached models are reused, and tokenizers are cached across jobs to avoid repeated loads of the same model.

- Model selection and configuration utilities
  - Helper functions such as `get_*_model_name` were refactored so fallback model names are not cached incorrectly and the correct models are used.
  - You can pin Hugging Face model revisions so pipeline runs are reproducible over time.
  - `trust_remote_code=True` was added to the AutoTokenizer for the `embedqa` model so it can load correctly from Hugging Face.
  - The scope of `get_hf_revision` was limited to the `nemo_retriever` package to avoid side effects elsewhere.

- Codebase consolidation and utilities
  - Helpers for parameter coercion and embed keyword arguments moved into `params/utils` so they are shared and consistent, and related utilities for LanceDB and recall were consolidated to reduce duplication.
  - Detection‑summary logic was consolidated, and logic for row‑count computation (`num_rows`) from Ray datasets was corrected so batch and reporting counts are accurate.
  - Noisy warnings from `input_embeds` and `pynvml` are now suppressed where they do not indicate real issues.
  - The Nano 12B model or configuration was reverted where it caused compatibility problems so existing workflows continue to function.

- Security, dependencies, and builds
  - Dependency and code changes address reported CVEs, including version bumps for `azure-core` (for example, to 1.38.0) and `requests` (for example, to 2.32.5), and a relaxation of Transformers version pinning where it is safe to do so.
  - The release container now includes libfreetype source so license and compliance requirements can be met in restricted environments.
  - Nightly Hugging Face builds and PyTorch are now built or selected for CUDA 13 (for example, PyTorch 2.9 with cu130), and associated build scripts were updated accordingly.
  - Docker build naming and references are sanitized so invalid characters in branch or commit names do not break builds, and Docker build and test were combined into a single CI job so artifacts are easier to work with.
  - ARM64 builds and CI steps were temporarily disabled where support is not yet stable.

- GitHub, CI, and release workflows
  - A GitHub release workflow now runs Helm actions and a perform‑release step so releases can be cut and published directly from the repository, and the overall release process was unified into a single GitHub Actions flow.
  - CI now uses `NVIDIA_API_KEY` instead of `NGC_API_KEY`, and event contexts and workflow triggers for pull requests were fixed so runs no longer duplicate or go missing.
  - A reusable integration test workflow had syntax and pytest issues corrected, and CI now cancels in‑progress runs when new commits are pushed to a pull request.
  - CI now runs unit tests for the retriever package and includes `SimpleMessageBroker` atexit handling and an upper bound on `pypdfium2` (less than 5.0.0) for compatibility.
  - A pre‑commit hook validates Docker Compose and Helm configurations to catch invalid YAML or settings before changes are committed.
  - The retriever Docker image build was fixed and the GHCR registry name is forced to lowercase so image pushes succeed.

- Documentation and content updates
  - The README quickstart and default‑branch guidance were updated so new users know when to use `main` versus a `release` branch.
  - A documentation resources section was added on the GitHub home so key docs are easier to find, and the PDF blueprint architecture diagram was updated to match the current pipeline.
  - Documentation was updated to use the term “blueprint” consistently, to clarify the heuristics section, and to cover multimodal embedding options, dataset prerequisites, updated NIM service configurations, VLM caption model selection, and NIM resource limits.
  - Documentation and links for the NIM image used for OCR (for example, image 5967130) were updated to match current deployment steps.
  - Deprecated docs, older GitHub Pages content, Ray JAR artifacts, and outdated pages were removed so the repository reflects current behavior, and unit tests were updated so the suite passes.
  - Workflow and docs build locations and steps were adjusted so docs builds, Conda publishing, and Docker builds run as expected.


NeMo Retriever Library currently does not support image captioning via VLM. It will be added in the next release.

## Release Notes for Previous Versions

| [26.1.2](https://docs.nvidia.com/nemo/retriever/26.1.2/extraction/releasenotes-nv-ingest/)
| [26.1.1](https://docs.nvidia.com/nemo/retriever/26.1.1/extraction/releasenotes-nv-ingest/)
| [25.9.0](https://docs.nvidia.com/nemo/retriever/25.9.0/extraction/releasenotes-nv-ingest/) 
| [25.6.3](https://docs.nvidia.com/nemo/retriever/25.6.3/extraction/releasenotes-nv-ingest/) 
| [25.6.2](https://docs.nvidia.com/nemo/retriever/25.6.2/extraction/releasenotes-nv-ingest/) 
| [25.4.2](https://docs.nvidia.com/nemo/retriever/25.4.2/extraction/releasenotes-nv-ingest/) 
| [25.3.0](https://docs.nvidia.com/nemo/retriever/25.3.0/extraction/releasenotes-nv-ingest/) 
| [24.12.1](https://docs.nvidia.com/nemo/retriever/25.3.0/extraction/releasenotes-nv-ingest/) 
| [24.12.0](https://docs.nvidia.com/nemo/retriever/25.3.0/extraction/releasenotes-nv-ingest/) 
|



## Related Topics

- [Prerequisites](prerequisites.md)
- [Deploy Without Containers (Library Mode)](quickstart-library-mode.md)
- [Deploy With Docker Compose (Self-Hosted)](quickstart-guide.md)
- [Deploy With Helm](helm.md)
