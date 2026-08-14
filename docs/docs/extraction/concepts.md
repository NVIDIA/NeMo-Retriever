# Concepts

These terms appear throughout NeMo Retriever Library documentation.

## Job { #job }

An **ingestion job** is a unit of work you run on input content (documents, audio, video, and other supported types). You submit jobs through the **ingestor Python API** (for example `Ingestor` task chains such as `.extract(...)`) or the **`retriever ingest` CLI**—not by posting a standalone JSON job document. Default tasks target strong recall; customize behavior with task keyword arguments (including chunking and splitting on `.extract()`) or custom UDF-style operations. For UDFs and other extension paths, refer to [Customize & extend](customize-extend.md). Results are structured metadata and annotations (Ray Dataset, pandas `DataFrame`, or similar).

## Pipeline and tasks { #pipeline-and-tasks }

NeMo Retriever Library does **not** run one static pipeline on every document. You configure **tasks** such as parsing, chunking, embedding, storage, and filtering per job. For UDFs, custom graph stages, and other extension paths, refer to [Customize & extend](customize-extend.md).

## Extraction metadata { #extraction-metadata }

Output is a **Ray Dataset** (Ray Data) or **pandas** `DataFrame` listing extracted objects (text regions, tables, images, and so on), processing notes, and timing or trace data. Field-level detail is in the [metadata reference](content-metadata.md).

## Embeddings and retrieval { #embeddings-and-retrieval }

Optionally, the library can compute **embeddings** for extracted content and store vectors in [LanceDB](https://lancedb.com/) for downstream semantic search in your application. For upload and retrieval APIs, refer to [Vector databases](vdbs.md). For multimodal (VLM) embedding options, refer to [Multimodal embeddings (VLM)](embedding.md).

## Chunking { #chunking }

Chunking is built into the `.extract()` task and depends on **content type**:

- **PDF, DOCX, and PPTX** — Text is grouped using built-in **page** boundaries (one chunk per page where the format has pages).
- **Plain text (`.txt`) and HTML** — Formats without natural page breaks are split into segments of **1024 tokens** by default, using the revision-pinned [Llama Nemotron Embed VL 1B v2 tokenizer](https://huggingface.co/nvidia/llama-nemotron-embed-vl-1b-v2) so chunk boundaries stay aligned with the default embedding model. Published service images bundle this tokenizer artifact without model weights, so default text chunking does not require Hugging Face access at runtime. Refer to [Token-based splitting](#token-based-splitting) and [Environment variables](environment-config.md) for overrides and other runtimes.
- **Audio and video** — Media is split into **segments** for decoding and ASR using ffmpeg-based rules (configurable **size**, **time**, or **frame** split modes in the media chunking stage). With the Parakeet ASR path, you can optionally emit **sentence-like segments** using `extract_audio_params={"segment_audio": True}`; refer to [Speech and audio extraction](audio-video.md#speech-and-audio-extraction).

For PDF parallelism before Ray processing (large files), refer to [PDF pre-splitting for parallel ingest](nemo-retriever-api-reference.md#pdf-pre-splitting-for-parallel-ingest).

### Token-based splitting { #token-based-splitting }

Token-based splitting uses the revision-pinned tokenizer for the default embedding model (`nvidia/llama-nemotron-embed-vl-1b-v2`) with configurable `max_tokens` and `overlap_tokens`. For graph and library ingest (`create_ingestor(run_mode="inprocess")` or `create_ingestor(run_mode="batch")`), set those values on `.extract(split_config={"text": {"max_tokens": ..., "overlap_tokens": ...}})`, or omit `split_config` to use default text segmentation for unstructured text. Do not call `.split()` on `GraphIngestor`; that method exists only on `ServiceIngestor` (`run_mode="service"`). Published service images and the documented source builds include the tokenizer locally; source builds enable this with `--build-arg DOWNLOAD_DEFAULT_TOKENIZER=True`. The `service` image disables runtime Hub access, while `service-gpu` remains online for its other Hugging Face models. The base library install includes the tokenizer Python dependencies; pre-populate the Hugging Face cache before offline use. For parameter details, refer to the [Python API guide](nemo-retriever-api-reference.md).

## Deployment modes { #deployment-modes }

- **Library mode** — Run without the full container stack where appropriate; refer to [Deployment options](deployment-options.md).
- **Kubernetes / Helm (self-hosted)** — Refer to [Deploy (Helm chart)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md) and [deployment options](deployment-options.md) for running the full microservices pipeline on your infrastructure.
- **Notebooks** — [Jupyter examples](https://github.com/NVIDIA/NeMo-Retriever/blob/main/examples/README.md) for experimentation and RAG demos.

For a concise comparison, refer to [Deployment options](deployment-options.md).
