# Concepts

These terms appear throughout NeMo Retriever Library documentation.

## Job

An **ingestion job** is a unit of work done on input content (documents, audio, video, and other supported types). There are default tasks with built-in configurations aimed at strong recall, but behavior can be customized via task keyword arguments and via custom UDF-style operations ([Graph operators](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/graph/README.md), [user-defined stages](user-defined-stages.md)). Results are retrieved as structured metadata and annotations.

## Pipeline and tasks

NeMo Retriever Library does **not** run one static pipeline on every document. You configure **tasks** such as parsing, chunking, embedding, storage, and filtering per job. Related topics: [Extending/Customizing NeMo Retriever Library with custom code](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/src/nemo_retriever/graph/README.md), [user-defined stages](user-defined-stages.md).

## Extraction metadata

Output is a **Ray Dataset** (Ray Data) or **pandas** `DataFrame` listing extracted objects (text regions, tables, images, and so on), processing notes, and timing or trace data. Field-level detail is in the [metadata reference](content-metadata.md).

## Embeddings and retrieval

Optionally, the library can compute **embeddings** for extracted content and store vectors in [LanceDB](https://lancedb.com/) for downstream semantic or hybrid search in your application. For multimodal (VLM) embedding options, see [Multimodal embeddings (VLM)](embedding.md).

## Chunking {#chunking}

Chunking is built into the `.extract()` task and depends on **content type**:

- **PDF, DOCX, and PPTX** — Text is grouped using built-in **page** boundaries (one chunk per page where the format has pages).
- **Plain text (`.txt`) and HTML** — Formats without natural page breaks are split into segments of **1024 tokens** by default, using the [Llama 3.2 1B tokenizer](https://huggingface.co/meta-llama/Llama-3.2-1B) so chunk boundaries stay aligned with the default embedding tokenizer. The NeMo Retriever container image bundles this tokenizer, so default text chunking does not require a Hugging Face access token. See [Token-based splitting](#token-based-splitting) and [Environment variables](environment-config.md) for overrides and other runtimes.
- **Audio and video** — Media is split into **segments** for decoding and ASR using ffmpeg-based rules (configurable **size**, **time**, or **frame** split modes in the media chunking stage). With the Parakeet ASR path, you can optionally emit **sentence-like segments** using `extract_audio_params={"segment_audio": True}`; see [Speech and audio extraction](audio-video.md#speech-and-audio-extraction).

For PDF parallelism before Ray processing (large files), see [PDF pre-splitting for parallel ingest](nemo-retriever-api-reference.md#pdf-pre-splitting-for-parallel-ingest).

### Token-based splitting {#token-based-splitting}

Token-based splitting uses the Llama 3.2 1B tokenizer (default `meta-llama/Llama-3.2-1B`) with configurable `max_tokens` and `overlap_tokens` when you add an explicit `.split(...)` stage or when the pipeline applies the default text segmentation for unstructured text. In the shipped NeMo Retriever container, tokenizer assets are included locally, so you do not need `HF_ACCESS_TOKEN` for this default path. If your runtime loads the tokenizer from the Hugging Face Hub instead (for example, some library-only installs), set `HF_ACCESS_TOKEN` or pass `hf_access_token` in task params when the Hub requires it. Details appear in the [Python API guide](nemo-retriever-api-reference.md).

## Deployment modes

- **Library mode** — Run without the full container stack where appropriate; see [Deployment options](deployment-options.md).
- **Kubernetes / Helm (self-hosted)** — See [Deploy (Helm chart)](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md) and [deployment options](deployment-options.md) for running the full microservices pipeline on your infrastructure.
- **Notebooks** — [Jupyter examples](notebooks.md) for experimentation and RAG demos.

For a concise comparison, refer to [Deployment options](deployment-options.md).
