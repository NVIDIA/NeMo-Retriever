# Multimodal extraction

NeMo Retriever Library classifies and extracts text, tables, charts, infographics, and related layout from documents and media. This page groups formats, extraction modes, structured outputs, and throughput guidance in one place. Use the table of contents to jump to a topic.

## On this page

- [Supported file types and formats](#supported-file-types-and-formats)
- [Text and layout extraction](#text-and-layout-extraction)
- [Tables](#tables)
- [Charts and infographics](#charts-and-infographics)
- [OCR and scanned documents](#ocr-and-scanned-documents)
- [Image captioning](#image-captioning)
- [Metadata and content schema](#metadata-and-content-schema)
- [Extraction limitations and quality](#extraction-limitations-and-quality)

## Supported file types and formats { #supported-file-types-and-formats }

NeMo Retriever Library accepts multiple document and media types. A current list (including PDF, Office formats, HTML, images, audio, and video, some early access) appears in [NeMo Retriever Library Overview](overview.md) under **NeMo Retriever Library supports the following file types**.

**Related**

- [Troubleshoot](troubleshoot.md) for format-specific issues
- [Speech and audio](audio-video.md)

## Text and layout extraction { #text-and-layout-extraction }

For PDFs, NeMo Retriever Library typically uses **pdfium**-based extraction with configurable depth and paths. Scanned or mixed pages may use hybrid, OCR-oriented, or Nemotron Parse methods. For `method` options such as `pdfium`, `pdfium_hybrid`, `ocr`, and `nemotron_parse`, refer to the [Python API reference](nemo-retriever-api-reference.md).

!!! note
    `method="nemotron_parse"` requires the Nemotron Parse NIM client dependencies. Install them with the `nemotron-parse` extra, for example `pip install "nemo-retriever[nemotron-parse]"`, before running PDF extraction through Nemotron Parse. Local inference defaults to `nvidia/NVIDIA-Nemotron-Parse-v1.2`, and self-hosted NIM inference defaults to `nvidia/nemotron-parse-v1.2`. To use Parse 2.0, set `nemotron_parse_model` to `nvidia/NVIDIA-Nemotron-Parse-2.0` for local inference or `nvidia/nemotron-parse-v2.0` for a compatible self-hosted endpoint.

**Related**

- [NeMo Retriever Library Overview](overview.md)
- [OCR and scanned documents](#ocr-and-scanned-documents)
- [Chunking](concepts.md#chunking)

## Tables { #tables }

NeMo Retriever Library detects tables as structured page elements, processes them through the appropriate NIMs, and exports formats suitable for downstream RAG (including Markdown-oriented representations where configured). Availability depends on pipeline and model configuration; refer to the [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md).

**Related**

- [NeMo Retriever Library Overview](overview.md) for artifact classification
- [Nemotron Parse](https://build.nvidia.com/nvidia/nemotron-parse) for advanced visual parsing
- [Metadata reference](content-metadata.md)

## Charts and infographics { #charts-and-infographics }

Charts and infographic regions are classified with other page layout elements (tables, text blocks, titles) and processed through layout detection and OCR. `extract_charts` and `extract_infographics` are enabled by default. Outputs use the same metadata schema as other extracted objects.

!!! note "Nemotron Parse chart routing"
    [NVIDIA Nemotron Parse 2.0](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-2.0) emits the tagged `Chart` class. When you explicitly select that model with `method="nemotron_parse"`, NeMo Retriever Library routes `<class_Chart>` regions to `chart` modality rows. The default Parse v1.2 model does not produce chart modality rows.

Chart-labeled PDF regions are **not** routed through the Omni caption stage; they remain on the layout-and-OCR path. For scope and validation guidance, refer to [Image captioning](#image-captioning).

For natural-language infographic descriptions, optionally enable [image captioning](#image-captioning) and set `caption_infographics=True` when you need VLM captions on infographic regions.

**Related**

- [NeMo Retriever Library Overview](overview.md)
- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
- [Multimodal embeddings (VLM)](embedding.md) when you treat graphics as images for embedding

## OCR and scanned documents { #ocr-and-scanned-documents }

Scanned PDFs and image-only pages rely on OCR and hybrid paths that combine native text extraction with OCR when needed. For extract methods such as `ocr` and `pdfium_hybrid`, refer to the [Python API reference](nemo-retriever-api-reference.md).

When you run extraction locally with Hugging Face weights, the default OCR engine is **Nemotron OCR v2**, which operates in **multilingual** mode by default. For CLI flags and API parameters, refer to [CLI — OCR language mode](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/docs/cli/README.md#ocr-language-mode). For Kubernetes image pins and overrides, refer to [OCR NIM configuration](https://github.com/NVIDIA/NeMo-Retriever/blob/main/nemo_retriever/helm/README.md#ocr-nim-configuration). For hosted OCR endpoints and the NVCF language-mode limitation, refer to [Default NVCF endpoints](prerequisites-support-matrix.md#default-nvcf-endpoints).

**Related**

- [Text and layout extraction](#text-and-layout-extraction)
- [Nemotron Parse](https://build.nvidia.com/nvidia/nemotron-parse)
- [Extraction limitations and quality](#extraction-limitations-and-quality)

## Image captioning { #image-captioning }

Image captioning generates natural-language descriptions for unstructured image content. Retrieval can then use text embeddings over captions and visual embeddings where you configure them.

**Captioning is optional** — enable it in your ingest configuration (for example, the `caption` API or pipeline flag) when you need natural-language descriptions of image content. Reasoning traces are disabled by default for captioning.

Direct local Hugging Face captioning defaults to `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`. Its BF16 weights are approximately 62 GiB. For this local vLLM profile, NeMo Retriever Library reserves `0.95` of a dedicated GPU's memory for the model and KV cache. An NVIDIA H100 with 80 GB of memory meets this local profile's minimum capacity when it is dedicated to captioning. The Nano caption profiles retain their `0.5` memory-utilization default and remain available through explicit model overrides. Hosted captioning defaults to `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` at the NVIDIA API endpoint.

The 80 GB requirement for the self-hosted Omni NIM describes the NIM deployment. It does not by itself establish that local Hugging Face vLLM inference has enough memory for both weights and its KV cache. For local vLLM, use a dedicated GPU and retain the model-profile default unless you need to tune its memory reservation explicitly.

For example, to override the local vLLM memory reservation from the SDK, pass `CaptionParams` to `caption`.

```python
from nemo_retriever import create_ingestor
from nemo_retriever.common.params import CaptionParams, ExtractParams

result = (
    create_ingestor(run_mode="inprocess")
    .files(["multimodal_test.png"])
    .extract(ExtractParams(extract_images=True))
    .caption(
        CaptionParams(
            model_name="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
            gpu_memory_utilization=0.95,
        )
    )
    .ingest()
)
```

Chart-classified PDF regions stay on the layout/OCR path; only non-chart image regions and optional infographics (`caption_infographics=True`) receive Omni captions.

**Related**

- [Multimodal embeddings (VLM)](embedding.md)
- [Metadata reference](content-metadata.md)
- [Image captioning](prerequisites-support-matrix.md#image-captioning)

## Metadata and content schema { #metadata-and-content-schema }

Extracted objects follow the schema and field descriptions in the [Metadata reference](content-metadata.md). Use that page for tables, types, and per-field notes.

## Extraction limitations and quality { #extraction-limitations-and-quality }

Hosted Page Elements, Table Structure, and Graphic Elements NIM endpoints cap inline base64 image payloads at about **180,000 characters** (roughly 180 KB). The NeMo Retriever pipeline downscales large page renders before remote NIM calls. Direct API integrations must keep inline payloads under that cap. Hosted Page Elements does not accept NVCF Asset API references. For limits, plus `dpi` and `render_mode` tuning, refer to [Hosted Page Elements NIM image size limits](troubleshoot.md#hosted-page-elements-nim-image-size-limits).

Image payload limits are separate from the throughput metrics in the rest of this section.

A single headline metric can drastically misrepresent system efficiency. The amount of compute that you need to process a dataset depends far more on its content and how your pipeline operates than on its disk size. This section explains why, and offers better ways to measure and report throughput.

Some common throughput measures, and their problems, include the following:

- **TB/day, GB/hour, MB/s** – Useful for capacity planning for storage and network, and the cost of data movement or archival. A weak proxy for compute due to compression and encoding differences.
- **docs/min (documents per minute)** – Easy to understand, but documents vary wildly in length and complexity.
- **pages/sec (pages per second)** – Usually correlates with work batching (sets-of-pages from PDFs). Varies with per-page complexity and modality mix.
- **images/sec** – Relevant when image transforms dominate. Sensitive to resolution.
- **tokens/sec** – Useful for LLM/VLM text-heavy stages. Ignores non-text work.
- **elements/sec (tables/sec, charts/sec, OCR pages/sec)** – Stage-specific and informative. Must be paired with prevalence (how many elements per page).

### Summary

- Disk size is not a reflection of expected processing time. Content complexity and enabled tasks dominate actual compute cost.
- Pages/sec is generally better than data-size-over-time metrics because it correlates more with work units, but it is still imperfect.
- Report throughput alongside dataset characteristics and stage-level metrics for meaningful, reproducible comparisons.

### Example use cases

The following two datasets can yield the reverse ranking if you evaluate by data-size-over-time versus by pages/sec:

- **Complex-but-small** – A 1000-page PDF where each page contains dense tables and charts. The PDF may be small on disk (vector text, compressed graphics) yet very expensive to process (table detection, OCR, structure reconstruction, chart parsing).
- **Large-but-simple** – A 1000-page PDF with one large image per page. The file may be huge on disk (high-DPI scans) but comparatively fast to process if your pipeline mostly routes images without heavy analysis.

### What drives processing cost

The following factors drive processing cost.
!!! important
    None of the following factors correlate with file size.

- Content modality and tasks enabled
  - Text OCR vs. native text extraction
  - Table structure detection and reconstruction
  - Chart detection and text extraction
  - Image captioning or vision-language models
  - Embedding generation and vector storage

- Content density and complexity per page
  - Number of elements (tables, figures, charts, text blocks)
  - Layout complexity (nested tables, merged cells, multi-column text)
  - Languages, scripts, and fonts (OCR difficulty)

- Resolution and quality
  - DPI for scanned pages (I/O and pre-processing cost)
  - Compression artifacts vs. vector graphics

- Pipeline configuration
  - Which stages are turned on/off
  - Model choices (accuracy vs. speed trade-offs)
  - Batch sizes, concurrency, hardware placement

- System factors
  - Warm-up vs. steady state
  - I/O bandwidth and storage latency
  - Network latency to inference services

### Why data-size-over-time is misleading

Use data-size-over-time metrics for storage and network planning, not for compute efficiency.

The following are examples of why data-size-over-time metrics are misleading:

- Compression breaks the proxy
  - Highly compressible vector PDFs may be tiny yet compute-heavy.
  - Scanned images may be huge but require minimal analysis.

- Format dependency
  - Two datasets with identical content can have wildly different byte sizes due to encoding/format.

- Incentivizes the wrong optimizations
  - Encourages selecting “big-byte” but easy datasets to inflate data-size-over-time without improving true efficiency.

- Not portable across stages
  - Bytes are not additive across pipeline stages (and often increase or decrease as formats change).

- Hard to reproduce
  - Data-size-over-time varies wildly with dataset encoding choices, not just system performance.

### Why pages/sec is better (but imperfect)

When you report pages/sec, you should also report dataset characterization.

The following are some reasons why pages/sec is better than data-size-over-time metrics:

- Closer to the work unit
  - Pipelines commonly schedule and process sets-of-pages from PDFs to saturate pipeline resources.
- Normalizes away compression and file format
  - A page is a page regardless of on-disk bytes.

However, pages/sec is still imperfect because of the following:

- Page complexity varies
  - Pages with many tables/charts/figures or dense text cost more than blank or simple pages.
- Modality mix differs
  - OCR-heavy pages vs. native text pages drive very different compute paths.
- Resolution matters
  - High-DPI scans require more I/O and pre-processing.

### Example: interpreting the two 1000-page PDFs

The supposedly fast dataset by data-size-over-time can be the slow one by pages/sec, and vice versa. Only context-rich reporting avoids this trap.

The following are reasons why:

- Complex tables + charts per page (small file size)
  - Data-size-over-time appears low due to tiny bytes, but compute is high → pages/sec and stage-level metrics reveal true cost.
  - Expect lower pages/sec and lower tables/sec/charts/sec to dominate.
- Single large image per page (large file size)
  - Data-size-over-time appears high due to big bytes, but compute can be low → fast pages/sec.
  - If table/chart stages are skipped, stage-level numbers show negligible table/chart work.

### Practical tips for fair comparisons

The following are practical tips for fair comparisons:

- Separate warm-up from steady-state measurements.
- Fix the pipeline configuration and model versions for a given comparison.
- Keep concurrency and resource limits identical across runs.
- Provide dataset characterization alongside throughput numbers.
