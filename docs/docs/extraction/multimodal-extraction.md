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

## Supported file types and formats {#supported-file-types-and-formats}

NeMo Retriever Library accepts multiple document and media types. A current list (including PDF, Office formats, HTML, images, audio, and video, some early access) appears in [What is NeMo Retriever Library?](overview.md) under **NeMo Retriever Library supports the following file types**.

**Related**

- [Troubleshoot](troubleshoot.md) for format-specific issues
- [Speech and audio](audio-video.md)

## Text and layout extraction {#text-and-layout-extraction}

For PDFs, NeMo Retriever Library typically uses **pdfium**-based extraction with configurable depth and paths. Scanned or mixed pages may use hybrid or OCR-oriented methods. For `extract_method` options such as `pdfium`, `pdfium_hybrid`, and `ocr`, refer to the [Python API reference](nemo-retriever-api-reference.md).

**Related**

- [What is NeMo Retriever Library?](overview.md)
- [OCR and scanned documents](#ocr-and-scanned-documents)
- [Chunking and splitting](chunking.md)

## Tables {#tables}

NeMo Retriever Library detects tables as structured page elements, processes them through the appropriate NIMs, and exports formats suitable for downstream RAG (including Markdown-oriented representations where configured). Availability depends on pipeline and model configuration; refer to the [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md).

**Related**

- [What is NeMo Retriever Library?](overview.md) for artifact classification
- [Nemotron Parse](https://build.nvidia.com/nvidia/nemotron-parse) for advanced visual parsing
- [Metadata reference](content-metadata.md)

## Charts and infographics {#charts-and-infographics}

Charts and infographic regions are classified as graphic elements and processed with the corresponding NVIDIA NIM workflows (for example, **yolox-graphic-elements** in current releases). Outputs use the same metadata schema as other extracted objects.

**Related**

- [What is NeMo Retriever Library?](overview.md)
- [Pre-Requisites & Support Matrix](prerequisites-support-matrix.md)
- [Multimodal embeddings (VLM)](embedding.md) when you treat graphics as images for embedding

## OCR and scanned documents {#ocr-and-scanned-documents}

Scanned PDFs and image-only pages rely on OCR and hybrid paths that combine native text extraction with OCR when needed. For extract methods such as `ocr` and `pdfium_hybrid`, refer to the [Python API reference](nemo-retriever-api-reference.md).

**Related**

- [Text and layout extraction](#text-and-layout-extraction)
- [Nemotron Parse](https://build.nvidia.com/nvidia/nemotron-parse)
- [Extraction limitations and quality](#extraction-limitations-and-quality)

## Image captioning {#image-captioning}

Image captioning generates natural-language descriptions for unstructured image content. Retrieval can then use text embeddings over captions and visual embeddings where you configure them.

**Related**

- [Multimodal embeddings (VLM)](embedding.md)
- [Metadata reference](content-metadata.md)
- [What is NeMo Retriever Library?](overview.md)

## Metadata and content schema {#metadata-and-content-schema}

Extracted objects follow the schema and field descriptions in the [Metadata reference](content-metadata.md). Use that page for tables, types, and per-field notes.

## Extraction limitations and quality {#extraction-limitations-and-quality}

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

> [!IMPORTANT]
> None of the following factors correlate with file size.

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
