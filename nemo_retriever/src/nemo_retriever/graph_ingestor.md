# GraphIngestor

`GraphIngestor` is the graph-native ingestion API used by NeMo Retriever
Library for local in-process runs and Ray Data batch runs. It records a fluent
pipeline, builds a `nemo_retriever.graph.Graph` with
`nemo_retriever.graph.ingestor_runtime.build_graph`, and executes that graph
with either `InprocessExecutor` or `RayDataExecutor`.

Most users should enter through `nemo_retriever.create_ingestor()`. For
`run_mode="inprocess"` and `run_mode="batch"`, that factory returns a
`GraphIngestor`. `run_mode="service"` returns `ServiceIngestor` instead.

## Basic Usage

```python
from nemo_retriever import create_ingestor
from nemo_retriever.params import EmbedParams, ExtractParams

result = (
    create_ingestor(run_mode="inprocess")
    .files(["/data/docs/**/*.pdf", "/data/docs/**/*.bmp"])
    .extract(
        ExtractParams(
            page_elements_invoke_url="http://localhost:8000/v1/infer",
            ocr_invoke_url="http://localhost:8009/v1/infer",
            extract_text=True,
            extract_images=True,
            extract_tables=True,
            extract_charts=True,
        )
    )
    .embed(EmbedParams(embed_modality="text_image", embed_granularity="page"))
    .ingest()
)
```

`files()` accepts a path, a glob, or a list of paths/globs. `GraphIngestor`
stores those inputs until `ingest()` is called.

## Run Modes

`create_ingestor()` defaults to `run_mode="inprocess"`. Directly constructing
`GraphIngestor` defaults to `run_mode="batch"`.

`run_mode="inprocess"` builds the graph and runs each operator sequentially in
the current Python process over pandas DataFrames. This is useful for unit
tests, debugging, and small local runs.

`run_mode="batch"` builds the same logical graph but executes it through Ray
Data. The executor reads input files with Ray, applies the graph as
`map_batches` stages, and materializes the result to a pandas DataFrame at the
end of `ingest()`.

In batch mode, `GraphIngestor` initializes Ray when needed and forwards the
current virtual environment, Python path, Hugging Face cache settings, and
remote-auth environment to Ray workers. Resource defaults come from constructor
arguments and are refined by `BatchTuningParams`; explicit `node_overrides`
take precedence.

## Extraction Stage

Extraction is configured with `.extract()` or one of the typed extraction
shortcuts. The unified `.extract()` method accepts an `ExtractParams` object,
keyword overrides for `ExtractParams`, optional `split_config`, and the
`extraction_mode` parameter.

```python
ingestor = (
    create_ingestor(run_mode="batch")
    .files("/data/corpus/**/*")
    .extract(ExtractParams(method="pdfium"), extraction_mode="auto")
)
```

### extraction_mode

`extraction_mode` determines the first extraction graph that is built.

| Value | Behavior |
| --- | --- |
| `"auto"` | Default for `.extract()`. Uses `MultiTypeExtractOperator` and dispatches each input by file extension. This is the normal mode for mixed corpora and for direct image files such as BMP and TIFF. |
| `"pdf"` | Uses the dedicated PDF/document graph directly: document-to-PDF conversion, PDF splitting, PDF extraction, page-element detection, optional table/chart stages, and optional OCR. This is an explicit path for PDF-like document extraction. It does not dispatch direct image inputs through `ImageLoadActor`. |
| `"image"` | Forces inputs through the image pipeline. This is what `.extract_image_files()` sets. Images are converted into one-page page-schema rows before page-element detection and OCR/table/chart stages. |
| `"text"` | Forces inputs through the text splitter. This is what `.extract_txt()` sets. |
| `"html"` | Forces inputs through the HTML splitter. This is what `.extract_html()` sets. |
| `"audio"` | Forces inputs through media chunking and ASR. This is what `.extract_audio()` sets. |

In `"auto"` mode, extension dispatch is handled by `MultiTypeExtractOperator`.
Supported dispatch groups are:

| Group | Extensions | Pipeline |
| --- | --- | --- |
| PDF/document | `.pdf`, `.docx`, `.pptx` | `DocToPdfConversionActor`, `PDFSplitActor`, PDF extraction, detection, and optional OCR/table/chart stages |
| Image | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.tif`, `.svg` | `ImageLoadActor`, then detection and optional OCR/table/chart stages |
| Text | `.txt` | `TxtSplitActor` |
| HTML | `.html` | `HtmlSplitActor` |
| Audio | `.mp3`, `.wav` | `MediaChunkActor`, then `ASRActor` |
| Video | `.mp4`, `.mov`, `.mkv` | video frame extraction/OCR, optional audio ASR, and optional audio-visual fusion |

Unsupported extensions are not assigned to an extraction group in `"auto"`
mode and therefore do not produce rows.

### Why auto is the default

The library documents multiple supported file types, and `.extract()` is the
general extraction entry point. Defaulting `.extract()` to `"auto"` means the
ingestor can route each supported file type through the matching graph branch.
For direct image inputs, that branch first materializes `page_image`, allowing
page-elements, OCR, table, chart, embed, and store stages to consume the image
payload. Use `extraction_mode="pdf"` when you intentionally want to force all
inputs through the PDF/document graph instead of extension dispatch.

### Typed extraction shortcuts

The typed shortcuts configure one extraction family directly:

```python
ingestor.extract_image_files(ExtractParams(...))  # extraction_mode="image"
ingestor.extract_txt(TextChunkParams(...))        # extraction_mode="text"
ingestor.extract_html(HtmlChunkParams(...))       # extraction_mode="html"
ingestor.extract_audio(AudioChunkParams(...))     # extraction_mode="audio"
ingestor.extract_video(...)                       # extraction_mode="auto" with video params
```

Use `.extract_video()` when you need to configure video-specific behavior such
as frame extraction, frame deduplication, audio extraction, or audio-visual
fusion. It shares OCR endpoint and API-key configuration through
`ExtractParams`.

## ExtractParams

`ExtractParams` controls what content is extracted and where inference runs.
Important fields include:

| Field | Purpose |
| --- | --- |
| `method` | PDF extraction method. `"pdfium"` is the default; `"pdfium_hybrid"` and `"ocr"` can trigger OCR for text extraction; `"nemotron_parse"` uses Nemotron Parse. |
| `extract_text`, `extract_images`, `extract_tables`, `extract_charts`, `extract_infographics` | Feature flags for downstream extraction work. |
| `page_elements_invoke_url`, `ocr_invoke_url`, `table_structure_invoke_url`, `graphic_elements_invoke_url`, `nemotron_parse_invoke_url` | Remote NIM endpoints. When omitted, graph operators may resolve to local model-backed variants. |
| `api_key` and per-stage API keys | Authentication for remote endpoints. If `api_key` is omitted, `GraphIngestor` attempts to resolve it from `NVIDIA_API_KEY` or `NGC_API_KEY`. |
| `batch_tuning` | Batch sizes, worker counts, and GPU allocation hints used to derive Ray node overrides. |

Providing `graphic_elements_invoke_url` auto-enables graphic-elements usage.
Providing `table_structure_invoke_url` auto-enables table-structure usage.

## split_config

`split_config` enables post-extraction text chunking per content type. Valid
keys are `text`, `html`, `pdf`, `audio`, `image`, and `video`.

```python
ingestor.extract(
    ExtractParams(),
    split_config={
        "pdf": {"max_tokens": 512, "overlap_tokens": 64},
        "image": {"max_tokens": 512},
    },
)
```

If a key is omitted or set to `None`, chunking for that key is off. If a key is
set to `False`, it is treated as an explicit opt-out. Dict values are converted
to `TextChunkParams`, except `html`, which uses `HtmlChunkParams`.

By default, unified `.extract()` does not enable post-extraction chunking.

## Post-Extraction Stages

After extraction, the fluent API can append transform and output stages:

| Method | Stage |
| --- | --- |
| `.dedup(DedupParams(...))` | Removes image crops that overlap with table, chart, or infographic detections. |
| `.caption(CaptionParams(...))` | Captions extracted visual elements. If captioning is enabled and dedup was not explicitly configured, `GraphIngestor` inserts dedup before captioning, except for explicit image-only extraction. |
| `.store(StoreParams(...))` | Persists extracted image/text payloads and can strip base64 from rows after storage. |
| `.embed(EmbedParams(...))` | Embeds text, image, or text-image content. For `pdf`, `image`, and `auto` extraction, content is reshaped to page or element rows before embedding. |
| `.webhook(WebhookParams(...))` | Sends processed rows to a webhook endpoint. This stage is appended last when configured. |

The order in which fluent stage methods are called is preserved for `dedup`,
`caption`, `store`, and `embed`, subject to the automatic dedup insertion
described above. `webhook` runs last.

## Outputs

Extraction produces pandas rows using the page/document schema consumed by
later graph stages. Common columns include:

| Column | Meaning |
| --- | --- |
| `path` | Resolved source file path. |
| `page_number` | Page number or page-like unit number. Direct image files are represented as page 1. |
| `source_id` | Source identifier when available. |
| `text` | Extracted or OCR text. |
| `page_image` | Page image payload for PDF/image branches when materialized. Direct image files are encoded as PNG page images. |
| `images`, `tables`, `charts`, `infographics` | Extracted visual or structured elements. |
| `metadata` | Source metadata and error state. |
| `page_elements_v3`, `page_elements_v3_num_detections`, `page_elements_v3_counts_by_label` | Page-element detection output when that stage runs. |

Embedding and storage stages add their own columns, such as embedding vectors,
embedding status fields, and stored image URI fields.

## Choosing an Entry Point

Use `.extract()` with its default `extraction_mode="auto"` for general
document ingestion and mixed corpora. This is the safest default when the input
set may contain supported non-PDF formats.

Use `.extract(..., extraction_mode="pdf")` when you intentionally want the
PDF/document graph for all inputs.

Use typed shortcuts when the corpus is known to contain one file family and
you want that mode explicitly, or when you need family-specific parameters such
as `AudioChunkParams` or video frame settings.
