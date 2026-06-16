# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated

import typer

from nemo_retriever.ingest.plan import (
    AudioSplitTypeValue,
    IngestProfileValue,
    LocalIngestEmbedBackendValue,
    OcrLangValue,
    OcrVersionValue,
    TableOutputFormatValue,
)

DocumentsArgument = Annotated[
    list[str],
    typer.Argument(help="One or more files, directories, or globs. Supported file types are detected automatically."),
]
ProfileOption = Annotated[
    IngestProfileValue,
    typer.Option(
        "--profile",
        help="Ingest profile: auto or fast-text. Use fast-text for large text-only PDF fallback.",
    ),
]
LanceDbUriOption = Annotated[
    str,
    typer.Option("--lancedb-uri", help="LanceDB database URI."),
]
TableNameOption = Annotated[
    str,
    typer.Option("--table-name", help="LanceDB table name."),
]
DryRunOption = Annotated[
    bool,
    typer.Option("--dry-run", help="Print the resolved ingest plan as JSON without creating an ingestor."),
]
MethodOption = Annotated[str | None, typer.Option("--method", help="PDF text extraction method.")]
DpiOption = Annotated[int | None, typer.Option("--dpi", min=72, help="Render DPI for PDF page images.")]
ExtractTextOption = Annotated[
    bool | None,
    typer.Option("--extract-text/--no-extract-text", help="Enable or disable PDF text extraction."),
]
ExtractImagesOption = Annotated[
    bool | None,
    typer.Option("--extract-images/--no-extract-images", help="Enable or disable PDF image extraction."),
]
ExtractTablesOption = Annotated[
    bool | None,
    typer.Option("--extract-tables/--no-extract-tables", help="Enable or disable PDF table extraction."),
]
ExtractChartsOption = Annotated[
    bool | None,
    typer.Option("--extract-charts/--no-extract-charts", help="Enable or disable PDF chart extraction."),
]
ExtractInfographicsOption = Annotated[
    bool | None,
    typer.Option(
        "--extract-infographics/--no-extract-infographics", help="Enable or disable PDF infographic extraction."
    ),
]
ExtractPageAsImageOption = Annotated[
    bool | None,
    typer.Option(
        "--extract-page-as-image/--no-extract-page-as-image", help="Enable or disable full-page image extraction."
    ),
]
UsePageElementsOption = Annotated[
    bool | None,
    typer.Option(
        "--use-page-elements/--no-use-page-elements",
        help="Enable or disable page-element detection for OCR/table/chart extraction.",
    ),
]
UseGraphicElementsOption = Annotated[
    bool | None,
    typer.Option(
        "--use-graphic-elements/--no-use-graphic-elements",
        help="Enable or disable graphic-element extraction for chart/infographic rows.",
    ),
]
UseTableStructureOption = Annotated[
    bool | None,
    typer.Option(
        "--use-table-structure/--no-use-table-structure",
        help="Enable or disable table-structure extraction for markdown table text.",
    ),
]
SegmentAudioOption = Annotated[
    bool | None,
    typer.Option("--segment-audio/--no-segment-audio", help="Enable or disable ASR-side audio segmentation."),
]
AudioSplitTypeOption = Annotated[
    AudioSplitTypeValue,
    typer.Option("--audio-split-type", help="Audio/video audio split type: size, time, or frame."),
]
AudioSplitIntervalOption = Annotated[
    int | None,
    typer.Option("--audio-split-interval", min=1, help="Audio/video audio split interval."),
]
VideoExtractAudioOption = Annotated[
    bool | None,
    typer.Option(
        "--video-extract-audio/--no-video-extract-audio", help="Enable or disable audio extraction from video."
    ),
]
VideoExtractFramesOption = Annotated[
    bool | None,
    typer.Option("--video-extract-frames/--no-video-extract-frames", help="Enable or disable video frame extraction."),
]
VideoFrameFpsOption = Annotated[
    float | None,
    typer.Option("--video-frame-fps", min=0.001, help="Video frame extraction frames per second."),
]
VideoFrameDedupOption = Annotated[
    bool | None,
    typer.Option(
        "--video-frame-dedup/--no-video-frame-dedup", help="Enable or disable perceptual video frame deduplication."
    ),
]
VideoFrameTextDedupOption = Annotated[
    bool | None,
    typer.Option(
        "--video-frame-text-dedup/--no-video-frame-text-dedup",
        help="Enable or disable OCR-text deduplication across adjacent video frames.",
    ),
]
VideoFrameTextDedupMaxDroppedFramesOption = Annotated[
    int | None,
    typer.Option(
        "--video-frame-text-dedup-max-dropped-frames",
        min=0,
        help="Maximum dropped frames bridged by video frame text deduplication.",
    ),
]
VideoAvFuseOption = Annotated[
    bool | None,
    typer.Option("--video-av-fuse/--no-video-av-fuse", help="Enable or disable audio/visual fusion rows for video."),
]
CaptionOption = Annotated[
    bool,
    typer.Option("--caption", help="Add an optional VLM captioning stage after extraction."),
]
CaptionInvokeUrlOption = Annotated[
    str | None,
    typer.Option(
        "--caption-invoke-url",
        help=(
            "VLM caption endpoint URL. If omitted with --caption, GPU hosts use local captioning; "
            "CPU-only runs use the hosted default endpoint with NVIDIA_API_KEY/NGC_API_KEY."
        ),
    ),
]
ApiKeyOption = Annotated[
    str | None,
    typer.Option("--api-key", help="Bearer token for remote extract, embed, and caption endpoints."),
]
CaptionModelNameOption = Annotated[
    str | None,
    typer.Option("--caption-model-name", help="Optional VLM caption model name override."),
]
CaptionContextTextMaxCharsOption = Annotated[
    int | None,
    typer.Option(
        "--caption-context-text-max-chars",
        min=0,
        help="Maximum nearby extracted text characters to include in caption prompts.",
    ),
]
CaptionInfographicsOption = Annotated[
    bool | None,
    typer.Option(
        "--caption-infographics/--no-caption-infographics",
        help="Caption infographic crops in addition to extracted images.",
    ),
]
DedupOption = Annotated[
    bool,
    typer.Option("--dedup/--no-dedup", help="Add a deduplication stage before optional captioning and embedding."),
]
DedupIouThresholdOption = Annotated[
    float | None,
    typer.Option(
        "--dedup-iou-threshold",
        min=0.0,
        max=1.0,
        help="Image bounding-box IoU threshold for --dedup. Defaults to 0.45.",
    ),
]
StoreImagesUriOption = Annotated[
    str | None,
    typer.Option("--store-images-uri", help="Store extracted images at this local path or fsspec-compatible URI."),
]
OverwriteOption = Annotated[
    bool,
    typer.Option(
        "--overwrite/--append",
        help=(
            "Overwrite the target LanceDB table by default. Use --append to add rows to an existing "
            "table without duplicate checks; rerunning the same inputs in append mode creates duplicates."
        ),
    ),
]
HybridOption = Annotated[
    bool,
    typer.Option(
        "--hybrid/--no-hybrid",
        help=(
            "Also build a full-text (BM25) index over ingested text so query --hybrid can fuse "
            "lexical and vector retrieval. Disabled by default."
        ),
    ),
]
RayAddressOption = Annotated[
    str | None, typer.Option("--ray-address", help="Batch mode only. Ray address for batch ingest.")
]
RayLogToDriverOption = Annotated[
    bool | None,
    typer.Option(
        "--ray-log-to-driver/--no-ray-log-to-driver", help="Batch mode only. Forward Ray worker logs to the driver."
    ),
]
PageElementsInvokeUrlOption = Annotated[
    str | None,
    typer.Option("--page-elements-invoke-url", help="Page-elements NIM endpoint URL."),
]
OcrInvokeUrlOption = Annotated[str | None, typer.Option("--ocr-invoke-url", help="OCR NIM endpoint URL.")]
OcrVersionOption = Annotated[
    OcrVersionValue | None,
    typer.Option("--ocr-version", help="OCR engine version for extraction."),
]
OcrLangOption = Annotated[
    OcrLangValue | None,
    typer.Option("--ocr-lang", help="OCR v2 language selector for local extraction."),
]
GraphicElementsInvokeUrlOption = Annotated[
    str | None,
    typer.Option("--graphic-elements-invoke-url", help="Graphic-elements NIM endpoint URL."),
]
TableStructureInvokeUrlOption = Annotated[
    str | None,
    typer.Option("--table-structure-invoke-url", help="Table-structure NIM endpoint URL."),
]
TableOutputFormatOption = Annotated[
    TableOutputFormatValue | None,
    typer.Option(
        "--table-output-format", help="Table text format. 'markdown' enables local table-structure extraction."
    ),
]
EmbedInvokeUrlOption = Annotated[str | None, typer.Option("--embed-invoke-url", help="Embedding NIM endpoint URL.")]
EmbedModelNameOption = Annotated[
    str | None,
    typer.Option("--embed-model-name", help="Optional embedding model name override."),
]
LocalIngestEmbedBackendOption = Annotated[
    LocalIngestEmbedBackendValue | None,
    typer.Option(
        "--local-ingest-embed-backend", help="Local ingest-time text embedder when --embed-invoke-url is unset."
    ),
]
EmbedModalityOption = Annotated[
    str | None,
    typer.Option("--embed-modality", help="Embedding modality for emitted rows: text, image, or text_image."),
]
EmbedGranularityOption = Annotated[
    str | None,
    typer.Option("--embed-granularity", help="Embedding granularity: element or page."),
]
TextElementsModalityOption = Annotated[
    str | None,
    typer.Option("--text-elements-modality", help="Embedding modality override for page text elements."),
]
StructuredElementsModalityOption = Annotated[
    str | None,
    typer.Option(
        "--structured-elements-modality", help="Embedding modality override for table/chart/infographic elements."
    ),
]
TextChunkOption = Annotated[
    bool,
    typer.Option(
        "--text-chunk", help="Enable token chunking during extraction. Defaults to 1024 tokens with 150-token overlap."
    ),
]
TextChunkMaxTokensOption = Annotated[
    int | None,
    typer.Option("--text-chunk-max-tokens", min=1, help="Maximum tokens per text chunk. Implies --text-chunk."),
]
TextChunkOverlapTokensOption = Annotated[
    int | None,
    typer.Option(
        "--text-chunk-overlap-tokens", min=0, help="Token overlap between adjacent text chunks. Implies --text-chunk."
    ),
]
PdfSplitBatchSizeOption = Annotated[
    int | None,
    typer.Option("--pdf-split-batch-size", min=1, help="Batch mode only. PDF split batch size."),
]
PdfExtractWorkersOption = Annotated[
    int | None,
    typer.Option("--pdf-extract-workers", min=1, help="Batch mode only. Maximum Ray tasks for PDF extraction."),
]
PdfExtractBatchSizeOption = Annotated[
    int | None,
    typer.Option("--pdf-extract-batch-size", min=1, help="Batch mode only. PDF extraction batch size per Ray task."),
]
PdfExtractCpusPerTaskOption = Annotated[
    float | None,
    typer.Option(
        "--pdf-extract-cpus-per-task", min=0.0, help="Batch mode only. CPUs reserved per PDF extraction Ray task."
    ),
]
PageElementsWorkersOption = Annotated[
    int | None,
    typer.Option(
        "--page-elements-workers", min=1, help="Batch mode only. Number of Ray actors for page-element detection."
    ),
]
PageElementsBatchSizeOption = Annotated[
    int | None,
    typer.Option(
        "--page-elements-batch-size", min=1, help="Batch mode only. Page-element detection batch size per actor."
    ),
]
PageElementsCpusPerActorOption = Annotated[
    float | None,
    typer.Option(
        "--page-elements-cpus-per-actor",
        min=0.0,
        help="Batch mode only. CPUs reserved per page-element detection actor.",
    ),
]
PageElementsGpusPerActorOption = Annotated[
    float | None,
    typer.Option(
        "--page-elements-gpus-per-actor",
        min=0.0,
        help="Batch mode only. GPUs reserved per local page-element detection actor.",
    ),
]
OcrWorkersOption = Annotated[
    int | None,
    typer.Option("--ocr-workers", min=1, help="Batch mode only. Number of Ray actors for OCR inference."),
]
OcrBatchSizeOption = Annotated[
    int | None,
    typer.Option("--ocr-batch-size", min=1, help="Batch mode only. OCR inference batch size per actor."),
]
OcrCpusPerActorOption = Annotated[
    float | None,
    typer.Option("--ocr-cpus-per-actor", min=0.0, help="Batch mode only. CPUs reserved per OCR actor."),
]
OcrGpusPerActorOption = Annotated[
    float | None,
    typer.Option("--ocr-gpus-per-actor", min=0.0, help="Batch mode only. GPUs reserved per local OCR actor."),
]
TableStructureWorkersOption = Annotated[
    int | None,
    typer.Option(
        "--table-structure-workers", min=1, help="Batch mode only. Number of Ray actors for table-structure extraction."
    ),
]
TableStructureBatchSizeOption = Annotated[
    int | None,
    typer.Option(
        "--table-structure-batch-size", min=1, help="Batch mode only. Table-structure extraction batch size per actor."
    ),
]
TableStructureCpusPerActorOption = Annotated[
    float | None,
    typer.Option(
        "--table-structure-cpus-per-actor", min=0.0, help="Batch mode only. CPUs reserved per table-structure actor."
    ),
]
TableStructureGpusPerActorOption = Annotated[
    float | None,
    typer.Option(
        "--table-structure-gpus-per-actor",
        min=0.0,
        help="Batch mode only. GPUs reserved per local table-structure actor.",
    ),
]
NemotronParseWorkersOption = Annotated[
    int | None,
    typer.Option("--nemotron-parse-workers", min=1, help="Batch mode only. Number of Ray actors for Nemotron Parse."),
]
NemotronParseBatchSizeOption = Annotated[
    int | None,
    typer.Option("--nemotron-parse-batch-size", min=1, help="Batch mode only. Nemotron Parse batch size per actor."),
]
NemotronParseGpusPerActorOption = Annotated[
    float | None,
    typer.Option(
        "--nemotron-parse-gpus-per-actor",
        min=0.0,
        help="Batch mode only. GPUs reserved per local Nemotron Parse actor.",
    ),
]
EmbedWorkersOption = Annotated[
    int | None,
    typer.Option("--embed-workers", min=1, help="Batch mode only. Number of Ray actors for embedding."),
]
EmbedBatchSizeOption = Annotated[
    int | None,
    typer.Option("--embed-batch-size", min=1, help="Batch mode only. Embedding batch size per actor."),
]
EmbedCpusPerActorOption = Annotated[
    float | None,
    typer.Option("--embed-cpus-per-actor", min=0.0, help="Batch mode only. CPUs reserved per embedding actor."),
]
EmbedGpusPerActorOption = Annotated[
    float | None,
    typer.Option("--embed-gpus-per-actor", min=0.0, help="Batch mode only. GPUs reserved per local embedding actor."),
]
QuietOption = Annotated[
    bool,
    typer.Option(
        "--quiet/--no-quiet",
        help=(
            "Suppress verbose progress output (progress bars, HuggingFace "
            "downloads, vLLM init logs). On success, prints only the final "
            "summary line. On error, flushes all captured output to stderr "
            "for debugging. Enabled by default; pass --no-quiet for the full "
            "verbose output."
        ),
    ),
]
