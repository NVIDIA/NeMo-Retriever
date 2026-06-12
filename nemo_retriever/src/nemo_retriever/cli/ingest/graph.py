# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Mapping

import typer

from nemo_retriever.cli.ingest import options as opts
from nemo_retriever.cli.ingest.shared import run_cli_workflow
from nemo_retriever.cli.ingest_workflow import run_ingest_workflow
from nemo_retriever.ingest.plan import (
    IngestCaptionOptions,
    IngestChunkOptions,
    IngestDedupOptions,
    IngestEmbedBatchOptions,
    IngestEmbedOptions,
    IngestExtractBatchOptions,
    IngestExtractOptions,
    IngestImageStoreOptions,
    IngestMediaOptions,
    IngestPlanRequest,
    IngestRunModeValue,
    IngestRuntimeOptions,
    IngestSourceOptions,
    IngestStorageOptions,
    resolve_ingest_plan,
)


def _run_graph_ingest_request(request: IngestPlanRequest, *, dry_run: bool, quiet: bool) -> None:
    run_cli_workflow(
        lambda: run_ingest_workflow(resolve_ingest_plan(request), dry_run=dry_run),
        quiet=quiet,
    )


def _source_options(values: Mapping[str, Any]) -> IngestSourceOptions:
    return IngestSourceOptions(documents=values["documents"], profile=values["profile"])


def _runtime_options(values: Mapping[str, Any], *, run_mode: IngestRunModeValue) -> IngestRuntimeOptions:
    return IngestRuntimeOptions(
        run_mode=run_mode,
        ray_address=values.get("ray_address"),
        ray_log_to_driver=values.get("ray_log_to_driver"),
    )


def _extract_batch_options(values: Mapping[str, Any], *, enabled: bool) -> IngestExtractBatchOptions:
    if not enabled:
        return IngestExtractBatchOptions()
    return IngestExtractBatchOptions(
        pdf_split_batch_size=values.get("pdf_split_batch_size"),
        pdf_extract_workers=values.get("pdf_extract_workers"),
        pdf_extract_batch_size=values.get("pdf_extract_batch_size"),
        pdf_extract_cpus_per_task=values.get("pdf_extract_cpus_per_task"),
        page_elements_workers=values.get("page_elements_workers"),
        page_elements_batch_size=values.get("page_elements_batch_size"),
        page_elements_cpus_per_actor=values.get("page_elements_cpus_per_actor"),
        page_elements_gpus_per_actor=values.get("page_elements_gpus_per_actor"),
        ocr_workers=values.get("ocr_workers"),
        ocr_batch_size=values.get("ocr_batch_size"),
        ocr_cpus_per_actor=values.get("ocr_cpus_per_actor"),
        ocr_gpus_per_actor=values.get("ocr_gpus_per_actor"),
        table_structure_workers=values.get("table_structure_workers"),
        table_structure_batch_size=values.get("table_structure_batch_size"),
        table_structure_cpus_per_actor=values.get("table_structure_cpus_per_actor"),
        table_structure_gpus_per_actor=values.get("table_structure_gpus_per_actor"),
        nemotron_parse_workers=values.get("nemotron_parse_workers"),
        nemotron_parse_batch_size=values.get("nemotron_parse_batch_size"),
        nemotron_parse_gpus_per_actor=values.get("nemotron_parse_gpus_per_actor"),
    )


def _extract_options(values: Mapping[str, Any], *, batch: IngestExtractBatchOptions) -> IngestExtractOptions:
    return IngestExtractOptions(
        method=values.get("method"),
        dpi=values.get("dpi"),
        extract_text=values.get("extract_text"),
        extract_images=values.get("extract_images"),
        extract_tables=values.get("extract_tables"),
        extract_charts=values.get("extract_charts"),
        extract_infographics=values.get("extract_infographics"),
        extract_page_as_image=values.get("extract_page_as_image"),
        use_page_elements=values.get("use_page_elements"),
        use_graphic_elements=values.get("use_graphic_elements"),
        use_table_structure=values.get("use_table_structure"),
        page_elements_invoke_url=values.get("page_elements_invoke_url"),
        ocr_invoke_url=values.get("ocr_invoke_url"),
        ocr_version=values.get("ocr_version"),
        ocr_lang=values.get("ocr_lang"),
        graphic_elements_invoke_url=values.get("graphic_elements_invoke_url"),
        table_structure_invoke_url=values.get("table_structure_invoke_url"),
        table_output_format=values.get("table_output_format"),
        extract_api_key=values.get("api_key"),
        batch=batch,
    )


def _media_options(values: Mapping[str, Any]) -> IngestMediaOptions:
    return IngestMediaOptions(
        segment_audio=values.get("segment_audio"),
        audio_split_type=values["audio_split_type"],
        audio_split_interval=values.get("audio_split_interval"),
        video_extract_audio=values.get("video_extract_audio"),
        video_extract_frames=values.get("video_extract_frames"),
        video_frame_fps=values.get("video_frame_fps"),
        video_frame_dedup=values.get("video_frame_dedup"),
        video_frame_text_dedup=values.get("video_frame_text_dedup"),
        video_frame_text_dedup_max_dropped_frames=values.get("video_frame_text_dedup_max_dropped_frames"),
        video_av_fuse=values.get("video_av_fuse"),
    )


def _caption_options(values: Mapping[str, Any]) -> IngestCaptionOptions:
    return IngestCaptionOptions(
        enabled=values["caption"],
        caption_invoke_url=values.get("caption_invoke_url"),
        caption_api_key=values.get("api_key"),
        caption_model_name=values.get("caption_model_name"),
        caption_context_text_max_chars=values.get("caption_context_text_max_chars"),
        caption_infographics=values.get("caption_infographics"),
    )


def _dedup_options(values: Mapping[str, Any]) -> IngestDedupOptions:
    return IngestDedupOptions(enabled=values["dedup"], iou_threshold=values.get("dedup_iou_threshold"))


def _chunk_options(values: Mapping[str, Any]) -> IngestChunkOptions:
    return IngestChunkOptions(
        enabled=values["text_chunk"],
        text_chunk_max_tokens=values.get("text_chunk_max_tokens"),
        text_chunk_overlap_tokens=values.get("text_chunk_overlap_tokens"),
    )


def _embed_batch_options(values: Mapping[str, Any], *, enabled: bool) -> IngestEmbedBatchOptions:
    if not enabled:
        return IngestEmbedBatchOptions()
    return IngestEmbedBatchOptions(
        embed_workers=values.get("embed_workers"),
        embed_batch_size=values.get("embed_batch_size"),
        embed_cpus_per_actor=values.get("embed_cpus_per_actor"),
        embed_gpus_per_actor=values.get("embed_gpus_per_actor"),
    )


def _embed_options(values: Mapping[str, Any], *, batch: IngestEmbedBatchOptions) -> IngestEmbedOptions:
    return IngestEmbedOptions(
        embed_invoke_url=values.get("embed_invoke_url"),
        embed_model_name=values.get("embed_model_name"),
        local_ingest_embed_backend=values.get("local_ingest_embed_backend"),
        embed_api_key=values.get("api_key"),
        embed_modality=values.get("embed_modality"),
        embed_granularity=values.get("embed_granularity"),
        text_elements_modality=values.get("text_elements_modality"),
        structured_elements_modality=values.get("structured_elements_modality"),
        batch=batch,
    )


def _image_store_options(values: Mapping[str, Any]) -> IngestImageStoreOptions:
    return IngestImageStoreOptions(images_uri=values.get("store_images_uri"))


def _storage_options(values: Mapping[str, Any]) -> IngestStorageOptions:
    return IngestStorageOptions(
        lancedb_uri=values["lancedb_uri"],
        table_name=values["table_name"],
        overwrite=values["overwrite"],
    )


def _graph_ingest_request(values: Mapping[str, Any], *, run_mode: IngestRunModeValue) -> IngestPlanRequest:
    batch_enabled = run_mode == "batch"
    extract_batch = _extract_batch_options(values, enabled=batch_enabled)
    embed_batch = _embed_batch_options(values, enabled=batch_enabled)

    return IngestPlanRequest(
        source=_source_options(values),
        runtime=_runtime_options(values, run_mode=run_mode),
        extract=_extract_options(values, batch=extract_batch),
        media=_media_options(values),
        caption=_caption_options(values),
        dedup=_dedup_options(values),
        chunk=_chunk_options(values),
        embed=_embed_options(values, batch=embed_batch),
        image_store=_image_store_options(values),
        storage=_storage_options(values),
    )


def _run_graph_command(ctx: typer.Context, *, run_mode: IngestRunModeValue) -> None:
    request = _graph_ingest_request(ctx.params, run_mode=run_mode)
    _run_graph_ingest_request(request, dry_run=ctx.params["dry_run"], quiet=ctx.params["quiet"])


def _local_command(
    ctx: typer.Context,
    documents: opts.DocumentsArgument,
    profile: opts.ProfileOption = "auto",
    lancedb_uri: opts.LanceDbUriOption = "lancedb",
    table_name: opts.TableNameOption = "nemo-retriever",
    dry_run: opts.DryRunOption = False,
    method: opts.MethodOption = None,
    dpi: opts.DpiOption = None,
    extract_text: opts.ExtractTextOption = None,
    extract_images: opts.ExtractImagesOption = None,
    extract_tables: opts.ExtractTablesOption = None,
    extract_charts: opts.ExtractChartsOption = None,
    extract_infographics: opts.ExtractInfographicsOption = None,
    extract_page_as_image: opts.ExtractPageAsImageOption = None,
    use_page_elements: opts.UsePageElementsOption = None,
    use_graphic_elements: opts.UseGraphicElementsOption = None,
    use_table_structure: opts.UseTableStructureOption = None,
    segment_audio: opts.SegmentAudioOption = None,
    audio_split_type: opts.AudioSplitTypeOption = "size",
    audio_split_interval: opts.AudioSplitIntervalOption = None,
    video_extract_audio: opts.VideoExtractAudioOption = None,
    video_extract_frames: opts.VideoExtractFramesOption = None,
    video_frame_fps: opts.VideoFrameFpsOption = None,
    video_frame_dedup: opts.VideoFrameDedupOption = None,
    video_frame_text_dedup: opts.VideoFrameTextDedupOption = None,
    video_frame_text_dedup_max_dropped_frames: opts.VideoFrameTextDedupMaxDroppedFramesOption = None,
    video_av_fuse: opts.VideoAvFuseOption = None,
    caption: opts.CaptionOption = False,
    caption_invoke_url: opts.CaptionInvokeUrlOption = None,
    api_key: opts.ApiKeyOption = None,
    caption_model_name: opts.CaptionModelNameOption = None,
    caption_context_text_max_chars: opts.CaptionContextTextMaxCharsOption = None,
    caption_infographics: opts.CaptionInfographicsOption = None,
    dedup: opts.DedupOption = False,
    dedup_iou_threshold: opts.DedupIouThresholdOption = None,
    store_images_uri: opts.StoreImagesUriOption = None,
    overwrite: opts.OverwriteOption = True,
    page_elements_invoke_url: opts.PageElementsInvokeUrlOption = None,
    ocr_invoke_url: opts.OcrInvokeUrlOption = None,
    ocr_version: opts.OcrVersionOption = None,
    ocr_lang: opts.OcrLangOption = None,
    graphic_elements_invoke_url: opts.GraphicElementsInvokeUrlOption = None,
    table_structure_invoke_url: opts.TableStructureInvokeUrlOption = None,
    table_output_format: opts.TableOutputFormatOption = None,
    embed_invoke_url: opts.EmbedInvokeUrlOption = None,
    embed_model_name: opts.EmbedModelNameOption = None,
    local_ingest_embed_backend: opts.LocalIngestEmbedBackendOption = None,
    embed_modality: opts.EmbedModalityOption = None,
    embed_granularity: opts.EmbedGranularityOption = None,
    text_elements_modality: opts.TextElementsModalityOption = None,
    structured_elements_modality: opts.StructuredElementsModalityOption = None,
    text_chunk: opts.TextChunkOption = False,
    text_chunk_max_tokens: opts.TextChunkMaxTokensOption = None,
    text_chunk_overlap_tokens: opts.TextChunkOverlapTokensOption = None,
    quiet: opts.QuietOption = True,
) -> None:
    _run_graph_command(ctx, run_mode="inprocess")


def _batch_command(
    ctx: typer.Context,
    documents: opts.DocumentsArgument,
    profile: opts.ProfileOption = "auto",
    lancedb_uri: opts.LanceDbUriOption = "lancedb",
    table_name: opts.TableNameOption = "nemo-retriever",
    dry_run: opts.DryRunOption = False,
    method: opts.MethodOption = None,
    dpi: opts.DpiOption = None,
    extract_text: opts.ExtractTextOption = None,
    extract_images: opts.ExtractImagesOption = None,
    extract_tables: opts.ExtractTablesOption = None,
    extract_charts: opts.ExtractChartsOption = None,
    extract_infographics: opts.ExtractInfographicsOption = None,
    extract_page_as_image: opts.ExtractPageAsImageOption = None,
    use_page_elements: opts.UsePageElementsOption = None,
    use_graphic_elements: opts.UseGraphicElementsOption = None,
    use_table_structure: opts.UseTableStructureOption = None,
    segment_audio: opts.SegmentAudioOption = None,
    audio_split_type: opts.AudioSplitTypeOption = "size",
    audio_split_interval: opts.AudioSplitIntervalOption = None,
    video_extract_audio: opts.VideoExtractAudioOption = None,
    video_extract_frames: opts.VideoExtractFramesOption = None,
    video_frame_fps: opts.VideoFrameFpsOption = None,
    video_frame_dedup: opts.VideoFrameDedupOption = None,
    video_frame_text_dedup: opts.VideoFrameTextDedupOption = None,
    video_frame_text_dedup_max_dropped_frames: opts.VideoFrameTextDedupMaxDroppedFramesOption = None,
    video_av_fuse: opts.VideoAvFuseOption = None,
    caption: opts.CaptionOption = False,
    caption_invoke_url: opts.CaptionInvokeUrlOption = None,
    api_key: opts.ApiKeyOption = None,
    caption_model_name: opts.CaptionModelNameOption = None,
    caption_context_text_max_chars: opts.CaptionContextTextMaxCharsOption = None,
    caption_infographics: opts.CaptionInfographicsOption = None,
    dedup: opts.DedupOption = False,
    dedup_iou_threshold: opts.DedupIouThresholdOption = None,
    store_images_uri: opts.StoreImagesUriOption = None,
    overwrite: opts.OverwriteOption = True,
    ray_address: opts.RayAddressOption = None,
    ray_log_to_driver: opts.RayLogToDriverOption = None,
    page_elements_invoke_url: opts.PageElementsInvokeUrlOption = None,
    ocr_invoke_url: opts.OcrInvokeUrlOption = None,
    ocr_version: opts.OcrVersionOption = None,
    ocr_lang: opts.OcrLangOption = None,
    graphic_elements_invoke_url: opts.GraphicElementsInvokeUrlOption = None,
    table_structure_invoke_url: opts.TableStructureInvokeUrlOption = None,
    table_output_format: opts.TableOutputFormatOption = None,
    embed_invoke_url: opts.EmbedInvokeUrlOption = None,
    embed_model_name: opts.EmbedModelNameOption = None,
    local_ingest_embed_backend: opts.LocalIngestEmbedBackendOption = None,
    embed_modality: opts.EmbedModalityOption = None,
    embed_granularity: opts.EmbedGranularityOption = None,
    text_elements_modality: opts.TextElementsModalityOption = None,
    structured_elements_modality: opts.StructuredElementsModalityOption = None,
    text_chunk: opts.TextChunkOption = False,
    text_chunk_max_tokens: opts.TextChunkMaxTokensOption = None,
    text_chunk_overlap_tokens: opts.TextChunkOverlapTokensOption = None,
    pdf_split_batch_size: opts.PdfSplitBatchSizeOption = None,
    pdf_extract_workers: opts.PdfExtractWorkersOption = None,
    pdf_extract_batch_size: opts.PdfExtractBatchSizeOption = None,
    pdf_extract_cpus_per_task: opts.PdfExtractCpusPerTaskOption = None,
    page_elements_workers: opts.PageElementsWorkersOption = None,
    page_elements_batch_size: opts.PageElementsBatchSizeOption = None,
    page_elements_cpus_per_actor: opts.PageElementsCpusPerActorOption = None,
    page_elements_gpus_per_actor: opts.PageElementsGpusPerActorOption = None,
    ocr_workers: opts.OcrWorkersOption = None,
    ocr_batch_size: opts.OcrBatchSizeOption = None,
    ocr_cpus_per_actor: opts.OcrCpusPerActorOption = None,
    ocr_gpus_per_actor: opts.OcrGpusPerActorOption = None,
    table_structure_workers: opts.TableStructureWorkersOption = None,
    table_structure_batch_size: opts.TableStructureBatchSizeOption = None,
    table_structure_cpus_per_actor: opts.TableStructureCpusPerActorOption = None,
    table_structure_gpus_per_actor: opts.TableStructureGpusPerActorOption = None,
    nemotron_parse_workers: opts.NemotronParseWorkersOption = None,
    nemotron_parse_batch_size: opts.NemotronParseBatchSizeOption = None,
    nemotron_parse_gpus_per_actor: opts.NemotronParseGpusPerActorOption = None,
    embed_workers: opts.EmbedWorkersOption = None,
    embed_batch_size: opts.EmbedBatchSizeOption = None,
    embed_cpus_per_actor: opts.EmbedCpusPerActorOption = None,
    embed_gpus_per_actor: opts.EmbedGpusPerActorOption = None,
    quiet: opts.QuietOption = True,
) -> None:
    _run_graph_command(ctx, run_mode="batch")
