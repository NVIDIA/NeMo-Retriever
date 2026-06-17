# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import fields
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


_GRAPH_COMMAND_RUN_MODES: dict[str, IngestRunModeValue] = {
    "local": "inprocess",
    "batch": "batch",
}

_BATCH_ONLY_FLAGS = {
    "ray_address": "--ray-address",
    "ray_log_to_driver": "--ray-log-to-driver",
    "pdf_split_batch_size": "--pdf-split-batch-size",
    "pdf_extract_workers": "--pdf-extract-workers",
    "pdf_extract_batch_size": "--pdf-extract-batch-size",
    "pdf_extract_cpus_per_task": "--pdf-extract-cpus-per-task",
    "page_elements_workers": "--page-elements-workers",
    "page_elements_batch_size": "--page-elements-batch-size",
    "page_elements_cpus_per_actor": "--page-elements-cpus-per-actor",
    "page_elements_gpus_per_actor": "--page-elements-gpus-per-actor",
    "ocr_workers": "--ocr-workers",
    "ocr_batch_size": "--ocr-batch-size",
    "ocr_cpus_per_actor": "--ocr-cpus-per-actor",
    "ocr_gpus_per_actor": "--ocr-gpus-per-actor",
    "table_structure_workers": "--table-structure-workers",
    "table_structure_batch_size": "--table-structure-batch-size",
    "table_structure_cpus_per_actor": "--table-structure-cpus-per-actor",
    "table_structure_gpus_per_actor": "--table-structure-gpus-per-actor",
    "nemotron_parse_workers": "--nemotron-parse-workers",
    "nemotron_parse_batch_size": "--nemotron-parse-batch-size",
    "nemotron_parse_gpus_per_actor": "--nemotron-parse-gpus-per-actor",
    "embed_workers": "--embed-workers",
    "embed_batch_size": "--embed-batch-size",
    "embed_cpus_per_actor": "--embed-cpus-per-actor",
    "embed_gpus_per_actor": "--embed-gpus-per-actor",
}


def _graph_run_mode_for_command(ctx: typer.Context) -> IngestRunModeValue:
    command_name = ctx.info_name or ctx.command.name
    if command_name in _GRAPH_COMMAND_RUN_MODES:
        return _GRAPH_COMMAND_RUN_MODES[command_name]
    raise typer.BadParameter(f"Unknown graph ingest command: {command_name!r}")


def _validate_graph_ingest_mode_options(values: Mapping[str, Any], *, run_mode: IngestRunModeValue) -> None:
    if run_mode == "batch":
        return
    batch_only_flags = sorted(flag for name, flag in _BATCH_ONLY_FLAGS.items() if values.get(name) is not None)
    if batch_only_flags:
        joined_flags = ", ".join(batch_only_flags)
        raise ValueError(f"Batch-only option(s) require `retriever ingest batch`: {joined_flags}")


def _matching_option_values(values: Mapping[str, Any], options_type: type[Any]) -> dict[str, Any]:
    return {field.name: values[field.name] for field in fields(options_type) if field.name in values}


def _build_source_options(values: Mapping[str, Any]) -> IngestSourceOptions:
    return IngestSourceOptions(**_matching_option_values(values, IngestSourceOptions))


def _build_runtime_options(values: Mapping[str, Any], *, run_mode: IngestRunModeValue) -> IngestRuntimeOptions:
    return IngestRuntimeOptions(
        run_mode=run_mode,
        ray_address=values.get("ray_address"),
        ray_log_to_driver=values.get("ray_log_to_driver"),
    )


def _build_extract_batch_options(values: Mapping[str, Any], *, enabled: bool) -> IngestExtractBatchOptions:
    if not enabled:
        return IngestExtractBatchOptions()
    return IngestExtractBatchOptions(**_matching_option_values(values, IngestExtractBatchOptions))


def _build_extract_options(values: Mapping[str, Any], *, batch: IngestExtractBatchOptions) -> IngestExtractOptions:
    option_values = _matching_option_values(values, IngestExtractOptions)
    option_values["extract_api_key"] = values.get("api_key")
    option_values["batch"] = batch
    return IngestExtractOptions(**option_values)


def _build_media_options(values: Mapping[str, Any]) -> IngestMediaOptions:
    return IngestMediaOptions(**_matching_option_values(values, IngestMediaOptions))


def _build_caption_options(values: Mapping[str, Any]) -> IngestCaptionOptions:
    option_values = _matching_option_values(values, IngestCaptionOptions)
    option_values["enabled"] = values["caption"]
    option_values["caption_api_key"] = values.get("api_key")
    return IngestCaptionOptions(**option_values)


def _build_dedup_options(values: Mapping[str, Any]) -> IngestDedupOptions:
    return IngestDedupOptions(enabled=values["dedup"], iou_threshold=values.get("dedup_iou_threshold"))


def _build_chunk_options(values: Mapping[str, Any]) -> IngestChunkOptions:
    option_values = _matching_option_values(values, IngestChunkOptions)
    option_values["enabled"] = values["text_chunk"]
    return IngestChunkOptions(**option_values)


def _build_embed_batch_options(values: Mapping[str, Any], *, enabled: bool) -> IngestEmbedBatchOptions:
    if not enabled:
        return IngestEmbedBatchOptions()
    return IngestEmbedBatchOptions(**_matching_option_values(values, IngestEmbedBatchOptions))


def _build_embed_options(values: Mapping[str, Any], *, batch: IngestEmbedBatchOptions) -> IngestEmbedOptions:
    option_values = _matching_option_values(values, IngestEmbedOptions)
    option_values["embed_api_key"] = values.get("api_key")
    option_values["batch"] = batch
    return IngestEmbedOptions(**option_values)


def _build_image_store_options(values: Mapping[str, Any]) -> IngestImageStoreOptions:
    return IngestImageStoreOptions(images_uri=values.get("store_images_uri"))


def _build_storage_options(values: Mapping[str, Any]) -> IngestStorageOptions:
    return IngestStorageOptions(**_matching_option_values(values, IngestStorageOptions))


def _build_graph_ingest_request(values: Mapping[str, Any], *, run_mode: IngestRunModeValue) -> IngestPlanRequest:
    batch_enabled = run_mode == "batch"
    return IngestPlanRequest(
        source=_build_source_options(values),
        runtime=_build_runtime_options(values, run_mode=run_mode),
        extract=_build_extract_options(values, batch=_build_extract_batch_options(values, enabled=batch_enabled)),
        media=_build_media_options(values),
        caption=_build_caption_options(values),
        dedup=_build_dedup_options(values),
        chunk=_build_chunk_options(values),
        embed=_build_embed_options(values, batch=_build_embed_batch_options(values, enabled=batch_enabled)),
        image_store=_build_image_store_options(values),
        storage=_build_storage_options(values),
    )


def _run_graph_ingest_from_parsed_options(parsed_options: Mapping[str, Any], *, run_mode: IngestRunModeValue) -> None:
    def _run() -> dict[str, Any]:
        _validate_graph_ingest_mode_options(parsed_options, run_mode=run_mode)
        request = _build_graph_ingest_request(parsed_options, run_mode=run_mode)
        return run_ingest_workflow(resolve_ingest_plan(request), dry_run=parsed_options["dry_run"])

    run_cli_workflow(_run, quiet=parsed_options["quiet"])


def _graph_ingest_command(
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
    hybrid: opts.HybridOption = False,
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
    _run_graph_ingest_from_parsed_options(ctx.params, run_mode=_graph_run_mode_for_command(ctx))
