# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.adapters.cli.ingest import options as opts
from nemo_retriever.adapters.cli.ingest.shared import run_cli_workflow
from nemo_retriever.adapters.cli.ingest_workflow import run_service_ingest_workflow
from nemo_retriever.ingest.service import (
    ServiceIngestCaptionOptions,
    ServiceIngestChunkOptions,
    ServiceIngestConnectionOptions,
    ServiceIngestDedupOptions,
    ServiceIngestEmbedOptions,
    ServiceIngestExtractOptions,
    ServiceIngestImageStoreOptions,
    ServiceIngestPlanRequest,
    ServiceIngestSourceOptions,
    resolve_service_ingest_request,
)


def service_command(
    documents: opts.DocumentsArgument,
    profile: opts.ProfileOption = "auto",
    service_url: opts.ServiceUrlOption = "http://localhost:7670",
    service_concurrency: opts.ServiceConcurrencyOption = 8,
    service_api_token: opts.ServiceApiTokenOption = None,
    dry_run: opts.ServiceDryRunOption = False,
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
    ocr_version: opts.OcrVersionOption = None,
    table_output_format: opts.ServiceTableOutputFormatOption = None,
    caption: opts.ServiceCaptionOption = False,
    caption_context_text_max_chars: opts.CaptionContextTextMaxCharsOption = None,
    caption_infographics: opts.CaptionInfographicsOption = None,
    dedup: opts.DedupOption = False,
    dedup_iou_threshold: opts.DedupIouThresholdOption = None,
    store_images_uri: opts.ServiceStoreImagesUriOption = None,
    embed_modality: opts.EmbedModalityOption = None,
    embed_granularity: opts.EmbedGranularityOption = None,
    text_elements_modality: opts.TextElementsModalityOption = None,
    structured_elements_modality: opts.StructuredElementsModalityOption = None,
    text_chunk: opts.TextChunkOption = False,
    text_chunk_max_tokens: opts.TextChunkMaxTokensOption = None,
    text_chunk_overlap_tokens: opts.TextChunkOverlapTokensOption = None,
    quiet: opts.ServiceQuietOption = True,
) -> None:
    request = ServiceIngestPlanRequest(
        source=ServiceIngestSourceOptions(documents=documents, profile=profile),
        connection=ServiceIngestConnectionOptions(
            service_url=service_url,
            service_concurrency=service_concurrency,
            service_api_token=service_api_token,
        ),
        extract=ServiceIngestExtractOptions(
            method=method,
            dpi=dpi,
            extract_text=extract_text,
            extract_images=extract_images,
            extract_tables=extract_tables,
            extract_charts=extract_charts,
            extract_infographics=extract_infographics,
            extract_page_as_image=extract_page_as_image,
            use_page_elements=use_page_elements,
            use_graphic_elements=use_graphic_elements,
            use_table_structure=use_table_structure,
            table_output_format=table_output_format,
            ocr_version=ocr_version,
        ),
        dedup=ServiceIngestDedupOptions(enabled=dedup, iou_threshold=dedup_iou_threshold),
        caption=ServiceIngestCaptionOptions(
            enabled=caption,
            context_text_max_chars=caption_context_text_max_chars,
            caption_infographics=caption_infographics,
        ),
        chunk=ServiceIngestChunkOptions(
            enabled=text_chunk,
            text_chunk_max_tokens=text_chunk_max_tokens,
            text_chunk_overlap_tokens=text_chunk_overlap_tokens,
        ),
        embed=ServiceIngestEmbedOptions(
            embed_modality=embed_modality,
            text_elements_modality=text_elements_modality,
            structured_elements_modality=structured_elements_modality,
            embed_granularity=embed_granularity,
        ),
        image_store=ServiceIngestImageStoreOptions(images_uri=store_images_uri),
    )
    run_cli_workflow(
        lambda: run_service_ingest_workflow(resolve_service_ingest_request(request), dry_run=dry_run),
        quiet=quiet,
    )
