# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for graph-backed ingestor implementations."""

from __future__ import annotations

from functools import partial
from typing import cast
from typing import Any

from nemo_retriever.caption.caption import CaptionActor
from nemo_retriever.audio import ASRActor
from nemo_retriever.audio import MediaChunkActor
from nemo_retriever.dedup.dedup import dedup_images
from nemo_retriever.graph import Graph, StoreOperator, UDFOperator
from nemo_retriever.graph.actor_selection import (
    embed_actor_class,
    graphic_elements_actor_class,
    nemotron_parse_actor_class,
    ocr_actor_class,
    page_elements_actor_class,
    table_structure_actor_class,
)
from nemo_retriever.graph.content_transforms import (
    _CONTENT_COLUMNS,
    collapse_content_to_page_rows,
    explode_content_to_rows,
)
from nemo_retriever.graph.multi_type_extract_operator import MultiTypeExtractOperator
from nemo_retriever.text_embed.operators import _BatchEmbedActor
from nemo_retriever.ocr.ocr import OCRActor
from nemo_retriever.parse.nemotron_parse import NemotronParseActor
from nemo_retriever.page_elements.page_elements import PageElementDetectionActor
from nemo_retriever.pdf.extract import PDFExtractionActor
from nemo_retriever.pdf.split import PDFSplitActor
from nemo_retriever.txt.ray_data import TextChunkActor
from nemo_retriever.utils.convert.to_pdf import DocToPdfConversionActor
from nemo_retriever.ingest_plans import IngestExecutionPlan
from nemo_retriever.utils.ray_resource_hueristics import (
    ClusterResources,
    resolve_requested_plan,
)


def _batch_tuning(params: Any) -> Any:
    return getattr(params, "batch_tuning", None)


def _positive(value: Any) -> Any:
    return value if value not in (None, 0, 0.0, "", False) else None


def batch_tuning_to_node_overrides(
    extract_params: Any | None,
    embed_params: Any | None,
    cluster_resources: ClusterResources | None = None,
) -> dict[str, dict[str, Any]]:
    """Translate BatchTuningParams from extract/embed params into RayDataExecutor node_overrides.

    Explicit (non-zero) values from BatchTuningParams always win.  When a field
    is absent or zero, the heuristic default from ``resolve_requested_plan`` is
    used instead — provided ``cluster_resources`` is supplied (i.e. Ray is
    already initialised).  Without ``cluster_resources`` only explicit values
    are emitted, matching the previous behaviour.

    PDF extract concurrency is capped so that it cannot exhaust the cluster CPU
    budget when all other persistent actors are running simultaneously.
    """
    plan = resolve_requested_plan(cluster_resources=cluster_resources) if cluster_resources is not None else None

    overrides: dict[str, dict[str, Any]] = {}

    def _resolve(explicit: Any, fallback: Any = None) -> Any:
        v = _positive(explicit)
        if v is None and fallback is not None:
            v = fallback
        return v

    def _set(node_name: str, key: str, explicit: Any, fallback: Any = None) -> None:
        v = _resolve(explicit, fallback)
        if v is not None:
            overrides.setdefault(node_name, {})[key] = v

    embed_tuning = _batch_tuning(embed_params)
    embed_concurrency: int = 0
    embed_cpus: float = 1.0
    if embed_params is not None:
        embed_invoke_url = _positive(getattr(embed_params, "embed_invoke_url", None))
        explicit_bs = getattr(embed_tuning, "embed_batch_size", None) if embed_tuning is not None else None
        embed_bs = _positive(explicit_bs) or (plan.embed_batch_size if plan else None)
        _set(_BatchEmbedActor.__name__, "batch_size", embed_bs)
        if embed_bs:
            overrides.setdefault(_BatchEmbedActor.__name__, {})["target_num_rows_per_block"] = embed_bs
        embed_concurrency = (
            _resolve(
                getattr(embed_tuning, "embed_workers", None) if embed_tuning is not None else None,
                plan.embed_initial_actors if plan else None,
            )
            or 0
        )
        _set(_BatchEmbedActor.__name__, "concurrency", embed_concurrency or None)
        embed_cpus = (
            _resolve(
                getattr(embed_tuning, "embed_cpus_per_actor", None) if embed_tuning is not None else None,
            )
            or 1.0
        )
        _set(_BatchEmbedActor.__name__, "num_cpus", embed_cpus if embed_cpus != 1.0 else None)
        if not embed_invoke_url:
            _set(
                _BatchEmbedActor.__name__,
                "num_gpus",
                getattr(embed_tuning, "gpu_embed", None) if embed_tuning is not None else None,
                plan.embed_gpus_per_actor if plan else None,
            )

    extract_tuning = _batch_tuning(extract_params)
    ocr_concurrency: int = 0
    ocr_cpus: float = 1.0
    page_elements_concurrency: int = 0
    page_elements_cpus: float = 1.0
    if extract_params is not None:
        ocr_invoke_url = _positive(getattr(extract_params, "ocr_invoke_url", None))
        page_elements_invoke_url = _positive(getattr(extract_params, "page_elements_invoke_url", None))

        ocr_bs = _positive(
            getattr(extract_tuning, "ocr_inference_batch_size", None) if extract_tuning is not None else None
        ) or (plan.ocr_batch_size if plan else None)
        _set(OCRActor.__name__, "batch_size", ocr_bs)
        ocr_concurrency = (
            _resolve(
                getattr(extract_tuning, "ocr_workers", None) if extract_tuning is not None else None,
                plan.ocr_initial_actors if plan else None,
            )
            or 0
        )
        _set(OCRActor.__name__, "concurrency", ocr_concurrency or None)
        ocr_cpus = (
            _resolve(
                getattr(extract_tuning, "ocr_cpus_per_actor", None) if extract_tuning is not None else None,
            )
            or 1.0
        )
        _set(OCRActor.__name__, "num_cpus", ocr_cpus if ocr_cpus != 1.0 else None)
        if not ocr_invoke_url:
            _set(
                OCRActor.__name__,
                "num_gpus",
                getattr(extract_tuning, "gpu_ocr", None) if extract_tuning is not None else None,
                plan.ocr_gpus_per_actor if plan else None,
            )

        pe_bs = _positive(
            getattr(extract_tuning, "page_elements_batch_size", None) if extract_tuning is not None else None
        ) or (plan.page_elements_batch_size if plan else None)
        _set(PageElementDetectionActor.__name__, "batch_size", pe_bs)
        if pe_bs:
            overrides.setdefault(PageElementDetectionActor.__name__, {})["target_num_rows_per_block"] = pe_bs
        page_elements_concurrency = (
            _resolve(
                getattr(extract_tuning, "page_elements_workers", None) if extract_tuning is not None else None,
                plan.page_elements_initial_actors if plan else None,
            )
            or 0
        )
        _set(PageElementDetectionActor.__name__, "concurrency", page_elements_concurrency or None)
        page_elements_cpus = (
            _resolve(
                getattr(extract_tuning, "page_elements_cpus_per_actor", None) if extract_tuning is not None else None,
            )
            or 1.0
        )
        _set(PageElementDetectionActor.__name__, "num_cpus", page_elements_cpus if page_elements_cpus != 1.0 else None)
        if not page_elements_invoke_url:
            _set(
                PageElementDetectionActor.__name__,
                "num_gpus",
                getattr(extract_tuning, "gpu_page_elements", None) if extract_tuning is not None else None,
                plan.page_elements_gpus_per_actor if plan else None,
            )

        np_bs = _positive(
            getattr(extract_tuning, "nemotron_parse_batch_size", None) if extract_tuning is not None else None
        ) or (plan.nemotron_parse_batch_size if plan else None)
        _set(NemotronParseActor.__name__, "batch_size", np_bs)
        _set(
            NemotronParseActor.__name__,
            "concurrency",
            getattr(extract_tuning, "nemotron_parse_workers", None) if extract_tuning is not None else None,
            plan.nemotron_parse_initial_actors if plan else None,
        )
        _set(
            NemotronParseActor.__name__,
            "num_gpus",
            getattr(extract_tuning, "gpu_nemotron_parse", None) if extract_tuning is not None else None,
            plan.nemotron_parse_gpus_per_actor if plan else None,
        )

        pdf_bs = _positive(
            getattr(extract_tuning, "pdf_extract_batch_size", None) if extract_tuning is not None else None
        ) or (plan.pdf_extract_batch_size if plan else None)
        pdf_extract_cpus = (
            _resolve(
                getattr(extract_tuning, "pdf_extract_num_cpus", None) if extract_tuning is not None else None,
                plan.pdf_extract_cpus_per_task if plan else None,
            )
            or 1.0
        )
        pdf_extract_tasks = _resolve(
            getattr(extract_tuning, "pdf_extract_workers", None) if extract_tuning is not None else None,
            plan.pdf_extract_tasks if plan else None,
        )

        # Cap PDF extract concurrency so persistent actors for page-elements,
        # OCR, and embed plus 4 fixed pipeline tasks (DocToPdf, PDFSplit,
        # UDFOperator, ReadBinary) cannot exhaust the cluster CPU budget.
        if pdf_extract_tasks is not None and cluster_resources is not None:
            non_pdf_cpu_overhead = (
                4
                + page_elements_concurrency * page_elements_cpus
                + ocr_concurrency * ocr_cpus
                + embed_concurrency * embed_cpus
            )
            pdf_extract_tasks = min(
                pdf_extract_tasks,
                max(1, int((cluster_resources.total_cpu_count() - non_pdf_cpu_overhead) // pdf_extract_cpus)),
            )

        _set(PDFExtractionActor.__name__, "batch_size", pdf_bs)
        _set(PDFExtractionActor.__name__, "concurrency", pdf_extract_tasks)
        _set(PDFExtractionActor.__name__, "num_cpus", pdf_extract_cpus if pdf_extract_cpus != 1.0 else None)

    return overrides


def _resolve_execution_inputs(
    *,
    execution_plan: IngestExecutionPlan | None,
    extraction_mode: str,
    extract_params: Any | None,
    text_params: Any | None,
    html_params: Any | None,
    audio_chunk_params: Any | None,
    asr_params: Any | None,
    dedup_params: Any | None,
    split_params: Any | None,
    caption_params: Any | None,
    store_params: Any | None,
    embed_params: Any | None,
    stage_order: tuple[str, ...],
) -> tuple[
    str,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    Any | None,
    tuple[str, ...],
]:
    """Resolve legacy builder args or a shared execution plan into one input tuple."""

    if execution_plan is None:
        return (
            extraction_mode,
            extract_params,
            text_params,
            html_params,
            audio_chunk_params,
            asr_params,
            dedup_params,
            split_params,
            caption_params,
            store_params,
            embed_params,
            stage_order,
        )

    stage_map = {stage.name: stage.params for stage in execution_plan.stages}
    return (
        execution_plan.extraction_mode,
        execution_plan.extract_params,
        execution_plan.text_params,
        execution_plan.html_params,
        execution_plan.audio_chunk_params,
        execution_plan.asr_params,
        stage_map.get("dedup"),
        stage_map.get("split"),
        stage_map.get("caption"),
        stage_map.get("store"),
        stage_map.get("embed"),
        tuple(stage.name for stage in execution_plan.stages),
    )


def _should_build_audio_graph(
    *,
    extract_params: Any | None,
    asr_params: Any | None,
) -> bool:
    method = str(getattr(extract_params, "method", "") or "").strip().lower()
    if method == "audio":
        return True
    if asr_params is not None:
        return True
    return False


def _append_ordered_transform_stages(
    graph: Graph,
    *,
    extraction_mode: str,
    dedup_params: Any | None,
    split_params: Any | None,
    caption_params: Any | None,
    store_params: Any | None,
    embed_params: Any | None,
    stage_order: tuple[str, ...],
    supports_dedup: bool,
    reshape_for_modal_content: bool,
) -> Graph:
    """Append post-extraction transform stages in the exact recorded plan order."""

    pending_stages = [
        stage
        for stage in stage_order
        if stage in {"dedup", "split", "caption", "store", "embed"} and (supports_dedup or stage != "dedup")
    ]
    if not pending_stages:
        if supports_dedup and dedup_params is not None:
            pending_stages.append("dedup")
        if caption_params is not None:
            pending_stages.append("caption")
        if store_params is not None:
            pending_stages.append("store")
        if split_params is not None:
            pending_stages.append("split")
        if embed_params is not None:
            pending_stages.append("embed")

    for stage_name in pending_stages:
        if stage_name == "store" and store_params is not None:
            graph = graph >> StoreOperator(params=store_params)
        elif stage_name == "dedup" and supports_dedup and dedup_params is not None:
            dedup_kwargs = cast(dict[str, Any], dedup_params.model_dump(mode="python"))
            graph = graph >> UDFOperator(partial(dedup_images, **dedup_kwargs), name="DedupImages")
        elif stage_name == "caption" and caption_params is not None:
            graph = graph >> CaptionActor(caption_params)
        elif stage_name == "split" and split_params is not None:
            graph = graph >> TextChunkActor(split_params)
        elif stage_name == "embed" and embed_params is not None:
            needs_content_reshape = reshape_for_modal_content and extraction_mode in {"pdf", "image", "auto"}
            if needs_content_reshape:
                content_columns = (_CONTENT_COLUMNS + ("images",)) if caption_params is not None else _CONTENT_COLUMNS
                if embed_params.embed_granularity == "page":
                    graph = graph >> UDFOperator(
                        partial(
                            collapse_content_to_page_rows,
                            modality=embed_params.embed_modality,
                            content_columns=content_columns,
                        ),
                        name="CollapseContentToPageRows",
                    )
                else:
                    graph = graph >> UDFOperator(
                        partial(
                            explode_content_to_rows,
                            modality=embed_params.embed_modality,
                            text_elements_modality=embed_params.text_elements_modality or embed_params.embed_modality,
                            structured_elements_modality=embed_params.structured_elements_modality
                            or embed_params.embed_modality,
                            content_columns=content_columns,
                        ),
                        name="ExplodeContentToRows",
                    )
            graph = graph >> embed_actor_class(embed_params=embed_params)(params=embed_params)

    return graph


def build_graph(
    *,
    execution_plan: IngestExecutionPlan | None = None,
    extraction_mode: str = "pdf",
    extract_params: Any | None = None,
    text_params: Any | None = None,
    html_params: Any | None = None,
    audio_chunk_params: Any | None = None,
    asr_params: Any | None = None,
    dedup_params: Any | None = None,
    embed_params: Any | None = None,
    split_params: Any | None = None,
    caption_params: Any | None = None,
    store_params: Any | None = None,
    stage_order: tuple[str, ...] = (),
) -> Graph:
    """Build a batch graph from explicit params or a shared execution plan."""

    (
        extraction_mode,
        extract_params,
        text_params,
        html_params,
        audio_chunk_params,
        asr_params,
        dedup_params,
        split_params,
        caption_params,
        store_params,
        embed_params,
        stage_order,
    ) = _resolve_execution_inputs(
        execution_plan=execution_plan,
        extraction_mode=extraction_mode,
        extract_params=extract_params,
        text_params=text_params,
        html_params=html_params,
        audio_chunk_params=audio_chunk_params,
        asr_params=asr_params,
        dedup_params=dedup_params,
        split_params=split_params,
        caption_params=caption_params,
        store_params=store_params,
        embed_params=embed_params,
        stage_order=stage_order,
    )

    if _should_build_audio_graph(
        extract_params=extract_params,
        asr_params=asr_params,
    ):
        graph = Graph() >> MediaChunkActor(params=audio_chunk_params) >> ASRActor(params=asr_params)
    elif extraction_mode in {"text", "html", "audio", "image", "auto"}:
        graph = Graph() >> MultiTypeExtractOperator(
            extraction_mode=extraction_mode,
            extract_params=extract_params,
            text_params=text_params,
            html_params=html_params,
            audio_chunk_params=audio_chunk_params,
            asr_params=asr_params,
            caption_params=caption_params,
        )
    else:
        graph = Graph()
        graph = graph >> DocToPdfConversionActor() >> PDFSplitActor()

        tuning = _batch_tuning(extract_params)
        parse_mode = extract_params.method == "nemotron_parse" or (
            tuning is not None
            and (_positive(getattr(tuning, "nemotron_parse_workers", None)) is not None)
            and (_positive(getattr(tuning, "gpu_nemotron_parse", None)) is not None)
            and (_positive(getattr(tuning, "nemotron_parse_batch_size", None)) is not None)
        )

        extract_kwargs: dict[str, Any] = {
            "method": extract_params.method,
            "dpi": int(extract_params.dpi),
            "extract_text": extract_params.extract_text,
            "extract_images": extract_params.extract_images,
            "extract_tables": extract_params.extract_tables,
            "extract_charts": extract_params.extract_charts,
            "extract_infographics": extract_params.extract_infographics,
            "extract_page_as_image": extract_params.extract_page_as_image,
            "api_key": extract_params.api_key,
        }

        if parse_mode:
            # PDF extraction renders pages to images required by Nemotron Parse.
            extract_kwargs["extract_page_as_image"] = True
            graph = graph >> PDFExtractionActor(**extract_kwargs)

            parse_kwargs: dict[str, Any] = {
                "extract_text": extract_params.extract_text,
                "extract_tables": extract_params.extract_tables,
                "extract_charts": extract_params.extract_charts,
                "extract_infographics": extract_params.extract_infographics,
            }
            if extract_params.nemotron_parse_invoke_url:
                parse_kwargs["nemotron_parse_invoke_url"] = extract_params.nemotron_parse_invoke_url
            elif extract_params.invoke_url:
                parse_kwargs["invoke_url"] = extract_params.invoke_url
            if extract_params.api_key:
                parse_kwargs["api_key"] = extract_params.api_key
            parse_actor = nemotron_parse_actor_class(
                invoke_url=getattr(extract_params, "nemotron_parse_invoke_url", None)
            )
            graph = graph >> parse_actor(**parse_kwargs)
        else:
            detect_kwargs: dict[str, Any] = {}
            if extract_params.page_elements_invoke_url:
                detect_kwargs["page_elements_invoke_url"] = extract_params.page_elements_invoke_url
            if extract_params.api_key:
                detect_kwargs["api_key"] = extract_params.api_key
            if extract_params.inference_batch_size:
                detect_kwargs["inference_batch_size"] = int(extract_params.inference_batch_size)

            ocr_kwargs: dict[str, Any] = {}
            if extract_params.method in ("pdfium_hybrid", "ocr") and extract_params.extract_text:
                ocr_kwargs["extract_text"] = True
            if extract_params.extract_tables and not extract_params.use_table_structure:
                ocr_kwargs["extract_tables"] = True
            if extract_params.extract_charts and not extract_params.use_graphic_elements:
                ocr_kwargs["extract_charts"] = True
            if extract_params.extract_infographics:
                ocr_kwargs["extract_infographics"] = True
            ocr_kwargs["use_graphic_elements"] = extract_params.use_graphic_elements
            if extract_params.ocr_invoke_url:
                ocr_kwargs["ocr_invoke_url"] = extract_params.ocr_invoke_url
            if extract_params.api_key:
                ocr_kwargs["api_key"] = extract_params.api_key
            detect_batch_size = _positive(
                getattr(tuning, "ocr_inference_batch_size", None) if tuning is not None else None
            )
            if detect_batch_size:
                ocr_kwargs["inference_batch_size"] = int(detect_batch_size)

            table_kwargs: dict[str, Any] = {}
            if extract_params.table_structure_invoke_url:
                table_kwargs["table_structure_invoke_url"] = extract_params.table_structure_invoke_url
            if extract_params.ocr_invoke_url:
                table_kwargs["ocr_invoke_url"] = extract_params.ocr_invoke_url
            if extract_params.api_key:
                table_kwargs["api_key"] = extract_params.api_key
            if extract_params.table_output_format:
                table_kwargs["table_output_format"] = extract_params.table_output_format

            graphic_kwargs: dict[str, Any] = {}
            if extract_params.graphic_elements_invoke_url:
                graphic_kwargs["graphic_elements_invoke_url"] = extract_params.graphic_elements_invoke_url
            if extract_params.ocr_invoke_url:
                graphic_kwargs["ocr_invoke_url"] = extract_params.ocr_invoke_url
            if extract_params.api_key:
                graphic_kwargs["api_key"] = extract_params.api_key

            page_elements_actor = page_elements_actor_class(extract_params=extract_params)
            table_actor = table_structure_actor_class(extract_params=extract_params)
            graphic_actor = graphic_elements_actor_class(extract_params=extract_params)
            ocr_actor = ocr_actor_class(extract_params=extract_params)

            graph = graph >> PDFExtractionActor(**extract_kwargs) >> page_elements_actor(**detect_kwargs)
            if extract_params.use_table_structure and extract_params.extract_tables:
                graph = graph >> table_actor(**table_kwargs)
            if extract_params.use_graphic_elements and extract_params.extract_charts:
                graph = graph >> graphic_actor(**graphic_kwargs)

            needs_ocr = any(
                bool(ocr_kwargs.get(key))
                for key in ("extract_text", "extract_tables", "extract_charts", "extract_infographics")
            )
            if needs_ocr:
                graph = graph >> ocr_actor(**ocr_kwargs)

    return _append_ordered_transform_stages(
        graph,
        extraction_mode=extraction_mode,
        dedup_params=dedup_params,
        split_params=split_params,
        caption_params=caption_params,
        store_params=store_params,
        embed_params=embed_params,
        stage_order=stage_order,
        supports_dedup=True,
        reshape_for_modal_content=True,
    )


# build_inprocess_graph previously maintained a separate graph shape.
# In-process execution now intentionally reuses the shared graph builder so
# both modes inherit the same defaults, node ordering, and optional stages.
build_inprocess_graph = build_graph
