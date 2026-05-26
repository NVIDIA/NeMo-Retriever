# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence, cast

from nemo_retriever.ingestor import create_ingestor
from nemo_retriever.ocr.config import OCRLang, OCRVersion
from nemo_retriever.params import (
    AudioChunkParams,
    AudioVisualFuseParams,
    BatchTuningParams,
    EmbedParams,
    ExtractParams,
    HtmlChunkParams,
    TextChunkParams,
    VdbUploadParams,
    VideoFrameParams,
    VideoFrameTextDedupParams,
)
from nemo_retriever.params.utils import normalize_embed_kwargs
from nemo_retriever.retriever import Retriever
from nemo_retriever.utils.input_files import (
    AUTO_INPUT_EXTENSIONS,
    INPUT_TYPE_EXTENSIONS,
    expand_input_file_patterns,
    input_type_for_path,
    resolve_input_files,
)
from nemo_retriever.utils.remote_auth import resolve_remote_api_key
from nemo_retriever.vdb.records import RetrievalHit


IngestInputTypeValue = Literal["auto", "pdf", "doc", "txt", "html", "image", "audio", "video"]
IngestRunModeValue = Literal["inprocess", "batch"]
LocalIngestEmbedBackendValue = Literal["vllm", "hf"]
OcrLangValue = OCRLang
OcrVersionValue = OCRVersion
TableOutputFormatValue = Literal["pseudo_markdown", "markdown"]
_SUPPORTED_RUN_MODES: tuple[IngestRunModeValue, ...] = ("inprocess", "batch")
_SUPPORTED_INPUT_TYPES: tuple[IngestInputTypeValue, ...] = (
    "auto",
    "pdf",
    "doc",
    "txt",
    "html",
    "image",
    "audio",
    "video",
)
_AUDIO_SPLIT_INTERVAL = 500000
_VIDEO_FRAME_FPS = 0.5


def _validate_run_mode(run_mode: str) -> IngestRunModeValue:
    if run_mode not in _SUPPORTED_RUN_MODES:
        raise ValueError(f"run_mode must be one of {', '.join(_SUPPORTED_RUN_MODES)}, got {run_mode!r}.")
    return cast(IngestRunModeValue, run_mode)


def _validate_input_type(input_type: str) -> IngestInputTypeValue:
    if input_type not in _SUPPORTED_INPUT_TYPES:
        raise ValueError(f"input_type must be one of {', '.join(_SUPPORTED_INPUT_TYPES)}, got {input_type!r}.")
    return cast(IngestInputTypeValue, input_type)


def _input_type_for_extension(path: str) -> IngestInputTypeValue | None:
    return cast(IngestInputTypeValue | None, input_type_for_path(path))


def _validate_ingest_document_types(
    documents: Sequence[str],
    *,
    input_type: IngestInputTypeValue,
) -> None:
    allowed = AUTO_INPUT_EXTENSIONS if input_type == "auto" else INPUT_TYPE_EXTENSIONS[input_type]
    unsupported = [
        document
        for document in documents
        if not any(ch in str(document) for ch in "*?[") and Path(document).suffix.lower() not in allowed
    ]
    if unsupported:
        examples = ", ".join(unsupported[:3])
        if input_type == "auto":
            raise ValueError(f"Unsupported input file type(s) for retriever ingest: {examples}")
        raise ValueError(f"Input file type(s) do not match --input-type={input_type!r}: {examples}")


# The ingest command accepts bare dataset directories; expand those to supported
# files before passing file/glob inputs through the shared input normalizer.
def _expand_ingest_documents(
    documents: Sequence[str],
    *,
    input_type: IngestInputTypeValue,
) -> list[str]:
    inputs: list[str] = []
    for document in documents:
        raw_document = str(document)
        path = Path(raw_document).expanduser()
        if path.is_dir():
            directory_files = resolve_input_files(path, input_type)
            if not directory_files:
                if input_type == "auto":
                    raise FileNotFoundError(f"No supported ingest files found under directory: {path}")
                raise FileNotFoundError(f"No {input_type} files found under directory: {path}")
            inputs.extend(str(file) for file in directory_files)
        else:
            inputs.append(raw_document)

    document_list = expand_input_file_patterns(inputs)
    _validate_ingest_document_types(document_list, input_type=input_type)
    return document_list


def _resolve_effective_input_type(
    documents: Sequence[str],
    *,
    input_type: IngestInputTypeValue,
) -> IngestInputTypeValue:
    if input_type != "auto":
        return "pdf" if input_type == "doc" else input_type

    observed = {
        resolved
        for document in documents
        if not any(ch in str(document) for ch in "*?[")
        if (resolved := _input_type_for_extension(str(document))) is not None
    }
    if not observed:
        return "auto"
    if observed <= {"pdf", "doc"}:
        return "pdf"
    if len(observed) == 1:
        only = next(iter(observed))
        return "pdf" if only == "doc" else only
    return "auto"


def _default_asr_params() -> Any:
    from nemo_retriever.audio import asr_params_from_env

    return asr_params_from_env()


def _attach_extract_stage(
    ingestor: Any,
    *,
    input_type: IngestInputTypeValue,
    extract_params: ExtractParams | None,
) -> Any:
    if input_type == "pdf":
        params = extract_params or ExtractParams()
        return ingestor.extract(params, extraction_mode="pdf")
    if input_type == "txt":
        return ingestor.extract_txt(TextChunkParams())
    if input_type == "html":
        return ingestor.extract_html(HtmlChunkParams())
    if input_type == "image":
        return ingestor.extract_image_files(extract_params or ExtractParams())
    if input_type == "audio":
        asr_params = _default_asr_params().model_copy(update={"segment_audio": False})
        return ingestor.extract_audio(
            params=AudioChunkParams(split_type="size", split_interval=_AUDIO_SPLIT_INTERVAL),
            asr_params=asr_params,
        )
    if input_type == "video":
        asr_params = _default_asr_params().model_copy(update={"segment_audio": False})
        return ingestor.extract_video(
            params=AudioChunkParams(
                enabled=True,
                split_type="size",
                split_interval=_AUDIO_SPLIT_INTERVAL,
            ),
            asr_params=asr_params,
            video_frame_params=VideoFrameParams(
                enabled=True,
                fps=_VIDEO_FRAME_FPS,
                dedup=True,
            ),
            video_text_dedup_params=VideoFrameTextDedupParams(enabled=True, max_dropped_frames=2),
            av_fuse_params=AudioVisualFuseParams(enabled=True),
            extract_params=extract_params or ExtractParams(),
        )
    return ingestor.extract(
        extract_params or ExtractParams(),
        extraction_mode="auto",
        text_params=TextChunkParams(),
        html_params=HtmlChunkParams(),
    )


def _build_embed_kwargs(
    embed_invoke_url: str | None,
    embed_model_name: str | None,
    local_ingest_embed_backend: LocalIngestEmbedBackendValue | None = None,
    embed_workers: int | None = None,
    embed_batch_size: int | None = None,
    embed_cpus_per_actor: float | None = None,
    embed_gpus_per_actor: float | None = None,
) -> dict[str, Any]:
    embed_kwargs: dict[str, Any] = {}
    if embed_invoke_url is not None:
        embed_kwargs["embed_invoke_url"] = embed_invoke_url
    if embed_model_name is not None:
        # Remote HTTP embedding reads model_name; local/GPU paths read embed_model_name.
        embed_kwargs["model_name"] = embed_model_name
        embed_kwargs["embed_model_name"] = embed_model_name
    if local_ingest_embed_backend is not None:
        embed_kwargs["local_ingest_embed_backend"] = local_ingest_embed_backend
    embed_tuning = _build_embed_batch_tuning(
        embed_workers=embed_workers,
        embed_batch_size=embed_batch_size,
        embed_cpus_per_actor=embed_cpus_per_actor,
        embed_gpus_per_actor=embed_gpus_per_actor,
    )
    if embed_tuning is not None:
        embed_kwargs["batch_tuning"] = embed_tuning
    return normalize_embed_kwargs(embed_kwargs)


def _build_extract_batch_tuning(
    *,
    pdf_extract_workers: int | None,
    pdf_extract_batch_size: int | None,
    pdf_extract_cpus_per_task: float | None,
    page_elements_workers: int | None,
    page_elements_batch_size: int | None,
    page_elements_cpus_per_actor: float | None,
    page_elements_gpus_per_actor: float | None,
    ocr_workers: int | None,
    ocr_batch_size: int | None,
    ocr_cpus_per_actor: float | None,
    ocr_gpus_per_actor: float | None,
    table_structure_workers: int | None,
    table_structure_batch_size: int | None,
    table_structure_cpus_per_actor: float | None,
    table_structure_gpus_per_actor: float | None,
) -> BatchTuningParams | None:
    tuning_kwargs = {
        key: value
        for key, value in {
            "pdf_extract_workers": pdf_extract_workers,
            "pdf_extract_batch_size": pdf_extract_batch_size,
            # BatchTuningParams names this per-Ray-task reservation num_cpus.
            "pdf_extract_num_cpus": pdf_extract_cpus_per_task,
            "page_elements_workers": page_elements_workers,
            "page_elements_batch_size": page_elements_batch_size,
            "page_elements_cpus_per_actor": page_elements_cpus_per_actor,
            "gpu_page_elements": page_elements_gpus_per_actor,
            "ocr_workers": ocr_workers,
            "ocr_inference_batch_size": ocr_batch_size,
            "ocr_cpus_per_actor": ocr_cpus_per_actor,
            "gpu_ocr": ocr_gpus_per_actor,
            "table_structure_workers": table_structure_workers,
            "table_structure_batch_size": table_structure_batch_size,
            "table_structure_cpus_per_actor": table_structure_cpus_per_actor,
            "gpu_table_structure": table_structure_gpus_per_actor,
        }.items()
        if value is not None
    }
    return BatchTuningParams(**tuning_kwargs) if tuning_kwargs else None


def _build_embed_batch_tuning(
    *,
    embed_workers: int | None,
    embed_batch_size: int | None,
    embed_cpus_per_actor: float | None,
    embed_gpus_per_actor: float | None,
) -> BatchTuningParams | None:
    tuning_kwargs = {
        key: value
        for key, value in {
            "embed_workers": embed_workers,
            "embed_batch_size": embed_batch_size,
            "embed_cpus_per_actor": embed_cpus_per_actor,
            "gpu_embed": embed_gpus_per_actor,
        }.items()
        if value is not None
    }
    return BatchTuningParams(**tuning_kwargs) if tuning_kwargs else None


_LOCAL_VL_RERANK_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2"


def _build_rerank_kwargs(
    reranker_invoke_url: str | None,
    reranker_model_name: str | None = None,
    reranker_backend: str | None = None,
) -> dict[str, str]:
    """Build kwargs for the rerank stage. Mirrors :func:`_build_embed_kwargs`:
    if ``reranker_invoke_url`` is given the remote NIM path is configured;
    otherwise the local GPU reranker runs with ``reranker_model_name`` (or the
    matching VL default to pair with the local VL embedder).

    ``reranker_backend`` only applies to the local path and selects the local
    inference backend (``"vllm"`` or ``"hf"``); ``None`` defers to the library
    default in ``_default_rerank_actor_kwargs``.
    """
    reranker_url = (reranker_invoke_url or "").strip()
    if reranker_url:
        rerank_kwargs: dict[str, str] = {"rerank_invoke_url": reranker_url}
        if reranker_model_name:
            rerank_kwargs["model_name"] = reranker_model_name
        api_key = resolve_remote_api_key()
        if api_key is not None:
            rerank_kwargs["api_key"] = api_key
        return rerank_kwargs

    # Local GPU reranker - VL by default to pair with the local VL embedder.
    # ``NemotronRerankGPUActor`` loads the model once per actor; the rerank
    # model is ~2 GB and coexists with the vLLM embedder (which respects
    # ``gpu_memory_utilization=0.45``).
    local: dict[str, str] = {"model_name": reranker_model_name or _LOCAL_VL_RERANK_MODEL}
    if reranker_backend:
        local["local_reranker_backend"] = reranker_backend
    return local


def ingest_documents(
    documents: Sequence[str],
    *,
    input_type: IngestInputTypeValue = "auto",
    run_mode: IngestRunModeValue = "inprocess",
    ray_address: str | None = None,
    ray_log_to_driver: bool | None = None,
    lancedb_uri: str = "lancedb",
    table_name: str = "nv-ingest",
    overwrite: bool = True,
    page_elements_invoke_url: str | None = None,
    ocr_invoke_url: str | None = None,
    ocr_version: OcrVersionValue | None = None,
    ocr_lang: OcrLangValue | None = None,
    graphic_elements_invoke_url: str | None = None,
    table_structure_invoke_url: str | None = None,
    table_output_format: TableOutputFormatValue | None = None,
    embed_invoke_url: str | None = None,
    embed_model_name: str | None = None,
    local_ingest_embed_backend: LocalIngestEmbedBackendValue | None = None,
    pdf_extract_workers: int | None = None,
    pdf_extract_batch_size: int | None = None,
    pdf_extract_cpus_per_task: float | None = None,
    page_elements_workers: int | None = None,
    page_elements_batch_size: int | None = None,
    page_elements_cpus_per_actor: float | None = None,
    page_elements_gpus_per_actor: float | None = None,
    ocr_workers: int | None = None,
    ocr_batch_size: int | None = None,
    ocr_cpus_per_actor: float | None = None,
    ocr_gpus_per_actor: float | None = None,
    table_structure_workers: int | None = None,
    table_structure_batch_size: int | None = None,
    table_structure_cpus_per_actor: float | None = None,
    table_structure_gpus_per_actor: float | None = None,
    embed_workers: int | None = None,
    embed_batch_size: int | None = None,
    embed_cpus_per_actor: float | None = None,
    embed_gpus_per_actor: float | None = None,
) -> dict[str, Any]:
    """Run the root CLI ingestion path through the SDK adapter.

    ``ray_address`` and ``ray_log_to_driver`` are forwarded only when the
    caller sets them, preserving the default ``create_ingestor`` behavior.
    Batch tuning arguments are opt-in and are translated into
    ``BatchTuningParams`` for extraction or embedding; they are meaningful for
    ``run_mode="batch"`` and ignored by callers that leave them unset.
    """
    validated_run_mode = _validate_run_mode(run_mode)
    validated_input_type = _validate_input_type(input_type)
    document_list = _expand_ingest_documents(documents, input_type=validated_input_type)
    effective_input_type = _resolve_effective_input_type(document_list, input_type=validated_input_type)
    extract_kwargs = {
        key: value
        for key, value in {
            "page_elements_invoke_url": page_elements_invoke_url,
            "ocr_invoke_url": ocr_invoke_url,
            "ocr_version": ocr_version,
            "ocr_lang": ocr_lang,
            "graphic_elements_invoke_url": graphic_elements_invoke_url,
            "table_structure_invoke_url": table_structure_invoke_url,
            "table_output_format": table_output_format,
        }.items()
        if value is not None
    }
    if table_output_format == "markdown":
        extract_kwargs["use_table_structure"] = True
    extract_tuning = _build_extract_batch_tuning(
        pdf_extract_workers=pdf_extract_workers,
        pdf_extract_batch_size=pdf_extract_batch_size,
        pdf_extract_cpus_per_task=pdf_extract_cpus_per_task,
        page_elements_workers=page_elements_workers,
        page_elements_batch_size=page_elements_batch_size,
        page_elements_cpus_per_actor=page_elements_cpus_per_actor,
        page_elements_gpus_per_actor=page_elements_gpus_per_actor,
        ocr_workers=ocr_workers,
        ocr_batch_size=ocr_batch_size,
        ocr_cpus_per_actor=ocr_cpus_per_actor,
        ocr_gpus_per_actor=ocr_gpus_per_actor,
        table_structure_workers=table_structure_workers,
        table_structure_batch_size=table_structure_batch_size,
        table_structure_cpus_per_actor=table_structure_cpus_per_actor,
        table_structure_gpus_per_actor=table_structure_gpus_per_actor,
    )
    if extract_tuning is not None:
        extract_kwargs["batch_tuning"] = extract_tuning
    embed_kwargs = _build_embed_kwargs(
        embed_invoke_url,
        embed_model_name,
        local_ingest_embed_backend=local_ingest_embed_backend,
        embed_workers=embed_workers,
        embed_batch_size=embed_batch_size,
        embed_cpus_per_actor=embed_cpus_per_actor,
        embed_gpus_per_actor=embed_gpus_per_actor,
    )
    extract_params = ExtractParams(**extract_kwargs) if extract_kwargs else None
    embed_params = EmbedParams(**embed_kwargs) if embed_kwargs else None
    vdb_params = VdbUploadParams(
        vdb_kwargs={"uri": lancedb_uri, "table_name": table_name, "overwrite": bool(overwrite)}
    )

    create_kwargs: dict[str, Any] = {"run_mode": validated_run_mode}
    if ray_address is not None:
        create_kwargs["ray_address"] = ray_address
    if ray_log_to_driver is not None:
        create_kwargs["ray_log_to_driver"] = ray_log_to_driver

    ingestor = create_ingestor(**create_kwargs).files(document_list)
    ingestor = _attach_extract_stage(
        ingestor,
        input_type=effective_input_type,
        extract_params=extract_params,
    )
    ingestor = ingestor.embed(embed_params) if embed_params is not None else ingestor.embed()
    result = ingestor.vdb_upload(vdb_params).ingest()
    return {
        "documents": document_list,
        "lancedb_uri": lancedb_uri,
        "result": result,
        "table_name": table_name,
    }


def query_documents(
    query: str,
    *,
    top_k: int = 10,
    lancedb_uri: str = "lancedb",
    table_name: str = "nv-ingest",
    embed_invoke_url: str | None = None,
    embed_model_name: str | None = None,
    reranker_invoke_url: str | None = None,
    reranker_model_name: str | None = None,
    reranker_backend: str | None = None,
    rerank: bool = False,
) -> list[RetrievalHit]:
    """Run the minimal SDK query path used by the root CLI.

    Reranking is opt-in: pass ``rerank=True`` (or any of the rerank-related
    args via the CLI, which implicitly set ``rerank=True``) to enable.
    """
    embed_kwargs = _build_embed_kwargs(embed_invoke_url, embed_model_name)
    retriever_kwargs: dict[str, Any] = {
        "top_k": top_k,
        "vdb_kwargs": {"uri": lancedb_uri, "table_name": table_name},
    }
    if embed_kwargs:
        retriever_kwargs["embed_kwargs"] = embed_kwargs
    if rerank:
        rerank_kwargs = _build_rerank_kwargs(reranker_invoke_url, reranker_model_name, reranker_backend)
        retriever_kwargs["rerank"] = True
        if rerank_kwargs:
            retriever_kwargs["rerank_kwargs"] = rerank_kwargs

    retriever = Retriever(**retriever_kwargs)
    return retriever.query(query)
