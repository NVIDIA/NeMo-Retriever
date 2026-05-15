# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence, cast

from nemo_retriever.ingestor import create_ingestor
from nemo_retriever.params import BatchTuningParams, EmbedParams, ExtractParams, VdbUploadParams
from nemo_retriever.params.utils import normalize_embed_kwargs
from nemo_retriever.retriever import Retriever
from nemo_retriever.utils.input_files import expand_input_file_patterns, resolve_input_files
from nemo_retriever.utils.remote_auth import resolve_remote_api_key


IngestRunModeValue = Literal["inprocess", "batch"]
OcrVersionValue = Literal["v1", "v2"]
_SUPPORTED_RUN_MODES: tuple[IngestRunModeValue, ...] = ("inprocess", "batch")


def _validate_run_mode(run_mode: str) -> IngestRunModeValue:
    if run_mode not in _SUPPORTED_RUN_MODES:
        raise ValueError(f"run_mode must be one of {', '.join(_SUPPORTED_RUN_MODES)}, got {run_mode!r}.")
    return cast(IngestRunModeValue, run_mode)


# The ingest command accepts bare dataset directories; expand those to PDFs
# before passing file/glob inputs through the shared input normalizer.
def _expand_pdf_ingest_documents(documents: Sequence[str]) -> list[str]:
    inputs: list[str] = []
    for document in documents:
        raw_document = str(document)
        path = Path(raw_document).expanduser()
        if path.is_dir():
            directory_files = resolve_input_files(path, "pdf")
            if not directory_files:
                raise FileNotFoundError(f"No PDF files found under directory: {path}")
            inputs.extend(str(file) for file in directory_files)
        else:
            inputs.append(raw_document)

    document_list = expand_input_file_patterns(inputs)
    non_pdf_documents = [document for document in document_list if Path(document).suffix.lower() != ".pdf"]
    if non_pdf_documents:
        examples = ", ".join(non_pdf_documents[:3])
        raise ValueError(
            "Only PDF inputs are supported by retriever ingest. "
            f"Use 'retriever pipeline run' for other input types. Non-PDF input(s): {examples}"
        )
    return document_list


def _build_embed_kwargs(
    embed_invoke_url: str | None,
    embed_model_name: str | None,
    embed_workers: int | None = None,
    embed_batch_size: int | None = None,
    embed_cpus_per_actor: float | None = None,
) -> dict[str, Any]:
    embed_kwargs: dict[str, Any] = {}
    if embed_invoke_url is not None:
        embed_kwargs["embed_invoke_url"] = embed_invoke_url
    if embed_model_name is not None:
        # Remote HTTP embedding reads model_name; local/GPU paths read embed_model_name.
        embed_kwargs["model_name"] = embed_model_name
        embed_kwargs["embed_model_name"] = embed_model_name
    embed_tuning = _build_embed_batch_tuning(
        embed_workers=embed_workers,
        embed_batch_size=embed_batch_size,
        embed_cpus_per_actor=embed_cpus_per_actor,
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
    ocr_workers: int | None,
    ocr_batch_size: int | None,
    ocr_cpus_per_actor: float | None,
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
            "ocr_workers": ocr_workers,
            "ocr_inference_batch_size": ocr_batch_size,
            "ocr_cpus_per_actor": ocr_cpus_per_actor,
        }.items()
        if value is not None
    }
    return BatchTuningParams(**tuning_kwargs) if tuning_kwargs else None


def _build_embed_batch_tuning(
    *,
    embed_workers: int | None,
    embed_batch_size: int | None,
    embed_cpus_per_actor: float | None,
) -> BatchTuningParams | None:
    tuning_kwargs = {
        key: value
        for key, value in {
            "embed_workers": embed_workers,
            "embed_batch_size": embed_batch_size,
            "embed_cpus_per_actor": embed_cpus_per_actor,
        }.items()
        if value is not None
    }
    return BatchTuningParams(**tuning_kwargs) if tuning_kwargs else None


def _build_rerank_kwargs(reranker_invoke_url: str | None) -> dict[str, str]:
    if reranker_invoke_url is None:
        return {}

    reranker_url = reranker_invoke_url.strip()
    if not reranker_url:
        return {}

    rerank_kwargs = {"rerank_invoke_url": reranker_url}
    api_key = resolve_remote_api_key()
    if api_key is not None:
        rerank_kwargs["api_key"] = api_key
    return rerank_kwargs


def ingest_documents(
    documents: Sequence[str],
    *,
    run_mode: IngestRunModeValue = "inprocess",
    ray_address: str | None = None,
    ray_log_to_driver: bool | None = None,
    lancedb_uri: str = "lancedb",
    table_name: str = "nv-ingest",
    overwrite: bool = True,
    page_elements_invoke_url: str | None = None,
    ocr_invoke_url: str | None = None,
    ocr_version: OcrVersionValue | None = None,
    graphic_elements_invoke_url: str | None = None,
    table_structure_invoke_url: str | None = None,
    embed_invoke_url: str | None = None,
    embed_model_name: str | None = None,
    pdf_extract_workers: int | None = None,
    pdf_extract_batch_size: int | None = None,
    pdf_extract_cpus_per_task: float | None = None,
    page_elements_workers: int | None = None,
    page_elements_batch_size: int | None = None,
    page_elements_cpus_per_actor: float | None = None,
    ocr_workers: int | None = None,
    ocr_batch_size: int | None = None,
    ocr_cpus_per_actor: float | None = None,
    embed_workers: int | None = None,
    embed_batch_size: int | None = None,
    embed_cpus_per_actor: float | None = None,
) -> dict[str, Any]:
    """Run the root CLI ingestion path through the SDK adapter.

    ``ray_address`` and ``ray_log_to_driver`` are forwarded only when the
    caller sets them, preserving the default ``create_ingestor`` behavior.
    Batch tuning arguments are opt-in and are translated into
    ``BatchTuningParams`` for extraction or embedding; they are meaningful for
    ``run_mode="batch"`` and ignored by callers that leave them unset.
    """
    validated_run_mode = _validate_run_mode(run_mode)
    document_list = _expand_pdf_ingest_documents(documents)
    extract_kwargs = {
        key: value
        for key, value in {
            "page_elements_invoke_url": page_elements_invoke_url,
            "ocr_invoke_url": ocr_invoke_url,
            "ocr_version": ocr_version,
            "graphic_elements_invoke_url": graphic_elements_invoke_url,
            "table_structure_invoke_url": table_structure_invoke_url,
        }.items()
        if value is not None
    }
    extract_tuning = _build_extract_batch_tuning(
        pdf_extract_workers=pdf_extract_workers,
        pdf_extract_batch_size=pdf_extract_batch_size,
        pdf_extract_cpus_per_task=pdf_extract_cpus_per_task,
        page_elements_workers=page_elements_workers,
        page_elements_batch_size=page_elements_batch_size,
        page_elements_cpus_per_actor=page_elements_cpus_per_actor,
        ocr_workers=ocr_workers,
        ocr_batch_size=ocr_batch_size,
        ocr_cpus_per_actor=ocr_cpus_per_actor,
    )
    if extract_tuning is not None:
        extract_kwargs["batch_tuning"] = extract_tuning
    embed_kwargs = _build_embed_kwargs(
        embed_invoke_url,
        embed_model_name,
        embed_workers=embed_workers,
        embed_batch_size=embed_batch_size,
        embed_cpus_per_actor=embed_cpus_per_actor,
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
    ingestor = ingestor.extract(extract_params) if extract_params is not None else ingestor.extract()
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
) -> list[dict[str, Any]]:
    """Run the minimal SDK query path used by the root CLI."""
    embed_kwargs = _build_embed_kwargs(embed_invoke_url, embed_model_name)
    rerank_kwargs = _build_rerank_kwargs(reranker_invoke_url)
    retriever_kwargs: dict[str, Any] = {
        "top_k": top_k,
        "vdb_kwargs": {"uri": lancedb_uri, "table_name": table_name},
    }
    if embed_kwargs:
        retriever_kwargs["embed_kwargs"] = embed_kwargs
    if rerank_kwargs:
        retriever_kwargs["rerank"] = True
        retriever_kwargs["rerank_kwargs"] = rerank_kwargs

    retriever = Retriever(**retriever_kwargs)
    return retriever.query(query)
