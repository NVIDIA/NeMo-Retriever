# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any

from pydantic import BaseModel, ConfigDict

from nemo_retriever.model import resolve_embed_model
from nemo_retriever.recall.beir import BeirConfig, evaluate_lancedb_beir
from nemo_retriever.recall.core import RecallConfig, retrieve_and_score
from nemo_retriever.utils.detection_summary import write_detection_summary
from nemo_retriever.utils.input_files import resolve_input_patterns

DEFAULT_LANCEDB_URI = "lancedb"
DEFAULT_LANCEDB_TABLE = "nv-ingest"

logger = logging.getLogger(__name__)


class ModePipelineConfigModel(BaseModel):
    """Strict config base for mode pipeline runners."""

    model_config = ConfigDict(extra="forbid")


def safe_pdf_page_count(path: Path) -> int | None:
    try:
        import pypdfium2 as pdfium  # type: ignore

        doc = pdfium.PdfDocument(str(path))
        try:
            try:
                count = int(len(doc))
            except Exception:
                count = int(doc.get_page_count())  # type: ignore[attr-defined]
        finally:
            try:
                doc.close()
            except Exception:
                pass
        return max(count, 0)
    except Exception:
        return None


def resolve_input_pages(input_type: str, input_files: list[Path]) -> int | None:
    if input_type == "image":
        return len(input_files)
    if input_type != "pdf":
        return None

    total_pages = 0
    counted_any = False
    for path in input_files:
        page_count = safe_pdf_page_count(path)
        if page_count is None:
            continue
        counted_any = True
        total_pages += page_count
    if counted_any:
        return total_pages
    return None


def resolve_mode_file_patterns(*, input_path: str | Path, input_type: str, file_patterns: list[str]) -> list[str]:
    if file_patterns:
        return list(file_patterns)
    return resolve_input_patterns(Path(input_path), input_type)


def resolve_lancedb_target(
    *,
    artifacts_lancedb_uri: str | None,
    artifacts_lancedb_table: str | None,
    vdb_lancedb_uri: str | None = None,
    vdb_lancedb_table: str | None = None,
) -> tuple[str, str]:
    uri = str(Path(vdb_lancedb_uri or artifacts_lancedb_uri or DEFAULT_LANCEDB_URI).expanduser().resolve())
    table = str(vdb_lancedb_table or artifacts_lancedb_table or DEFAULT_LANCEDB_TABLE)
    return uri, table


def persist_detection_summary_artifact(
    *,
    detection_summary_file: str | None,
    detection_payload: dict[str, Any] | None,
) -> str | None:
    if detection_summary_file is None:
        return None
    detection_path = Path(detection_summary_file).expanduser().resolve()
    write_detection_summary(detection_path, detection_payload)
    return str(detection_path)


def shutdown_ray_safely() -> None:
    try:
        import ray

        ray.shutdown()
    except Exception:
        pass


def evaluate_lancedb_metrics(
    evaluation_cfg,
    *,
    lancedb_uri: str,
    lancedb_table: str,
    embed_model_name: str,
    embed_invoke_url: str | None,
    embed_api_key: str | None,
    hybrid: bool,
):
    from .reports import EvaluationSummary
    from nemo_retriever.vector_store.lancedb_store import open_lancedb_table_with_retry

    if evaluation_cfg.evaluation_mode not in {"recall", "beir"}:
        raise ValueError(f"Unsupported evaluation mode: {evaluation_cfg.evaluation_mode}")

    table = open_lancedb_table_with_retry(lancedb_uri, lancedb_table)
    try:
        if int(table.count_rows()) == 0:
            logger.warning(
                "LanceDB table %r exists but is empty; skipping %s evaluation.",
                lancedb_table,
                evaluation_cfg.evaluation_mode,
            )
            return EvaluationSummary(label="BEIR" if evaluation_cfg.evaluation_mode == "beir" else "Recall"), 0.0
    except Exception:
        pass

    resolved_model = resolve_embed_model(embed_model_name)
    evaluation_label = "BEIR" if evaluation_cfg.evaluation_mode == "beir" else "Recall"

    if evaluation_cfg.evaluation_mode == "beir":
        if not evaluation_cfg.beir_loader:
            raise ValueError("--beir-loader is required when --evaluation-mode=beir")
        if not evaluation_cfg.beir_dataset_name:
            raise ValueError("--beir-dataset-name is required when --evaluation-mode=beir")

        beir_cfg = BeirConfig(
            lancedb_uri=lancedb_uri,
            lancedb_table=lancedb_table,
            embedding_model=resolved_model,
            loader=str(evaluation_cfg.beir_loader),
            dataset_name=str(evaluation_cfg.beir_dataset_name),
            split=str(evaluation_cfg.beir_split),
            query_language=evaluation_cfg.beir_query_language,
            doc_id_field=str(evaluation_cfg.beir_doc_id_field),
            ks=tuple(evaluation_cfg.beir_ks) if evaluation_cfg.beir_ks else (1, 3, 5, 10),
            embedding_http_endpoint=embed_invoke_url,
            embedding_api_key=(embed_api_key or "").strip(),
            hybrid=hybrid,
            reranker=bool(evaluation_cfg.reranker),
            reranker_model_name=str(evaluation_cfg.reranker_model_name),
        )
        evaluation_start = time.perf_counter()
        beir_dataset, _raw_hits, _run, metrics = evaluate_lancedb_beir(beir_cfg)
        return (
            EvaluationSummary(label=evaluation_label, query_count=len(beir_dataset.query_ids), metrics=metrics),
            time.perf_counter() - evaluation_start,
        )

    query_csv = Path(str(evaluation_cfg.query_csv)).expanduser() if evaluation_cfg.query_csv else None
    if query_csv is None or not query_csv.exists():
        logger.warning("Query CSV not found at %s; skipping recall evaluation.", query_csv)
        return EvaluationSummary(label=evaluation_label), 0.0

    recall_cfg = RecallConfig(
        lancedb_uri=lancedb_uri,
        lancedb_table=lancedb_table,
        embedding_model=resolved_model,
        embedding_http_endpoint=embed_invoke_url,
        embedding_api_key=(embed_api_key or "").strip(),
        top_k=10,
        ks=(1, 5, 10),
        hybrid=hybrid,
        match_mode=str(evaluation_cfg.recall_match_mode),
        reranker=str(evaluation_cfg.reranker_model_name) if evaluation_cfg.reranker else None,
    )
    evaluation_start = time.perf_counter()
    df_query, _gold, _raw_hits, _retrieved_keys, metrics = retrieve_and_score(query_csv=query_csv, cfg=recall_cfg)
    return (
        EvaluationSummary(label=evaluation_label, query_count=len(df_query.index), metrics=metrics),
        time.perf_counter() - evaluation_start,
    )
