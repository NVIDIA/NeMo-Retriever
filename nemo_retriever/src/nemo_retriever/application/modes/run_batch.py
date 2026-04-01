# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Batch-mode orchestration with structured run report artifacts."""

import json
from pathlib import Path
import time

from pydantic import Field

from nemo_retriever.ingest_modes.batch import BatchIngestor
from nemo_retriever.params import EmbedParams
from nemo_retriever.params import ExtractParams
from nemo_retriever.params import IngestExecuteParams
from nemo_retriever.params import IngestorCreateParams
from nemo_retriever.params import TextChunkParams
from nemo_retriever.utils.input_files import resolve_input_files
from nemo_retriever.vector_store.lancedb_store import (
    ensure_lancedb_table,
    estimate_processed_pages,
    handle_lancedb,
)

from .executor import run_mode_ingest
from .reports import (
    RunArtifactConfig,
    RunArtifacts,
    RunEvaluationConfig,
    RunMetrics,
    RunReport,
    render_run_report,
    persist_run_report_artifacts,
)
from .shared import (
    ModePipelineConfigModel,
    evaluate_lancedb_metrics,
    persist_detection_summary_artifact,
    resolve_lancedb_target,
    resolve_mode_file_patterns,
    resolve_input_pages,
    shutdown_ray_safely,
)


class BatchPipelineConfig(ModePipelineConfigModel):
    input_path: str
    input_type: str = "pdf"
    file_patterns: list[str] = Field(default_factory=list)
    create_params: IngestorCreateParams = Field(default_factory=IngestorCreateParams)
    execute_params: IngestExecuteParams = Field(default_factory=IngestExecuteParams)
    extract_params: ExtractParams | None = None
    embed_params: EmbedParams = Field(default_factory=EmbedParams)
    text_chunk_params: TextChunkParams | None = None
    enable_text_chunk: bool = False
    evaluation: RunEvaluationConfig = Field(default_factory=RunEvaluationConfig)
    artifacts: RunArtifactConfig = Field(default_factory=RunArtifactConfig)
    hybrid: bool = False


def _build_ingestor(cfg: BatchPipelineConfig):
    from nemo_retriever.ingestor import create_ingestor

    file_patterns = resolve_mode_file_patterns(
        input_path=cfg.input_path,
        input_type=cfg.input_type,
        file_patterns=cfg.file_patterns,
    )
    ingestor = create_ingestor(run_mode="batch", params=cfg.create_params)
    chunk_params = cfg.text_chunk_params or TextChunkParams()

    if cfg.input_type == "txt":
        ingestor = ingestor.files(file_patterns).extract_txt(chunk_params)
    elif cfg.input_type == "html":
        ingestor = ingestor.files(file_patterns).extract_html(chunk_params)
    elif cfg.input_type == "image":
        ingestor = ingestor.files(file_patterns).extract_image_files(cfg.extract_params or ExtractParams())
    else:
        ingestor = ingestor.files(file_patterns).extract(cfg.extract_params or ExtractParams())

    if cfg.enable_text_chunk:
        ingestor = ingestor.split(chunk_params)

    ingestor = ingestor.embed(cfg.embed_params)
    return ingestor, file_patterns


def _write_error_rows(error_rows, output_dir: str | None) -> None:
    error_count = int(error_rows.count())
    if error_count <= 0:
        return

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else Path.cwd()
    target_dir.mkdir(parents=True, exist_ok=True)
    error_file = target_dir / "ingest_errors.json"
    error_rows_to_write = error_rows.take(min(5, error_count))
    with error_file.open("w", encoding="utf-8") as fh:
        json.dump(error_rows_to_write, fh, indent=2, default=str)
        fh.write("\n")
    raise RuntimeError(
        "Detected "
        f"{error_count} error row(s) in ingest results. "
        f"Wrote first {len(error_rows_to_write)} row(s) to {error_file}."
    )


def run_batch_pipeline(cfg: BatchPipelineConfig) -> RunReport:
    try:
        input_path = Path(cfg.input_path).expanduser().resolve()
        input_files = resolve_input_files(input_path, cfg.input_type)
        resolved_input_pages = resolve_input_pages(cfg.input_type, input_files)
        lancedb_uri, lancedb_table = resolve_lancedb_target(
            artifacts_lancedb_uri=cfg.artifacts.lancedb_uri,
            artifacts_lancedb_table=cfg.artifacts.lancedb_table,
        )

        ensure_lancedb_table(lancedb_uri, lancedb_table)
        ingestor, file_patterns = _build_ingestor(cfg)

        ingest_start = time.perf_counter()
        ingest_results = ingestor.ingest(params=cfg.execute_params).get_dataset().materialize()
        ingest_secs = time.perf_counter() - ingest_start

        materialize_start = time.perf_counter()
        ingest_local_results = ingest_results.take_all()
        materialize_secs = time.perf_counter() - materialize_start

        vdb_write_start = time.perf_counter()
        handle_lancedb(ingest_local_results, lancedb_uri, lancedb_table, hybrid=cfg.hybrid, mode="overwrite")
        vdb_write_secs = time.perf_counter() - vdb_write_start

        if isinstance(ingestor, BatchIngestor):
            error_rows = ingestor.get_error_rows(dataset=ingest_results).materialize()
            _write_error_rows(error_rows, cfg.execute_params.runtime_metrics_dir)

        detection_summary = estimate_processed_pages(lancedb_uri, lancedb_table)
        evaluation_summary, evaluation_secs = evaluate_lancedb_metrics(
            cfg.evaluation,
            lancedb_uri=lancedb_uri,
            lancedb_table=lancedb_table,
            embed_model_name=str(cfg.embed_params.model_name or "nvidia/llama-nemotron-embed-1b-v2"),
            embed_invoke_url=cfg.embed_params.embed_invoke_url,
            embed_api_key=cfg.embed_params.api_key,
            hybrid=cfg.hybrid,
        )
        total_secs = time.perf_counter() - ingest_start

        from nemo_retriever.utils.detection_summary import collect_detection_summary_from_lancedb

        detection_payload = collect_detection_summary_from_lancedb(lancedb_uri, lancedb_table)
        detection_summary_file = persist_detection_summary_artifact(
            detection_summary_file=cfg.artifacts.detection_summary_file,
            detection_payload=detection_payload,
        )

        processed_pages = (
            detection_payload.get("pages_seen")
            if isinstance(detection_payload, dict) and detection_payload.get("pages_seen") is not None
            else detection_summary
        )

        report = RunReport(
            run_mode="batch",
            input_path=str(input_path),
            input_type=cfg.input_type,
            evaluation_mode=cfg.evaluation.evaluation_mode,
            metrics=RunMetrics(
                input_files=len(input_files) or None,
                input_pages=resolved_input_pages,
                processed_pages=processed_pages,
                rows_processed=len(ingest_local_results),
                ingest_secs=ingest_secs,
                materialize_secs=materialize_secs,
                vdb_write_secs=vdb_write_secs,
                evaluation_secs=evaluation_secs,
                total_secs=total_secs,
            ),
            evaluation=evaluation_summary,
            detection_summary=detection_payload,
            artifacts=RunArtifacts(
                detection_summary_file=detection_summary_file,
                lancedb_uri=lancedb_uri,
                lancedb_table=lancedb_table,
            ),
            extras={"file_patterns": file_patterns},
        )
        return persist_run_report_artifacts(
            report,
            runtime_metrics_dir=cfg.execute_params.runtime_metrics_dir,
            prefix=cfg.execute_params.runtime_metrics_prefix,
        )
    finally:
        shutdown_ray_safely()


def render_batch_run_report(report: RunReport, *, hybrid: bool) -> None:
    _ = hybrid
    render_run_report(report)


def run_batch(
    *,
    create_params: IngestorCreateParams | None = None,
    ingest_params: IngestExecuteParams | None = None,
) -> object:
    return run_mode_ingest(run_mode="batch", create_params=create_params, ingest_params=ingest_params)
