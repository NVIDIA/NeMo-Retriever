# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
import time

from pydantic import BaseModel, ConfigDict, Field

from nemo_retriever.params import EmbedParams
from nemo_retriever.params import ExtractParams
from nemo_retriever.params import IngestExecuteParams
from nemo_retriever.params import IngestorCreateParams
from nemo_retriever.params import VdbUploadParams
from nemo_retriever.utils.detection_summary import (
    print_detection_summary,
    print_pages_per_second,
    write_detection_summary,
)
from nemo_retriever.utils.input_files import resolve_input_files, resolve_input_patterns

from .executor import run_mode_ingest
from .reports import (
    RunArtifactConfig,
    RunArtifacts,
    RunEvaluationConfig,
    RunMetrics,
    RunReport,
    persist_run_report_artifacts,
)
from .shared import (
    DEFAULT_LANCEDB_TABLE,
    DEFAULT_LANCEDB_URI,
    count_lancedb_rows,
    ensure_lancedb_table,
    estimate_processed_pages,
    evaluate_lancedb_metrics,
    resolve_input_pages,
)


class _RunnerConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FusedPipelineConfig(_RunnerConfigModel):
    input_path: str
    input_type: str = "pdf"
    file_patterns: list[str] = Field(default_factory=list)
    create_params: IngestorCreateParams = Field(default_factory=IngestorCreateParams)
    execute_params: IngestExecuteParams = Field(default_factory=IngestExecuteParams)
    extract_params: ExtractParams = Field(default_factory=ExtractParams)
    embed_params: EmbedParams = Field(default_factory=EmbedParams)
    vdb_upload_params: VdbUploadParams = Field(default_factory=VdbUploadParams)
    evaluation: RunEvaluationConfig = Field(default_factory=RunEvaluationConfig)
    artifacts: RunArtifactConfig = Field(default_factory=RunArtifactConfig)


def _resolve_file_patterns(cfg: FusedPipelineConfig) -> list[str]:
    if cfg.file_patterns:
        return list(cfg.file_patterns)
    return resolve_input_patterns(Path(cfg.input_path), cfg.input_type)


def run_fused_pipeline(cfg: FusedPipelineConfig) -> RunReport:
    try:
        from nemo_retriever.ingestor import create_ingestor

        if cfg.input_type != "pdf":
            raise ValueError("Fused mode currently supports only pdf input_type.")

        input_path = Path(cfg.input_path).expanduser().resolve()
        input_files = resolve_input_files(input_path, cfg.input_type)
        file_patterns = _resolve_file_patterns(cfg)
        lancedb_uri = str(
            Path(cfg.vdb_upload_params.lancedb.lancedb_uri or cfg.artifacts.lancedb_uri or DEFAULT_LANCEDB_URI)
            .expanduser()
            .resolve()
        )
        lancedb_table = str(
            cfg.vdb_upload_params.lancedb.table_name or cfg.artifacts.lancedb_table or DEFAULT_LANCEDB_TABLE
        )

        ensure_lancedb_table(lancedb_uri, lancedb_table)
        ingestor = create_ingestor(run_mode="fused", params=cfg.create_params)
        ingestor = (
            ingestor.files(file_patterns)
            .extract(cfg.extract_params)
            .embed(cfg.embed_params)
            .vdb_upload(cfg.vdb_upload_params)
        )

        ingest_start = time.perf_counter()
        ingestor.ingest(params=cfg.execute_params)
        ingest_secs = time.perf_counter() - ingest_start

        from nemo_retriever.utils.detection_summary import collect_detection_summary_from_lancedb

        detection_payload = collect_detection_summary_from_lancedb(lancedb_uri, lancedb_table)
        if cfg.artifacts.detection_summary_file is not None:
            detection_path = Path(cfg.artifacts.detection_summary_file).expanduser().resolve()
            write_detection_summary(detection_path, detection_payload)
            detection_summary_file = str(detection_path)
        else:
            detection_summary_file = None

        evaluation_summary, evaluation_secs = evaluate_lancedb_metrics(
            cfg.evaluation,
            lancedb_uri=lancedb_uri,
            lancedb_table=lancedb_table,
            embed_model_name=str(cfg.embed_params.model_name or "nemo_retriever_v1"),
            embed_invoke_url=cfg.embed_params.embed_invoke_url,
            embed_api_key=cfg.embed_params.api_key,
            hybrid=bool(cfg.vdb_upload_params.lancedb.hybrid),
        )
        total_secs = ingest_secs + evaluation_secs

        report = RunReport(
            run_mode="fused",
            input_path=str(input_path),
            input_type=cfg.input_type,
            evaluation_mode=cfg.evaluation.evaluation_mode,
            metrics=RunMetrics(
                input_files=len(input_files) or None,
                input_pages=resolve_input_pages(cfg.input_type, input_files),
                processed_pages=(
                    detection_payload.get("pages_seen")
                    if isinstance(detection_payload, dict) and detection_payload.get("pages_seen") is not None
                    else estimate_processed_pages(lancedb_uri, lancedb_table)
                ),
                rows_processed=count_lancedb_rows(lancedb_uri, lancedb_table),
                ingest_secs=ingest_secs,
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
        try:
            import ray

            ray.shutdown()
        except Exception:
            pass


def render_fused_run_report(report: RunReport) -> None:
    if report.detection_summary is not None:
        print_detection_summary(report.detection_summary)
    if report.metrics.ingest_secs is not None:
        print_pages_per_second(report.metrics.processed_pages, report.metrics.ingest_secs)
    if report.evaluation.metrics:
        print(f"\n{report.evaluation.label} metrics:")
        for key, value in sorted(report.evaluation.metrics.items()):
            print(f"  {key}: {value:.4f}")


def run_fused(
    *,
    create_params: IngestorCreateParams | None = None,
    ingest_params: IngestExecuteParams | None = None,
) -> object:
    return run_mode_ingest(run_mode="fused", create_params=create_params, ingest_params=ingest_params)
