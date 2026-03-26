# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nemo_retriever.params import RunMode


class _ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunArtifacts(_ReportModel):
    runtime_metrics_dir: str | None = None
    report_file: str | None = None
    runtime_summary_file: str | None = None
    detection_summary_file: str | None = None
    log_file: str | None = None
    lancedb_uri: str | None = None
    lancedb_table: str | None = None


class RunMetrics(_ReportModel):
    input_files: int | None = None
    input_pages: int | None = None
    processed_pages: int | None = None
    rows_processed: int | None = None
    ingest_secs: float | None = None
    materialize_secs: float | None = None
    vdb_write_secs: float | None = None
    evaluation_secs: float | None = None
    total_secs: float | None = None
    pages_per_sec_ingest: float | None = None
    rows_per_sec_ingest: float | None = None


class EvaluationSummary(_ReportModel):
    label: str = "Recall"
    query_count: int | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


class RunEvaluationConfig(_ReportModel):
    evaluation_mode: str = "recall"
    query_csv: str | None = None
    recall_match_mode: str = "pdf_page"
    beir_loader: str | None = None
    beir_dataset_name: str | None = None
    beir_split: str = "test"
    beir_query_language: str | None = None
    beir_doc_id_field: str = "pdf_basename"
    beir_ks: tuple[int, ...] = (1, 3, 5, 10)
    reranker: bool = False
    reranker_model_name: str = "nvidia/llama-nemotron-rerank-1b-v2"


class RunArtifactConfig(_ReportModel):
    lancedb_uri: str = "lancedb"
    lancedb_table: str = "nv-ingest"
    detection_summary_file: str | None = None
    log_file: str | None = None


class RunReport(_ReportModel):
    run_mode: RunMode
    input_path: str
    input_type: str
    evaluation_mode: str
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    evaluation: EvaluationSummary = Field(default_factory=EvaluationSummary)
    detection_summary: dict[str, Any] | None = None
    runtime_summary: dict[str, Any] = Field(default_factory=dict)
    artifacts: RunArtifacts = Field(default_factory=RunArtifacts)
    extras: dict[str, Any] = Field(default_factory=dict)


def normalize_metric_key(key: str) -> str:
    metric = str(key).strip().lower()
    return metric.replace("@", "_").replace("-", "_")


def _safe_ratio(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator in {None, 0, 0.0}:
        return None
    try:
        value = float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return round(value, 2)


def canonical_pages(report: RunReport) -> int | None:
    if report.metrics.processed_pages is not None:
        return report.metrics.processed_pages
    return report.metrics.input_pages


def flatten_report_metrics(report: RunReport) -> dict[str, Any]:
    flat: dict[str, Any] = {
        "files": report.metrics.input_files,
        "pages": canonical_pages(report),
        "input_files": report.metrics.input_files,
        "input_pages": report.metrics.input_pages,
        "processed_pages": report.metrics.processed_pages,
        "rows_processed": report.metrics.rows_processed,
        "ingest_secs": report.metrics.ingest_secs,
        "materialize_secs": report.metrics.materialize_secs,
        "vdb_write_secs": report.metrics.vdb_write_secs,
        "evaluation_secs": report.metrics.evaluation_secs,
        "total_secs": report.metrics.total_secs,
        "pages_per_sec_ingest": report.metrics.pages_per_sec_ingest,
        "rows_per_sec_ingest": report.metrics.rows_per_sec_ingest,
        "evaluation_query_count": report.evaluation.query_count,
    }
    for key, value in report.evaluation.metrics.items():
        flat[normalize_metric_key(key)] = value
    return flat


def project_summary_metrics(report: RunReport) -> dict[str, Any]:
    flat = flatten_report_metrics(report)
    return {
        "pages": flat.get("pages"),
        "ingest_secs": flat.get("ingest_secs"),
        "pages_per_sec_ingest": flat.get("pages_per_sec_ingest"),
        "recall_5": flat.get("recall_5"),
        "ndcg_10": flat.get("ndcg_10"),
    }


def build_runtime_summary(report: RunReport) -> dict[str, Any]:
    summary = dict(report.runtime_summary)
    summary.update(
        {
            "run_mode": report.run_mode,
            "input_type": report.input_type,
            "input_files": report.metrics.input_files,
            "input_pages": report.metrics.input_pages,
            "processed_pages": report.metrics.processed_pages,
            "rows_processed": report.metrics.rows_processed,
            "ingest_secs": report.metrics.ingest_secs,
            "materialize_secs": report.metrics.materialize_secs,
            "vdb_write_secs": report.metrics.vdb_write_secs,
            "evaluation_secs": report.metrics.evaluation_secs,
            "elapsed_secs": report.metrics.total_secs,
            "pages_per_sec_ingest": report.metrics.pages_per_sec_ingest,
            "rows_per_sec_ingest": report.metrics.rows_per_sec_ingest,
        }
    )
    return summary


def update_metric_derivatives(report: RunReport) -> RunReport:
    updated_metrics = report.metrics.model_copy(
        update={
            "pages_per_sec_ingest": (
                report.metrics.pages_per_sec_ingest
                if report.metrics.pages_per_sec_ingest is not None
                else _safe_ratio(canonical_pages(report), report.metrics.ingest_secs)
            ),
            "rows_per_sec_ingest": (
                report.metrics.rows_per_sec_ingest
                if report.metrics.rows_per_sec_ingest is not None
                else _safe_ratio(report.metrics.rows_processed, report.metrics.ingest_secs)
            ),
        }
    )
    updated_report = report.model_copy(update={"metrics": updated_metrics})
    return updated_report.model_copy(update={"runtime_summary": build_runtime_summary(updated_report)})


def persist_run_report_artifacts(
    report: RunReport, *, runtime_metrics_dir: str | None, prefix: str | None
) -> RunReport:
    if runtime_metrics_dir is None:
        return update_metric_derivatives(report)

    root = Path(runtime_metrics_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_prefix = str(prefix or "run")
    report_path = root / f"{run_prefix}.run_report.json"
    runtime_summary_path = root / f"{run_prefix}.runtime.summary.json"

    updated_report = update_metric_derivatives(
        report.model_copy(
            update={
                "artifacts": report.artifacts.model_copy(
                    update={
                        "runtime_metrics_dir": str(root),
                        "report_file": str(report_path),
                        "runtime_summary_file": str(runtime_summary_path),
                    }
                )
            }
        )
    )

    report_path.write_text(
        json.dumps(updated_report.model_dump(mode="python"), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    runtime_summary_path.write_text(
        json.dumps(updated_report.runtime_summary, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return updated_report
