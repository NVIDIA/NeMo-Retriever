# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GET /v1/ingest_metrics — per-page processing statistics with full
detection breakdowns pulled from the ``processing_metrics`` table via the
existing ``document_id`` foreign key.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from nemo_retriever.service.db.repository import Repository

router = APIRouter(tags=["metrics"])


# ------------------------------------------------------------------
# Response models
# ------------------------------------------------------------------


class LabelCount(BaseModel):
    """Detection count for a single label within a model."""

    label: str
    count: int


class ModelDetections(BaseModel):
    """Per-model detection breakdown for a single page."""

    model_name: str
    invocation_count: int
    detections_count: int
    labels: list[LabelCount] = Field(default_factory=list)


class DetectionsSummary(BaseModel):
    """Nested detections object attached to each page entry."""

    total_detections: int
    models: list[ModelDetections] = Field(default_factory=list)


class PageMetricEntry(BaseModel):
    """One page's processing stats, including timing and detection detail."""

    id: str = Field(description="Composite key: <source_file>:<page_number>")
    source_file: str
    page_number: int
    processing_duration_ms: float
    started_at: str
    completed_at: str
    detections: DetectionsSummary
    job_id: str | None = None
    document_id: str


class FileMetricsSummary(BaseModel):
    """Aggregated stats for a single source file."""

    source_file: str
    total_pages: int
    total_detections: int
    total_duration_ms: float
    avg_duration_ms: float
    detections: DetectionsSummary
    pages: list[PageMetricEntry]


class IngestMetricsResponse(BaseModel):
    """Top-level response for ``GET /v1/ingest_metrics``."""

    total_pages_processed: int
    total_detections: int
    total_duration_ms: float
    avg_duration_per_page_ms: float
    detections: DetectionsSummary
    files: list[FileMetricsSummary]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _build_detections_for_document(repo: Repository, document_id: str) -> DetectionsSummary:
    """Pull ``processing_metrics`` rows for *document_id* and aggregate."""
    metrics = repo.get_metrics(document_id)
    models: list[ModelDetections] = []
    total = 0
    for m in metrics:
        labels_dict: dict[str, int] = {}
        try:
            labels_dict = json.loads(m.counts_by_label_json)
        except (json.JSONDecodeError, TypeError):
            pass

        labels = [LabelCount(label=lbl, count=cnt) for lbl, cnt in sorted(labels_dict.items(), key=lambda x: -x[1])]
        models.append(
            ModelDetections(
                model_name=m.model_name,
                invocation_count=m.invocation_count,
                detections_count=m.detections_count,
                labels=labels,
            )
        )
        total += m.detections_count

    return DetectionsSummary(total_detections=total, models=models)


def _merge_detections(summaries: list[DetectionsSummary]) -> DetectionsSummary:
    """Merge multiple per-page detection summaries into one aggregate."""
    grand_total = 0
    by_model: dict[str, dict[str, Any]] = {}
    for s in summaries:
        grand_total += s.total_detections
        for m in s.models:
            if m.model_name not in by_model:
                by_model[m.model_name] = {
                    "invocation_count": 0,
                    "detections_count": 0,
                    "labels": {},
                }
            agg = by_model[m.model_name]
            agg["invocation_count"] += m.invocation_count
            agg["detections_count"] += m.detections_count
            for lc in m.labels:
                agg["labels"][lc.label] = agg["labels"].get(lc.label, 0) + lc.count

    models = [
        ModelDetections(
            model_name=name,
            invocation_count=vals["invocation_count"],
            detections_count=vals["detections_count"],
            labels=[
                LabelCount(label=lbl, count=cnt) for lbl, cnt in sorted(vals["labels"].items(), key=lambda x: -x[1])
            ],
        )
        for name, vals in sorted(by_model.items())
    ]
    return DetectionsSummary(total_detections=grand_total, models=models)


# ------------------------------------------------------------------
# Endpoint
# ------------------------------------------------------------------


@router.get(
    "/ingest_metrics",
    response_model=IngestMetricsResponse,
    summary="Per-page processing metrics with full detection breakdowns",
)
async def get_ingest_metrics(request: Request) -> JSONResponse:
    repo: Repository = request.app.state.repository

    def _build() -> IngestMetricsResponse:
        logs = repo.get_all_page_processing_logs()

        by_file: dict[str, list[Any]] = {}
        for entry in logs:
            by_file.setdefault(entry.source_file, []).append(entry)

        file_summaries: list[FileMetricsSummary] = []
        grand_pages = 0
        grand_detections = 0
        grand_duration = 0.0
        all_page_detections: list[DetectionsSummary] = []

        for source_file in sorted(by_file):
            entries = sorted(by_file[source_file], key=lambda e: e.page_number)
            file_page_detections: list[DetectionsSummary] = []
            pages: list[PageMetricEntry] = []

            for e in entries:
                page_det = _build_detections_for_document(repo, e.document_id)
                file_page_detections.append(page_det)

                pages.append(
                    PageMetricEntry(
                        id=e.id,
                        source_file=e.source_file,
                        page_number=e.page_number,
                        processing_duration_ms=round(e.processing_duration_ms, 2),
                        started_at=e.started_at,
                        completed_at=e.completed_at,
                        detections=page_det,
                        job_id=e.job_id,
                        document_id=e.document_id,
                    )
                )

            file_det = _merge_detections(file_page_detections)
            all_page_detections.extend(file_page_detections)

            total_dur = sum(p.processing_duration_ms for p in pages)
            n = len(pages)
            file_summaries.append(
                FileMetricsSummary(
                    source_file=source_file,
                    total_pages=n,
                    total_detections=file_det.total_detections,
                    total_duration_ms=round(total_dur, 2),
                    avg_duration_ms=round(total_dur / n, 2) if n else 0.0,
                    detections=file_det,
                    pages=pages,
                )
            )
            grand_pages += n
            grand_detections += file_det.total_detections
            grand_duration += total_dur

        grand_det = _merge_detections(all_page_detections)

        return IngestMetricsResponse(
            total_pages_processed=grand_pages,
            total_detections=grand_detections,
            total_duration_ms=round(grand_duration, 2),
            avg_duration_per_page_ms=round(grand_duration / grand_pages, 2) if grand_pages else 0.0,
            detections=grand_det,
            files=file_summaries,
        )

    payload = await asyncio.to_thread(_build)
    return JSONResponse(
        content=payload.model_dump(),
        media_type="application/json",
    )
