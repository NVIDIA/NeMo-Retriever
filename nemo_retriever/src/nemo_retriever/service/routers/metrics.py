# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metrics endpoints for inspecting ingest activity at job, document, and page level."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from nemo_retriever.service.services.metrics import (
    DocumentMetric,
    IngestMetricsSnapshot,
    JobMetric,
    PageMetric,
    get_metrics,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["metrics"])

_HTML_TEMPLATE = """\
<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><title>{title}</title>
<style>
  body {{ font-family: monospace; background: #1e1e2e; color: #cdd6f4; padding: 2rem; }}
  pre  {{ background: #181825; padding: 1.5rem; border-radius: 8px;
          overflow-x: auto; font-size: 14px; line-height: 1.5; }}
  h1   {{ color: #89b4fa; font-size: 1.4rem; }}
</style>
</head><body><h1>{title}</h1><pre>{payload}</pre></body></html>
"""


def _require_metrics(request: Request):
    m = get_metrics()
    if m is None:
        mode = getattr(request.app.state, "config", None)
        mode_label = mode.mode if mode else "unknown"
        raise HTTPException(
            status_code=404,
            detail=(
                f"IngestMetrics is not available on this pod (mode={mode_label}). "
                "In split topology, metrics are tracked on the gateway pod. "
                "Query the gateway's /v1/ingest/metrics endpoint instead."
            ),
        )
    return m


def _html_or_json(request: Request, model, *, title: str = "Metrics"):
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        payload = json.dumps(
            (
                model.model_dump(mode="json")
                if hasattr(model, "model_dump")
                else [m.model_dump(mode="json") for m in model]
            ),
            indent=2,
        )
        return HTMLResponse(_HTML_TEMPLATE.format(title=title, payload=payload))
    return model


# ------------------------------------------------------------------
# GET /v1/ingest/metrics  — full snapshot
# ------------------------------------------------------------------


@router.get(
    "/ingest/metrics",
    response_model=IngestMetricsSnapshot,
    summary="Full ingest metrics snapshot",
)
async def ingest_metrics(request: Request):
    m = _require_metrics(request)
    return _html_or_json(request, m.snapshot(), title="Ingest Metrics")


# ------------------------------------------------------------------
# GET /v1/ingest/metrics/job/{job_id}  — single job
# ------------------------------------------------------------------


@router.get(
    "/ingest/metrics/job/{job_id}",
    response_model=JobMetric,
    summary="Metrics for a single job",
)
async def ingest_metrics_job(job_id: str, request: Request):
    m = _require_metrics(request)
    job = m.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return _html_or_json(request, job, title=f"Job {job_id[:12]}")


# ------------------------------------------------------------------
# GET /v1/ingest/metrics/job/{job_id}/errors  — error documents in job
# ------------------------------------------------------------------


@router.get(
    "/ingest/metrics/job/{job_id}/errors",
    response_model=list[DocumentMetric],
    summary="Documents with errors for a specific job",
)
async def ingest_metrics_job_errors(job_id: str, request: Request):
    m = _require_metrics(request)
    if m.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    errors = m.get_job_error_documents(job_id)
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        payload = json.dumps([d.model_dump(mode="json") for d in errors], indent=2)
        return HTMLResponse(
            _HTML_TEMPLATE.format(
                title=f"Job {job_id[:12]} — {len(errors)} error(s)",
                payload=payload,
            )
        )
    return errors


# ------------------------------------------------------------------
# GET /v1/ingest/metrics/document/{document_id}  — single document
# ------------------------------------------------------------------


@router.get(
    "/ingest/metrics/document/{document_id}",
    response_model=DocumentMetric,
    summary="Metrics for a single document",
)
async def ingest_metrics_document(document_id: str, request: Request):
    m = _require_metrics(request)
    doc = m.get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
    return _html_or_json(request, doc, title=f"Document {document_id[:12]}")


# ------------------------------------------------------------------
# GET /v1/ingest/metrics/page/{page_id}  — single page
# ------------------------------------------------------------------


@router.get(
    "/ingest/metrics/page/{page_id}",
    response_model=PageMetric,
    summary="Metrics for a single page",
)
async def ingest_metrics_page(page_id: str, request: Request):
    m = _require_metrics(request)
    page = m.get_page(page_id)
    if page is None:
        raise HTTPException(status_code=404, detail=f"Page '{page_id}' not found")
    return _html_or_json(request, page, title=f"Page {page_id[:12]}")
