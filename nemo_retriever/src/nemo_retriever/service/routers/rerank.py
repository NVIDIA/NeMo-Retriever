# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""POST /v1/rerank — rerank passages against a query via the reranker NIM."""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from nemo_retriever.service.config import RerankerConfig, ServiceConfig
from nemo_retriever.service.db.repository import Repository
from nemo_retriever.service.event_logger import record_event
from nemo_retriever.service.failure_types import EventCategory
from nemo_retriever.service.models.event_log import EventOutcome, EventSeverity
from nemo_retriever.service.models.requests import RerankRequest
from nemo_retriever.service.models.responses import RerankHit, RerankResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rerank"])

_rerank_rr_idx = 0


def _pick_endpoint(csv_url: str) -> str:
    """Round-robin select one URL from a comma-separated NIM endpoint string."""
    global _rerank_rr_idx
    urls = [u.strip() for u in csv_url.split(",") if u.strip()]
    if not urls:
        return csv_url
    if len(urls) == 1:
        return urls[0]
    chosen = urls[_rerank_rr_idx % len(urls)]
    _rerank_rr_idx += 1
    return chosen


# ------------------------------------------------------------------
# Reranking helper (runs in a worker thread)
# ------------------------------------------------------------------


def _rerank_passages(
    query: str,
    passages: list[dict[str, Any]],
    *,
    endpoint: str,
    model_name: str,
    api_key: str,
    top_n: int | None,
) -> list[dict[str, Any]]:
    """Call the reranker NIM and return passages sorted by score.

    Each returned dict is the original passage dict with a
    ``_rerank_score`` key added.  Raises with a descriptive message
    on any failure.
    """
    from nemo_retriever.rerank.rerank import _rerank_via_endpoint

    documents = []
    for i, p in enumerate(passages):
        text = p.get("text")
        if text is None:
            raise ValueError(
                f"Passage at index {i} is missing the required 'text' key. "
                "Every passage dict must contain a 'text' field."
            )
        documents.append(str(text))

    if not documents:
        return []

    try:
        scores = _rerank_via_endpoint(
            query,
            documents,
            endpoint=endpoint,
            model_name=model_name,
            api_key=api_key,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Reranker NIM request failed (endpoint={endpoint!r}, model={model_name!r}): "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if len(scores) != len(passages):
        raise RuntimeError(
            f"Reranker NIM returned {len(scores)} scores for {len(passages)} passages "
            f"(endpoint={endpoint!r}, model={model_name!r}). Response may be malformed."
        )

    ranked = sorted(
        [{"_rerank_score": s, **p} for s, p in zip(scores, passages)],
        key=lambda x: x["_rerank_score"],
        reverse=True,
    )

    if top_n is not None:
        ranked = ranked[:top_n]

    return ranked


# ------------------------------------------------------------------
# Route
# ------------------------------------------------------------------


@router.post(
    "/rerank",
    response_model=RerankResponse,
    summary="Rerank passages against a query using the reranker NIM",
    responses={
        400: {
            "description": "Invalid request (missing text key, empty passages).",
        },
        503: {
            "description": "Reranker NIM is unreachable or returned an error.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Reranker NIM request failed: ConnectionError: ...",
                    }
                }
            },
        },
    },
)
async def rerank(request: Request, body: RerankRequest) -> RerankResponse:
    config: ServiceConfig = request.app.state.config
    reranker_cfg: RerankerConfig = config.reranker

    # --- Validate endpoint ---
    rerank_raw = (config.nim_endpoints.rerank_invoke_url or "").strip()
    rerank_endpoint = _pick_endpoint(rerank_raw) if rerank_raw else ""
    if not rerank_endpoint:
        logger.error("Rerank rejected: nim_endpoints.rerank_invoke_url is not configured")
        raise HTTPException(
            status_code=503,
            detail=(
                "The reranker NIM endpoint is not configured on this server. "
                "Set nim_endpoints.rerank_invoke_url in retriever-service.yaml "
                "(e.g. 'http://localhost:8015')."
            ),
        )

    query_text = body.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text must not be empty or whitespace-only.")

    if not body.passages:
        raise HTTPException(status_code=400, detail="At least one passage is required.")

    for i, p in enumerate(body.passages):
        if "text" not in p:
            raise HTTPException(
                status_code=400,
                detail=f"Passage at index {i} is missing the required 'text' key.",
            )

    model_name = body.model_name or reranker_cfg.model_name
    top_n = body.top_n or reranker_cfg.default_top_n
    api_key = (config.nim_endpoints.api_key or "").strip()

    logger.info(
        "Rerank request: query=%r, %d passages, model=%s, top_n=%d",
        query_text[:80],
        len(body.passages),
        model_name,
        top_n,
    )

    repo: Repository = request.app.state.repository
    req_id = getattr(request.state, "request_id", "")

    # --- Call the reranker NIM ---
    try:
        ranked = await asyncio.to_thread(
            _rerank_passages,
            query_text,
            body.passages,
            endpoint=rerank_endpoint,
            model_name=model_name,
            api_key=api_key,
            top_n=top_n,
        )
    except ValueError as exc:
        record_event(
            repo,
            category=EventCategory.VALIDATION.value,
            severity=EventSeverity.WARNING,
            outcome=EventOutcome.FAILED,
            summary=f"Rerank validation error: {exc}",
            detail=str(exc),
            stage="rerank",
            endpoint="/v1/rerank",
            request_id=req_id,
        )
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        logger.error("Reranking failed: %s", exc)
        record_event(
            repo,
            category=EventCategory.RERANK.value,
            severity=EventSeverity.ERROR,
            outcome=EventOutcome.FAILED,
            summary=f"Reranker NIM failure: {exc}",
            detail=str(exc),
            stack_trace=traceback.format_exc(),
            stage="rerank",
            endpoint="/v1/rerank",
            request_id=req_id,
        )
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Unexpected error during reranking: %s\n%s", exc, tb)
        record_event(
            repo,
            category=EventCategory.RERANK.value,
            severity=EventSeverity.ERROR,
            outcome=EventOutcome.FAILED,
            summary=f"Unexpected reranker error: {type(exc).__name__}: {exc}",
            detail=str(exc),
            stack_trace=tb,
            stage="rerank",
            endpoint="/v1/rerank",
            request_id=req_id,
        )
        raise HTTPException(
            status_code=503,
            detail=(f"Unexpected error while reranking: " f"{type(exc).__name__}: {exc}"),
        )

    # --- Build typed response ---
    results: list[RerankHit] = []
    for hit in ranked:
        metadata_raw = hit.get("metadata", {})
        if isinstance(metadata_raw, str):
            try:
                import ast

                metadata_dict = ast.literal_eval(metadata_raw) if metadata_raw else {}
            except Exception:
                metadata_dict = {"_raw": metadata_raw} if metadata_raw else {}
        elif isinstance(metadata_raw, dict):
            metadata_dict = metadata_raw
        else:
            metadata_dict = {}

        results.append(
            RerankHit(
                rerank_score=float(hit.get("_rerank_score", 0.0)),
                text=str(hit.get("text", "")),
                source=str(hit.get("source", "")),
                page_number=int(hit.get("page_number", 0)),
                pdf_page=str(hit.get("pdf_page", "")),
                pdf_basename=str(hit.get("pdf_basename", "")),
                source_id=str(hit.get("source_id", "")),
                path=str(hit.get("path", "")),
                stored_image_uri=str(hit.get("stored_image_uri", "")),
                content_type=str(hit.get("content_type", "")),
                bbox_xyxy_norm=str(hit.get("bbox_xyxy_norm", "")),
                metadata=metadata_dict,
            )
        )

    logger.info(
        "Rerank complete: %d results returned for query %r",
        len(results),
        query_text[:80],
    )

    return RerankResponse(
        query=query_text,
        results=results,
        total_results=len(results),
        model_name=model_name,
    )
