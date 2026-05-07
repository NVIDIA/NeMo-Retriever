# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""POST /v1/ingest/internal_results — webhook callback from the pipeline."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request

from nemo_retriever.service.db.repository import Repository
from nemo_retriever.service.event_bus import EventBus
from nemo_retriever.service.models.page_result import PageResult

logger = logging.getLogger(__name__)

router = APIRouter(tags=["internal"])


def _safe_value(v):
    """Best-effort conversion to JSON-safe type."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, bytes):
        return f"<bytes len={len(v)}>"
    if isinstance(v, (list, tuple)):
        return [_safe_value(i) for i in v]
    if isinstance(v, dict):
        return {str(k): _safe_value(val) for k, val in v.items()}
    return str(v)


@router.post(
    "/ingest/internal_results",
    status_code=200,
    summary="Receive raw results from the WebhookNotifyOperator",
)
async def receive_internal_results(
    request: Request,
) -> dict[str, str]:
    """Called by the ``WebhookNotifyOperator`` (e.g. from a Ray Data DAG).

    Accepts the raw ``list[dict]`` format that the webhook operator POSTs
    (serialised DataFrame rows).  Each row is stored as a page result.

    The ``content_sha256`` header (``X-Content-SHA256``) or a ``path`` field
    in each row is used for document correlation.  If neither is available the
    rows are stored but cannot be associated with a tracked document.
    """
    body = await request.json()
    repo: Repository = request.app.state.repository
    event_bus: EventBus = request.app.state.event_bus

    records: list[dict] = body if isinstance(body, list) else [body]

    stored = 0
    for idx, row in enumerate(records):
        safe_row = {k: _safe_value(v) for k, v in row.items() if k != "bytes"}
        content_json = json.dumps(safe_row)

        doc_id: str | None = None
        sha = row.get("_content_sha256") or request.headers.get("x-content-sha256")
        if sha:
            doc = repo.get_document_by_sha(sha)
            if doc:
                doc_id = doc.id

        if doc_id:
            page = PageResult(
                document_id=doc_id,
                page_number=idx,
                content_json=content_json,
            )
            repo.insert_page_result(page)
            new_count = repo.increment_pages_received(doc_id)

            await event_bus.publish(
                doc_id,
                {
                    "event": "page_complete",
                    "document_id": doc_id,
                    "page_number": idx,
                    "pages_received": new_count,
                    "total_pages": len(records),
                },
            )
            stored += 1
        else:
            logger.debug(
                "internal_results: row %d has no document correlation key, skipping DB storage",
                idx,
            )

    logger.info("internal_results: stored %d/%d rows", stored, len(records))
    return {"status": "ok", "rows_stored": str(stored)}
