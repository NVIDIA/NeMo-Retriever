# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""REST endpoints for querying the provenance event log.

Provides paginated listing, per-event detail, and aggregated summary
views suitable for driving an auditing / provenance UI.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query, Request

from nemo_retriever.service.db.repository import Repository
from nemo_retriever.service.models.responses import (
    EventBulkAcknowledgeRequest,
    EventBulkDeleteRequest,
    EventCategorySummary,
    EventLogEntry,
    EventLogList,
    EventMutationResponse,
    EventSummaryResponse,
    EventUpdateRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["events"])


def _record_to_entry(rec) -> EventLogEntry:
    """Convert an :class:`EventRecord` to the REST response model."""
    return EventLogEntry(
        id=rec.id,
        timestamp=rec.timestamp,
        job_id=rec.job_id,
        document_id=rec.document_id,
        source_file=rec.source_file,
        page_number=rec.page_number,
        category=rec.category,
        severity=rec.severity.value if hasattr(rec.severity, "value") else str(rec.severity),
        outcome=rec.outcome.value if hasattr(rec.outcome, "value") else str(rec.outcome),
        stage=rec.stage,
        summary=rec.summary,
        detail=rec.detail,
        stack_trace=rec.stack_trace,
        endpoint=rec.endpoint,
        request_id=rec.request_id,
        extra_json=rec.extra_json,
        created_at=rec.created_at,
    )


@router.get(
    "/events",
    response_model=EventLogList,
    summary="Paginated listing of provenance events",
    responses={
        200: {
            "description": "A page of events matching the filters.",
        },
    },
)
async def list_events(
    request: Request,
    job_id: str | None = Query(default=None, description="Filter by job ID."),
    document_id: str | None = Query(default=None, description="Filter by document ID."),
    category: str | None = Query(default=None, description="Filter by event category."),
    severity: str | None = Query(default=None, description="Filter by severity (info, warning, error)."),
    outcome: str | None = Query(default=None, description="Filter by outcome (failed, recovered, in_progress)."),
    since: str | None = Query(default=None, description="ISO-8601 lower bound on timestamp (inclusive)."),
    until: str | None = Query(default=None, description="ISO-8601 upper bound on timestamp (inclusive)."),
    limit: int = Query(default=100, ge=1, le=1000, description="Page size."),
    offset: int = Query(default=0, ge=0, description="Number of events to skip."),
) -> EventLogList:
    repo: Repository = request.app.state.repository

    records = await asyncio.to_thread(
        repo.list_events,
        job_id=job_id,
        document_id=document_id,
        category=category,
        severity=severity,
        outcome=outcome,
        since=since,
        until=until,
        limit=limit,
        offset=offset,
    )

    entries = [_record_to_entry(r) for r in records]
    return EventLogList(
        events=entries,
        limit=limit,
        offset=offset,
        returned=len(entries),
    )


@router.get(
    "/events/summary",
    response_model=EventSummaryResponse,
    summary="Aggregated event counts by category",
)
async def event_summary(
    request: Request,
    job_id: str | None = Query(default=None, description="Scope summary to a single job."),
    severity: str | None = Query(default=None, description="Filter by severity before aggregating."),
) -> EventSummaryResponse:
    repo: Repository = request.app.state.repository

    counts = await asyncio.to_thread(
        repo.count_events_by_category,
        job_id=job_id,
        severity=severity,
    )

    categories = [EventCategorySummary(category=cat, count=cnt) for cat, cnt in counts.items()]
    total = sum(c.count for c in categories)

    return EventSummaryResponse(categories=categories, total=total)


@router.get(
    "/events/{event_id}",
    response_model=EventLogEntry,
    summary="Retrieve a single provenance event by ID",
    responses={
        404: {"description": "Event not found."},
    },
)
async def get_event(request: Request, event_id: str) -> EventLogEntry:
    repo: Repository = request.app.state.repository

    record = await asyncio.to_thread(repo.get_event, event_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Event {event_id!r} not found.")

    return _record_to_entry(record)


@router.patch(
    "/events/{event_id}",
    response_model=EventMutationResponse,
    summary="Update an event's outcome (e.g. mark as recovered/acknowledged)",
    responses={404: {"description": "Event not found."}},
)
async def update_event(
    request: Request,
    event_id: str,
    body: EventUpdateRequest,
) -> EventMutationResponse:
    repo: Repository = request.app.state.repository

    valid_outcomes = {"failed", "recovered", "in_progress"}
    if body.outcome not in valid_outcomes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid outcome {body.outcome!r}. Must be one of: {valid_outcomes}",
        )

    found = await asyncio.to_thread(repo.update_event_outcome, event_id, body.outcome)
    if not found:
        raise HTTPException(status_code=404, detail=f"Event {event_id!r} not found.")

    return EventMutationResponse(affected=1, action="updated")


@router.delete(
    "/events/{event_id}",
    response_model=EventMutationResponse,
    summary="Delete a single provenance event",
    responses={404: {"description": "Event not found."}},
)
async def delete_event(request: Request, event_id: str) -> EventMutationResponse:
    repo: Repository = request.app.state.repository

    found = await asyncio.to_thread(repo.delete_event, event_id)
    if not found:
        raise HTTPException(status_code=404, detail=f"Event {event_id!r} not found.")

    return EventMutationResponse(affected=1, action="deleted")


@router.post(
    "/events/acknowledge",
    response_model=EventMutationResponse,
    summary="Bulk-acknowledge events (set outcome to recovered)",
)
async def acknowledge_events(
    request: Request,
    body: EventBulkAcknowledgeRequest,
) -> EventMutationResponse:
    repo: Repository = request.app.state.repository

    valid_outcomes = {"failed", "recovered", "in_progress"}
    if body.outcome not in valid_outcomes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid outcome {body.outcome!r}. Must be one of: {valid_outcomes}",
        )

    count = await asyncio.to_thread(
        repo.bulk_update_event_outcome,
        body.event_ids,
        body.outcome,
    )
    return EventMutationResponse(affected=count, action="updated")


@router.post(
    "/events/delete",
    response_model=EventMutationResponse,
    summary="Bulk-delete events by IDs or filters",
)
async def bulk_delete_events(
    request: Request,
    body: EventBulkDeleteRequest,
) -> EventMutationResponse:
    repo: Repository = request.app.state.repository

    if body.event_ids is None and body.category is None and body.severity is None and body.outcome is None:
        raise HTTPException(
            status_code=400,
            detail="At least one filter (event_ids, category, severity, outcome) is required.",
        )

    count = await asyncio.to_thread(
        repo.bulk_delete_events,
        event_ids=body.event_ids,
        category=body.category,
        severity=body.severity,
        outcome=body.outcome,
    )
    return EventMutationResponse(affected=count, action="deleted")
