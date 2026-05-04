# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SSE streaming endpoints for real-time document processing events.

Three endpoints are provided:

- ``GET  /v1/ingest/stream/{document_id}`` — single-document stream (legacy)
- ``POST /v1/ingest/stream`` — session-level stream for multiple documents
  on a single SSE connection (preferred by the CLI client)
- ``POST /v1/ingest/stream/jobs`` — job-level stream for tracking pages
  across one or more jobs

All three honour the standard SSE ``Last-Event-ID`` header.  Generators use
a subscribe-first-then-replay pattern: the subscription is created BEFORE
replaying buffered events so no events published during the replay window
are lost.  Events that appear in both the replay and the live queue are
deduplicated by their monotonic ``seq`` number.  Each frame includes an
``id: <seq>`` line suitable for the browser ``EventSource`` API to track
and re-send on reconnect.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nemo_retriever.service.db.repository import Repository
from nemo_retriever.service.event_bus import EventBus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stream"])


# ------------------------------------------------------------------
# Per-endpoint event-type interest sets
# ------------------------------------------------------------------
#
# Each SSE endpoint declares the set of event types its subscribers
# actually consume.  The EventBus uses these as a hard filter so chatty
# per-page events that the endpoint doesn't forward (``metrics_update``,
# ``page_complete``, ``document_complete``) never even reach the
# subscriber's queue.  This is the principal mechanism that keeps
# ``stream_overflow`` from being triggered under normal load.

# Doc-level (single-document) endpoint — caller wants the whole story
# for one document, so accept everything.
_DOC_STREAM_EVENT_TYPES: frozenset[str] | None = None

# Multi-doc / session endpoint — caller needs status transitions and
# per-page completion + (optionally) per-page content; aggregate metrics
# don't help a session-level consumer.
_SESSION_STREAM_BASE_EVENTS: frozenset[str] = frozenset(
    {
        "page_complete",
        "document_complete",
        "status_change",
        "stream_overflow",
    }
)

# Job-level endpoint — caller is interested in job life-cycle events
# only (plus optionally the full per-page content payloads when content
# was requested).  Per-model metrics and per-page progress events are
# server-side-only details for this endpoint.
_JOB_STREAM_BASE_EVENTS: frozenset[str] = frozenset(
    {
        "job_started",
        "job_complete",
        "status_change",
        "stream_overflow",
    }
)


# ------------------------------------------------------------------
# Shared SSE formatting
# ------------------------------------------------------------------


def _format_event(event: dict, include_content: bool) -> str | None:
    """Return the wire-format SSE frame for *event*, or ``None`` to skip."""
    event_type = event.get("event", "message")
    if event_type == "page_result" and not include_content:
        return None
    seq = event.get("seq")
    id_line = f"id: {seq}\n" if isinstance(seq, int) else ""
    data = json.dumps(event)
    return f"{id_line}event: {event_type}\ndata: {data}\n\n"


def _parse_last_event_id(value: str | None) -> int:
    """Parse the ``Last-Event-ID`` header into an int (default 0)."""
    if not value:
        return 0
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return 0


async def _single_doc_generator(
    event_bus: EventBus,
    document_id: str,
    *,
    after_seq: int = 0,
    include_content: bool = True,
) -> AsyncIterator[str]:
    """Yield SSE frames for a single document until ``document_complete``.

    Uses subscribe-first-then-replay to guarantee no events are lost
    between initial buffering and the live subscription.
    """
    queue = event_bus.subscribe(document_id)
    replayed_max_seq = after_seq
    try:
        for evt in event_bus.replay([document_id], after_seq=after_seq):
            frame = _format_event(evt, include_content)
            if frame is not None:
                yield frame
            seq = evt.get("seq", 0)
            if isinstance(seq, int) and seq > replayed_max_seq:
                replayed_max_seq = seq

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                yield f"event: keepalive\ndata: {json.dumps({'pending': 1})}\n\n"
                continue

            seq = event.get("seq", 0)
            if isinstance(seq, int) and seq <= replayed_max_seq:
                continue

            event_type = event.get("event", "message")
            frame = _format_event(event, include_content)
            if frame is not None:
                yield frame

            if event_type in ("document_complete", "stream_overflow"):
                break
    finally:
        event_bus.unsubscribe(document_id, queue)


async def _multi_doc_generator(
    event_bus: EventBus,
    document_ids: list[str],
    *,
    after_seq: int = 0,
    include_content: bool = False,
    repository: Repository | None = None,
) -> AsyncIterator[str]:
    """Yield SSE frames for *all* listed documents on a single connection.

    Uses subscribe-first-then-replay to guarantee no events are lost
    between initial buffering and the live subscription.
    """
    wanted = set(_SESSION_STREAM_BASE_EVENTS)
    if include_content:
        wanted.add("page_result")
    queue = event_bus.subscribe_many(document_ids, event_types=wanted)
    replayed_max_seq = after_seq
    pending = set(document_ids)
    stale_keepalives = 0
    try:
        for evt in event_bus.replay(document_ids, after_seq=after_seq):
            frame = _format_event(evt, include_content)
            if frame is not None:
                yield frame
            seq = evt.get("seq", 0)
            if isinstance(seq, int) and seq > replayed_max_seq:
                replayed_max_seq = seq
            event_type = evt.get("event", "message")
            doc_id = evt.get("document_id")
            if event_type == "document_complete" and doc_id in pending:
                pending.discard(doc_id)
            elif event_type == "status_change" and evt.get("status") == "failed":
                if doc_id in pending:
                    pending.discard(doc_id)

        while pending:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=_STALE_POLL_INTERVAL_S)
            except asyncio.TimeoutError:
                stale_keepalives += 1
                if stale_keepalives >= 1 and repository is not None:
                    resolved = await _poll_terminal_documents(repository, pending)
                    for doc_id in resolved:
                        pending.discard(doc_id)
                        synth = {"event": "document_complete", "document_id": doc_id, "synthetic": True}
                        frame = _format_event(synth, include_content)
                        if frame is not None:
                            yield frame
                    if not pending:
                        break
                yield f"event: keepalive\ndata: {json.dumps({'pending': len(pending)})}\n\n"
                continue

            seq = event.get("seq", 0)
            if isinstance(seq, int) and seq <= replayed_max_seq:
                continue

            stale_keepalives = 0
            event_type = event.get("event", "message")

            if event_type == "stream_overflow":
                frame = _format_event(event, include_content)
                if frame is not None:
                    yield frame
                break

            frame = _format_event(event, include_content)
            if frame is not None:
                yield frame

            doc_id = event.get("document_id")
            if event_type == "document_complete" and doc_id in pending:
                pending.discard(doc_id)
            elif event_type == "status_change" and event.get("status") == "failed":
                if doc_id in pending:
                    pending.discard(doc_id)

        yield f"event: session_complete\ndata: {json.dumps({'completed': len(document_ids)})}\n\n"
    finally:
        event_bus.unsubscribe_many(document_ids, queue)


async def _poll_terminal_documents(repo: Repository, pending: set[str]) -> list[str]:
    """Check DB for documents that reached terminal state without an SSE event."""
    resolved: list[str] = []
    for doc_id in list(pending):
        try:
            doc = await asyncio.to_thread(repo.get_document, doc_id)
            if doc is None:
                continue
            if doc.processing_status in ("complete", "failed", "cancelled"):
                resolved.append(doc_id)
        except Exception:
            pass
    return resolved


_STALE_POLL_INTERVAL_S = 30.0


async def _job_stream_generator(
    event_bus: EventBus,
    job_ids: list[str],
    *,
    after_seq: int = 0,
    include_content: bool = False,
    repository: Repository | None = None,
) -> AsyncIterator[str]:
    """Yield SSE frames for one or more jobs until every job completes.

    Uses subscribe-first-then-replay to guarantee no events are lost
    between initial buffering and the live subscription.  Events that
    appear in both the replay buffer and the live queue are deduplicated
    by their monotonic ``seq`` number.
    """
    wanted = set(_JOB_STREAM_BASE_EVENTS)
    if include_content:
        wanted.add("page_result")
    queue = event_bus.subscribe_many(job_ids, event_types=wanted)
    replayed_max_seq = after_seq
    pending = set(job_ids)
    stale_keepalives = 0
    try:
        for evt in event_bus.replay(job_ids, after_seq=after_seq):
            frame = _format_event(evt, include_content)
            if frame is not None:
                yield frame
            seq = evt.get("seq", 0)
            if isinstance(seq, int) and seq > replayed_max_seq:
                replayed_max_seq = seq
            event_type = evt.get("event", "message")
            jid = evt.get("job_id")
            if event_type == "job_complete" and jid in pending:
                pending.discard(jid)
            elif event_type == "status_change" and evt.get("status") == "failed":
                if jid in pending:
                    pending.discard(jid)

        while pending:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=_STALE_POLL_INTERVAL_S)
            except asyncio.TimeoutError:
                stale_keepalives += 1
                if stale_keepalives >= 1 and repository is not None:
                    resolved = await _poll_terminal_jobs(repository, pending)
                    for jid in resolved:
                        pending.discard(jid)
                        synth = {"event": "job_complete", "job_id": jid, "synthetic": True}
                        frame = _format_event(synth, include_content)
                        if frame is not None:
                            yield frame
                    if not pending:
                        break
                yield f"event: keepalive\ndata: {json.dumps({'pending': len(pending)})}\n\n"
                continue

            seq = event.get("seq", 0)
            if isinstance(seq, int) and seq <= replayed_max_seq:
                continue

            stale_keepalives = 0
            event_type = event.get("event", "message")

            if event_type == "stream_overflow":
                frame = _format_event(event, include_content)
                if frame is not None:
                    yield frame
                break

            frame = _format_event(event, include_content)
            if frame is not None:
                yield frame

            jid = event.get("job_id")
            if event_type == "job_complete" and jid in pending:
                pending.discard(jid)
            elif event_type == "status_change" and event.get("status") == "failed":
                if jid in pending:
                    pending.discard(jid)

        yield f"event: session_complete\ndata: {json.dumps({'completed': len(job_ids)})}\n\n"
    finally:
        event_bus.unsubscribe_many(job_ids, queue)


async def _poll_terminal_jobs(repo: Repository, pending: set[str]) -> list[str]:
    """Check DB for jobs that reached terminal state without an SSE event.

    Jobs that return None (unknown to the server — e.g. all uploads failed)
    are also treated as resolved to prevent infinite hangs.
    """
    resolved: list[str] = []
    for jid in list(pending):
        try:
            job = await asyncio.to_thread(repo.get_job, jid)
            if job is None:
                logger.warning("Job %s not found in DB during stale poll — treating as resolved", jid[:8])
                resolved.append(jid)
                continue
            if job.processing_status in ("complete", "failed", "cancelled"):
                resolved.append(jid)
        except Exception:
            pass
    return resolved


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.get(
    "/ingest/stream/{document_id}",
    summary="Stream processing events for a single document via SSE",
)
async def stream_document_events(
    request: Request,
    document_id: str,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    repo: Repository = request.app.state.repository
    doc = repo.get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    event_bus: EventBus = request.app.state.event_bus
    return StreamingResponse(
        _single_doc_generator(
            event_bus,
            document_id,
            after_seq=_parse_last_event_id(last_event_id),
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


class StreamSessionRequest(BaseModel):
    """Body for the multi-document SSE endpoint."""

    document_ids: list[str] = Field(..., min_length=1)
    include_content: bool = Field(
        default=False,
        description="If true, include `page_result` events carrying the full per-page extracted content.",
    )


@router.post(
    "/ingest/stream",
    summary="Stream processing events for multiple documents on a single SSE connection",
)
async def stream_session_events(
    request: Request,
    body: StreamSessionRequest,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    repo: Repository = request.app.state.repository
    for doc_id in body.document_ids:
        doc = repo.get_document(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

    event_bus: EventBus = request.app.state.event_bus
    return StreamingResponse(
        _multi_doc_generator(
            event_bus,
            body.document_ids,
            after_seq=_parse_last_event_id(last_event_id),
            include_content=body.include_content,
            repository=repo,
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


class JobStreamRequest(BaseModel):
    """Body for the job-level SSE endpoint."""

    job_ids: list[str] = Field(..., min_length=1)
    include_content: bool = Field(
        default=False,
        description="If true, include `page_result` events carrying the full per-page extracted content.",
    )


@router.post(
    "/ingest/stream/jobs",
    summary="Stream processing events for one or more jobs on a single SSE connection",
)
async def stream_job_events(
    request: Request,
    body: JobStreamRequest,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    # Subscribe immediately without validating job existence.
    # Jobs are created on-the-fly as the first page for each file arrives,
    # so the SSE stream must be open BEFORE uploads begin to avoid missing
    # early events.  The EventBus is purely in-memory and doesn't need
    # the job row to exist.
    event_bus: EventBus = request.app.state.event_bus
    repo: Repository = request.app.state.repository
    return StreamingResponse(
        _job_stream_generator(
            event_bus,
            body.job_ids,
            after_seq=_parse_last_event_id(last_event_id),
            include_content=body.include_content,
            repository=repo,
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
