# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convenience helper for writing provenance events to SQLite.

All instrumentation sites (routers, processing pool, middleware) call
:func:`record_event` instead of constructing an :class:`EventRecord` and
calling ``repo.insert_event()`` directly.  This keeps the call sites
terse and centralises any future logic (sampling, throttling, enrichment).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from nemo_retriever.service.db.repository import Repository
from nemo_retriever.service.models.event_log import EventOutcome, EventRecord, EventSeverity

logger = logging.getLogger(__name__)


def record_event(
    repo: Repository,
    *,
    category: str,
    severity: EventSeverity = EventSeverity.ERROR,
    outcome: EventOutcome = EventOutcome.FAILED,
    summary: str = "",
    detail: str = "",
    stack_trace: str = "",
    stage: str = "",
    endpoint: str = "",
    job_id: str | None = None,
    document_id: str | None = None,
    source_file: str = "",
    page_number: int | None = None,
    request_id: str = "",
    extra: dict[str, Any] | None = None,
) -> EventRecord:
    """Build an :class:`EventRecord`, persist it, and return it.

    This is the **single instrumentation entry-point** for the entire
    provenance system.  Every error path, recovery, and lifecycle marker
    should flow through here.

    Parameters
    ----------
    repo
        An active :class:`Repository` instance (thread-local connection).
    category
        One of the :class:`EventCategory` values (passed as a string so
        callers don't need to import the enum).
    severity / outcome
        Classify the event for the UI.
    summary
        One-line human-readable description shown in lists/tables.
    detail
        Extended context or full error message.
    stack_trace
        Python traceback string (``traceback.format_exc()``).
    stage
        Pipeline stage name or endpoint step (e.g. ``"page_elements"``).
    endpoint
        Originating HTTP endpoint (e.g. ``"/v1/query"``) or
        ``"pipeline"`` for worker-process events.
    job_id / document_id / source_file / page_number
        Correlation keys linking the event to tracked entities.
    request_id
        Groups events from the same HTTP request or batch.
    extra
        Arbitrary dict serialised to ``extra_json``.  Use for
        future provenance metadata without schema changes.

    Returns
    -------
    EventRecord
        The persisted record (useful for tests and chaining).
    """
    extra_json = json.dumps(extra) if extra else "{}"

    event = EventRecord(
        category=category,
        severity=severity,
        outcome=outcome,
        summary=summary,
        detail=detail,
        stack_trace=stack_trace,
        stage=stage,
        endpoint=endpoint,
        job_id=job_id,
        document_id=document_id,
        source_file=source_file,
        page_number=page_number,
        request_id=request_id,
        extra_json=extra_json,
    )

    try:
        repo.insert_event(event)
    except Exception:
        logger.exception(
            "Failed to persist provenance event (category=%s, summary=%s)",
            category,
            summary[:120],
        )

    return event
