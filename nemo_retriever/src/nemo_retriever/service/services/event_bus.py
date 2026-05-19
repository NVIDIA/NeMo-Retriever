# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process event bus for SSE push notifications.

Per-job routing: every published event carries a ``job_id`` routing
key. Subscribers register with the ``job_id`` they want to listen to
and only receive events tagged with the matching key. A privileged
"firehose" subscription (``job_id=None``) sees everything — used by
the dashboard's live event log panel.

Usage::

    bus = get_event_bus()
    sub_id, queue = bus.subscribe(job_id="abc-123")
    ...
    bus.publish_sync({"type": "completed", "id": "doc1"}, job_id="abc-123")
    ...
    bus.unsubscribe(sub_id)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class EventBus:
    """Broadcast events to zero or more ``asyncio.Queue`` subscribers.

    Subscribers are keyed by an integer id; each entry stores the
    target ``asyncio.Queue`` and an optional ``job_id`` filter. When
    a publisher calls :meth:`publish_sync`, only subscribers whose
    filter matches the event's ``job_id`` (or whose filter is ``None``)
    receive the event.
    """

    def __init__(self) -> None:
        self._subscribers: dict[int, tuple[asyncio.Queue[dict[str, Any]], Optional[str]]] = {}
        self._counter: int = 0

    def subscribe(self, *, job_id: Optional[str] = None) -> tuple[int, asyncio.Queue[dict[str, Any]]]:
        """Create a new subscription scoped to *job_id*.

        ``job_id=None`` opts into the firehose — every event from
        every job. Used by the gateway dashboard's overview SSE
        endpoint; should not be exposed to untrusted callers.
        """
        self._counter += 1
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=4096)
        self._subscribers[self._counter] = (q, job_id)
        logger.info(
            "EventBus: subscriber %d registered (filter=%s; total=%d)",
            self._counter,
            job_id if job_id is not None else "*",
            len(self._subscribers),
        )
        return self._counter, q

    def unsubscribe(self, sub_id: int) -> None:
        removed = self._subscribers.pop(sub_id, None)
        if removed:
            logger.debug("EventBus: subscriber %d removed", sub_id)

    def publish_sync(self, event: dict[str, Any], *, job_id: Optional[str] = None) -> None:
        """Publish *event* to all matching subscribers (non-blocking).

        ``job_id`` is the routing key. When ``None`` the event is
        delivered only to firehose subscribers (those that themselves
        passed ``job_id=None`` to :meth:`subscribe`) — this is the
        explicit signal for "system event, no per-job context".

        Uses ``put_nowait`` so this can be called from synchronous code
        on the event loop thread. Subscribers whose queues are full
        are dropped.
        """
        delivered = 0
        dead: list[int] = []
        for sub_id, (q, filter_id) in self._subscribers.items():
            if filter_id is not None and filter_id != job_id:
                continue
            try:
                q.put_nowait(event)
                delivered += 1
            except asyncio.QueueFull:
                dead.append(sub_id)
        for sub_id in dead:
            self._subscribers.pop(sub_id, None)
            logger.warning("EventBus: dropped subscriber %d (queue full)", sub_id)
        if delivered == 0 and self._subscribers:
            logger.debug(
                "EventBus: event id=%s job_id=%s published but no matching subscribers",
                event.get("id", "?"),
                job_id,
            )

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def subscribers_for(self, job_id: Optional[str]) -> int:
        """Count subscribers whose filter matches *job_id* (or are firehose)."""
        return sum(1 for _, (_q, f) in self._subscribers.items() if f is None or f == job_id)


# ── Module-level singleton ───────────────────────────────────────────

_instance: EventBus | None = None


def init_event_bus() -> EventBus:
    global _instance
    _instance = EventBus()
    logger.info("Event bus initialised")
    return _instance


def get_event_bus() -> EventBus | None:
    return _instance


def shutdown_event_bus() -> None:
    global _instance
    if _instance is not None:
        logger.info("Event bus shut down (subscribers=%d)", _instance.subscriber_count)
        _instance = None
