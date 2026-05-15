# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process event bus for SSE push notifications.

Provides a lightweight pub/sub mechanism so that internal state changes
(job completed, job failed) can be pushed to all connected SSE clients
without polling.

Usage::

    bus = get_event_bus()
    sub_id, queue = bus.subscribe()
    ...
    bus.publish_sync({"type": "completed", "id": "item1"})
    ...
    bus.unsubscribe(sub_id)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class EventBus:
    """Broadcast events to zero or more ``asyncio.Queue`` subscribers."""

    def __init__(self) -> None:
        self._subscribers: dict[int, asyncio.Queue[dict[str, Any]]] = {}
        self._counter: int = 0

    def subscribe(self) -> tuple[int, asyncio.Queue[dict[str, Any]]]:
        """Create a new subscription that receives all events."""
        self._counter += 1
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=4096)
        self._subscribers[self._counter] = q
        logger.info("EventBus: subscriber %d registered (total=%d)", self._counter, len(self._subscribers))
        return self._counter, q

    def unsubscribe(self, sub_id: int) -> None:
        removed = self._subscribers.pop(sub_id, None)
        if removed:
            logger.debug("EventBus: subscriber %d removed", sub_id)

    def publish_sync(self, event: dict[str, Any]) -> None:
        """Publish *event* to all subscribers (non-blocking).

        Uses ``put_nowait`` so this can be called from synchronous code
        running on the event loop thread.  Subscribers whose queues are
        full are silently dropped.
        """
        delivered = 0
        dead: list[int] = []
        for sub_id, q in self._subscribers.items():
            try:
                q.put_nowait(event)
                delivered += 1
            except asyncio.QueueFull:
                dead.append(sub_id)
        for sub_id in dead:
            self._subscribers.pop(sub_id, None)
            logger.warning("EventBus: dropped subscriber %d (queue full)", sub_id)
        if not self._subscribers:
            logger.debug(
                "EventBus: event id=%s published but no subscribers",
                event.get("id", "?"),
            )

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


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
