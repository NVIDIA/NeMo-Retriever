# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight in-memory pub/sub for SSE streaming.

Each document (or job) can have zero or more subscriber queues.  When an
event is published it is fanned out to all connected queues and stored in
a per-key ring buffer so reconnecting clients can replay missed events.
If no subscribers exist when the event is published the event is still
buffered (so a late subscriber gets it) but not delivered to anyone.

Backpressure model
------------------
Each subscription is a **bounded** ``asyncio.Queue`` (default 8192 events).
``publish()`` always uses ``put_nowait`` so a slow subscriber can never
block worker processes from completing batches.

Two layered defenses keep a slow consumer from collapsing the stream:

1. **Per-subscription event-type filter.**  A subscription may declare a
   ``frozenset`` of event types it is interested in; events of other types
   never even hit the queue.

2. **Priority-drop load shedding.**  When the queue is full, a configurable
   set of low-priority event types (``page_complete``) are silently dropped.
   Terminal events (``job_complete``, ``document_complete``) are always
   preserved.  If a terminal event arrives on a full queue, the queue is
   drained and a sentinel is injected.

Resumable streams
-----------------
Every published event is annotated with a monotonic ``seq`` integer
(unique within this process) and stored in a fixed-size per-key ring
buffer (default 1024 events).  When a client reconnects with the
``Last-Event-ID`` header the SSE generator replays all events with
``seq > last_event_id`` before pulling new ones from the queue.

Supports both per-document and multi-document (session) subscriptions so
a single SSE connection can receive events for many documents at once.
"""

from __future__ import annotations

import asyncio
import collections
import itertools
import logging
import threading
from typing import Any, Iterable, Literal

logger = logging.getLogger(__name__)


_DEFAULT_MAXSIZE = 8192
_DEFAULT_BUFFER_SIZE = 4096
_OVERFLOW_EVENT = {"event": "stream_overflow", "reason": "subscriber_too_slow"}

# How `publish()` reacts when a subscriber's queue is full:
#
# - ``drop_low_priority`` (default): synchronous, never blocks the
#   publisher.  Watermark + priority drop preserve terminal events; if
#   even a terminal event hits a full queue, ``stream_overflow`` is
#   signalled.  Lowest latency, lossy under sustained overload.
# - ``backpressure``: ``publish()`` becomes async and ``await``s
#   ``Queue.put()`` with a configurable timeout.  The publisher (the
#   worker callback thread) blocks, which back-propagates 503s to
#   ingest.  No events dropped until the timeout expires.
# - ``block``: same as ``backpressure`` but waits forever — zero loss
#   at the cost of unbounded latency.
OverflowPolicy = Literal["drop_low_priority", "backpressure", "block"]

# Event types that are safe to silently drop under back-pressure.  These
# carry progress information that a recovering client can reconstruct
# from the REST status endpoints; losing one does not change the final
# result of a job.  Terminal events (``job_complete``, ``document_complete``)
# are intentionally NOT in this set.
_DEFAULT_PRIORITY_DROP: frozenset[str] = frozenset(
    {
        "page_complete",
    }
)

# Fraction of queue capacity above which low-priority events are shed
# proactively, so headroom is always reserved for terminal-class events.
# At 0.75 a 8192-slot queue starts dropping ``page_complete`` when more
# than ~6100 events are already buffered, leaving 2000+ slots free for
# terminal events even under sustained burst load.
_PRIORITY_DROP_WATERMARK: float = 0.75


# Process-wide monotonic event sequence number.  ``itertools.count`` is
# thread-safe in CPython for ``next()`` calls (single bytecode op).
_seq_counter = itertools.count(1)


def _next_seq() -> int:
    return next(_seq_counter)


class _Subscription:
    """A single subscriber's bounded queue with overflow detection.

    The public ``queue`` attribute is a vanilla ``asyncio.Queue`` so that
    SSE generators can ``await queue.get()`` exactly as before.  All writes
    go through :meth:`put` which performs filtering, priority-drop load
    shedding, and overflow handling.
    """

    __slots__ = (
        "queue",
        "maxsize",
        "_overflowed",
        "_event_types",
        "_priority_drop",
        "_watermark",
        "_dropped_count",
    )

    def __init__(
        self,
        maxsize: int = _DEFAULT_MAXSIZE,
        *,
        event_types: frozenset[str] | None = None,
        priority_drop: frozenset[str] | None = None,
        watermark_fraction: float = _PRIORITY_DROP_WATERMARK,
    ) -> None:
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self.maxsize = maxsize
        self._overflowed = False
        # ``None`` means "accept every event"; a frozenset means "only accept
        # events whose type is in this set".  Filtering happens BEFORE the
        # queue is touched so an interest-mismatch never costs queue space.
        self._event_types = event_types
        self._priority_drop = priority_drop or _DEFAULT_PRIORITY_DROP
        # Pre-computed integer threshold for the proactive watermark drop.
        # ``max(1, ...)`` keeps very small queues sensible (the test path
        # uses maxsize=4).
        self._watermark = max(1, int(maxsize * watermark_fraction))
        self._dropped_count = 0

    @property
    def overflowed(self) -> bool:
        return self._overflowed

    @property
    def dropped_count(self) -> int:
        """Number of events silently shed under back-pressure (cumulative)."""
        return self._dropped_count

    def _filter(self, event_type: str | None) -> bool:
        """Return True if this subscription accepts the given event type."""
        if self._event_types is None:
            return True
        return event_type in self._event_types

    def put(self, event: dict[str, Any]) -> bool:
        """Try to enqueue ``event`` synchronously (drop-on-overflow path).

        Returns ``True`` when the event has been accepted *or* deliberately
        dropped (filter mismatch / priority-drop shed under load), and
        ``False`` only when this call has just put the subscription into
        the irrecoverable overflowed state.  Returning ``True`` for a
        deliberate drop is by design: the publisher should not interpret a
        load-shed as a failure.
        """
        if self._overflowed:
            return False

        event_type = event.get("event")

        # 1) Per-subscription event-type filter.  An event the subscriber
        #    explicitly opted out of never even reaches the queue.
        if not self._filter(event_type):
            return True

        # 2) Watermark-based proactive shedding.  When the queue is already
        #    crowded, low-priority events are dropped EARLY so that
        #    terminal events arriving later still have room to enqueue.
        #    Without this guard, a sustained burst of `page_complete`
        #    events fills the queue right up to the hard cap and then
        #    `job_complete` is the unlucky event that trips overflow.
        is_droppable = event_type in self._priority_drop
        if is_droppable and self.queue.qsize() >= self._watermark:
            self._dropped_count += 1
            return True

        # 3) Try to enqueue.
        try:
            self.queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            pass

        # 4) Last-chance load shedding for low-priority events that race
        #    past the watermark check (e.g. between the qsize() probe and
        #    the put_nowait()).  Silently absorbed.
        if is_droppable:
            self._dropped_count += 1
            return True

        # 5) Terminal-class event arrived on a full queue.  Mark overflow
        #    and signal the consumer so it can reconnect with
        #    ``Last-Event-ID`` and resume from the per-key replay buffer.
        self._overflowed = True
        self._drain_and_signal()
        return False

    async def aput(self, event: dict[str, Any], *, timeout: float | None) -> bool:
        """Async enqueue with optional timeout (back-pressure path).

        ``timeout=None`` means wait forever (``block`` policy).  A finite
        timeout means ``backpressure`` policy: wait up to *timeout*
        seconds for the consumer to drain enough room.  If the wait
        expires, fall back to the same overflow signal as :meth:`put` so
        the consumer learns it should reconnect.
        """
        if self._overflowed:
            return False

        event_type = event.get("event")
        if not self._filter(event_type):
            return True

        # Try the cheap path first.  This also lets us avoid touching
        # the loop scheduler when the queue has free space.
        try:
            self.queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            pass

        try:
            if timeout is None:
                await self.queue.put(event)
            else:
                await asyncio.wait_for(self.queue.put(event), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            # Back-pressure exhausted: the consumer is genuinely stuck.
            # Fall back to the overflow signal so the client reconnects.
            self._overflowed = True
            self._drain_and_signal()
            return False

    def _drain_and_signal(self) -> None:
        """Empty the queue and inject the overflow sentinel."""
        try:
            while True:
                self.queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            self.queue.put_nowait(dict(_OVERFLOW_EVENT))
        except asyncio.QueueFull:
            pass


class EventBus:
    """Per-key fan-out event bus backed by bounded ``asyncio.Queue`` instances.

    Keys are usually ``document_id`` or ``job_id`` strings.  The bus does not
    care which — events published under a key go to every subscription for
    that key.
    """

    def __init__(
        self,
        *,
        default_maxsize: int = _DEFAULT_MAXSIZE,
        buffer_size: int = _DEFAULT_BUFFER_SIZE,
        overflow_policy: OverflowPolicy = "drop_low_priority",
        publish_timeout_s: float = 30.0,
    ) -> None:
        self._subscribers: dict[str, list[_Subscription]] = {}
        self._buffers: dict[str, collections.deque[dict[str, Any]]] = {}
        self._buffers_lock = threading.Lock()
        self._default_maxsize = default_maxsize
        self._buffer_size = buffer_size
        self._overflow_policy: OverflowPolicy = overflow_policy
        self._publish_timeout_s = publish_timeout_s

    @property
    def overflow_policy(self) -> OverflowPolicy:
        return self._overflow_policy

    # ------------------------------------------------------------------
    # Subscription
    # ------------------------------------------------------------------

    def subscribe(
        self,
        document_id: str,
        *,
        maxsize: int | None = None,
        event_types: Iterable[str] | None = None,
        priority_drop: Iterable[str] | None = None,
    ) -> asyncio.Queue[dict[str, Any]]:
        """Subscribe to events for a single document/job.

        Parameters
        ----------
        document_id
            Key (document_id or job_id) the subscriber is interested in.
        maxsize
            Per-subscription queue size.  Defaults to the bus default.
        event_types
            If set, only events of these types reach the subscriber's
            queue.  Strongly recommended for high-throughput subscribers
            that only care about a subset of events — every filtered-out
            event saves a queue slot and an SSE write.
        priority_drop
            Override the default set of event types that may be silently
            dropped under back-pressure (default:
            ``{"page_complete"}``).
        """
        sub = _Subscription(
            maxsize=maxsize or self._default_maxsize,
            event_types=frozenset(event_types) if event_types is not None else None,
            priority_drop=frozenset(priority_drop) if priority_drop is not None else None,
        )
        self._subscribers.setdefault(document_id, []).append(sub)
        return sub.queue

    def subscribe_many(
        self,
        document_ids: list[str],
        *,
        maxsize: int | None = None,
        event_types: Iterable[str] | None = None,
        priority_drop: Iterable[str] | None = None,
    ) -> asyncio.Queue[dict[str, Any]]:
        """Subscribe to events for multiple documents on a single queue.

        See :meth:`subscribe` for parameter semantics.  All documents share
        a single underlying ``_Subscription``, so the filter and queue
        size apply across the whole session.
        """
        sub = _Subscription(
            maxsize=maxsize or self._default_maxsize,
            event_types=frozenset(event_types) if event_types is not None else None,
            priority_drop=frozenset(priority_drop) if priority_drop is not None else None,
        )
        for doc_id in document_ids:
            self._subscribers.setdefault(doc_id, []).append(sub)
        return sub.queue

    def unsubscribe(self, document_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        subs = self._subscribers.get(document_id)
        if not subs:
            return
        self._subscribers[document_id] = [s for s in subs if s.queue is not queue]
        if not self._subscribers[document_id]:
            del self._subscribers[document_id]

    def unsubscribe_many(
        self,
        document_ids: list[str],
        queue: asyncio.Queue[dict[str, Any]],
    ) -> None:
        for doc_id in document_ids:
            self.unsubscribe(doc_id, queue)

    # ------------------------------------------------------------------
    # Replay buffer
    # ------------------------------------------------------------------

    def replay(self, document_ids: list[str], *, after_seq: int) -> list[dict[str, Any]]:
        """Return buffered events for *document_ids* whose ``seq > after_seq``.

        Returned events are sorted by ``seq`` so a multi-key replay still
        produces a coherent stream.  Events too old to be in the buffer are
        silently lost; the client should fall back to polling the REST
        endpoints in that case.
        """
        out: list[dict[str, Any]] = []
        with self._buffers_lock:
            for key in document_ids:
                buf = self._buffers.get(key)
                if not buf:
                    continue
                for evt in buf:
                    seq = evt.get("seq")
                    if isinstance(seq, int) and seq > after_seq:
                        out.append(evt)
        out.sort(key=lambda e: e.get("seq", 0))
        return out

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish(self, document_id: str, event: dict[str, Any]) -> None:
        """Fan ``event`` out to every subscriber and store it for replay.

        Mutates ``event`` to add a ``seq`` field if not already present.
        Honors :attr:`overflow_policy` — under ``backpressure`` or
        ``block`` modes, the call may take an extended time waiting for
        slow subscribers to drain.  This is the entire point of those
        modes: back-pressure here propagates up the worker callback
        thread to the worker pool, eventually 503-ing fresh ingest.
        """
        if "seq" not in event:
            event["seq"] = _next_seq()

        # Buffer first so even an event with no subscribers is replayable.
        with self._buffers_lock:
            buf = self._buffers.get(document_id)
            if buf is None:
                buf = collections.deque(maxlen=self._buffer_size)
                self._buffers[document_id] = buf
            buf.append(event)

        subs = self._subscribers.get(document_id, [])
        if not subs:
            return

        if self._overflow_policy == "drop_low_priority":
            for sub in subs:
                ok = sub.put(event)
                if not ok:
                    logger.warning(
                        "SSE subscription for %s overflowed on terminal-class "
                        "event %r (maxsize=%d, dropped=%d before overflow) — "
                        "signalling stream_overflow; client should reconnect "
                        "with Last-Event-ID to resume from the replay buffer",
                        document_id,
                        event.get("event"),
                        sub.maxsize,
                        sub.dropped_count,
                    )
            return

        # backpressure / block — await each subscriber's queue with the
        # appropriate timeout.  We await sequentially so a slow consumer
        # naturally throttles ALL upstream production.  Subscribers are
        # rare (one per active SSE client) so the latency is bounded.
        timeout: float | None = None if self._overflow_policy == "block" else self._publish_timeout_s
        for sub in subs:
            ok = await sub.aput(event, timeout=timeout)
            if not ok:
                logger.warning(
                    "SSE subscription for %s back-pressure timeout on event %r "
                    "(maxsize=%d, timeout=%ss); falling back to stream_overflow",
                    document_id,
                    event.get("event"),
                    sub.maxsize,
                    timeout,
                )

    def has_subscribers(self, document_id: str) -> bool:
        return bool(self._subscribers.get(document_id))
