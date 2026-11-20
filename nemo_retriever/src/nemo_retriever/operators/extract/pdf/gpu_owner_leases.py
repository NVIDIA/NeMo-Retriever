# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Bounded producer references retained through final image consumption."""
import threading
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class _Lease:
    charge: int
    count: int
    remaining: set
    reference: object = None


class OwnerLeaseBook:
    def __init__(self, limit_bytes):
        if not isinstance(limit_bytes, int) or isinstance(limit_bytes, bool) or limit_bytes <= 0:
            raise ValueError("positive byte limit required")
        self.limit = limit_bytes
        self._lock = threading.Lock()
        self._entries = {}
        self._waiting = OrderedDict()
        self._acknowledged_pages = 0
        self._next = 0
        self._bytes = self._peak = self._completed = self._rejections = 0

    def reserve(self, charge, pages, request_id=None):
        if (
            not isinstance(charge, int)
            or isinstance(charge, bool)
            or charge <= 0
            or not isinstance(pages, int)
            or isinstance(pages, bool)
            or pages <= 0
        ):
            raise ValueError("positive integer charge and page count required")
        if charge > self.limit:
            raise ValueError("one batch exceeds owner byte limit")
        if request_id is not None and (not isinstance(request_id, str) or not 0 < len(request_id) <= 128):
            raise ValueError("admission request ID must be a nonempty string of at most 128 characters")
        with self._lock:
            if request_id is not None:
                expected = (charge, pages)
                if request_id in self._waiting:
                    if self._waiting[request_id] != expected:
                        raise ValueError("admission request changed its batch charge or page count")
                else:
                    if len(self._waiting) >= 1024:
                        raise RuntimeError("GPU image owner admission queue limit exceeded")
                    self._waiting[request_id] = expected
                first = next(iter(self._waiting))
                ahead = first != request_id
            else:
                ahead = bool(self._waiting)
            if ahead or self._bytes + charge > self.limit:
                self._rejections += 1
                return None
            if request_id is not None:
                self._waiting.pop(request_id)
            self._next += 1
            token = self._next
            self._entries[token] = _Lease(charge, pages, set(range(pages)))
            self._bytes += charge
            self._peak = max(self._peak, self._bytes)
            return token

    def cancel_request(self, request_id):
        """Cancel only a queued request; published images still require ACKs."""
        with self._lock:
            return self._waiting.pop(request_id, None) is not None

    def admission_progress(self):
        with self._lock:
            return self._acknowledged_pages

    def commit(self, token, reference):
        if reference is None:
            raise ValueError("source reference required")
        with self._lock:
            entry = self._entries[token]
            if entry.reference is not None:
                raise RuntimeError("already published")
            entry.reference = reference

    def cancel(self, token):
        with self._lock:
            entry = self._entries[token]
            if entry.reference is not None:
                raise RuntimeError("published source needs consumer acknowledgement")
            self._entries.pop(token)
            self._bytes -= entry.charge

    def acknowledge(self, token, indices):
        indices = tuple(indices)
        if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in indices):
            raise ValueError("page indices must be nonnegative integers")
        released = None
        with self._lock:
            entry = self._entries.get(token)
            if entry is None:
                return False
            if entry.reference is None:
                raise RuntimeError("source not published")
            # Full index domain is retained even after partial acknowledgement.
            if any(i >= entry.count for i in indices):
                raise ValueError("page index outside source batch")
            before = len(entry.remaining)
            entry.remaining.difference_update(indices)
            self._acknowledged_pages += before - len(entry.remaining)
            if not entry.remaining:
                released = self._entries.pop(token)
                self._bytes -= entry.charge
                self._completed += 1
        # Decref outside the lock: Ray/native finalizers may execute here.
        del released
        return True

    def status(self):
        with self._lock:
            return dict(
                limit_bytes=self.limit,
                active_bytes=self._bytes,
                peak_bytes=self._peak,
                active_batches=len(self._entries),
                retained_batches=sum(e.reference is not None for e in self._entries.values()),
                pending_pages=sum(len(e.remaining) for e in self._entries.values()),
                completed_batches=self._completed,
                acknowledged_pages=self._acknowledged_pages,
                queued_requests=len(self._waiting),
                capacity_rejections=self._rejections,
            )
