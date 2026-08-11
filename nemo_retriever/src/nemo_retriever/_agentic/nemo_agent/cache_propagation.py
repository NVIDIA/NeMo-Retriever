# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adaptive propagation pacing for back-to-back LLM calls.

Anthropic prompt-cache writes via Bedrock-via-NIM propagate across cache
replicas with a small lag (empirically ~5-15s on this stack). When a
downstream LLM call needs to read a cache entry written by an immediately
preceding upstream call, the natural inter-call gap may be too short and
the read misses.

:class:`PropagationPacer` lets a caller declare "ensure at least N seconds
between consecutive marked calls" without inserting a blind sleep on every
call: it sleeps only the *remaining* time relative to the most recent mark,
so calls that take longer than ``target_s`` on their own incur no extra
wait at all.

Set ``target_s = 0`` (the default for non-Anthropic providers) to disable
pacing entirely; both ``await_propagation`` and ``mark`` become cheap
no-ops.
"""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, Optional


class PropagationPacer:
    """Maintain a minimum-wall-time gap between consecutive LLM calls.

    Typical usage::

        pacer = PropagationPacer(target_s=30.0)
        for k in topk_list:
            await pacer.await_propagation()
            result = await llm_call(k)
            pacer.mark()
    """

    def __init__(
        self,
        target_s: float,
        *,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.target_s = max(0.0, float(target_s))
        self._last_ts: Optional[float] = None
        self._now = now
        self._sleep = sleep

    @property
    def primed(self) -> bool:
        """True once :meth:`mark` has been called at least once."""
        return self._last_ts is not None

    def reset(self) -> None:
        """Clear the most recent mark (next ``await_propagation`` is a no-op)."""
        self._last_ts = None

    def mark(self) -> None:
        """Record the current time as the latest call's finish timestamp.

        Always call this AFTER the LLM call returns; ``await_propagation``
        reads back from this timestamp on the next iteration.
        """
        if self.target_s <= 0:
            return
        self._last_ts = self._now()

    async def await_propagation(self) -> float:
        """Sleep just long enough to meet the target gap. Returns the seconds slept.

        No-op (returns 0.0) when:
        - ``target_s`` is zero (pacing disabled), OR
        - the pacer has not been marked yet (first call in the sequence), OR
        - the natural gap since the last mark already meets or exceeds
          ``target_s``.
        """
        if self.target_s <= 0 or self._last_ts is None:
            return 0.0
        elapsed = self._now() - self._last_ts
        remaining = self.target_s - elapsed
        if remaining <= 0:
            return 0.0
        await self._sleep(remaining)
        return remaining
