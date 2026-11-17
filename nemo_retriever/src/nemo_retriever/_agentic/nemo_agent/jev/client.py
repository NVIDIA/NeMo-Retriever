# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client for System One decision endpoints (``POST .../v1/systemone``).

A request carries a ``state`` and a map of typed questions (``noul``,
``choice``, ``score``); the response carries one typed answer per question.
Usage is attributed exactly like LLM usage: to the ambient query id and stage
bindings from :mod:`nemo_agent.llm.usage`, so callers pop it per query.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from copy import deepcopy
from typing import Any, Dict, Mapping, Optional, Sequence

import httpx
from pydantic import BaseModel, ConfigDict

from ..llm.helpers import resolve_api_key
from ..llm.usage import UNSET_QUERY, UNSET_STAGE, deep_merge_usage, get_query_id, get_stage

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class JevCallError(RuntimeError):
    """A System One request failed after retries or returned an unusable body."""


class JevContextLengthError(JevCallError):
    """The request exceeded the endpoint's context window; a smaller request may succeed."""


class JevClientConfig(BaseModel):
    """Connection settings for a System One endpoint.

    ``url`` is the full endpoint URL (including ``/v1/systemone``). ``api_key``
    accepts a literal key or an ``"os.environ/VAR_NAME"`` reference.
    """

    model_config = ConfigDict(extra="forbid")

    url: str
    model: str = "jev-latest"
    api_key: Optional[str] = None
    timeout_s: float = 60.0
    max_retries: int = 4
    retry_sleep_s: float = 2.0


def noul(instructions: Any, *, true: Any = None, false: Any = None) -> Dict[str, Any]:
    """Build a yes/no question; the answer's ``noul`` is P(yes)."""
    question: Dict[str, Any] = {"type": "noul", "instructions": instructions}
    if true is not None or false is not None:
        question["criteria"] = {"true": true, "false": false}
    return question


def choice(instructions: Any, options: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a question that picks one of ``options`` (option name -> description)."""
    if len(options) < 2:
        raise ValueError("a choice question needs at least two options")
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


def score(instructions: Any, levels: Sequence[Any]) -> Dict[str, Any]:
    """Build a question rated against ordered ``levels`` (lowest first)."""
    if not 2 <= len(levels) <= 10:
        raise ValueError("a score question needs between 2 and 10 levels")
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


class JevClient:
    """Async System One client with retry and per-query usage accounting.

    Usage per call is ``{"input_tokens", "output_tokens", "requests",
    "latency_ms"}``, summed per ``(query_id, stage)``. One instance safely
    serves concurrent runs; each call opens its own HTTP connection so the
    client never binds to a particular event loop.
    """

    def __init__(self, config: JevClientConfig, *, transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        self.config = config
        self._api_key = resolve_api_key(config.api_key) or None
        self._transport = transport
        self._usage: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._usage_lock = threading.Lock()

    async def aevaluate(self, state: Any, questions: Mapping[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Ask ``questions`` about ``state``; return the answers keyed like the questions."""
        if not questions:
            return {}
        payload = {"model": self.config.model, "state": state, "questions": dict(questions)}
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        started = time.monotonic()
        body = await self._post(payload, headers)
        answers = body.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(questions):
            raise JevCallError(
                f"System One response is missing answers for {sorted(set(questions) - set(answers or {}))}"
            )
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        self._record_usage(
            {
                "input_tokens": int(usage.get("input_tokens") or 0),
                "output_tokens": int(usage.get("output_tokens") or 0),
                "requests": 1,
                "latency_ms": int((time.monotonic() - started) * 1000),
            }
        )
        return answers

    async def _post(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        last_error = ""
        async with httpx.AsyncClient(timeout=self.config.timeout_s, transport=self._transport) as client:
            for attempt in range(self.config.max_retries + 1):
                try:
                    response = await client.post(self.config.url, json=payload, headers=headers)
                except httpx.HTTPError as e:
                    last_error = f"{type(e).__name__}: {e}"
                else:
                    if response.status_code == 200:
                        try:
                            return response.json()
                        except ValueError as e:
                            raise JevCallError(f"System One returned a non-JSON body: {e}") from e
                    last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                    if "maximum context length" in response.text:
                        # Gateways can wrap an upstream context-length 400 in a 5xx; a retry cannot succeed.
                        raise JevContextLengthError(f"System One request to {self.config.url} failed: {last_error}")
                    if response.status_code not in _RETRYABLE_STATUS:
                        break
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_sleep_s * (2**attempt))
        raise JevCallError(f"System One request to {self.config.url} failed: {last_error}")

    def _record_usage(self, usage: Dict[str, Any]) -> None:
        query_id = get_query_id() or UNSET_QUERY
        stage = get_stage() or UNSET_STAGE
        with self._usage_lock:
            deep_merge_usage(self._usage.setdefault(query_id, {}).setdefault(stage, {}), usage)

    def get_usage(self, query_id: str) -> Dict[str, Any]:
        """Deep copy of one query's ``{stage: usage}`` breakdown."""
        with self._usage_lock:
            return deepcopy(self._usage.get(query_id, {}))

    def pop_query_usage(self, query_id: str) -> Dict[str, Any]:
        """Remove and return one query's ``{stage: usage}`` breakdown (``{}`` if unknown)."""
        with self._usage_lock:
            return self._usage.pop(query_id, {})
