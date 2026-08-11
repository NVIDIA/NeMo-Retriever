# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The result envelope every LLM backend returns from a completion call."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CompletionResult:
    """Normalized result of one chat-completion call.

    The stable/unstable boundary: anything a consumer needs to *branch* on is a
    typed field with a fixed meaning; :attr:`extra_response_info` is explicitly
    schema-unstable and may only be logged.

    Attributes
    ----------
    message:
        OpenAI-chat-format assistant message dict, directly appendable to the
        conversation history and valid to send back on the next call.
        ``content`` is ``str | None``. ``tool_calls`` (when present) follow the
        OpenAI shape ``{"id", "type", "function": {"name", "arguments"}}`` with
        ``arguments`` kept as the raw JSON *string* — parsing (and the error
        message when parsing fails) belongs to the caller. Exactly one
        assistant message per call.
    finish_reason:
        Normalized finish reason; ``"stop"`` and ``"tool_calls"`` are the two
        values agent loops branch on. Backends map anything unknown to a
        non-empty descriptive string.
    reasoning:
        The model's reasoning trace for this turn, if the backend can extract
        one (see ``helpers.extract_reasoning_from_message``), else ``None``.
    usage:
        Token usage as a JSON-ish nested dict with int leaves, or ``None`` when
        the backend/server does not report usage. No fixed schema is required:
        a minimal backend may return just ``{"prompt_tokens", "completion_tokens",
        "total_tokens"}``; richer backends may nest arbitrarily (cache tiers,
        reasoning tokens, ...). Consumers must tolerate missing keys. The base
        class also merges this into its per-(query, stage) usage tracker.
    extra_response_info:
        Always populated (possibly ``{}``), built best-effort — an exotic
        response must degrade this to ``{}``, never fail a successful call.
        **Backend-defined and schema-unstable: consumers may log it, never
        branch on it.** Anything decision-worthy must be promoted to a typed
        field. (The litellm backend puts the full response dump minus
        ``choices`` here, plus rate-limit headers under ``"ratelimit"``.)
    raw_request:
        The request as sent, as a plain JSON-serializable dict — populated only
        when ``config.capture_raw_io`` is true, else ``None``. MUST be
        credential-redacted by the backend (no ``api_key`` / authorization
        material).
    raw_response:
        Full JSON-serializable dump of the provider response — populated only
        when ``config.capture_raw_io`` is true, else ``None``.
    """

    message: Dict[str, Any]
    finish_reason: str
    reasoning: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    extra_response_info: Dict[str, Any] = field(default_factory=dict)
    raw_request: Optional[Dict[str, Any]] = None
    raw_response: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.message, dict):
            raise TypeError(
                f"CompletionResult.message must be an OpenAI-format assistant message dict, "
                f"got {type(self.message).__name__}"
            )
        if self.message.get("role") != "assistant":
            raise ValueError(
                f"CompletionResult.message must have role='assistant', got {self.message.get('role')!r}"
            )
        if not isinstance(self.finish_reason, str) or not self.finish_reason:
            raise ValueError(
                f"CompletionResult.finish_reason must be a non-empty string, got {self.finish_reason!r}"
            )
        if not isinstance(self.extra_response_info, dict):
            raise TypeError(
                f"CompletionResult.extra_response_info must be a dict, "
                f"got {type(self.extra_response_info).__name__}"
            )
