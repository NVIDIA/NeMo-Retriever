# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map completion-callable failures onto the error hierarchy the agent branches on.

Why this module exists
----------------------
:class:`~nemo_agent.llm.callable_backend.CallableLLMBackend` receives an injected
completion callable and therefore cannot know which client library raised. The
agent, however, *does* branch on the distinction: ``loop.py`` records
:class:`ContextLimitError` / :class:`ContentPolicyError` as expected outcomes,
and ``selection_agent.py`` retries a shrunken candidate list only when the run
failed with a context limit. Flattening every failure to a bare
:class:`LLMCallError` silently disables both.

Deliberately HTTP-library-free
------------------------------
This package must import neither ``requests`` nor ``httpx``. An exception that
carries a response is therefore **duck-typed** on ``.status_code`` / ``.text`` /
``.headers`` / ``.url`` — a surface both ``requests.Response`` and
``httpx.Response`` satisfy.

Sole entry point is :func:`classify_call_exception`, which is **total**: it
always returns an :class:`LLMCallError` subclass and never raises.

Caveat on message content
-------------------------
Some clients embed the raw endpoint URL and the full response body in the
exception message. :func:`~nemo_agent.llm.helpers.excerpt` and
:func:`~nemo_agent.llm.helpers.redact_url` bound and scrub the message this
module *builds*, but the original text survives on ``__cause__``, which
``loop.py`` logs with ``exc_info=True``. That is tidiness, not containment.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Mapping, Optional, Tuple

from .errors import ContentPolicyError, ContextLimitError, LLMCallError, RateLimitError
from .helpers import excerpt, redact_url, redact_urls_in_text

logger = logging.getLogger(__name__)

#: ``Retry-After`` values above this are treated as unusable; the base class's
#: configured sleep is a better answer than a 15-minute stall.
_RETRY_AFTER_MAX_S = 300.0

_CONTEXT_LIMIT_CODES = frozenset({"context_length_exceeded", "string_above_max_length"})
_CONTENT_POLICY_CODES = frozenset({"content_filter", "content_policy_violation"})

_CONTENT_POLICY_MARKERS = (
    "content filter",
    "content_filter",
    "content policy",
    "content_policy",
    "content management policy",
    "guardrail",
    "responsible ai",
)


# ----------------------------------------------------------------------
# Pure helpers. None of these log — their callers do.
# ----------------------------------------------------------------------


def _header(headers: Optional[Mapping[str, Any]], name: str) -> Optional[Any]:
    """Case-insensitive header lookup that works on any mapping."""
    if not headers:
        return None
    for key, value in headers.items():
        if str(key).lower() == name:
            return value
    return None


def _parse_retry_after(headers: Optional[Mapping[str, Any]]) -> Optional[float]:
    """Seconds from a ``Retry-After`` header, or None when unusable.

    Accepts delta-seconds and RFC-9110 HTTP-date. Returns None for ``<= 0`` and
    for implausibly large values: the base class accepts any finite ``>= 0`` and
    caps at its own ceiling, so ``Retry-After: 0`` would burn the whole
    rate-limit budget instantly and a skewed date would stall every retry.
    """
    raw = _header(headers, "retry-after")
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    try:
        seconds = float(text)
    except ValueError:
        try:
            when = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        if when is None:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - datetime.now(timezone.utc)).total_seconds()

    if seconds <= 0 or seconds > _RETRY_AFTER_MAX_S:
        return None
    return seconds


def _error_fields(body_json: Any) -> Tuple[str, str]:
    """``(code, type)`` from an OpenAI-style error envelope, lowercased; ``("", "")`` if absent."""
    if not isinstance(body_json, dict):
        return "", ""
    error = body_json.get("error")
    if not isinstance(error, dict):
        return "", ""
    code = str(error.get("code") or "").strip().lower()
    type_ = str(error.get("type") or "").strip().lower()
    return code, type_


def _looks_like_context_limit(lowered: str) -> bool:
    """Prose fallback shared by the response and no-response paths.

    Provider wording drifts, so these markers are a heuristic layered *under* the
    structured ``error.code`` check — never a replacement for it.
    """
    return (
        "contextwindowexceedederror" in lowered
        or ("context" in lowered and "window" in lowered)
        or ("context" in lowered and "reduce" in lowered)
        or "maximum context length" in lowered
        or "is longer than the maximum model length" in lowered
        or "please reduce the length" in lowered
        # Prompt so long the completion budget went negative — a context-overflow
        # symptom rather than a literal context-window message.
        or ("max_tokens must be at least 1" in lowered and "got -" in lowered)
    )


def _looks_like_content_policy(lowered: str) -> bool:
    return any(marker in lowered for marker in _CONTENT_POLICY_MARKERS)


# ----------------------------------------------------------------------
# Response duck-typing. Every accessor is total.
# ----------------------------------------------------------------------


def _status_code(response: Any) -> Optional[int]:
    """Status code from a duck-typed response, or None when there isn't one.

    Compares ``is None`` and NEVER truthiness: ``requests.Response.__bool__``
    returns ``status_code < 400``, so every 4xx/5xx response object — exactly the
    ones this module exists to classify — is falsy.
    """
    if response is None:
        return None
    try:
        return int(getattr(response, "status_code", None))
    except (TypeError, ValueError):
        return None


def _response_text(response: Any) -> str:
    # getattr INSIDE the try: `.text` is a property that decodes `.content` and can
    # itself raise (chunked-encoding errors, "content already consumed", ...).
    try:
        value = getattr(response, "text", None)
        return "" if value is None else str(value)
    except Exception:
        return ""


def _response_json(response: Any) -> Any:
    """Best-effort decode; None when the body is not JSON.

    Broad on purpose: ``.json()`` raises a ``ValueError`` subclass, but the
    ``.content`` read underneath it can raise non-``ValueError``s.
    """
    getter = getattr(response, "json", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception:
        return None


def _response_headers(response: Any) -> Mapping[str, Any]:
    try:
        headers = getattr(response, "headers", None)
        return headers if isinstance(headers, Mapping) else {}
    except Exception:
        return {}


def _response_url(response: Any) -> str:
    try:
        return str(getattr(response, "url", "") or "")
    except Exception:
        return ""


def _safe_str(exc: BaseException) -> str:
    """``str(exc)`` that cannot itself raise.

    Not paranoia: a custom exception with a broken ``__str__`` would otherwise
    defeat this module's totality guarantee at the one moment it matters most —
    while something is already going wrong.
    """
    try:
        return str(exc)
    except Exception:
        return f"<unprintable {type(exc).__name__}>"


# ----------------------------------------------------------------------
# Classification.
# ----------------------------------------------------------------------


def classify_http_error(
    status_code: int,
    body_text: str,
    body_json: Any,
    headers: Optional[Mapping[str, Any]],
    url: str,
) -> LLMCallError:
    """Map an HTTP error response to the exception class the agent branches on.

    Structured ``error.code`` / ``error.type`` are checked first because they are
    stable; prose markers are a fallback because provider wording drifts.
    ``body_json`` may be None (an error body is not always JSON) — every prose
    check runs against ``body_text``, which is a superset of it.

    Pure: an unclassified error is returned as a plain ``LLMCallError`` and the
    caller decides whether to log it.
    """
    # ``body_text`` is scrubbed for the message only; the marker matching below
    # runs against the raw text so redaction can never cost a classification.
    message = f"HTTP {status_code} from {redact_url(url)}: {excerpt(redact_urls_in_text(body_text))}"

    if status_code == 429:
        return RateLimitError(message, retry_after=_parse_retry_after(headers))

    code, type_ = _error_fields(body_json)
    if code in _CONTEXT_LIMIT_CODES or type_ in _CONTEXT_LIMIT_CODES:
        return ContextLimitError(message)
    if code in _CONTENT_POLICY_CODES or type_ in _CONTENT_POLICY_CODES:
        return ContentPolicyError(message)

    lowered = str(body_text or "").lower()
    if _looks_like_context_limit(lowered):
        return ContextLimitError(message)
    if _looks_like_content_policy(lowered):
        return ContentPolicyError(message)

    return LLMCallError(message)


def classify_prose_error(message: str, detail: str) -> LLMCallError:
    """Classify from an exception's text alone, for failures carrying no response.

    ``message`` is what the returned exception says; ``detail`` is the raw text
    the markers are matched against. They are separate arguments so a caller can
    scrub/bound the former without narrowing the latter.
    """
    lowered = str(detail or "").lower()
    if _looks_like_context_limit(lowered):
        return ContextLimitError(message)
    if _looks_like_content_policy(lowered):
        return ContentPolicyError(message)
    return LLMCallError(message)


def classify_call_exception(exc: Exception, *, context: str = "completion callable failed") -> LLMCallError:
    """Total exception -> :class:`LLMCallError` subclass. NEVER raises.

    Classification quality degrades gracefully by design: a failure carrying a
    response object gets full HTTP classification; anything else falls back to
    prose matching and, failing that, to a plain ``LLMCallError``. In-process
    backends land in the latter buckets, which is acceptable — the agent treats
    an unclassified failure as a terminal call error, which it is.
    """
    try:
        return _classify(exc, context)
    except Exception:
        logger.warning("LLM error classification failed; degrading to LLMCallError.", exc_info=True)
        # ``_safe_str`` again, not ``str(exc)``: a raising ``__str__`` is one of the
        # few things that can land us here, so re-reading it would raise from the
        # handler itself. Scrubbed like every other path — a degraded message is
        # still a message we surface.
        return LLMCallError(f"{context}: {redact_urls_in_text(_safe_str(exc))}")


def _classify(exc: Exception, context: str) -> LLMCallError:
    if isinstance(exc, LLMCallError):
        # Already one of ours (a backend-shaped callable). Re-wrapping would erase
        # the subclass the agent branches on.
        return exc

    detail = _safe_str(exc)
    response = getattr(exc, "response", None)
    status = _status_code(response)
    if status is None:
        # No response to inspect: timeouts, transport errors, retries-exhausted
        # wrappers, and every in-process backend failure. Prose is all we have.
        #
        # The message is scrubbed but ``detail`` is passed through raw: the
        # message is what we surface, while ``detail`` is only ever matched
        # against markers, and narrowing it could lose a classification.
        return classify_prose_error(f"{context}: {excerpt(redact_urls_in_text(detail))}", detail)

    return classify_http_error(
        status,
        _response_text(response),
        _response_json(response),
        _response_headers(response),
        _response_url(response),
    )
