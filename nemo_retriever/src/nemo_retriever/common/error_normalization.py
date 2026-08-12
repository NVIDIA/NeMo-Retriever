"""Safe, consistent error rendering for public service surfaces."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

PUBLIC_ERROR_MESSAGE_LIMIT = 2048
_LIST_LIMIT = 5
_SENSITIVE_VALUE = re.compile(
    r"(?i)\b(?:api[_-]?key|token|authorization|password|secret|credential)\s*[:=]\s*(?:bearer\s+)?[^\s,;]+"
)
_BEARER_VALUE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_URL = re.compile(r"https?://[^\s\]\[},;]+")


@dataclass(frozen=True)
class NormalizedError:
    """Bounded, redacted error data safe for APIs, logs, and metrics."""

    type: str
    stage: str | None
    endpoint: str | None
    message: str

    @property
    def summary(self) -> str:
        context = []
        if self.stage:
            context.append(f"stage={self.stage}")
        if self.endpoint:
            context.append(f"endpoint={self.endpoint}")
        prefix = self.type
        if context:
            prefix += " [" + " ".join(context) + "]"
        return f"{prefix}: {self.message}"

    def as_dict(self) -> dict[str, str | None]:
        return {"type": self.type, "stage": self.stage, "endpoint": self.endpoint, "message": self.message}


def normalize_error(value: Any, *, limit: int = PUBLIC_ERROR_MESSAGE_LIMIT) -> NormalizedError:
    """Normalize an exception or error-shaped value without exposing secrets.

    Strings are intentionally handled before all iterable types: a string is one
    message, never a sequence of individual error items.
    """
    normalized = _normalize(value, seen=set())
    return NormalizedError(
        type=_safe_text(normalized.type, limit=128) or "Error",
        stage=_safe_text(normalized.stage, limit=128),
        endpoint=_safe_endpoint(normalized.endpoint),
        message=_safe_text(normalized.message, limit=limit) or "Unknown error",
    )


def _normalize(value: Any, *, seen: set[int]) -> NormalizedError:
    if isinstance(value, str):
        return NormalizedError("Error", None, None, value)
    if value is None:
        return NormalizedError("Error", None, None, "Unknown error")
    if isinstance(value, (list, tuple)):
        return _normalize_sequence(value, seen=seen)
    if isinstance(value, dict):
        return _normalize_mapping(value, seen=seen)
    if isinstance(value, BaseException):
        return _normalize_exception(value, seen=seen)
    return NormalizedError(type(value).__name__, None, None, str(value))


def _normalize_sequence(values: list[Any] | tuple[Any, ...], *, seen: set[int]) -> NormalizedError:
    items = [_normalize(item, seen=seen) for item in values[:_LIST_LIMIT]]
    if not items:
        return NormalizedError("Error", None, None, "Unknown error")
    first = items[0]
    message = "; ".join(item.summary for item in items)
    if len(values) > _LIST_LIMIT:
        message += f" ({len(values) - _LIST_LIMIT} more)"
    return NormalizedError(first.type if len(items) == 1 else "MultipleErrors", first.stage, first.endpoint, message)


def _normalize_mapping(value: dict[Any, Any], *, seen: set[int]) -> NormalizedError:
    error_type = value.get("type") or value.get("error_type") or "Error"
    stage = value.get("stage")
    endpoint = value.get("endpoint") or value.get("url") or value.get("invoke_url")
    message = value.get("message") or value.get("detail")
    status = _status_code(value)
    nested = value.get("error") or value.get("exception") or value.get("errors")
    nested_error = _normalize(nested, seen=seen) if nested is not None and nested is not value else None
    if message is None and nested_error is not None:
        message = nested_error.message
    if stage is None and nested_error is not None:
        stage = nested_error.stage
    if endpoint is None and nested_error is not None:
        endpoint = nested_error.endpoint
    if error_type == "Error" and nested_error is not None:
        error_type = nested_error.type
    if status is not None:
        prefix = f"HTTP {status}"
        message = f"{prefix}: {message}" if message else prefix
    if message is None:
        message = str(error_type)
    return NormalizedError(str(error_type), _as_text(stage), _as_text(endpoint), _as_text(message) or "Unknown error")


def _normalize_exception(exc: BaseException, *, seen: set[int]) -> NormalizedError:
    ident = id(exc)
    if ident in seen:
        return NormalizedError(type(exc).__name__, None, None, "Exception chain cycle")
    seen.add(ident)
    try:
        if type(exc).__name__ == "GraphIngestionError" and hasattr(exc, "records"):
            return _normalize_graph_ingestion_error(exc, seen=seen)
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None) if response is not None else None
        endpoint = getattr(response, "url", None) if response is not None else None
        message = str(exc) or type(exc).__name__
        if response is not None:
            try:
                response_body = response.json()
            except (AttributeError, TypeError, ValueError):
                response_body = None
            if isinstance(response_body, dict):
                response_message = _normalize_mapping(response_body, seen=seen).message
                if response_message and response_message != "Error":
                    message = response_message
        if status is not None:
            message = f"HTTP {status}: {message}"
        cause = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
        if cause is not None:
            causal = _normalize(cause, seen=seen)
            if not endpoint:
                endpoint = causal.endpoint
            message = f"{message}; caused by {causal.summary}"
        return NormalizedError(type(exc).__name__, None, _as_text(endpoint), message)
    finally:
        seen.discard(ident)


def _normalize_graph_ingestion_error(exc: BaseException, *, seen: set[int]) -> NormalizedError:
    records = getattr(exc, "records", [])
    # Old callers may have passed a string despite the historical list annotation.
    if isinstance(records, str):
        records = [records]
    if not isinstance(records, (list, tuple)):
        records = [records]
    first = records[0] if records else str(exc)
    record = first if isinstance(first, dict) else {"error": first}
    nested = _normalize(record.get("error"), seen=seen)
    diagnostics = getattr(exc, "stage_diagnostics", {}) or {}
    column = record.get("column")
    diagnostic = diagnostics.get(column) if isinstance(diagnostics, dict) and isinstance(column, str) else None
    role = getattr(diagnostic, "role", None)
    display_name = getattr(diagnostic, "display_name", None)
    stage = _as_text(role).upper() if _as_text(role) else _as_text(display_name)
    endpoint = getattr(diagnostic, "invoke_url", None) or nested.endpoint
    return NormalizedError("GraphIngestionError", stage or nested.stage, _as_text(endpoint), nested.message)


def _status_code(value: dict[Any, Any]) -> int | None:
    for key in ("status_code", "http_status", "status", "code"):
        candidate = value.get(key)
        if isinstance(candidate, int) and not isinstance(candidate, bool) and 100 <= candidate < 1000:
            return candidate
        if isinstance(candidate, str) and candidate.strip().isdigit():
            parsed = int(candidate.strip())
            if 100 <= parsed < 1000:
                return parsed
    return None


def _safe_endpoint(value: str | None) -> str | None:
    text = " ".join(str(value).split()) if value is not None else ""
    if len(text) > 1024:
        text = text[:1024].rstrip() + "..."
    if not text:
        return None
    try:
        parsed = urlsplit(text)
        if parsed.scheme and parsed.netloc:
            host = parsed.hostname or ""
            if parsed.port:
                host = f"{host}:{parsed.port}"
            return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except ValueError:
        pass
    return _BEARER_VALUE.sub("Bearer <redacted>", _SENSITIVE_VALUE.sub("<redacted>", text))


def _safe_text(value: Any, *, limit: int) -> str | None:
    if value is None:
        return None
    text = _redact_text(str(value))
    text = "".join(ch if ch.isprintable() else " " for ch in text).strip()
    text = " ".join(text.split())
    if not text:
        return None
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
    return text


def _redact_text(text: str) -> str:
    text = _SENSITIVE_VALUE.sub("<redacted>", text)
    text = _BEARER_VALUE.sub("Bearer <redacted>", text)

    def replace_url(match: re.Match[str]) -> str:
        return _safe_endpoint(match.group(0)) or "<redacted-url>"

    # `_safe_endpoint` only calls `_redact_text` for non-URLs, preventing recursion.
    return _URL.sub(replace_url, text)


def _as_text(value: Any) -> str | None:
    return None if value is None else str(value)
