# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP server integration for the Harness Portal.

Builds a ``FastMCP`` server whose tools are populated from the
``@portal_tool`` registry.  Each invocation is wrapped with an audit
logger that writes to the ``mcp_audit_log`` SQLite table.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from typing import Any

from fastmcp import FastMCP

from nemo_retriever.harness import history
from nemo_retriever.harness.portal.mcp_registry import get_tool_registry

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "Harness Portal",
    instructions=(
        "This MCP server exposes tools for interacting with the NVIDIA "
        "nemo_retriever Harness Portal.  You can trigger benchmark runs, "
        "list datasets/presets/runners, check run results, and manage "
        "schedules.  Use `list_datasets` and `list_presets` to discover "
        "available options before triggering runs."
    ),
)


def _get_disabled_tools() -> set[str]:
    """Return the set of tool registry keys that the admin has disabled."""
    raw = history.get_portal_setting("mcp_disabled_tools") or "[]"
    try:
        return set(json.loads(raw))
    except (json.JSONDecodeError, TypeError):
        return set()


def _make_audited_wrapper(tool_name: str, fn: Any):
    """Return a wrapper that logs every call to the audit table."""

    if asyncio.iscoroutinefunction(fn):

        async def _wrapper(**kwargs: Any) -> Any:
            t0 = time.monotonic()
            try:
                result = await fn(**kwargs)
                duration = (time.monotonic() - t0) * 1000
                _log_audit(tool_name, kwargs, result, duration, success=True)
                return result
            except Exception as exc:
                duration = (time.monotonic() - t0) * 1000
                _log_audit(tool_name, kwargs, None, duration, success=False, error=str(exc))
                raise

    else:

        def _wrapper(**kwargs: Any) -> Any:
            t0 = time.monotonic()
            try:
                result = fn(**kwargs)
                duration = (time.monotonic() - t0) * 1000
                _log_audit(tool_name, kwargs, result, duration, success=True)
                return result
            except Exception as exc:
                duration = (time.monotonic() - t0) * 1000
                _log_audit(tool_name, kwargs, None, duration, success=False, error=str(exc))
                raise

    _wrapper.__name__ = fn.__name__
    _wrapper.__doc__ = fn.__doc__
    _wrapper.__module__ = fn.__module__
    sig = inspect.signature(fn)
    _wrapper.__signature__ = sig  # type: ignore[attr-defined]
    _wrapper.__annotations__ = fn.__annotations__.copy()
    return _wrapper


def _log_audit(
    tool_name: str,
    arguments: Any,
    result: Any,
    duration_ms: float,
    *,
    success: bool,
    error: str | None = None,
) -> None:
    try:
        args_str = json.dumps(arguments, default=str)[:4096] if arguments else None
        result_str = None
        if result is not None:
            try:
                result_str = json.dumps(result, default=str)[:2048]
            except (TypeError, ValueError):
                result_str = str(result)[:2048]

        history.insert_mcp_audit_entry(
            tool_name=tool_name,
            arguments=args_str,
            result_summary=result_str,
            duration_ms=round(duration_ms, 2),
            success=success,
            error=error,
        )
    except Exception:
        logger.exception("Failed to write MCP audit log entry for tool %s", tool_name)


def register_tools_from_registry() -> int:
    """Scan the ``@portal_tool`` registry and register enabled tools with the
    MCP server.  Returns the number of tools registered."""
    registry = get_tool_registry()
    disabled = _get_disabled_tools()
    count = 0

    for key, entry in registry.items():
        if key in disabled:
            logger.debug("Skipping disabled MCP tool: %s", entry["name"])
            continue

        fn = entry["fn"]
        wrapper = _make_audited_wrapper(entry["name"], fn)

        mcp.tool(
            name=entry["name"],
            description=entry["description"],
        )(wrapper)
        count += 1

    logger.info("Registered %d MCP tools (%d disabled)", count, len(disabled))
    return count


def register_resources() -> None:
    """Register MCP resources for read-only portal state."""

    @mcp.resource("portal://datasets")
    def list_datasets_resource() -> str:
        """List all managed benchmark datasets."""
        names = history.get_dataset_names()
        return json.dumps(names)

    @mcp.resource("portal://presets")
    def list_presets_resource() -> str:
        """List all managed presets."""
        names = history.get_preset_names()
        return json.dumps(names)

    @mcp.resource("portal://runners")
    def list_runners_resource() -> str:
        """List all runners and their current status."""
        runners = history.get_runners()
        safe = [
            {
                "id": r["id"],
                "name": r.get("name"),
                "status": r.get("status"),
                "gpu_type": r.get("gpu_type"),
                "gpu_count": r.get("gpu_count"),
            }
            for r in runners
        ]
        return json.dumps(safe)

    @mcp.resource("portal://recent-runs")
    def recent_runs_resource() -> str:
        """Return the 20 most recent benchmark runs."""
        runs = history.get_runs(limit=20)
        safe = [
            {
                "id": r["id"],
                "dataset": r.get("dataset"),
                "preset": r.get("preset"),
                "success": r.get("success"),
                "pages_per_sec": r.get("pages_per_sec"),
                "recall_5": r.get("recall_5"),
                "timestamp": r.get("timestamp"),
            }
            for r in runs
        ]
        return json.dumps(safe)


class _MCPGuardMiddleware:
    """ASGI middleware that enforces ``mcp_allowed_origins`` and ``mcp_rate_limit``."""

    def __init__(self, app):
        self._app = app
        self._request_timestamps: list[float] = []

    def _allowed_origins(self) -> set[str] | None:
        raw = history.get_portal_setting("mcp_allowed_origins")
        if not raw or raw.strip() == "*":
            return None
        return {o.strip().rstrip("/") for o in raw.split(",") if o.strip()}

    def _rate_limit(self) -> int:
        raw = history.get_portal_setting("mcp_rate_limit")
        if raw:
            try:
                val = int(raw)
                if val > 0:
                    return val
            except (ValueError, TypeError):
                pass
        return 0

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self._app(scope, receive, send)

        origins = self._allowed_origins()
        if origins is not None:
            headers = dict(scope.get("headers") or [])
            origin = headers.get(b"origin", b"").decode()
            if origin.rstrip("/") not in origins:
                return await _reject(send, 403, "Origin not allowed")

        rpm = self._rate_limit()
        if rpm:
            now = time.monotonic()
            cutoff = now - 60.0
            self._request_timestamps = [t for t in self._request_timestamps if t > cutoff]
            if len(self._request_timestamps) >= rpm:
                return await _reject(send, 429, "Rate limit exceeded")
            self._request_timestamps.append(now)

        return await self._app(scope, receive, send)


async def _reject(send, status: int, detail: str):
    body = json.dumps({"detail": detail}).encode()
    await send({"type": "http.response.start", "status": status, "headers": [[b"content-type", b"application/json"]]})
    await send({"type": "http.response.body", "body": body})


def build_mcp_app():
    """Create the ASGI app from the MCP server, suitable for mounting."""
    return _MCPGuardMiddleware(mcp.http_app(path="/"))
