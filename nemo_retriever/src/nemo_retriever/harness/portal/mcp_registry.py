# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decorator-driven registration of MCP-exposed portal tools.

Usage::

    from nemo_retriever.harness.portal.mcp_registry import portal_tool

    @portal_tool(
        name="trigger_benchmark_run",
        category="Jobs",
        description="Trigger a benchmark run with a dataset and optional preset",
    )
    async def trigger_run(dataset: str, preset: str | None = None) -> dict:
        ...

Decorated functions are collected into a module-level registry at import time.
The MCP server reads this registry at startup to build its tool list.  The same
``pkgutil.walk_packages`` scan that activates ``@designer_component`` also
activates ``@portal_tool``, so no separate discovery pass is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ToolMeta:
    """Metadata attached to every ``@portal_tool``-decorated function."""

    name: str
    category: str
    description: str
    tags: list[str] = field(default_factory=list)


_TOOL_REGISTRY: dict[str, dict[str, Any]] = {}


def portal_tool(
    *,
    name: str,
    category: str = "General",
    description: str = "",
    tags: list[str] | None = None,
):
    """Register a function as an MCP-exposed portal tool.

    Parameters
    ----------
    name : str
        Friendly tool name exposed to MCP clients.
    category : str
        Grouping label shown in the Settings UI (e.g. ``"Jobs"``).
    description : str
        Human-readable description sent to MCP clients so agents understand
        what this tool does.
    tags : list[str] | None
        Optional free-form tags for filtering in the UI.
    """

    def decorator(fn: Callable) -> Callable:
        key = f"{fn.__module__}.{fn.__qualname__}"
        entry = {
            "fn": fn,
            "name": name,
            "category": category,
            "description": description,
            "tags": tags or [],
        }
        _TOOL_REGISTRY[key] = entry
        fn._portal_tool_meta = ToolMeta(  # type: ignore[attr-defined]
            name=name,
            category=category,
            description=description,
            tags=tags or [],
        )
        return fn

    return decorator


def get_tool_registry() -> dict[str, dict[str, Any]]:
    """Return a shallow copy of the tool registry."""
    return dict(_TOOL_REGISTRY)
