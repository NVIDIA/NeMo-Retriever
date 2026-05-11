# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from nemo_retriever.agent_mcp.backend import InProcessRetrieverBackend
from nemo_retriever.agent_mcp.paths import PathPolicy
from nemo_retriever.agent_mcp.registry import CollectionRegistry
from nemo_retriever.agent_mcp.tools import AgentMcpToolService


def create_mcp_server(service: AgentMcpToolService) -> FastMCP:
    @asynccontextmanager
    async def lifespan(_server: FastMCP) -> AsyncIterator[dict[str, Any]]:
        try:
            yield {}
        finally:
            shutdown = getattr(service.backend, "shutdown", None)
            if callable(shutdown):
                shutdown(wait=True)

    mcp = FastMCP(
        "NeMo Retriever Agent MCP",
        instructions=(
            "Expose NeMo Retriever collection, ingestion, query, and reranking tools "
            "for agents. Tools return structured evidence and operational status, not "
            "generated answers."
        ),
        lifespan=lifespan,
    )

    mcp.tool(
        name="list_collections",
        description="List retriever collections available to the agent.",
    )(service.list_collections)
    mcp.tool(
        name="create_collection",
        description="Create a retriever collection for local evidence ingestion.",
    )(service.create_collection)
    mcp.tool(
        name="describe_collection",
        description="Describe collection configuration, status, and recent ingestion jobs.",
    )(service.describe_collection)
    mcp.tool(
        name="delete_collection",
        description="Delete a retriever collection and optionally its stored data.",
    )(service.delete_collection)
    mcp.tool(
        name="start_ingestion",
        description="Start ingesting allowed local paths into a collection.",
    )(service.start_ingestion)
    mcp.tool(
        name="get_ingestion_status",
        description="Return the current status for an ingestion job.",
    )(service.get_ingestion_status)
    mcp.tool(
        name="ingest_local_paths",
        description="Ingest allowed local paths into a collection, optionally waiting for completion.",
    )(service.ingest_local_paths)
    mcp.tool(
        name="query_collection",
        description="Query a collection and return evidence hits for agent grounding.",
    )(service.query_collection)
    mcp.tool(
        name="rerank_results",
        description="Rerank evidence hits for a query and return the highest scoring evidence.",
    )(service.rerank_results)

    return mcp


def create_default_service(
    data_root: str | Path,
    allowed_roots: list[str | Path],
    registry_path: str | Path | None = None,
) -> AgentMcpToolService:
    root = Path(data_root)
    registry = CollectionRegistry(registry_path or root / "registry.sqlite", data_root=root)
    backend = InProcessRetrieverBackend(
        registry=registry,
        path_policy=PathPolicy(allowed_roots=[Path(path) for path in allowed_roots]),
    )
    return AgentMcpToolService(backend)


def build_asgi_app(
    data_root: str | Path,
    allowed_roots: list[str | Path],
    registry_path: str | Path | None = None,
) -> Any:
    service = create_default_service(data_root, allowed_roots, registry_path=registry_path)
    return create_mcp_server(service).http_app(path="/")
