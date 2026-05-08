# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import typer
import uvicorn

from nemo_retriever.agent_mcp.server import build_asgi_app

app = typer.Typer(help="Start the NeMo Retriever agent MCP server.")


@app.command()
def start(
    data_root: Path = typer.Option(
        Path(".nemo-retriever-mcp"),
        "--data-root",
        help="Directory for agent MCP registry and collection data.",
    ),
    allowed_root: list[Path] = typer.Option(
        [],
        "--allowed-root",
        help="Local root path agents may ingest from. May be supplied multiple times.",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host interface for the agent MCP server.",
    ),
    port: int = typer.Option(
        8099,
        "--port",
        help="Port for the agent MCP server.",
    ),
    log_level: str = typer.Option(
        "info",
        "--log-level",
        help="Uvicorn log level.",
    ),
) -> None:
    roots = allowed_root or [Path.cwd()]
    application = build_asgi_app(data_root=data_root, allowed_roots=roots)
    uvicorn.run(application, host=host, port=port, log_level=log_level)
