# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typer sub-application for ``retriever service``."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Run the retriever ingest service or submit documents to it.")


@app.command("start")
def start(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to a retriever-service.yaml configuration file.",
    ),
    host: Optional[str] = typer.Option(None, "--host", help="Bind address (overrides YAML)."),
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Listen port (overrides YAML)."),
    log_level: Optional[str] = typer.Option(None, "--log-level", help="Logging level (overrides YAML)."),
    log_file: Optional[str] = typer.Option(None, "--log-file", help="Log file path (overrides YAML)."),
    nim_api_key: Optional[str] = typer.Option(
        None, "--nim-api-key", help="API key for NIM endpoints (overrides YAML / $NVIDIA_API_KEY)."
    ),
    llm_api_key: Optional[str] = typer.Option(
        None,
        "--llm-api-key",
        help="API key for LLM answer generation endpoints (overrides YAML / $NEMO_RETRIEVER_LLM_API_KEY).",
        envvar="NEMO_RETRIEVER_LLM_API_KEY",
    ),
    gpu_devices: Optional[str] = typer.Option(
        None, "--gpu-devices", help="Comma-separated GPU device IDs (overrides YAML)."
    ),
    local_models: Optional[bool] = typer.Option(
        None,
        "--local-models/--no-local-models",
        help="Load Hugging Face models in-pod instead of remote NIMs (overrides YAML).",
    ),
    local_embed_backend: Optional[str] = typer.Option(
        None,
        "--local-embed-backend",
        help="In-pod embed backend when --local-models is set: hf or vllm (overrides YAML).",
    ),
    local_embed_model: Optional[str] = typer.Option(
        None,
        "--local-embed-model",
        help="HF model id for in-pod embedding (overrides YAML).",
    ),
    hf_cache_dir: Optional[str] = typer.Option(
        None,
        "--hf-cache-dir",
        help="Hugging Face model cache directory for in-pod models (overrides YAML).",
    ),
    local_models_warmup: bool = typer.Option(
        False,
        "--local-models-warmup/--no-local-models-warmup",
        help="Load HF models in each process-pool worker at startup (overrides YAML).",
    ),
    max_process_pool_workers: Optional[int] = typer.Option(
        None,
        "--max-process-pool-workers",
        min=1,
        help="Cap each ingest process pool when --local-models is set (default 1).",
    ),
    api_token: Optional[str] = typer.Option(
        None,
        "--api-token",
        help=(
            "Bearer-token required on every request when set (overrides YAML / $NEMO_RETRIEVER_API_TOKEN). "
            "Leave unset to disable authentication."
        ),
        envvar="NEMO_RETRIEVER_API_TOKEN",
    ),
) -> None:
    """Start the retriever ingest web server."""
    import uvicorn

    from nemo_retriever.service.config import load_config

    overrides: dict[str, object] = {}
    if host is not None:
        overrides["server.host"] = host
    if port is not None:
        overrides["server.port"] = port
    if log_level is not None:
        overrides["logging.level"] = log_level
    if log_file is not None:
        overrides["logging.file"] = log_file
    if nim_api_key is not None:
        overrides["nim_endpoints.api_key"] = nim_api_key
    if llm_api_key is not None:
        overrides["llm.api_key"] = llm_api_key
    if gpu_devices is not None:
        overrides["resources.gpu_devices"] = [d.strip() for d in gpu_devices.split(",") if d.strip()]
    if local_models is not None:
        overrides["local_models.enabled"] = local_models
    if local_embed_backend is not None:
        overrides["local_models.embed.local_ingest_embed_backend"] = local_embed_backend
    if local_embed_model is not None:
        overrides["local_models.embed.model_name"] = local_embed_model
    if hf_cache_dir is not None:
        overrides["local_models.hf_cache_dir"] = hf_cache_dir
    if local_models_warmup:
        overrides["local_models.warmup_on_startup"] = True
    if max_process_pool_workers is not None:
        overrides["local_models.max_process_pool_workers"] = max_process_pool_workers
    if api_token is not None:
        overrides["auth.api_token"] = api_token

    cfg = load_config(config_path=str(config) if config else None, overrides=overrides or None)

    from nemo_retriever.service.app import create_app

    application = create_app(cfg)

    try:
        import setproctitle

        setproctitle.setproctitle("nemo-retriever-server")
    except ImportError:
        pass

    uvicorn.run(
        application,
        host=cfg.server.host,
        port=cfg.server.port,
        log_level=cfg.logging.level.lower(),
    )


@app.command("ingest")
def ingest(
    files: list[Path] = typer.Argument(..., help="One or more document files to ingest."),
    server_url: str = typer.Option("http://localhost:7670", "--server-url", "-s", help="Retriever service base URL."),
    use_sse: bool = typer.Option(True, "--sse/--no-sse", help="Use SSE streaming (default) or poll."),
    poll_interval: float = typer.Option(2.0, "--poll-interval", help="Seconds between status polls (no-SSE mode)."),
    concurrency: int = typer.Option(8, "--concurrency", help="Max concurrent uploads."),
    api_token: Optional[str] = typer.Option(
        None,
        "--api-token",
        help="Bearer-token to send with every request ($NEMO_RETRIEVER_API_TOKEN env var also accepted).",
        envvar="NEMO_RETRIEVER_API_TOKEN",
    ),
) -> None:
    """Submit documents to a running retriever service for ingestion."""
    from nemo_retriever.service.client import RetrieverServiceClient

    async def _run() -> None:
        client = RetrieverServiceClient(
            base_url=server_url,
            max_concurrency=concurrency,
            api_token=api_token,
        )
        await client.ingest_documents(
            files=files,
            use_sse=use_sse,
            poll_interval=poll_interval,
        )

    asyncio.run(_run())


@app.command("mcp-stdio")
def mcp_stdio(
    service_url: str = typer.Option(
        "http://localhost:7670",
        "--service-url",
        "-s",
        help="Retriever service base URL that MCP tools call.",
        envvar="NEMO_RETRIEVER_SERVICE_URL",
    ),
    api_token: Optional[str] = typer.Option(
        None,
        "--api-token",
        help="Bearer-token sent to the retriever service by MCP tools.",
        envvar="NEMO_RETRIEVER_API_TOKEN",
    ),
    auth_header_name: str = typer.Option(
        "Authorization",
        "--auth-header-name",
        help="Header used for bearer-token authentication.",
    ),
    concurrency: int = typer.Option(8, "--concurrency", min=1, help="Max concurrent MCP document uploads."),
    request_timeout_s: float = typer.Option(60.0, "--request-timeout", min=0.1, help="HTTP request timeout."),
    ingest_timeout_s: float = typer.Option(1800.0, "--ingest-timeout", min=1.0, help="Document ingest timeout."),
    poll_interval_s: float = typer.Option(2.0, "--poll-interval", min=0.1, help="Status polling interval."),
    enable_write_tools: bool = typer.Option(
        True,
        "--write-tools/--read-only",
        help="Expose write-capable MCP tools such as ingest_documents.",
    ),
) -> None:
    """Run the retriever service MCP server over stdio for local agents."""
    from nemo_retriever.service.mcp_server import ServiceMCPSettings, build_mcp

    settings = ServiceMCPSettings(
        base_url=service_url,
        api_token=api_token,
        auth_header_name=auth_header_name,
        max_concurrency=concurrency,
        request_timeout_s=request_timeout_s,
        ingest_timeout_s=ingest_timeout_s,
        poll_interval_s=poll_interval_s,
        enable_write_tools=enable_write_tools,
    )
    build_mcp(settings).run(transport="stdio")
