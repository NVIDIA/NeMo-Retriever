# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typer sub-application for operating ``retriever service``."""

from __future__ import annotations

import os

from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen
from typing import Optional

import typer

app = typer.Typer(help="Operate the Retriever service. Use `retriever ingest service` to submit documents.")


def _stop_local_vectordb(process, *, timeout_s: float = 10) -> None:
    """Stop a supervised VectorDB child without masking the caller error."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _start_local_vectordb(cfg):
    vdb = cfg.vectordb
    parsed = urlparse(vdb.vectordb_url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise typer.BadParameter("vectordb.vectordb_url must use localhost when vectordb.launch_on_start is enabled.")
    local_embed = cfg.local_models.enabled and cfg.local_models.embed.enabled and not cfg.nim_endpoints.embed_invoke_url
    if not cfg.nim_endpoints.embed_invoke_url and not local_embed:
        raise typer.BadParameter(
            "vectordb.launch_on_start requires nim_endpoints.embed_invoke_url or enabled local_models.embed."
            + "Configure one, then retry."
        )
    port = parsed.port or 7671
    child_env = os.environ.copy()
    command = [
        sys.executable,
        "-m",
        "nemo_retriever.service.vectordb_app",
        "--host",
        parsed.hostname,
        "--port",
        str(port),
        "--lancedb-uri",
        vdb.lancedb_uri,
        "--table-name",
        vdb.table_name,
        "--index-mode",
        vdb.index_mode,
        "--embed-model",
        vdb.embed_model,
    ]
    if local_embed:
        command += ["--local-embed", "--local-embed-backend", cfg.local_models.embed.local_ingest_embed_backend]
        if cfg.local_models.hf_cache_dir:
            command += ["--hf-cache-dir", cfg.local_models.hf_cache_dir]
        if cfg.local_models.device:
            command += ["--device", cfg.local_models.device]
        command += ["--gpu-memory-utilization", str(cfg.local_models.embed.gpu_memory_utilization)]
    else:
        command += ["--embed-endpoint", cfg.nim_endpoints.embed_invoke_url]
    if cfg.nim_endpoints.api_key and not local_embed:
        child_env["NVIDIA_API_KEY"] = cfg.nim_endpoints.api_key
    if vdb.embed_model_provider_prefix:
        command += ["--embed-model-provider-prefix", vdb.embed_model_provider_prefix]
    if vdb.internal_api_token:
        child_env["NRL_INTERNAL_VDB_TOKEN"] = vdb.internal_api_token
    if not vdb.expiration_cleanup_enabled:
        command += ["--disable-expiration-cleanup"]
    command += ["--reconciliation-interval-seconds", str(vdb.reconciliation_interval_seconds)]
    if cfg.agentic.enabled:
        command += [
            "--agentic",
            "--agentic-llm-model",
            cfg.agentic.llm_model or "",
            "--agentic-invoke-url",
            cfg.agentic.invoke_url or "",
            "--agentic-reasoning-effort",
            cfg.agentic.reasoning_effort or "",
            "--agentic-backend-top-k",
            str(cfg.agentic.backend_top_k),
            "--agentic-react-max-steps",
            str(cfg.agentic.react_max_steps),
            "--agentic-text-truncation",
            str(cfg.agentic.text_truncation),
            "--agentic-temperature",
            str(cfg.agentic.temperature),
            "--agentic-request-timeout",
            str(cfg.agentic.request_timeout_s),
        ]
    process = subprocess.Popen(command, env=child_env)
    health_url = f"http://{parsed.hostname}:{port}/v1/health"
    for _ in range(150):
        if process.poll() is not None:
            raise RuntimeError(
                f"VectorDB exited during startup with status {process.returncode}. "
                "Inspect the VectorDB logs above and verify the embedding configuration, "
                "LanceDB path, and port 7671 availability."
            )
        try:
            with urlopen(health_url, timeout=1) as response:
                if response.status == 200:
                    return process
        except URLError:
            time.sleep(0.2)
    _stop_local_vectordb(process, timeout_s=5)
    raise RuntimeError(
        f"VectorDB did not become ready at {health_url} within 30 seconds. "
        "Inspect the VectorDB logs above and verify the embedding configuration, "
        "LanceDB path, and port 7671 availability."
    )


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
    launch_vectordb: bool = typer.Option(
        False, "--launch-vectordb", help="Start and supervise a local VectorDB process on 127.0.0.1:7671."
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
    if launch_vectordb:
        overrides["vectordb.enabled"] = True
        overrides["vectordb.launch_on_start"] = True
        overrides["vectordb.vectordb_url"] = "http://127.0.0.1:7671"
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

    vectordb_process = _start_local_vectordb(cfg) if cfg.vectordb.launch_on_start else None
    try:
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
    finally:
        if vectordb_process is not None:
            _stop_local_vectordb(vectordb_process)


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
    query_methods: str = typer.Option(
        "classic",
        "--query-methods",
        help="Retrieval MCP tools to expose: classic, agentic, or all.",
    ),
    agentic_request_timeout_s: float = typer.Option(
        1800.0,
        "--agentic-request-timeout",
        min=1.0,
        help="HTTP timeout for agentic_query.",
    ),
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

    normalized = query_methods.strip().lower()
    if normalized not in {"classic", "agentic", "all"}:
        raise typer.BadParameter("query-methods must be one of: classic, agentic, all")

    settings = ServiceMCPSettings(
        base_url=service_url,
        api_token=api_token,
        auth_header_name=auth_header_name,
        max_concurrency=concurrency,
        request_timeout_s=request_timeout_s,
        agentic_request_timeout_s=agentic_request_timeout_s,
        ingest_timeout_s=ingest_timeout_s,
        poll_interval_s=poll_interval_s,
        enable_write_tools=enable_write_tools,
        query_methods=normalized,  # type: ignore[arg-type]
    )
    build_mcp(settings).run(transport="stdio")
