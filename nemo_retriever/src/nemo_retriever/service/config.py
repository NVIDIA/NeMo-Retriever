# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service-mode configuration backed by ``retriever-service.yaml``."""

from __future__ import annotations

from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ConfigDict, Field

from nemo_retriever.service.models.base import RichModel

ServiceMode = Literal["standalone", "gateway", "realtime", "batch"]


class ServerConfig(RichModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "0.0.0.0"
    port: int = 7670


class LoggingConfig(RichModel):
    model_config = ConfigDict(extra="forbid")

    level: str = "INFO"
    file: str = "retriever-service.log"
    format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


class NimEndpointsConfig(RichModel):
    """Remote NIM microservice endpoints used instead of local GPU models."""

    model_config = ConfigDict(extra="forbid")

    page_elements_invoke_url: str | None = None
    ocr_invoke_url: str | None = None
    table_structure_invoke_url: str | None = None
    graphic_elements_invoke_url: str | None = None
    embed_invoke_url: str | None = None
    rerank_invoke_url: str | None = None
    api_key: str | None = None


class ResourceLimitsConfig(RichModel):
    model_config = ConfigDict(extra="forbid")

    max_memory_mb: int | None = None
    max_cpu_cores: int | None = None
    gpu_devices: list[str] = Field(default_factory=list)
    max_upload_bytes: int = Field(
        default=500_000_000,
        ge=1,
        description="Max upload file size in bytes (default 500 MB). Rejected before buffering.",
    )


class AuthConfig(RichModel):
    """Optional bearer-token authentication."""

    model_config = ConfigDict(extra="forbid")

    api_token: str | None = None
    header_name: str = "Authorization"
    bypass_paths: list[str] = Field(default_factory=lambda: ["/v1/health", "/docs", "/openapi.json", "/redoc"])


class GatewayConfig(RichModel):
    """Backend service URLs used when ``mode`` is ``gateway``.

    Defaults use Kubernetes in-cluster DNS names that match the Helm chart
    service names generated when ``topology.mode: split``.
    """

    model_config = ConfigDict(extra="forbid")

    realtime_url: str = "http://nemo-retriever-realtime:7670"
    batch_url: str = "http://nemo-retriever-batch:7670"
    timeout_s: float = Field(default=300.0, description="Per-request forwarding timeout in seconds")
    max_connections: int = Field(default=100, description="httpx connection pool limit per backend")


class PipelinePoolConfig(RichModel):
    """Worker pool sizing for realtime and batch ingestion pipelines.

    Workers are abstract dispatchers — the actual work function is plugged
    in at startup.  Defaults are tuned for a CPU-only node that forwards
    work to remote NIM endpoints over HTTP.
    """

    model_config = ConfigDict(extra="forbid")

    realtime_workers: int = Field(default=8, ge=1, description="Concurrent workers for low-latency page processing")
    realtime_queue_size: int = Field(default=2048, ge=1, description="Max queued items before realtime pool rejects")
    batch_workers: int = Field(default=16, ge=1, description="Concurrent workers for bulk document processing")
    batch_queue_size: int = Field(default=4096, ge=1, description="Max queued items before batch pool rejects")


class VectorDbConfig(RichModel):
    """Configuration for the dedicated VectorDB pod (LanceDB + query endpoint)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    lancedb_uri: str = "/data/vectordb"
    table_name: str = "nemo_retriever"
    embed_model: str = "nvidia/llama-nemotron-embed-1b-v2"
    vectordb_url: str = Field(
        default="http://nemo-retriever-vectordb:7671",
        description="URL of the vectordb service (for workers to POST embeddings to)",
    )


class ServiceConfig(RichModel):
    """Top-level configuration for the retriever service mode.

    Every section has sensible defaults so a zero-config launch works out of
    the box.  Values can be overridden per-field from CLI flags.

    The ``mode`` field selects the runtime role:

    * **standalone** — single pod runs both realtime and batch pools (default).
    * **gateway** — thin proxy that routes uploads to backend worker pods.
    * **realtime** — worker pod that only runs the realtime pool.
    * **batch** — worker pod that only runs the batch pool.
    """

    model_config = ConfigDict(extra="ignore")

    mode: ServiceMode = "standalone"
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    nim_endpoints: NimEndpointsConfig = Field(default_factory=NimEndpointsConfig)
    resources: ResourceLimitsConfig = Field(default_factory=ResourceLimitsConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    pipeline: PipelinePoolConfig = Field(default_factory=PipelinePoolConfig)
    vectordb: VectorDbConfig = Field(default_factory=VectorDbConfig)


def _bundled_yaml_path() -> Path:
    """Return the path to the default ``retriever-service.yaml`` shipped with the package."""
    ref = importlib_resources.files("nemo_retriever.service") / "retriever-service.yaml"
    return Path(str(ref))


def _discover_config_path(explicit: str | None = None) -> Path | None:
    """Locate a config file using the standard precedence rules.

    1. *explicit* path supplied via ``--config``
    2. ``./retriever-service.yaml`` in the current working directory
    3. Bundled default inside the package
    """
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        return p

    cwd_candidate = Path.cwd() / "retriever-service.yaml"
    if cwd_candidate.is_file():
        return cwd_candidate

    bundled = _bundled_yaml_path()
    if bundled.is_file():
        return bundled

    return None


def load_config(
    config_path: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> ServiceConfig:
    """Load a :class:`ServiceConfig` from YAML with optional CLI overrides."""
    path = _discover_config_path(config_path)
    if path is not None:
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    else:
        raw = {}

    if overrides:
        for dotted_key, value in overrides.items():
            if value is None:
                continue
            parts = dotted_key.split(".")
            target = raw
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value

    config = ServiceConfig(**raw)

    _REDACTED_FIELDS = frozenset({"api_key", "api_token", "password", "secret"})

    from rich.console import Console
    from rich.tree import Tree

    console = Console(stderr=True)
    tree = Tree(f"[bold]ServiceConfig[/bold]  (source: {path or 'defaults'})")
    for section_name, section_value in config:
        if isinstance(section_value, RichModel):
            branch = tree.add(f"[cyan]{section_name}[/cyan]")
            for field_name, field_value in section_value:
                display = "****" if field_name in _REDACTED_FIELDS and field_value else repr(field_value)
                branch.add(f"[dim]{field_name}[/dim] = [white]{display}[/white]")
        else:
            tree.add(f"[cyan]{section_name}[/cyan] = [white]{section_value!r}[/white]")
    console.print(tree)

    return config
