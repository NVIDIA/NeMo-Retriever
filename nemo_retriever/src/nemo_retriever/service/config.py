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

from nemo_retriever.common.schemas.base import RichModel

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
    embed_model_name: str | None = Field(
        default=None,
        description=(
            "Model identifier passed to the remote embedding endpoint. "
            "Server-owned — clients cannot override the deployed embed NIM SKU."
        ),
    )
    rerank_invoke_url: str | None = None
    audio_grpc_endpoint: str | None = Field(
        default=None,
        description=(
            "gRPC endpoint for the Parakeet ASR NIM (e.g. parakeet-nim:50051). "
            "When set, audio/video pipelines use remote ASR instead of loading "
            "the local Parakeet model (which requires torch)."
        ),
    )
    caption_invoke_url: str | None = Field(
        default=None,
        description=(
            "Remote VLM endpoint that fulfills .caption(...) requests. "
            "When set, clients may submit caption_params overrides "
            "(prompt, system_prompt, batch_size, etc.) without being able "
            "to redirect the endpoint or API key."
        ),
    )
    caption_model_name: str | None = Field(
        default=None,
        description=(
            "Model identifier passed to the remote caption endpoint. "
            "Server-owned — clients cannot override the deployed VLM SKU."
        ),
    )
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
    embed_model: str = "nvidia/llama-nemotron-embed-vl-1b-v2"
    vectordb_url: str = Field(
        default="http://nemo-retriever-vectordb:7671",
        description="URL of the vectordb service (for workers to POST embeddings to)",
    )


class SinksConfig(RichModel):
    """Per-sink-type egress allowlists for client-driven pipeline stages.

    Each list gates one of the three sinks (image store, webhook
    notifier, vector-DB upload). Leaving a list empty disables that
    sink — clients that ask for it receive HTTP 403 with a message
    explaining how to enable it.

    Use ``"*"`` as a wildcard entry to bypass enforcement entirely;
    this is intended for dev clusters only and is highly unsafe in
    multi-tenant deployments.
    """

    model_config = ConfigDict(extra="forbid")

    storage_uri_schemes: list[str] = Field(
        default_factory=list,
        description=(
            "URI schemes (e.g. 's3://', 'gs://', 'azure://') the worker is "
            "allowed to write extracted images to via .store(...). "
            "Empty disables storage sinks."
        ),
    )
    webhook_url_prefixes: list[str] = Field(
        default_factory=list,
        description=(
            "URL prefixes (e.g. 'https://hooks.example.com/') the worker is "
            "allowed to POST webhook notifications to. Empty disables webhooks."
        ),
    )
    vdb_uri_schemes: list[str] = Field(
        default_factory=list,
        description=(
            "URI schemes the worker is allowed to write LanceDB tables to "
            "via .vdb_upload(...). Empty disables per-request VDB overrides "
            "(the server's preconfigured vectordb pod still receives writes)."
        ),
    )


class PipelineOverridesConfig(RichModel):
    """How permissively to accept per-request ``PipelineSpec`` overrides.

    * ``mode='reject'`` — clients may not override pipeline config at all.
      Server-side YAML is the only source of truth.
    * ``mode='allow_list'`` (default) — only the keys enumerated by the
      built-in defaults plus the ``extra_*_keys`` extensions below are
      accepted. Endpoint URLs and API keys are *always* denied.
    * ``mode='allow_all'`` — every key is accepted **except** the
      endpoint/api_key denylist. Useful in dev clusters but unsafe in
      multi-tenant deployments.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["reject", "allow_list", "allow_all"] = "allow_list"
    extra_extract_keys: list[str] = Field(default_factory=list)
    extra_embed_keys: list[str] = Field(default_factory=list)
    extra_dedup_keys: list[str] = Field(default_factory=list)
    extra_split_keys: list[str] = Field(default_factory=list)
    extra_store_keys: list[str] = Field(default_factory=list)
    extra_storage_options_keys: list[str] = Field(default_factory=list)
    extra_webhook_keys: list[str] = Field(default_factory=list)
    extra_vdb_upload_keys: list[str] = Field(default_factory=list)
    extra_vdb_kwargs_keys: list[str] = Field(default_factory=list)
    extra_caption_keys: list[str] = Field(default_factory=list)
    sinks: SinksConfig = Field(default_factory=SinksConfig)

    def to_policy(self, *, caption_enabled: bool = False) -> "PipelineOverridesPolicy":  # noqa: F821
        """Return a :class:`PipelineOverridesPolicy` configured from this section.

        ``caption_enabled`` is derived from ``NimEndpointsConfig.caption_invoke_url``
        by the caller — clients can only override caption settings when the
        operator has actually wired up a VLM endpoint.
        """
        from nemo_retriever.common.policy import PipelineOverridesPolicy, SinkUrlAllowlist

        return PipelineOverridesPolicy(
            mode=self.mode,
            extra_extract_keys=frozenset(self.extra_extract_keys),
            extra_embed_keys=frozenset(self.extra_embed_keys),
            extra_dedup_keys=frozenset(self.extra_dedup_keys),
            extra_split_keys=frozenset(self.extra_split_keys),
            extra_store_keys=frozenset(self.extra_store_keys),
            extra_storage_options_keys=frozenset(self.extra_storage_options_keys),
            extra_webhook_keys=frozenset(self.extra_webhook_keys),
            extra_vdb_upload_keys=frozenset(self.extra_vdb_upload_keys),
            extra_vdb_kwargs_keys=frozenset(self.extra_vdb_kwargs_keys),
            extra_caption_keys=frozenset(self.extra_caption_keys),
            sinks=SinkUrlAllowlist(
                storage_uri_schemes=list(self.sinks.storage_uri_schemes),
                webhook_url_prefixes=list(self.sinks.webhook_url_prefixes),
                vdb_uri_schemes=list(self.sinks.vdb_uri_schemes),
            ),
            caption_enabled=caption_enabled,
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
    pipeline_overrides: PipelineOverridesConfig = Field(default_factory=PipelineOverridesConfig)


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
