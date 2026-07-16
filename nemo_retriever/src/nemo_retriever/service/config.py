# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service-mode configuration backed by ``retriever-service.yaml``."""

from __future__ import annotations

from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ConfigDict, Field, model_validator

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


class LocalExtractConfig(RichModel):
    """In-pod Hugging Face settings for PDF/image extraction stages."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    use_table_structure: bool = True
    ocr_version: Literal["v1", "v2"] = "v2"
    ocr_lang: Literal["multi", "english"] | None = None


class LocalEmbedConfig(RichModel):
    """In-pod Hugging Face settings for the embedding stage."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    model_name: str = "nvidia/llama-nemotron-embed-vl-1b-v2"
    local_ingest_embed_backend: str = "hf"
    gpu_memory_utilization: float = 0.45

    @model_validator(mode="after")
    def _validate_backend(self) -> "LocalEmbedConfig":
        from nemo_retriever.models import (
            _LOCAL_INGEST_EMBED_BACKENDS,
            normalize_backend,
        )

        self.local_ingest_embed_backend = normalize_backend(
            self.local_ingest_embed_backend,
            _LOCAL_INGEST_EMBED_BACKENDS,
            field_name="local_ingest_embed_backend",
            default="hf",
        )
        return self


class LocalAsrConfig(RichModel):
    """In-pod Hugging Face Parakeet ASR when no Parakeet NIM gRPC endpoint is set."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True


class LocalModelsConfig(RichModel):
    """Load Nemotron Hugging Face weights inside the service worker pod.

    When ``enabled`` is true, pipeline stages without a matching
    ``nim_endpoints.*`` URL load models from ``nemo_retriever.models.local``
    instead of calling remote NIMs. Requires the ``[local]`` (and usually
    ``[multimedia]``) install extras plus GPU resources.

    NIM URLs always take precedence when both are configured for a stage.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    hf_cache_dir: str | None = None
    device: str | None = None
    warmup_on_startup: bool = Field(
        default=False,
        description=(
            "When true, each process-pool worker loads local HF models at "
            "startup (before the first ingest). Increases VRAM use by "
            "num_workers × model stack; pair with low worker counts."
        ),
    )
    max_tasks_per_child: int | None = Field(
        default=None,
        description=(
            "Override process pool max_tasks_per_child when warmup_on_startup "
            "is true. Defaults to 10000 so warmed weights stay loaded."
        ),
    )
    max_process_pool_workers: int = Field(
        default=1,
        ge=1,
        description=(
            "When enabled, caps each ingest process pool (realtime and batch) "
            "to this many workers. Each worker loads the full local model stack "
            "into GPU memory, so keep this low (often 1)."
        ),
    )
    extract: LocalExtractConfig = Field(default_factory=LocalExtractConfig)
    embed: LocalEmbedConfig = Field(default_factory=LocalEmbedConfig)
    asr: LocalAsrConfig = Field(default_factory=LocalAsrConfig)


class NimEndpointsConfig(RichModel):
    """Remote NIM microservice endpoints used instead of local GPU models."""

    model_config = ConfigDict(extra="forbid")

    page_elements_invoke_url: str | None = None
    ocr_invoke_url: str | None = None
    table_structure_invoke_url: str | None = None
    embed_invoke_url: str | None = None
    embed_model_name: str | None = Field(
        default=None,
        description=(
            "Model identifier passed to the remote embedding endpoint. "
            "Server-owned — clients cannot override the deployed embed NIM SKU."
        ),
    )
    embed_model_provider_prefix: str | None = Field(
        default=None,
        description=(
            "Optional LiteLLM provider prefix prepended to embed_model_name for "
            "remote embedding endpoints that require namespaced model IDs."
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


class LLMConfig(RichModel):
    """Remote LLM configuration for service-mode RAG answer generation."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    model: str = "openai/nvidia/llama-3.3-nemotron-super-49b-v1.5"
    api_base: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    top_p: float | None = None
    max_tokens: int = 512
    extra_params: dict[str, Any] = Field(default_factory=dict)
    num_retries: int = 3
    timeout: float = 180.0
    rag_system_prompt: str | None = None
    rag_system_prompt_prefix: str | None = None
    reasoning_enabled: bool = True

    @model_validator(mode="after")
    def _validate_enabled_model(self) -> "LLMConfig":
        if self.enabled and not self.model.strip():
            raise ValueError("llm.model must be set when llm.enabled is true")
        return self


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


class MCPConfig(RichModel):
    """FastMCP transport configuration for service-mode agent integrations."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    path: str = Field(default="/mcp", description="HTTP mount path for the service FastMCP app.")
    base_url: str | None = Field(
        default=None,
        description=(
            "Service URL MCP tools call internally. Defaults to loopback on server.port "
            "when mounted by retriever service start."
        ),
    )
    enable_write_tools: bool = True
    max_concurrency: int = Field(default=8, ge=1)
    request_timeout_s: float = Field(default=60.0, gt=0)
    ingest_timeout_s: float = Field(default=1800.0, gt=0)
    poll_interval_s: float = Field(default=2.0, gt=0)

    @model_validator(mode="after")
    def _validate_path(self) -> "MCPConfig":
        if not self.path.startswith("/"):
            raise ValueError("mcp.path must start with '/'")
        if self.path == "/":
            raise ValueError("mcp.path must not be '/' because it would shadow service routes")
        if self.path.endswith("/"):
            raise ValueError("mcp.path must not end with '/'")
        return self


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

    realtime_workers: int = Field(
        default=8,
        ge=1,
        description="Concurrent workers for low-latency page processing",
    )
    realtime_queue_size: int = Field(default=2048, ge=1, description="Max queued items before realtime pool rejects")
    batch_workers: int = Field(default=16, ge=1, description="Concurrent workers for bulk document processing")
    batch_queue_size: int = Field(default=4096, ge=1, description="Max queued items before batch pool rejects")


class WorkQueueConfig(RichModel):
    """Gateway-owned split-topology work broker configuration."""

    model_config = ConfigDict(extra="forbid")

    gateway_url: str = Field(
        default="http://nemo-retriever-gateway:7670",
        description="Gateway Service URL used by split-mode workers.",
    )
    spool_directory: str = "/tmp/nemo-retriever-work"
    spool_limit_bytes: int = Field(default=20 * 1024**3, ge=1)
    claim_timeout_s: float = Field(default=30.0, gt=0)
    lease_ttl_s: float = Field(default=60.0, gt=0)
    heartbeat_interval_s: float = Field(default=20.0, gt=0)
    max_delivery_attempts: int = Field(default=3, ge=1)
    max_active_leases_realtime: int = Field(default=8, ge=1)
    max_active_leases_batch: int = Field(default=48, ge=1)

    @model_validator(mode="after")
    def _validate_heartbeat(self) -> "WorkQueueConfig":
        if self.heartbeat_interval_s >= self.lease_ttl_s / 2:
            raise ValueError("work_queue.heartbeat_interval_s must be below half work_queue.lease_ttl_s")
        return self


class VectorDbConfig(RichModel):
    """Configuration for the dedicated VectorDB pod (LanceDB + query endpoint)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    lancedb_uri: str = "/data/vectordb"
    table_name: str = "nemo_retriever"
    embed_model: str = "nvidia/llama-nemotron-embed-vl-1b-v2"
    embed_model_provider_prefix: str | None = None
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
        from nemo_retriever.common.policy import (
            PipelineOverridesPolicy,
            SinkUrlAllowlist,
        )

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
    local_models: LocalModelsConfig = Field(default_factory=LocalModelsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    resources: ResourceLimitsConfig = Field(default_factory=ResourceLimitsConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    pipeline: PipelinePoolConfig = Field(default_factory=PipelinePoolConfig)
    work_queue: WorkQueueConfig = Field(default_factory=WorkQueueConfig)
    vectordb: VectorDbConfig = Field(default_factory=VectorDbConfig)
    pipeline_overrides: PipelineOverridesConfig = Field(default_factory=PipelineOverridesConfig)

    @model_validator(mode="after")
    def _cap_process_pool_workers_for_local_models(self) -> "ServiceConfig":
        """Each process-pool worker loads the full HF stack — cap pool size."""
        if not self.local_models.enabled:
            return self
        cap = self.local_models.max_process_pool_workers
        updates: dict[str, int] = {}
        if self.pipeline.realtime_workers > cap:
            updates["realtime_workers"] = cap
        if self.pipeline.batch_workers > cap:
            updates["batch_workers"] = cap
        if updates:
            import logging

            logging.getLogger(__name__).warning(
                "local_models.enabled: capping pipeline workers to %d per pool "
                "(was realtime=%d batch=%d). Raise local_models.max_process_pool_workers "
                "only if GPU memory allows num_workers × model stack.",
                cap,
                self.pipeline.realtime_workers,
                self.pipeline.batch_workers,
            )
            self.pipeline = self.pipeline.model_copy(update=updates)
        return self


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
