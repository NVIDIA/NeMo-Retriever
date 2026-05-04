# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service-mode configuration backed by ``retriever-service.yaml``."""

from __future__ import annotations

from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "0.0.0.0"
    port: int = 7670


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str = "INFO"
    file: str = "retriever-service.log"
    format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


class DatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = "retriever-service.db"


class NimEndpointsConfig(BaseModel):
    """Remote NIM microservice endpoints used instead of local GPU models."""

    model_config = ConfigDict(extra="forbid")

    page_elements_invoke_url: str | None = None
    ocr_invoke_url: str | None = None
    table_structure_invoke_url: str | None = None
    graphic_elements_invoke_url: str | None = None
    embed_invoke_url: str | None = None
    rerank_invoke_url: str | None = None
    api_key: str | None = None


class ProcessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_workers: int = 16
    batch_size: int = 32
    batch_timeout_s: float = 2.0
    store_page_results: bool = True


class ResourceLimitsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_memory_mb: int | None = None
    max_cpu_cores: int | None = None
    gpu_devices: list[str] = Field(default_factory=list)


class AuthConfig(BaseModel):
    """Optional bearer-token authentication.

    When ``api_token`` is set, every request must carry a matching
    ``Authorization: Bearer <token>`` header (or whatever ``header_name``
    is configured to).  Paths in ``bypass_paths`` are exempt — useful for
    Kubernetes liveness probes hitting ``/v1/health``.
    """

    model_config = ConfigDict(extra="forbid")

    api_token: str | None = None
    header_name: str = "Authorization"
    bypass_paths: list[str] = Field(default_factory=lambda: ["/v1/health", "/docs", "/openapi.json", "/redoc"])


class DrainConfig(BaseModel):
    """Graceful-shutdown tunables."""

    model_config = ConfigDict(extra="forbid")

    timeout_s: float = 60.0


class SpoolConfig(BaseModel):
    """Durable spool of accepted-but-not-yet-processed page bytes.

    When ``enabled`` is true, ingest writes every accepted page to disk
    (atomic write + fsync) before returning 202.  On startup the service
    scans the spool and re-enqueues any non-terminal documents — a pod
    crash, OOM kill, or rolling upgrade no longer loses accepted work.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    # Filesystem root for spooled bytes.  Defaults to ``<db_dir>/spool``
    # so it sits on the same persistence volume as the SQLite DB.
    # Override to point at a faster (or larger) volume.
    path: str | None = None
    # How often the background sweeper unlinks files for documents that
    # have reached a terminal state.
    cleanup_interval_s: float = 60.0
    # Per-cleanup batch cap so the sweeper never holds the loop hostage
    # when a backlog builds up.
    cleanup_batch_size: int = 1000


class RerankerConfig(BaseModel):
    """Defaults for the ``/v1/rerank`` endpoint."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = "nvidia/llama-nemotron-rerank-1b-v2"
    default_top_n: int = 10


class VectorStoreConfig(BaseModel):
    """LanceDB vector store settings used by the ``/v1/query`` endpoint."""

    model_config = ConfigDict(extra="forbid")

    lancedb_uri: str = "/var/lib/nemo-retriever/lancedb"
    lancedb_table: str = "nv-ingest"
    top_k: int = 10
    vector_column_name: str = "vector"
    nprobes: int = 16
    refine_factor: int = 10
    embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2"


class EventBusConfig(BaseModel):
    """SSE event-bus back-pressure policy.

    ``overflow_policy`` controls what happens when a subscriber queue
    fills up faster than the SSE consumer can drain it:

    - ``drop_low_priority`` (default): silently shed
      ``metrics_update``/``page_complete``/``document_complete`` events
      to preserve terminal events.  Lowest latency, no caller blocking,
      but accepts that *some* events are dropped under load.
    - ``backpressure``: the publisher awaits ``Queue.put()`` with the
      configured timeout, which back-pressures the worker callback
      thread, which in turn back-pressures the worker pool, which
      eventually 503s ingest with ``Retry-After``.  No events dropped
      until the timeout expires; after that, falls back to overflow.
    - ``block``: same as ``backpressure`` but waits forever.  Use only
      when the caller can tolerate unbounded latency to guarantee
      zero loss.
    """

    model_config = ConfigDict(extra="forbid")

    overflow_policy: Literal["drop_low_priority", "backpressure", "block"] = "drop_low_priority"
    # Per-subscription queue size.  Larger queues smooth bigger bursts
    # before back-pressure or shedding kicks in, at the cost of memory.
    queue_maxsize: int = 16384
    # Per-key replay buffer size.  When a client reconnects with
    # ``Last-Event-ID`` the server replays from this deque, so it should
    # comfortably cover one document's full event history.
    replay_buffer_size: int = 1024
    # Used by ``backpressure`` mode: max time the publisher is willing
    # to wait before falling back to the overflow path.  Ignored in
    # ``drop_low_priority`` and ``block`` modes.
    publish_timeout_s: float = 30.0


class ServiceConfig(BaseModel):
    """Top-level configuration for the retriever service mode.

    Every section has sensible defaults so a zero-config launch works out of
    the box.  Values can be overridden per-field from CLI flags.
    """

    model_config = ConfigDict(extra="forbid")

    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    nim_endpoints: NimEndpointsConfig = Field(default_factory=NimEndpointsConfig)
    resources: ResourceLimitsConfig = Field(default_factory=ResourceLimitsConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    drain: DrainConfig = Field(default_factory=DrainConfig)
    spool: SpoolConfig = Field(default_factory=SpoolConfig)
    event_bus: EventBusConfig = Field(default_factory=EventBusConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)


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
    """Load a :class:`ServiceConfig` from YAML with optional CLI overrides.

    Parameters
    ----------
    config_path
        Explicit path to a YAML config file.  ``None`` triggers auto-discovery.
    overrides
        Flat ``section.key`` overrides (e.g. ``{"server.port": 9000}``).
    """
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

    return ServiceConfig(**raw)
