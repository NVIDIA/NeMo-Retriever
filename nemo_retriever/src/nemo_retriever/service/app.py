# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI application factory for the retriever service mode."""

from __future__ import annotations

import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from nemo_retriever.service.config import ServiceConfig

logger = logging.getLogger(__name__)


def _configure_logging(config: ServiceConfig) -> None:
    """Set up root logger with both console and rotating-file handlers."""
    root = logging.getLogger()
    root.setLevel(config.logging.level.upper())

    fmt = logging.Formatter(config.logging.format)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        config.logging.file,
        maxBytes=50 * 1024 * 1024,  # 50 MiB
        backupCount=5,
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    logger.info("Logging configured: level=%s file=%s", config.logging.level, config.logging.file)


def _apply_resource_limits(config: ServiceConfig) -> None:
    """Best-effort resource capping (Linux only for some features)."""
    res = config.resources

    if res.gpu_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(res.gpu_devices)
        logger.info("CUDA_VISIBLE_DEVICES set to %s", os.environ["CUDA_VISIBLE_DEVICES"])

    if res.max_cpu_cores is not None:
        try:
            cpus = set(range(res.max_cpu_cores))
            os.sched_setaffinity(0, cpus)
            logger.info("CPU affinity restricted to %d cores", res.max_cpu_cores)
        except (AttributeError, OSError) as exc:
            logger.warning("Could not set CPU affinity: %s", exc)

    if res.max_memory_mb is not None:
        try:
            import resource as _resource

            limit_bytes = res.max_memory_mb * 1024 * 1024
            _resource.setrlimit(_resource.RLIMIT_AS, (limit_bytes, limit_bytes))
            logger.info("Memory limit set to %d MiB", res.max_memory_mb)
        except (ImportError, ValueError, OSError) as exc:
            logger.warning("Could not set memory limit: %s", exc)


def _check_media_dependencies(mode: str) -> None:
    """Log a startup banner when ``ffmpeg``/``ffprobe`` are missing.

    Audio and video ingestion uploads fail with HTTP 501 when these
    binaries are absent (see :func:`enforce_media_dependencies`). Surfacing
    a clear WARNING at startup gives cluster operators a chance to fix
    the deployment (set ``service.installFfmpeg=true`` or bake FFmpeg
    into a custom image) before the first media upload arrives, instead
    of debugging a worker traceback after the fact.

    The gateway pod does not run pipeline workers, so its missing FFmpeg
    is only a problem if it also classifies media uploads — which it
    does (it computes the routing category before forwarding). The
    warning therefore applies to every service role.
    """
    from nemo_retriever.common.modality.audio.media_interface import (
        HELM_FFMPEG_INSTALL_VALUE,
        MANUAL_FFMPEG_INSTALL_COMMAND,
        is_media_available,
        missing_media_dependencies,
    )

    if is_media_available():
        logger.info("Media dependencies (ffmpeg, ffprobe) detected — audio/video ingestion enabled (mode=%s)", mode)
        return

    missing = ", ".join(missing_media_dependencies()) or "ffmpeg, ffprobe"
    logger.warning(
        "Media dependencies missing in this container: %s. Audio and video "
        "uploads will be rejected with HTTP 501 (mode=%s). To enable "
        "media ingestion, redeploy the Helm chart with "
        "`--set %s`, install FFmpeg manually with `%s`, or build a "
        "custom image that includes ffmpeg/ffprobe.",
        missing,
        mode,
        HELM_FFMPEG_INSTALL_VALUE,
        MANUAL_FFMPEG_INSTALL_COMMAND,
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle for the service.

    Note:
        The config object (`app.state.config`) is constructed prior to app startup,
        typically via a factory that parses YAML, environment variables, or other
        runtime configuration sources, and validates it as a `ServiceConfig` object.
    """
    # The config is built externally (before this function is called) and stored on `app.state.config`.
    config: ServiceConfig = app.state.config
    mode = config.mode

    from nemo_retriever.service.services.event_bus import init_event_bus, shutdown_event_bus
    from nemo_retriever.service.services.job_tracker import init_job_tracker, shutdown_job_tracker
    from nemo_retriever.service.services.metrics import init_metrics, shutdown_metrics
    from nemo_retriever.service.services.pipeline_pool import init_pipeline_pool, shutdown_pipeline_pool
    from nemo_retriever.service.services.proxy import init_proxy, shutdown_proxy
    from nemo_retriever.service.services.sidecar_store import init_sidecar_store, shutdown_sidecar_store

    if mode in ("gateway", "standalone"):
        app.state.metrics = init_metrics()
    else:
        app.state.metrics = None

    tracker = init_job_tracker()
    event_bus = init_event_bus()
    tracker.set_event_bus(event_bus)
    app.state.sidecar_store = init_sidecar_store()

    if mode == "gateway":
        app.state.proxy = init_proxy(config.gateway)
        app.state.pipeline_pool = None
    else:
        from nemo_retriever.service.services.pipeline_executor import (
            create_batch_work_fn,
            create_realtime_work_fn,
        )

        rt_fn = create_realtime_work_fn(config) if mode in ("standalone", "realtime") else None
        bt_fn = create_batch_work_fn(config) if mode in ("standalone", "batch") else None
        app.state.proxy = None
        app.state.pipeline_pool = init_pipeline_pool(
            config.pipeline,
            mode=mode,
            realtime_work_fn=rt_fn,
            batch_work_fn=bt_fn,
        )

    _check_media_dependencies(mode)

    logger.info(
        "Retriever service started — mode=%s host=%s port=%d",
        mode,
        config.server.host,
        config.server.port,
    )

    yield

    from nemo_retriever.service.services.pipeline_executor import shutdown_process_executors

    shutdown_process_executors()
    await shutdown_proxy()
    await shutdown_pipeline_pool()
    shutdown_sidecar_store()
    shutdown_event_bus()
    shutdown_job_tracker()
    shutdown_metrics()
    logger.info("Retriever service stopped")


class _RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach a unique ``request_id`` to every incoming HTTP request."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request.state.request_id = uuid.uuid4().hex
        response = await call_next(request)
        return response


class _GatewayBodyCacheMiddleware:
    """Pure ASGI middleware: cap and buffer POST bodies for proxy forwarding.

    FastAPI's dependency injection parses ``UploadFile`` / ``Form`` parameters
    by consuming the ASGI body stream *before* the route handler runs.  When
    the gateway's proxy later calls ``request.body()`` the stream is already
    exhausted and Starlette raises ``RuntimeError: Stream consumed``.

    This middleware rejects bodies above ``max_body_bytes`` while reading,
    stores accepted bodies on the ASGI scope as ``scope["_cached_body"]``,
    and replays them through a synthetic ``receive`` callable so that form
    parsing works normally.  The proxy then reads from
    ``request.scope["_cached_body"]`` directly.

    Only active for ``POST`` requests; ``GET`` / ``OPTIONS`` etc. pass through
    untouched.
    """

    def __init__(self, app: Any, max_body_bytes: int | None = None) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope.get("method", "GET") != "POST":
            await self.app(scope, receive, send)
            return

        limit = self.max_body_bytes
        declared_size = self._content_length(scope)
        if limit is not None and declared_size is not None and declared_size > limit:
            await self._payload_too_large(scope, receive, send, limit, declared_size)
            return

        body_parts: list[bytes] = []
        body_size = 0
        while True:
            message = await receive()
            chunk = message.get("body", b"")
            if chunk:
                body_size += len(chunk)
                if limit is not None and body_size > limit:
                    await self._payload_too_large(scope, receive, send, limit, body_size)
                    return
                body_parts.append(chunk)
            if not message.get("more_body", False):
                break

        body = b"".join(body_parts)
        scope["_cached_body"] = body

        replayed = False

        async def replay_receive() -> dict:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)

    @staticmethod
    def _content_length(scope: dict) -> int | None:
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    parsed = int(value)
                except ValueError:
                    return None
                return parsed if parsed >= 0 else None
        return None

    @staticmethod
    async def _payload_too_large(
        scope: dict,
        receive: Any,
        send: Any,
        limit: int,
        actual_size: int,
    ) -> None:
        response = JSONResponse(
            status_code=413,
            content={
                "detail": f"Request body size {actual_size:,} bytes exceeds limit of {limit:,} bytes",
            },
        )
        await response(scope, receive, send)


def create_app(config: ServiceConfig) -> FastAPI:
    """Build and return a fully-configured :class:`FastAPI` application."""
    _configure_logging(config)
    _apply_resource_limits(config)

    app = FastAPI(
        title="Retriever Service",
        description="Low-latency document ingestion service powered by nemo-retriever",
        version="26.5.0",
        docs_url="/docs",
        lifespan=_lifespan,
    )
    app.state.config = config

    app.add_middleware(_RequestIdMiddleware)

    if config.mode == "gateway":
        app.add_middleware(_GatewayBodyCacheMiddleware, max_body_bytes=config.resources.max_upload_bytes)
        logger.info("Gateway body-cache middleware ENABLED")

    if config.auth.api_token:
        from nemo_retriever.service.auth import BearerAuthMiddleware

        app.add_middleware(BearerAuthMiddleware, config=config.auth)
        logger.info(
            "Bearer-token authentication ENABLED (header=%s, bypass=%s)",
            config.auth.header_name,
            config.auth.bypass_paths,
        )
    else:
        logger.info("Bearer-token authentication DISABLED (no api_token configured)")

    from nemo_retriever.service.routers import admin, ingest, metrics
    from nemo_retriever.service.services.prometheus import instrument_app

    app.include_router(ingest.router, prefix="/v1")
    app.include_router(metrics.router, prefix="/v1")
    # Admin/internal endpoints — pool_stats etc. Registered on every
    # role; the handler self-reports an empty pool dict on gateway pods.
    app.include_router(admin.router, prefix="/v1")
    instrument_app(app, role=config.mode)

    if config.mode == "gateway":
        from pathlib import Path as _Path

        from fastapi.staticfiles import StaticFiles

        from nemo_retriever.service.routers import dashboard

        app.include_router(dashboard.router, prefix="/v1/dashboard")
        _dashboard_static = _Path(__file__).parent / "dashboard" / "static"
        if _dashboard_static.is_dir():
            app.mount(
                "/v1/dashboard/static",
                StaticFiles(directory=str(_dashboard_static)),
                name="dashboard-static",
            )

    @app.get("/v1/health", tags=["system"], summary="Liveness / readiness probe")
    async def health() -> dict:
        base: dict = {"status": "ok", "mode": config.mode}
        if config.mode == "gateway":
            from nemo_retriever.service.services.proxy import get_proxy

            proxy = get_proxy()
            if proxy is not None:
                from nemo_retriever.service.services.pipeline_pool import PoolType

                base["backends"] = {
                    "realtime": await proxy.check_backend(PoolType.REALTIME),
                    "batch": await proxy.check_backend(PoolType.BATCH),
                }
        return base

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"{type(exc).__name__}: {exc}",
                "method": request.method,
                "path": request.url.path,
                "mode": config.mode,
            },
        )

    return app
