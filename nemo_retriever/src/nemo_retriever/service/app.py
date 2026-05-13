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

    if mode in ("gateway", "standalone"):
        app.state.metrics = init_metrics()
    else:
        app.state.metrics = None

    tracker = init_job_tracker()
    event_bus = init_event_bus()
    tracker.set_event_bus(event_bus)

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
    """Pure ASGI middleware: buffer POST bodies so the proxy can forward them.

    FastAPI's dependency injection parses ``UploadFile`` / ``Form`` parameters
    by consuming the ASGI body stream *before* the route handler runs.  When
    the gateway's proxy later calls ``request.body()`` the stream is already
    exhausted and Starlette raises ``RuntimeError: Stream consumed``.

    This middleware reads the entire body once, stores it on the ASGI scope
    as ``scope["_cached_body"]``, and replays it through a synthetic
    ``receive`` callable so that form parsing works normally.  The proxy
    then reads from ``request.scope["_cached_body"]`` directly.

    Only active for ``POST`` requests; ``GET`` / ``OPTIONS`` etc. pass through
    untouched.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope.get("method", "GET") != "POST":
            await self.app(scope, receive, send)
            return

        body_parts: list[bytes] = []
        while True:
            message = await receive()
            body_parts.append(message.get("body", b""))
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


def create_app(config: ServiceConfig) -> FastAPI:
    """Build and return a fully-configured :class:`FastAPI` application."""
    _configure_logging(config)
    _apply_resource_limits(config)

    app = FastAPI(
        title="Retriever Service",
        description="Low-latency document ingestion service powered by nemo-retriever",
        version="1.0.0",
        docs_url="/docs",
        lifespan=_lifespan,
    )
    app.state.config = config

    app.add_middleware(_RequestIdMiddleware)

    if config.mode == "gateway":
        app.add_middleware(_GatewayBodyCacheMiddleware)
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

    from nemo_retriever.service.routers import ingest, metrics
    from nemo_retriever.service.services.prometheus import instrument_app

    app.include_router(ingest.router, prefix="/v1")
    app.include_router(metrics.router, prefix="/v1")
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
