# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP proxy client for gateway mode.

When the service runs in ``gateway`` mode it does not maintain local worker
pools.  Instead it forwards each ingest request to the appropriate backend
(realtime or batch) worker pod over HTTP, using Kubernetes-internal DNS for
service discovery.

The proxy streams the multipart request body through without buffering the
entire file in memory, then returns the backend response verbatim.

Singleton lifecycle follows the same pattern as metrics and pipeline_pool::

    proxy = get_proxy()          # None when not in gateway mode
    if proxy is not None:
        response = await proxy.forward(request, PoolType.REALTIME)
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import Response

from nemo_retriever.service.config import GatewayConfig
from nemo_retriever.service.services.pipeline_pool import PoolType

logger = logging.getLogger(__name__)


def _error_response(status_code: int, detail: str) -> Response:
    """Build a JSON error response with full diagnostic detail."""
    import json as _json

    return Response(
        content=_json.dumps(
            {
                "detail": detail,
                "gateway_error": True,
                "status_code": status_code,
            }
        ),
        status_code=status_code,
        media_type="application/json",
    )


_PATH_TO_POOL: dict[str, PoolType] = {
    "/v1/ingest/page": PoolType.REALTIME,
    "/v1/ingest/document": PoolType.BATCH,
}


class GatewayProxy:
    """Forwards ingest requests to backend worker Services."""

    def __init__(self, config: GatewayConfig) -> None:
        self._config = config
        limits = httpx.Limits(
            max_connections=config.max_connections,
            max_keepalive_connections=config.max_connections,
        )
        timeout = httpx.Timeout(timeout=config.timeout_s)

        self._realtime = httpx.AsyncClient(
            base_url=config.realtime_url,
            limits=limits,
            timeout=timeout,
        )
        self._batch = httpx.AsyncClient(
            base_url=config.batch_url,
            limits=limits,
            timeout=timeout,
        )
        logger.info(
            "Gateway proxy initialised — realtime=%s  batch=%s",
            config.realtime_url,
            config.batch_url,
        )

    def _client_for(self, pool_type: PoolType) -> httpx.AsyncClient:
        if pool_type is PoolType.REALTIME:
            return self._realtime
        return self._batch

    async def forward(
        self,
        request: Request,
        pool_type: PoolType,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> Response:
        """Stream the incoming *request* to the backend for *pool_type*.

        Returns a :class:`fastapi.responses.Response` that can be returned
        directly from a route handler.
        """
        client = self._client_for(pool_type)
        backend_label = pool_type.value
        backend_url = self._config.realtime_url if pool_type is PoolType.REALTIME else self._config.batch_url
        target_path = request.url.path

        body = request.scope.get("_cached_body") or await request.body()
        if not body:
            logger.warning(
                "Gateway forwarding empty body to %s %s%s — " "the body-cache middleware may not have run",
                backend_label,
                backend_url,
                target_path,
            )

        fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "transfer-encoding")}
        if extra_headers:
            fwd_headers.update(extra_headers)

        try:
            backend_resp = await client.request(
                method=request.method,
                url=target_path,
                content=body,
                headers=fwd_headers,
            )
        except httpx.ConnectError as exc:
            detail = f"Gateway failed to connect to {backend_label} backend " f"at {backend_url}{target_path}: {exc}"
            logger.error(detail)
            return _error_response(502, detail)
        except httpx.TimeoutException as exc:
            detail = (
                f"Gateway timed out forwarding to {backend_label} backend "
                f"at {backend_url}{target_path} after {self._config.timeout_s}s: {exc}"
            )
            logger.error(detail)
            return _error_response(504, detail)
        except httpx.HTTPError as exc:
            detail = (
                f"Gateway transport error forwarding to {backend_label} backend "
                f"at {backend_url}{target_path}: {type(exc).__name__}: {exc}"
            )
            logger.error(detail)
            return _error_response(502, detail)

        if backend_resp.status_code >= 400:
            body_preview = backend_resp.text[:500]
            logger.warning(
                "Backend %s returned HTTP %d for %s %s — body: %s",
                backend_label,
                backend_resp.status_code,
                request.method,
                target_path,
                body_preview,
            )

        excluded_headers = {"transfer-encoding", "content-encoding", "content-length"}
        resp_headers = {k: v for k, v in backend_resp.headers.items() if k.lower() not in excluded_headers}

        return Response(
            content=backend_resp.content,
            status_code=backend_resp.status_code,
            headers=resp_headers,
            media_type=backend_resp.headers.get("content-type"),
        )

    async def forward_get(
        self,
        request: Request,
        pool_type: PoolType,
        path: str,
    ) -> Response:
        """Forward a GET request to the backend for *pool_type*."""
        client = self._client_for(pool_type)
        backend_label = pool_type.value
        fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host",)}

        try:
            backend_resp = await client.get(path, headers=fwd_headers)
        except httpx.ConnectError as exc:
            return _error_response(502, f"Gateway failed to connect to {backend_label}: {exc}")
        except httpx.TimeoutException as exc:
            return _error_response(504, f"Gateway timed out on {backend_label}: {exc}")
        except httpx.HTTPError as exc:
            return _error_response(502, f"Gateway transport error on {backend_label}: {type(exc).__name__}: {exc}")

        excluded = {"transfer-encoding", "content-encoding", "content-length"}
        resp_headers = {k: v for k, v in backend_resp.headers.items() if k.lower() not in excluded}
        return Response(
            content=backend_resp.content,
            status_code=backend_resp.status_code,
            headers=resp_headers,
            media_type=backend_resp.headers.get("content-type"),
        )

    async def check_backend(self, pool_type: PoolType) -> dict[str, Any]:
        """Quick health probe against a backend."""
        client = self._client_for(pool_type)
        try:
            resp = await client.get("/v1/health", timeout=5.0)
            return {"status": "ok", "code": resp.status_code}
        except httpx.HTTPError as exc:
            return {"status": "unreachable", "error": str(exc)}

    async def shutdown(self) -> None:
        await self._realtime.aclose()
        await self._batch.aclose()
        logger.info("Gateway proxy shut down")


# ── Module-level singleton access ────────────────────────────────────

_instance: GatewayProxy | None = None


def init_proxy(config: GatewayConfig) -> GatewayProxy:
    """Create the global gateway proxy (call once at startup in gateway mode)."""
    global _instance
    _instance = GatewayProxy(config)
    return _instance


def get_proxy() -> GatewayProxy | None:
    """Return the proxy singleton, or ``None`` if not in gateway mode."""
    return _instance


async def shutdown_proxy() -> None:
    """Tear down the singleton (call during app shutdown)."""
    global _instance
    if _instance is not None:
        await _instance.shutdown()
        _instance = None
