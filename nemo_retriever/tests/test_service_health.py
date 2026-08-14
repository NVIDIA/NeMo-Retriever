# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for service liveness and readiness endpoints."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from nemo_retriever.service.app import create_app
from nemo_retriever.service.config import AuthConfig, GatewayConfig, ServiceConfig
from nemo_retriever.service.services import proxy as proxy_module
from nemo_retriever.service.services.pipeline_pool import PoolType
from nemo_retriever.service.services.proxy import GatewayProxy


class _HealthProxy:
    def __init__(self, statuses: dict[str, dict[str, object]]) -> None:
        self._statuses = statuses

    async def check_backend(self, pool_type: PoolType) -> dict[str, object]:
        return self._statuses[pool_type.value]


class _HealthClient:
    def __init__(self, status_code: int) -> None:
        self._status_code = status_code

    async def get(self, path: str, *, timeout: float) -> httpx.Response:
        assert path == "/v1/health"
        assert timeout == 5.0
        return httpx.Response(self._status_code)


def test_live_is_shallow_and_unauthenticated() -> None:
    app = create_app(ServiceConfig(mode="gateway", auth=AuthConfig(enabled=True, api_token="secret")))

    with TestClient(app) as client:
        response = client.get("/v1/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "mode": "gateway"}


@pytest.mark.parametrize(
    ("statuses", "expected_status"),
    [
        (
            {"realtime": {"status": "ok", "code": 200}, "batch": {"status": "ok", "code": 200}},
            200,
        ),
        (
            {
                "realtime": {"status": "unreachable", "error": "connection refused"},
                "batch": {"status": "ok", "code": 200},
            },
            503,
        ),
        (
            {"realtime": {"status": "unhealthy", "code": 503}, "batch": {"status": "ok", "code": 200}},
            503,
        ),
    ],
)
def test_gateway_health_reflects_required_backend_readiness(
    monkeypatch: pytest.MonkeyPatch,
    statuses: dict[str, dict[str, object]],
    expected_status: int,
) -> None:
    monkeypatch.setattr(proxy_module, "get_proxy", lambda: _HealthProxy(statuses))
    app = create_app(ServiceConfig(mode="gateway"))

    with TestClient(app) as client:
        response = client.get("/v1/health")

    assert response.status_code == expected_status
    assert response.json()["backends"] == statuses
    assert response.json()["status"] == ("ok" if expected_status == 200 else "unavailable")


@pytest.mark.parametrize("status_code", [400, 500, 503])
def test_backend_non_success_response_is_unhealthy(status_code: int) -> None:
    proxy = GatewayProxy.__new__(GatewayProxy)
    proxy._config = GatewayConfig(realtime_url="http://realtime.test", batch_url="http://batch.test")  # noqa: SLF001
    proxy._realtime = _HealthClient(status_code)  # noqa: SLF001

    result = asyncio.run(proxy.check_backend(PoolType.REALTIME))

    assert result == {"status": "unhealthy", "code": status_code}
