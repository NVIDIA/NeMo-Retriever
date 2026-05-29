# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Split-topology worker callback must not POST full result_data payloads."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from nemo_retriever.service.services.pipeline_pool import _fire_gateway_callback
from nemo_retriever.service.services.worker_result_store import (
    clear_for_tests,
    consume_result_data,
    store_result_data,
)


@pytest.fixture(autouse=True)
def _clear_worker_store() -> None:
    clear_for_tests()
    yield
    clear_for_tests()


def test_fire_gateway_callback_omits_result_data() -> None:
    posted: dict[str, Any] = {}

    class _Resp:
        status_code = 200

    class _Client:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def post(self, url: str, json: dict[str, Any]) -> _Resp:
            posted["url"] = url
            posted["json"] = json
            return _Resp()

    rows = [{"page": 1, "text": "x" * 10_000}]

    async def _run() -> None:
        with patch("httpx.AsyncClient", _Client):
            store_result_data("doc-1", rows)
            await _fire_gateway_callback(
                "http://gateway/v1/internal/job-callback",
                "doc-1",
                "completed",
                result_rows=42,
            )

    asyncio.run(_run())

    assert posted["json"] == {"id": "doc-1", "status": "completed", "result_rows": 42}
    assert "result_data" not in posted["json"]
    assert consume_result_data("doc-1") == rows


def test_worker_document_result_endpoint() -> None:
    from fastapi.testclient import TestClient

    from nemo_retriever.service.app import create_app
    from nemo_retriever.service.config import PipelineOverridesConfig, PipelinePoolConfig, ServiceConfig

    cfg = ServiceConfig(
        mode="batch",
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        pipeline_overrides=PipelineOverridesConfig(),
    )
    store_result_data("doc-x", [{"text": "hello"}])
    with TestClient(create_app(cfg)) as client:
        resp = client.get("/v1/internal/document-result/doc-x")
        assert resp.status_code == 200
        assert resp.json()["result_data"] == [{"text": "hello"}]
        assert client.get("/v1/internal/document-result/doc-x").status_code == 404
