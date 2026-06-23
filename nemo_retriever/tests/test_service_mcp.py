# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import httpx
from fastapi.testclient import TestClient

from nemo_retriever.service.app import create_app
from nemo_retriever.service.config import (
    AuthConfig,
    LoggingConfig,
    MCPConfig,
    PipelinePoolConfig,
    ServiceConfig,
)
from nemo_retriever.service.mcp_server import (
    MCPDocumentInput,
    ServiceMCPClient,
    ServiceMCPSettings,
    settings_from_service_config,
)
from nemo_retriever.service.services.pipeline_pool import WorkItem


def _run(coro):
    return asyncio.run(coro)


def test_settings_from_service_config_defaults_to_loopback_for_mounted_mcp(tmp_path) -> None:
    cfg = ServiceConfig(
        logging=LoggingConfig(file=str(tmp_path / "service.log")),
        auth=AuthConfig(api_token="secret", header_name="X-Token"),
        mcp=MCPConfig(max_concurrency=3, poll_interval_s=0.5),
    )

    settings = settings_from_service_config(cfg)

    assert settings.base_url == "http://127.0.0.1:7670"
    assert settings.api_token == "secret"
    assert settings.auth_header_name == "X-Token"
    assert settings.max_concurrency == 3
    assert settings.poll_interval_s == 0.5


def test_query_tool_client_posts_payload_and_auth_header() -> None:
    seen: dict[str, Any] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": [{"hits": [{"text": "match"}]}]})

    client = ServiceMCPClient(
        ServiceMCPSettings(base_url="http://service:7670", api_token="tok"),
        transport=httpx.MockTransport(_handler),
    )

    result = _run(client.query("What is indexed?", top_k=2, payload={"filters": {"source": "a.pdf"}}))

    assert seen == {
        "path": "/v1/query",
        "auth": "Bearer tok",
        "body": {"filters": {"source": "a.pdf"}, "query": "What is indexed?", "top_k": 2},
    }
    assert result["results"][0]["hits"][0]["text"] == "match"


def test_ingest_documents_accepts_inline_base64_upload() -> None:
    calls: list[tuple[str, str]] = []
    upload_body = b""

    def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal upload_body
        calls.append((request.method, request.url.path))
        if request.method == "POST" and request.url.path == "/v1/ingest/job":
            assert json.loads(request.content) == {
                "expected_documents": 1,
                "label": "agent-upload",
                "metadata": {"source": "mcp-test"},
                "retain_results": False,
            }
            return httpx.Response(
                201,
                json={
                    "job_id": "job-1",
                    "expected_documents": 1,
                    "status": "pending",
                    "created_at": "2026-06-23T00:00:00Z",
                },
            )
        if request.method == "POST" and request.url.path == "/v1/ingest/job/job-1/document":
            upload_body = request.content
            return httpx.Response(
                202,
                json={
                    "document_id": "doc-1",
                    "job_id": "job-1",
                    "content_sha256": "sha",
                    "status": "accepted",
                    "created_at": "2026-06-23T00:00:01Z",
                },
            )
        if request.method == "GET" and request.url.path == "/v1/ingest/job/job-1/documents":
            return httpx.Response(
                200,
                json={
                    "job_id": "job-1",
                    "total": 1,
                    "total_filtered": 1,
                    "offset": 0,
                    "limit": 1000,
                    "items": [
                        {
                            "document_id": "doc-1",
                            "job_id": "job-1",
                            "status": "completed",
                            "filename": "remote.txt",
                            "result_rows": 2,
                        }
                    ],
                },
            )
        return httpx.Response(404, text=f"unexpected {request.method} {request.url.path}")

    client = ServiceMCPClient(
        ServiceMCPSettings(base_url="http://service:7670", poll_interval_s=0.01),
        transport=httpx.MockTransport(_handler),
    )
    encoded = base64.b64encode(b"hello from remote agent").decode("ascii")

    result = _run(
        client.ingest_documents(
            [
                MCPDocumentInput(
                    filename="remote.txt",
                    content_base64=encoded,
                    content_type="text/plain",
                    metadata={"category": "test"},
                )
            ],
            label="agent-upload",
            job_metadata={"source": "mcp-test"},
        )
    )

    assert result["job_id"] == "job-1"
    assert result["upload_errors"] == []
    assert result["documents"]["items"][0]["status"] == "completed"
    assert b"hello from remote agent" in upload_body
    assert b"remote.txt" in upload_body
    assert calls == [
        ("POST", "/v1/ingest/job"),
        ("POST", "/v1/ingest/job/job-1/document"),
        ("GET", "/v1/ingest/job/job-1/documents"),
    ]


def test_service_start_mounts_mcp_and_auth_protects_it(monkeypatch, tmp_path) -> None:
    async def _stub_work(item: WorkItem) -> tuple[int, list[dict[str, Any]]]:
        return 1, [{"id": item.id, "stub": True}]

    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_realtime_work_fn",
        lambda _c: _stub_work,
    )
    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_batch_work_fn",
        lambda _c: _stub_work,
    )

    cfg = ServiceConfig(
        mode="standalone",
        logging=LoggingConfig(file=str(tmp_path / "service.log")),
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        auth=AuthConfig(api_token="secret"),
    )

    app = create_app(cfg)
    with TestClient(app) as client:
        unauthorized = client.get("/mcp")
        authorized = client.get("/mcp", headers={"Authorization": "Bearer secret"})

    assert unauthorized.status_code == 401
    assert authorized.status_code != 401

