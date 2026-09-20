# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Service-mode SSE streaming for POST /v1/answer/stream."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from nemo_retriever.models.llm.text_utils import ThinkTagStreamFilter, strip_think_tags
from nemo_retriever.service.app import create_app
from nemo_retriever.service.config import LLMConfig, LoggingConfig, PipelinePoolConfig, ServiceConfig, VectorDbConfig


def _parse_sse_response(raw: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    event_type = ""
    data_buf = ""
    for line in raw.splitlines():
        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data_buf = line[5:].strip()
        elif line == "" and data_buf:
            payload = json.loads(data_buf)
            events.append({"event": event_type, **payload})
            data_buf = ""
            event_type = ""
    return events


@pytest.fixture
def app_with_answer_config(monkeypatch: pytest.MonkeyPatch, tmp_path):
    async def _stub_work(_item):
        return 0, []

    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_realtime_work_fn",
        lambda _config: _stub_work,
    )
    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_batch_work_fn",
        lambda _config: _stub_work,
    )

    cfg = ServiceConfig(
        mode="standalone",
        logging=LoggingConfig(file=str(tmp_path / "service.log")),
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        vectordb=VectorDbConfig(enabled=True, vectordb_url="http://vectordb:7671"),
        llm=LLMConfig(
            enabled=True,
            model="openai/nvidia/llama-3.3-nemotron-super-49b-v1.5",
            api_base="http://llama-3-3-nemotron-super-49b-v1-5:8000/v1",
            api_key="not-needed",
            max_tokens=128,
            timeout=180.0,
            reasoning_enabled=False,
        ),
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        yield client


def _install_fake_vectordb(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResponse:
        status_code = 200
        content = json.dumps(
            {
                "results": [
                    {
                        "hits": [
                            {"text": "Super-49B is the answer generator.", "source": "doc.pdf"},
                        ]
                    }
                ]
            }
        ).encode()

        def json(self) -> dict[str, Any]:
            return json.loads(self.content.decode())

    class _FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, **kwargs) -> _FakeResponse:
            return _FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", _FakeAsyncClient)


def test_think_tag_stream_filter_matches_batch_strip() -> None:
    raw = "<think>secret</think>Visible answer"
    filt = ThinkTagStreamFilter()
    emitted: list[str] = []
    for chunk in ("<think>secret</think>Visible ", "answer"):
        emitted.extend(filt.feed(chunk))
    assert "".join(emitted) == strip_think_tags(raw)


def test_answer_stream_emits_retrieval_tokens_and_done(
    app_with_answer_config: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vectordb(monkeypatch)

    async def fake_stream_generate(
        query: str,
        chunks: list[str],
        *,
        reasoning_enabled: bool | None = None,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        assert query == "What generates answers?"
        assert chunks == ["Super-49B is the answer generator."]
        yield "metrics", {"ttft_s": 0.05}
        yield "token", {"delta": "Super-49B", "index": 0}
        yield "token", {"delta": " does.", "index": 1}
        yield "metrics", {"ttft_s": 0.05, "generation_latency_s": 0.2}
        yield "complete", {
            "answer": "Super-49B does.",
            "latency_s": 0.2,
            "model": "openai/nvidia/llama-3.3-nemotron-super-49b-v1.5",
            "error": None,
            "ttft_s": 0.05,
        }

    fake_llm = SimpleNamespace(stream_generate=fake_stream_generate)

    with patch("nemo_retriever.models.llm.clients.LiteLLMClient.from_kwargs", return_value=fake_llm):
        with app_with_answer_config.stream(
            "POST",
            "/v1/answer/stream",
            json={"query": "What generates answers?", "include_chunks": True},
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            body = "".join(resp.iter_text())

    events = _parse_sse_response(body)
    assert [event["event"] for event in events] == [
        "retrieval_done",
        "metrics",
        "token",
        "token",
        "metrics",
        "done",
    ]
    assert events[0]["chunk_count"] == 1
    assert events[0]["chunks"] == ["Super-49B is the answer generator."]
    assert events[2]["delta"] == "Super-49B"
    assert events[-1]["answer"] == "Super-49B does."
    assert events[-1]["chunks"] == ["Super-49B is the answer generator."]


def test_answer_stream_emits_error_event_when_generation_fails(
    app_with_answer_config: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vectordb(monkeypatch)

    async def fake_stream_generate(
        query: str,
        chunks: list[str],
        *,
        reasoning_enabled: bool | None = None,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        yield "complete", {
            "answer": "",
            "latency_s": 0.0,
            "model": "m",
            "error": "connection refused",
            "ttft_s": None,
        }

    fake_llm = SimpleNamespace(stream_generate=fake_stream_generate)

    with patch("nemo_retriever.models.llm.clients.LiteLLMClient.from_kwargs", return_value=fake_llm):
        with app_with_answer_config.stream("POST", "/v1/answer/stream", json={"query": "q"}) as resp:
            body = "".join(resp.iter_text())

    events = _parse_sse_response(body)
    assert events[0]["event"] == "retrieval_done"
    assert events[-1]["event"] == "error"
    assert "connection refused" in events[-1]["detail"]


def test_answer_stream_returns_404_when_llm_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    async def _stub_work(_item):
        return 0, []

    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_realtime_work_fn",
        lambda _config: _stub_work,
    )
    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_batch_work_fn",
        lambda _config: _stub_work,
    )

    app = create_app(
        ServiceConfig(
            mode="standalone",
            logging=LoggingConfig(file=str(tmp_path / "service.log")),
            pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
            vectordb=VectorDbConfig(enabled=True, vectordb_url="http://vectordb:7671"),
            llm=LLMConfig(enabled=False),
        )
    )

    with TestClient(app) as client:
        resp = client.post("/v1/answer/stream", json={"query": "q"})

    assert resp.status_code == 404
    assert "LLM answer generation is not enabled" in resp.json()["detail"]
