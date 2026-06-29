# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

import nemo_retriever.service.client as service_client_module
from nemo_retriever.service.client import RetrieverServiceClient


def _install_query_response(
    monkeypatch: pytest.MonkeyPatch,
    body: dict[str, Any],
    calls: list[dict[str, Any]] | None = None,
) -> None:
    class FakeResponse:
        status_code = 200
        text = ""

        def json(self) -> dict[str, Any]:
            return body

    class FakeHttpClient:
        def __init__(self, *, timeout: Any, headers: dict[str, str]) -> None:
            if calls is not None:
                calls.append({"timeout": timeout, "headers": headers})

        def __enter__(self) -> "FakeHttpClient":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def post(self, url: str, *, json: dict[str, Any]) -> FakeResponse:
            if calls is not None:
                calls.append({"url": url, "json": json})
            return FakeResponse()

    monkeypatch.setattr(service_client_module.httpx, "Client", FakeHttpClient)


def test_service_client_query_posts_to_v1_query_with_auth(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    _install_query_response(
        monkeypatch,
        {"results": [{"hits": [{"text": "passage", "source": "doc.pdf"}]}]},
        calls,
    )

    client = RetrieverServiceClient(base_url="http://svc:7670", api_token="secret")

    assert client.query("deployment?", top_k=2) == [[{"text": "passage", "source": "doc.pdf"}]]
    assert calls[0]["headers"] == {"Authorization": "Bearer secret"}
    assert calls[1] == {
        "url": "http://svc:7670/v1/query",
        "json": {"query": "deployment?", "top_k": 2},
    }


def test_service_client_query_accepts_empty_hits(monkeypatch) -> None:
    _install_query_response(monkeypatch, {"results": [{"hits": []}]})

    assert RetrieverServiceClient(base_url="http://svc:7670").query("deployment?", top_k=2) == [[]]


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ({"results": {}}, "results"),
        ({"results": []}, r"expected 1 result set\(s\), got 0"),
        ({"results": [None]}, "results.0"),
        ({"results": [{"hits": {}}]}, "results.0.hits"),
        ({"results": [{"hits": [None]}]}, "results.0.hits.0"),
    ],
)
def test_service_client_query_rejects_malformed_responses(monkeypatch, body: dict[str, Any], match: str) -> None:
    _install_query_response(monkeypatch, body)

    with pytest.raises(RuntimeError, match=match):
        RetrieverServiceClient(base_url="http://svc:7670").query("deployment?", top_k=2)


def test_service_client_query_rejects_result_count_mismatch_for_multi_query_request(monkeypatch) -> None:
    _install_query_response(monkeypatch, {"results": [{"hits": []}]})

    with pytest.raises(RuntimeError, match=r"expected 2 result set\(s\), got 1"):
        RetrieverServiceClient(base_url="http://svc:7670").query(["deployment?", "scaling?"], top_k=2)
