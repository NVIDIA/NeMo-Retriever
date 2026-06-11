# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import nemo_retriever.service.client as service_client_module
from nemo_retriever.service.client import RetrieverServiceClient


def test_service_client_query_posts_to_v1_query_with_auth(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self) -> dict[str, Any]:
            return {"results": [{"hits": [{"text": "passage", "source": "doc.pdf"}]}]}

    class FakeHttpClient:
        def __init__(self, *, timeout: Any, headers: dict[str, str]) -> None:
            calls.append({"timeout": timeout, "headers": headers})

        def __enter__(self) -> "FakeHttpClient":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def post(self, url: str, *, json: dict[str, Any]) -> FakeResponse:
            calls.append({"url": url, "json": json})
            return FakeResponse()

    monkeypatch.setattr(service_client_module.httpx, "Client", FakeHttpClient)

    client = RetrieverServiceClient(base_url="http://svc:7670", api_token="secret")

    assert client.query("deployment?", top_k=2) == [[{"text": "passage", "source": "doc.pdf"}]]
    assert calls[0]["headers"] == {"Authorization": "Bearer secret"}
    assert calls[1] == {
        "url": "http://svc:7670/v1/query",
        "json": {"query": "deployment?", "top_k": 2},
    }
