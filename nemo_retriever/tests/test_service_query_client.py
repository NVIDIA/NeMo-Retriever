# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import inspect
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
        content = b"{}"

        def json(self) -> dict[str, Any]:
            return body

    # The synchronous methods are facades over the async implementation, so
    # even a blocking ``query()`` call goes out over ``httpx.AsyncClient``.
    class FakeHttpClient:
        def __init__(self, *, timeout: Any, headers: dict[str, str]) -> None:
            if calls is not None:
                calls.append({"timeout": timeout, "headers": headers})

        async def __aenter__(self) -> "FakeHttpClient":
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def request(self, method: str, url: str, *, json: dict[str, Any]) -> FakeResponse:
            assert method == "POST"
            if calls is not None:
                calls.append({"url": url, "json": json})
            return FakeResponse()

    monkeypatch.setattr(service_client_module.httpx, "AsyncClient", FakeHttpClient)


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


def test_sync_and_async_query_agree_on_the_wire_payload(monkeypatch) -> None:
    """The sync facade must send exactly what the async implementation sends."""
    sync_calls: list[dict[str, Any]] = []
    _install_query_response(monkeypatch, {"results": [{"hits": []}]}, sync_calls)
    RetrieverServiceClient(base_url="http://svc:7670").query("deployment?", top_k=3)

    async_calls: list[dict[str, Any]] = []
    _install_query_response(monkeypatch, {"results": [{"hits": []}]}, async_calls)
    asyncio.run(RetrieverServiceClient(base_url="http://svc:7670").aquery("deployment?", top_k=3))

    assert sync_calls[1] == async_calls[1]


def test_query_requires_top_k_while_aquery_defaults_to_ten() -> None:
    """A released signature asymmetry: keep it, do not unify the two.

    ``query`` has always required ``top_k``; ``aquery`` has always defaulted
    it. Sharing one implementation must not quietly change either.
    """
    sync_top_k = inspect.signature(RetrieverServiceClient.query).parameters["top_k"]
    async_top_k = inspect.signature(RetrieverServiceClient.aquery).parameters["top_k"]

    assert sync_top_k.default is inspect.Parameter.empty
    assert async_top_k.default == 10


def test_sync_facade_works_from_inside_a_running_event_loop(monkeypatch) -> None:
    """``asyncio.run`` cannot nest, so the facade must fall back to a thread."""
    _install_query_response(monkeypatch, {"results": [{"hits": [{"text": "passage"}]}]})
    client = RetrieverServiceClient(base_url="http://svc:7670")

    async def _call_sync_from_async() -> Any:
        # Deliberately blocking: this is the call that used to be impossible.
        return client.query("deployment?", top_k=1)

    assert asyncio.run(_call_sync_from_async()) == [[{"text": "passage"}]]


# ----------------------------------------------------------------------
# Sync/async parity for the collection, document and job lifecycle
#
# Each operation is implemented once as a coroutine, with the synchronous
# method as a thin facade. These pin that both entry points issue the same
# request and parse the same response, which nothing covered before.
# ----------------------------------------------------------------------


def _install_lifecycle_response(
    monkeypatch: pytest.MonkeyPatch,
    body: dict[str, Any],
    calls: list[dict[str, Any]],
) -> None:
    class FakeResponse:
        status_code = 200
        text = ""
        content = b"{}"

        def json(self) -> dict[str, Any]:
            return body

    class FakeAsyncClient:
        def __init__(self, *, timeout: Any, headers: dict[str, str]) -> None:
            self._headers = headers

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
            calls.append({"method": method, "url": url, "headers": self._headers, **kwargs})
            return FakeResponse()

    monkeypatch.setattr(service_client_module.httpx, "AsyncClient", FakeAsyncClient)


_TS = "2026-01-01T00:00:00+00:00"
_COLLECTION_BODY = {
    "name": "research",
    "scope": "workspace",
    "status": "active",
    "created_at": _TS,
    "updated_at": _TS,
}
_DOCUMENT_BODY = {
    "document_id": "document-1",
    "collection_name": "research",
    "scope": "workspace",
    "filename": "paper.pdf",
    "content_sha256": "a" * 64,
    "document_version": "v1",
    "status": "completed",
    "chunk_count": 3,
    "created_at": _TS,
    "updated_at": _TS,
}
_JOB_BODY = {
    "job_id": "job-1",
    "expected_documents": 1,
    "status": "completed",
    "created_at": _TS,
}

_LIFECYCLE_CASES = [
    ("get_collection", ("research",), {}, _COLLECTION_BODY, "GET", "/v1/collections/research"),
    ("list_collections", (), {}, {"items": [], "next_token": None}, "GET", "/v1/collections"),
    ("update_collection", ("research",), {"description": "d"}, _COLLECTION_BODY, "PATCH", "/v1/collections/research"),
    (
        "delete_collection",
        ("research",),
        {},
        {"name": "research", "scope": "workspace", "existed": True, "deleted": False, "status": "deleting"},
        "DELETE",
        "/v1/collections/research",
    ),
    (
        "list_documents",
        ("research",),
        {},
        {"items": [], "next_token": None},
        "GET",
        "/v1/collections/research/documents",
    ),
    (
        "get_document",
        ("research", "document-1"),
        {},
        _DOCUMENT_BODY,
        "GET",
        "/v1/collections/research/documents/document-1",
    ),
    (
        "delete_document",
        ("research", "document-1"),
        {},
        {
            "document_id": "document-1",
            "collection_name": "research",
            "scope": "workspace",
            "existed": True,
            "deleted": False,
            "status": "deleting",
        },
        "DELETE",
        "/v1/collections/research/documents/document-1",
    ),
    ("get_job", ("job-1",), {}, _JOB_BODY, "GET", "/v1/ingest/job/job-1"),
    (
        "list_job_documents",
        ("job-1",),
        {},
        {"job_id": "job-1", "total": 0, "total_filtered": 0, "offset": 0, "limit": 100, "items": []},
        "GET",
        "/v1/ingest/job/job-1/documents",
    ),
]


@pytest.mark.parametrize(
    ("name", "args", "kwargs", "body", "method", "path"),
    _LIFECYCLE_CASES,
    ids=[case[0] for case in _LIFECYCLE_CASES],
)
def test_sync_and_async_lifecycle_methods_issue_identical_requests(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    body: dict[str, Any],
    method: str,
    path: str,
) -> None:
    sync_calls: list[dict[str, Any]] = []
    _install_lifecycle_response(monkeypatch, body, sync_calls)
    client = RetrieverServiceClient(base_url="http://svc:7670", scope="workspace")
    sync_result = getattr(client, name)(*args, **kwargs)

    async_calls: list[dict[str, Any]] = []
    _install_lifecycle_response(monkeypatch, body, async_calls)
    async_result = asyncio.run(getattr(client, f"a{name}")(*args, **kwargs))

    assert sync_calls == async_calls
    assert sync_calls[0]["method"] == method
    assert sync_calls[0]["url"] == f"http://svc:7670{path}"
    assert sync_result == async_result
