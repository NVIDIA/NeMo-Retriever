# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Split-topology callback and retained-result storage coverage."""

from __future__ import annotations

import asyncio
import errno
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from nemo_retriever.service.services import worker_result_store
from nemo_retriever.service.services.job_tracker import (
    DEFAULT_STALE_JOB_TTL_S,
    DEFAULT_TTL_S,
)
from nemo_retriever.service.services.pipeline_pool import (
    _CallbackDeliveryOutcome,
    _fire_gateway_callback,
)
from nemo_retriever.service.services.worker_result_store import (
    ResultStoreTemporarilyUnavailable,
    clear_for_tests,
    discard_local_result_data,
    get_result_data,
    store_result_data,
    validate_result_store,
)


@pytest.fixture(autouse=True)
def _clear_worker_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEMO_RETRIEVER_RESULTS_DIR", raising=False)
    monkeypatch.delenv("NEMO_RETRIEVER_RESULTS_TTL_SECONDS", raising=False)
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
    assert get_result_data("doc-1") == rows


def test_worker_document_result_endpoint_is_idempotent() -> None:
    from fastapi.testclient import TestClient

    from nemo_retriever.service.app import create_app
    from nemo_retriever.service.config import (
        PipelineOverridesConfig,
        PipelinePoolConfig,
        ServiceConfig,
    )

    cfg = ServiceConfig(
        mode="batch",
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        pipeline_overrides=PipelineOverridesConfig(),
    )
    rows = [{"text": "hello"}]
    store_result_data("doc-x", rows)
    with TestClient(create_app(cfg)) as client:
        assert client.get("/v1/internal/document-result/doc-x").json()["result_data"] == rows
        assert client.get("/v1/internal/document-result/doc-x").json()["result_data"] == rows


def test_shared_result_store_is_traversal_safe_and_cross_process_visible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    rows = [{"page": 1, "text": "shared"}]

    store_result_data("../unsafe/document-id", rows)
    clear_for_tests()  # Simulate reading from another pod/process.

    assert get_result_data("../unsafe/document-id") == rows
    assert get_result_data("../unsafe/document-id") == rows
    assert worker_result_store._document_dir(tmp_path, "../unsafe/document-id").is_relative_to(tmp_path)


def test_shared_result_store_supports_concurrent_idempotent_readers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    rows = [{"text": "read repeatedly"}]
    store_result_data("doc-concurrent", rows)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: get_result_data("doc-concurrent"), range(8)))

    assert results == [rows] * 8


def test_shared_result_store_preserves_generation_after_read_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    rows = [{"text": "retry read"}]
    store_result_data("doc-read-error", rows)
    generation = next(worker_result_store._document_dir(tmp_path, "doc-read-error").glob("*.json"))
    original_open = Path.open
    fail_read = True

    def fail_generation_read_once(path: Path, *args: Any, **kwargs: Any) -> Any:
        nonlocal fail_read
        if fail_read and path == generation:
            fail_read = False
            raise OSError(errno.EIO, "I/O error")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_generation_read_once)

    with pytest.raises(ResultStoreTemporarilyUnavailable):
        get_result_data("doc-read-error")
    assert generation.exists()
    assert get_result_data("doc-read-error") == rows


@pytest.mark.parametrize("payload", ["{", '{"unexpected":true}', "[1]"])
def test_shared_result_store_preserves_invalid_payload_for_diagnosis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, payload: str
) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    document_dir = worker_result_store._document_dir(tmp_path, "invalid")
    document_dir.mkdir(parents=True)
    generation = document_dir / f"{('a' * 32)}.json"
    generation.write_text(payload, encoding="utf-8")

    with pytest.raises(ResultStoreTemporarilyUnavailable):
        get_result_data("invalid")

    assert generation.read_text(encoding="utf-8") == payload


def test_shared_result_store_chooses_newest_completed_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    store_result_data("regenerated", [{"version": 1}])
    first = next(worker_result_store._document_dir(tmp_path, "regenerated").glob("*.json"))
    old = time.time() - 10
    os.utime(first, (old, old))

    store_result_data("regenerated", [{"version": 2}])

    assert get_result_data("regenerated") == [{"version": 2}]


def test_expiry_sweep_cannot_delete_concurrent_fresh_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_TTL_SECONDS", "60")
    store_result_data("replaced", [{"version": 1}])
    old_generation = next(worker_result_store._document_dir(tmp_path, "replaced").glob("*.json"))
    old = time.time() - 61
    os.utime(old_generation, (old, old))
    original_unlink = Path.unlink
    published_replacement = False

    def publish_before_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        nonlocal published_replacement
        if path == old_generation and not published_replacement:
            published_replacement = True
            store_result_data("replaced", [{"version": 2}])
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", publish_before_unlink)
    worker_result_store._remove_expired_file(old_generation, cutoff=time.time() - 60)

    assert get_result_data("replaced") == [{"version": 2}]


def test_shared_result_store_removes_only_expired_owned_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_TTL_SECONDS", "60")
    document_dir = worker_result_store._document_dir(tmp_path, "abandoned")
    document_dir.mkdir(parents=True)
    stale_files = [
        document_dir / f"{('a' * 32)}.json",
        document_dir / f".{('b' * 32)}.tmp",
    ]
    for path in stale_files:
        path.write_text("[]", encoding="utf-8")
        old = time.time() - 61
        os.utime(path, (old, old))
    unrelated = worker_result_store._results_root(tmp_path) / "keep-me.json"
    unrelated.write_text("[]", encoding="utf-8")

    store_result_data("fresh", [{"text": "available"}])

    assert all(not path.exists() for path in stale_files)
    assert unrelated.exists()
    assert get_result_data("fresh") == [{"text": "available"}]


def test_sweep_preserves_fresh_empty_document_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    document_dir = worker_result_store._document_dir(tmp_path, "about-to-publish")
    document_dir.mkdir(parents=True)

    worker_result_store._sweep_expired_files(tmp_path, now=time.time(), ttl_s=60)

    assert document_dir.is_dir()


def test_in_memory_result_store_is_idempotent_and_ttl_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_TTL_SECONDS", "60")
    monotonic_now = 100.0
    monkeypatch.setattr(worker_result_store.time, "monotonic", lambda: monotonic_now)
    rows = [{"text": "memory"}]
    store_result_data("memory", rows)

    assert get_result_data("memory") == rows
    assert get_result_data("memory") == rows

    monotonic_now = 161.0
    assert get_result_data("memory") is None


@pytest.mark.parametrize("shared", [False, True])
def test_result_store_isolates_stored_and_returned_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    shared: bool,
) -> None:
    if shared:
        monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    rows = [{"metadata": {"tags": ["original"]}}]
    expected = [{"metadata": {"tags": ["original"]}}]

    store_result_data("isolated", rows)
    rows[0]["metadata"]["tags"].append("producer-mutation")

    first_read = get_result_data("isolated")
    assert first_read == expected
    assert first_read is not None
    first_read[0]["metadata"]["tags"].append("reader-mutation")

    assert get_result_data("isolated") == expected


def test_shared_result_store_default_ttl_covers_full_job_lifecycle() -> None:
    assert worker_result_store._results_ttl_s() == DEFAULT_STALE_JOB_TTL_S + DEFAULT_TTL_S


def test_result_store_validation_probes_required_operations(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))

    validate_result_store()

    assert not list(worker_result_store._results_root(tmp_path).iterdir())


def test_result_store_validation_rejects_unsupported_atomic_rename(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fastapi.testclient import TestClient

    from nemo_retriever.service.app import create_app
    from nemo_retriever.service.config import ServiceConfig

    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))

    def unsupported_replace(*_: object) -> None:
        raise OSError(errno.EOPNOTSUPP, "Atomic rename is not supported")

    monkeypatch.setattr(worker_result_store.os, "replace", unsupported_replace)

    with pytest.raises(RuntimeError, match="same-directory atomic rename"):
        with TestClient(create_app(ServiceConfig(mode="gateway"))):
            pass


def test_worker_result_endpoint_returns_retryable_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi.testclient import TestClient

    from nemo_retriever.service.app import create_app
    from nemo_retriever.service.config import ServiceConfig
    from nemo_retriever.service.routers import ingest

    def unavailable(_: str) -> None:
        raise ResultStoreTemporarilyUnavailable("shared result store unavailable")

    monkeypatch.setattr(ingest, "get_result_data", unavailable)

    with TestClient(create_app(ServiceConfig(mode="batch"))) as client:
        response = client.get("/v1/internal/document-result/doc-unavailable")

    assert response.status_code == 503
    assert response.headers["retry-after"] == "60"
    assert response.json()["detail"] == "shared result store unavailable"


def test_gateway_fetch_returns_retryable_503_when_shared_store_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import HTTPException

    from nemo_retriever.service.routers import ingest

    def unavailable(_: str) -> None:
        raise ResultStoreTemporarilyUnavailable("shared result store unavailable")

    monkeypatch.setattr(ingest, "get_result_data", unavailable)

    with pytest.raises(HTTPException) as error:
        asyncio.run(ingest._fetch_result_data_from_workers("doc-unavailable"))

    assert error.value.status_code == 503
    assert error.value.headers == {"Retry-After": "60"}


def test_gateway_fetches_shared_result_before_proxy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from nemo_retriever.service.routers.ingest import _fetch_result_data_from_workers

    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    rows = [{"text": "gateway"}]
    store_result_data("doc-gateway", rows)

    assert asyncio.run(_fetch_result_data_from_workers("doc-gateway")) == rows


def test_gateway_returns_503_for_missing_configured_shared_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fastapi import HTTPException

    from nemo_retriever.service.routers.ingest import _fetch_result_data_from_workers

    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))

    with pytest.raises(HTTPException) as error:
        asyncio.run(_fetch_result_data_from_workers("doc-missing"))

    assert error.value.status_code == 503
    assert error.value.headers == {"Retry-After": "60"}


def test_gateway_status_routes_read_shared_results_idempotently(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fastapi.testclient import TestClient

    from nemo_retriever.service.app import create_app
    from nemo_retriever.service.config import ServiceConfig
    from nemo_retriever.service.services.job_tracker import get_job_tracker

    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    rows = [{"text": "shared route"}]

    with TestClient(create_app(ServiceConfig(mode="gateway"))) as client:
        tracker = get_job_tracker()
        assert tracker is not None
        tracker.register_job("job-shared", expected_documents=2, retain_results=True)

        tracker.register_document("doc-job-route", job_id="job-shared")
        tracker.mark_completed("doc-job-route", result_rows=1)
        store_result_data("doc-job-route", rows)
        for _ in range(2):
            response = client.get("/v1/ingest/job/job-shared/document/doc-job-route")
            assert response.status_code == 200
            assert response.json()["result_data"] == rows

        tracker.register_document("doc-status-route", job_id="job-shared")
        tracker.mark_completed("doc-status-route", result_rows=1)
        store_result_data("doc-status-route", rows)
        for _ in range(2):
            response = client.get("/v1/ingest/status/doc-status-route")
            assert response.status_code == 200
            assert response.json()["result_data"] == rows


def test_fire_gateway_callback_retries_and_advertises_worker_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_retriever.service.services import pipeline_pool

    attempts: list[dict[str, Any]] = []

    class _Resp:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    class _Client:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def post(self, url: str, json: dict[str, Any]) -> _Resp:
            attempts.append({"url": url, "json": json})
            return _Resp(503 if len(attempts) == 1 else 200)

    monkeypatch.setattr(pipeline_pool, "_CALLBACK_RETRY_DELAYS_S", (0.0,))
    with patch("httpx.AsyncClient", _Client):
        outcome = asyncio.run(
            _fire_gateway_callback(
                "http://gateway/v1/internal/job-callback",
                "doc-retry",
                "completed",
                result_rows=2,
                result_worker_ip="10.1.2.3",
            )
        )

    assert outcome == _CallbackDeliveryOutcome.ACKNOWLEDGED
    assert len(attempts) == 2
    assert attempts[-1]["json"]["result_worker_ip"] == "10.1.2.3"
    assert "result_data" not in attempts[-1]["json"]


def test_fire_gateway_callback_does_not_retry_permanent_4xx() -> None:
    attempts = 0

    class _Resp:
        status_code = 401

    class _Client:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def post(self, url: str, json: dict[str, Any]) -> _Resp:
            nonlocal attempts
            attempts += 1
            return _Resp()

    with patch("httpx.AsyncClient", _Client):
        outcome = asyncio.run(
            _fire_gateway_callback(
                "http://gateway/v1/internal/job-callback",
                "permanently-rejected-doc",
                "completed",
            )
        )

    assert outcome == _CallbackDeliveryOutcome.PERMANENT_FAILURE
    assert attempts == 1


def test_discard_local_result_data_removes_acknowledged_worker_rows() -> None:
    store_result_data("acknowledged", [{"text": "copied"}])

    discard_local_result_data("acknowledged")

    assert get_result_data("acknowledged") is None


def test_gateway_callback_copies_result_before_completing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fastapi.testclient import TestClient

    from nemo_retriever.service.app import create_app
    from nemo_retriever.service.config import ServiceConfig
    from nemo_retriever.service.routers import ingest
    from nemo_retriever.service.services.job_tracker import (
        DocumentStatus,
        get_job_tracker,
    )

    rows = [{"text": "owned by worker 10.1.2.3"}]
    requested_urls: list[str] = []

    class _Resp:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"result_data": rows}

    class _Client:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def get(self, url: str) -> _Resp:
            requested_urls.append(url)
            return _Resp()

    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    with TestClient(create_app(ServiceConfig(mode="gateway"))) as client:
        tracker = get_job_tracker()
        assert tracker is not None
        tracker.register_job("handoff-job", expected_documents=1, retain_results=True)
        tracker.register_document("handoff-doc", job_id="handoff-job")
        tracker.mark_processing("handoff-doc")

        monkeypatch.setattr(ingest.httpx, "AsyncClient", _Client)
        response = client.post(
            "/v1/internal/job-callback",
            json={
                "id": "handoff-doc",
                "status": "completed",
                "result_rows": 1,
                "result_worker_ip": "10.1.2.3",
            },
        )

        assert response.status_code == 200
        assert requested_urls == ["http://10.1.2.3:7670/v1/internal/document-result/handoff-doc"]
        record = tracker.get_document("handoff-doc")
        assert record is not None
        assert record.status == DocumentStatus.COMPLETED
        assert get_result_data("handoff-doc") == rows

        status_response = client.get("/v1/ingest/status/handoff-doc")
        assert status_response.status_code == 200
        assert status_response.json()["result_data"] == rows


def test_gateway_callback_does_not_complete_when_result_handoff_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fastapi.testclient import TestClient

    from nemo_retriever.service.app import create_app
    from nemo_retriever.service.config import ServiceConfig
    from nemo_retriever.service.routers import ingest
    from nemo_retriever.service.services.job_tracker import (
        DocumentStatus,
        get_job_tracker,
    )

    class _Resp:
        status_code = 404

    class _Client:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def get(self, url: str) -> _Resp:
            return _Resp()

    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    with TestClient(create_app(ServiceConfig(mode="gateway"))) as client:
        tracker = get_job_tracker()
        assert tracker is not None
        tracker.register_job("failed-handoff-job", expected_documents=1, retain_results=True)
        tracker.register_document("failed-handoff-doc", job_id="failed-handoff-job")
        tracker.mark_processing("failed-handoff-doc")

        monkeypatch.setattr(ingest.httpx, "AsyncClient", _Client)
        response = client.post(
            "/v1/internal/job-callback",
            json={
                "id": "failed-handoff-doc",
                "status": "completed",
                "result_rows": 1,
                "result_worker_ip": "10.1.2.3",
            },
        )

        assert response.status_code == 503
        assert response.headers["retry-after"] == "1"
        record = tracker.get_document("failed-handoff-doc")
        assert record is not None
        assert record.status == DocumentStatus.PROCESSING
        assert get_result_data("failed-handoff-doc") is None


def test_worker_result_url_supports_ipv6_and_rejects_spoofed_peer() -> None:
    from types import SimpleNamespace

    from fastapi import HTTPException
    from starlette.requests import Request

    from nemo_retriever.service.config import ServiceConfig
    from nemo_retriever.service.routers.ingest import _worker_result_url

    app = SimpleNamespace(state=SimpleNamespace(config=ServiceConfig(mode="gateway")))

    def make_request(peer: str) -> Request:
        return Request(
            {
                "type": "http",
                "app": app,
                "client": (peer, 12345),
                "headers": [],
                "method": "POST",
                "path": "/v1/internal/job-callback",
                "query_string": b"",
                "scheme": "http",
                "server": ("gateway", 7670),
            }
        )

    assert (
        _worker_result_url(make_request("fd00::1"), "ipv6-doc", "fd00::1")
        == "http://[fd00::1]:7670/v1/internal/document-result/ipv6-doc"
    )

    with pytest.raises(HTTPException) as mismatch:
        _worker_result_url(make_request("10.1.2.4"), "spoofed-doc", "10.1.2.3")
    assert mismatch.value.status_code == 400

    with pytest.raises(HTTPException) as missing:
        _worker_result_url(make_request("testclient"), "missing-doc", None)
    assert missing.value.status_code == 503


def test_internal_auth_headers_support_default_and_custom_header_names() -> None:
    from nemo_retriever.service.auth import auth_headers
    from nemo_retriever.service.config import AuthConfig

    assert auth_headers(AuthConfig(api_token="secret")) == {"Authorization": "Bearer secret"}
    assert auth_headers(AuthConfig(api_token="secret", header_name="X-Service-Token")) == {"X-Service-Token": "secret"}


def test_fire_gateway_callback_sends_internal_auth_headers() -> None:
    client_kwargs: dict[str, Any] = {}

    class _Resp:
        status_code = 200

    class _Client:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            client_kwargs.update(kwargs)

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def post(self, url: str, json: dict[str, Any]) -> _Resp:
            return _Resp()

    with patch("httpx.AsyncClient", _Client):
        outcome = asyncio.run(
            _fire_gateway_callback(
                "http://gateway/v1/internal/job-callback",
                "authenticated-doc",
                "completed",
                callback_headers={"X-Service-Token": "secret"},
            )
        )

    assert outcome == _CallbackDeliveryOutcome.ACKNOWLEDGED
    assert client_kwargs["headers"] == {"X-Service-Token": "secret"}


def test_deferred_callback_retries_without_rerunning_and_discards_after_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_retriever.service.services import pipeline_pool
    from nemo_retriever.service.services.pipeline_pool import _Pool

    attempts = 0

    async def succeeds(*args: Any, **kwargs: Any) -> _CallbackDeliveryOutcome:
        nonlocal attempts
        attempts += 1
        return _CallbackDeliveryOutcome.ACKNOWLEDGED

    async def run() -> None:
        pool = _Pool("deferred-test", num_workers=1, max_queue_size=1)
        pool._running = True
        pool._handoff_slots = asyncio.BoundedSemaphore(pool.num_workers)
        store_result_data("deferred-doc", [{"text": "retained"}])
        monkeypatch.setattr(pipeline_pool, "_fire_gateway_callback", succeeds)
        monkeypatch.setattr(pipeline_pool, "_CALLBACK_DEFERRED_INITIAL_DELAY_S", 0.0)
        await pool._schedule_gateway_callback_retry(
            callback_url="http://gateway/v1/internal/job-callback",
            item_id="deferred-doc",
            status="completed",
            result_rows=1,
            result_worker_ip="10.1.2.3",
            callback_headers={"Authorization": "Bearer secret"},
            retain_results=True,
        )
        task = pool._handoff_tasks["deferred-doc"]
        await asyncio.wait_for(task, timeout=1.0)
        pool._running = False

    asyncio.run(run())

    assert attempts == 1
    assert get_result_data("deferred-doc") is None


def test_deferred_callback_tasks_are_bounded_and_cancellation_releases_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_retriever.service.services.pipeline_pool import _Pool

    async def run() -> None:
        pool = _Pool("bounded-deferred-test", num_workers=1, max_queue_size=1)
        pool._running = True
        pool._handoff_slots = asyncio.BoundedSemaphore(pool.num_workers)
        second_release = asyncio.Event()

        async def wait_for_release(self: _Pool, *, item_id: str, **kwargs: Any) -> None:
            if item_id == "second-doc":
                await second_release.wait()
            else:
                await asyncio.Event().wait()

        monkeypatch.setattr(_Pool, "_retry_gateway_callback_until_expired", wait_for_release)

        await pool._schedule_gateway_callback_retry(
            callback_url="http://gateway/v1/internal/job-callback",
            item_id="first-doc",
            status="completed",
        )
        first_task = pool._handoff_tasks["first-doc"]
        second_schedule = asyncio.create_task(
            pool._schedule_gateway_callback_retry(
                callback_url="http://gateway/v1/internal/job-callback",
                item_id="second-doc",
                status="completed",
            )
        )
        await asyncio.sleep(0)

        assert len(pool._handoff_tasks) == pool.num_workers
        assert not second_schedule.done()

        first_task.cancel()
        await asyncio.gather(first_task, return_exceptions=True)
        await asyncio.wait_for(second_schedule, timeout=1.0)

        assert len(pool._handoff_tasks) == pool.num_workers
        assert "second-doc" in pool._handoff_tasks

        second_task = pool._handoff_tasks["second-doc"]
        second_release.set()
        await asyncio.wait_for(second_task, timeout=1.0)
        await asyncio.sleep(0)
        assert pool._handoff_tasks == {}
        pool._running = False

    asyncio.run(run())


def test_shared_result_publication_fsyncs_created_parents_and_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    fsynced: list[Path] = []
    original = worker_result_store._fsync_directory

    def record_fsync(path: Path) -> None:
        fsynced.append(path)
        original(path)

    monkeypatch.setattr(worker_result_store, "_fsync_directory", record_fsync)
    store_result_data("durable-doc", [{"text": "durable"}])

    root = worker_result_store._results_root(tmp_path)
    document_dir = worker_result_store._document_dir(tmp_path, "durable-doc")
    assert fsynced == [tmp_path, root, document_dir]
    assert get_result_data("durable-doc") == [{"text": "durable"}]


def test_result_store_validation_rejects_unsupported_directory_fsync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))

    def unsupported_directory_fsync(_: Path) -> None:
        raise OSError(errno.EOPNOTSUPP, "Directory fsync is not supported")

    monkeypatch.setattr(worker_result_store, "_fsync_directory", unsupported_directory_fsync)

    with pytest.raises(RuntimeError, match="fsync"):
        validate_result_store()


def test_gateway_result_pull_sends_configured_internal_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from starlette.requests import Request

    from nemo_retriever.service.config import AuthConfig, ServiceConfig
    from nemo_retriever.service.routers import ingest

    client_kwargs: dict[str, Any] = {}

    class _Resp:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"result_data": [{"text": "authenticated handoff"}]}

    class _Client:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            client_kwargs.update(kwargs)

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def get(self, url: str) -> _Resp:
            return _Resp()

    app = SimpleNamespace(
        state=SimpleNamespace(
            config=ServiceConfig(
                mode="gateway",
                auth=AuthConfig(api_token="secret", header_name="X-Service-Token"),
            )
        )
    )
    request = Request(
        {
            "type": "http",
            "app": app,
            "client": ("10.1.2.3", 12345),
            "headers": [],
            "method": "POST",
            "path": "/v1/internal/job-callback",
            "query_string": b"",
            "scheme": "http",
            "server": ("gateway", 7670),
        }
    )

    monkeypatch.setattr(ingest.httpx, "AsyncClient", _Client)
    asyncio.run(ingest._pull_and_store_worker_result(request, "auth-pull-doc", "10.1.2.3"))

    assert client_kwargs["headers"] == {"X-Service-Token": "secret"}
    assert get_result_data("auth-pull-doc") == [{"text": "authenticated handoff"}]


def test_fire_gateway_callback_honors_capped_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nemo_retriever.service.services import pipeline_pool

    sleeps: list[float] = []
    attempts = 0

    class _Resp:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code
            self.headers = {"Retry-After": "120"}

    class _Client:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def post(self, url: str, json: dict[str, Any]) -> _Resp:
            nonlocal attempts
            attempts += 1
            return _Resp(503 if attempts == 1 else 200)

    async def record_sleep(delay_s: float) -> None:
        sleeps.append(delay_s)

    monkeypatch.setattr(pipeline_pool, "_CALLBACK_RETRY_DELAYS_S", (0.0,))
    monkeypatch.setattr(pipeline_pool.asyncio, "sleep", record_sleep)
    with patch("httpx.AsyncClient", _Client):
        outcome = asyncio.run(
            _fire_gateway_callback(
                "http://gateway/v1/internal/job-callback",
                "retry-after-doc",
                "completed",
                retry_after_cap_s=2.0,
            )
        )

    assert outcome == _CallbackDeliveryOutcome.ACKNOWLEDGED
    assert sleeps == [2.0]
