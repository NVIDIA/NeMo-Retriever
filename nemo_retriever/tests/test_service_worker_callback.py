# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Split-topology worker callback must not POST full result_data payloads."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from nemo_retriever.service.services import worker_result_store
from nemo_retriever.service.services.job_tracker import DEFAULT_STALE_JOB_TTL_S, DEFAULT_TTL_S
from nemo_retriever.service.services.pipeline_pool import _fire_gateway_callback
from nemo_retriever.service.services.worker_result_store import (
    ResultStoreTemporarilyUnavailable,
    clear_for_tests,
    consume_result_data,
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


def test_shared_result_store_is_visible_across_memory_stores(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    rows = [{"page": 1, "text": "shared"}]

    store_result_data("../unsafe/document-id", rows)
    clear_for_tests()  # Simulate reading from another pod/process.

    assert consume_result_data("../unsafe/document-id") == rows
    assert consume_result_data("../unsafe/document-id") is None
    assert not list(tmp_path.iterdir())


def test_shared_result_store_has_single_consumer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    rows = [{"text": "consume once"}]
    store_result_data("doc-concurrent", rows)

    with ThreadPoolExecutor(max_workers=8) as executor:
        consumed = list(executor.map(lambda _: consume_result_data("doc-concurrent"), range(8)))

    assert consumed.count(rows) == 1
    assert consumed.count(None) == 7


def test_shared_result_store_tolerates_concurrently_swept_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    store_result_data("doc-expiring", [{"text": "expiring"}])
    original_replace = os.replace

    def replace_then_remove_claim(source: Path, destination: Path) -> None:
        original_replace(source, destination)
        if destination.name.endswith(".claim"):
            destination.unlink()

    monkeypatch.setattr(worker_result_store.os, "replace", replace_then_remove_claim)

    assert consume_result_data("doc-expiring") is None
    assert not list(tmp_path.iterdir())


def test_shared_result_store_retries_after_claim_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    rows = [{"text": "retry claim"}]
    store_result_data("doc-claim-error", rows)
    target = worker_result_store._result_path(tmp_path, "doc-claim-error")
    original_replace = os.replace
    fail_claim = True

    def fail_claim_once(source: Path, destination: Path) -> None:
        nonlocal fail_claim
        if fail_claim and source == target and destination.name.endswith(".claim"):
            fail_claim = False
            raise OSError(errno.ESTALE, "Stale file handle")
        original_replace(source, destination)

    monkeypatch.setattr(worker_result_store.os, "replace", fail_claim_once)

    with pytest.raises(ResultStoreTemporarilyUnavailable):
        consume_result_data("doc-claim-error")
    assert consume_result_data("doc-claim-error") == rows


def test_shared_result_store_restores_claim_after_read_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    rows = [{"text": "retry read"}]
    store_result_data("doc-read-error", rows)
    target = worker_result_store._result_path(tmp_path, "doc-read-error")
    original_open = Path.open
    fail_read = True

    def fail_claim_read_once(path: Path, *args: Any, **kwargs: Any) -> Any:
        nonlocal fail_read
        if fail_read and path.name.endswith(".claim"):
            fail_read = False
            raise OSError(errno.EIO, "I/O error")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_claim_read_once)

    with pytest.raises(ResultStoreTemporarilyUnavailable):
        consume_result_data("doc-read-error")
    assert target.exists()
    assert consume_result_data("doc-read-error") == rows


def test_shared_result_store_recovers_abandoned_claim_after_restore_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    rows = [{"text": "preserve"}]
    store_result_data("doc-restore-error", rows)
    original_open = Path.open
    original_link = os.link

    def fail_claim_read(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.name.endswith(".claim"):
            raise OSError(errno.EIO, "I/O error")
        return original_open(path, *args, **kwargs)

    def fail_restore(source: Path, destination: Path) -> None:
        if source.name.endswith(".claim"):
            raise OSError(errno.ESTALE, "Stale file handle")
        original_link(source, destination)

    with monkeypatch.context() as context:
        context.setattr(Path, "open", fail_claim_read)
        context.setattr(worker_result_store.os, "link", fail_restore)
        with pytest.raises(ResultStoreTemporarilyUnavailable):
            consume_result_data("doc-restore-error")

    claims = list(tmp_path.glob("*.claim"))
    assert len(claims) == 1
    expired_lease = time.time() - worker_result_store._CLAIM_LEASE_S - 1
    os.utime(claims[0], (expired_lease, expired_lease))

    assert consume_result_data("doc-restore-error") == rows
    assert not list(tmp_path.iterdir())


def test_shared_result_store_does_not_steal_active_claim(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    store_result_data("doc-active", [{"text": "active"}])
    target = worker_result_store._result_path(tmp_path, "doc-active")
    claimed = tmp_path / f".{target.name}.{('a' * 32)}.claim"
    os.replace(target, claimed)
    os.utime(claimed, None)

    assert consume_result_data("doc-active") is None
    assert claimed.exists()


def test_shared_result_store_ignores_claim_cleanup_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    rows = [{"text": "successful read"}]
    store_result_data("doc-cleanup-error", rows)
    original_unlink = Path.unlink

    def fail_claim_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        if path.name.endswith(".claim"):
            raise OSError(errno.EIO, "I/O error")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_claim_unlink)

    assert consume_result_data("doc-cleanup-error") == rows
    assert len(list(tmp_path.glob("*.claim"))) == 1


@pytest.mark.parametrize("payload", ["{", '{"unexpected":true}'])
def test_shared_result_store_discards_invalid_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture, payload: str
) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    target = worker_result_store._result_path(tmp_path, "invalid")
    target.write_text(payload, encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger=worker_result_store.__name__):
        assert consume_result_data("invalid") is None

    assert "Unable to decode shared result payload for 'invalid'" in caplog.text
    assert not list(tmp_path.iterdir())


def test_shared_result_store_default_ttl_covers_full_job_lifecycle() -> None:
    assert worker_result_store._results_ttl_s() == DEFAULT_STALE_JOB_TTL_S + DEFAULT_TTL_S


def test_result_store_validation_probes_required_operations(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))

    validate_result_store()

    assert not list(tmp_path.iterdir())


def test_result_store_validation_rejects_unsupported_hard_links(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from fastapi.testclient import TestClient

    from nemo_retriever.service.app import create_app
    from nemo_retriever.service.config import ServiceConfig

    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))

    def unsupported_link(*_: object) -> None:
        raise OSError(errno.EOPNOTSUPP, "Hard links are not supported")

    monkeypatch.setattr(worker_result_store.os, "link", unsupported_link)

    with pytest.raises(RuntimeError, match="must support file creation"):
        with TestClient(create_app(ServiceConfig(mode="gateway"))):
            pass
    assert not list(tmp_path.iterdir())


def test_shared_result_store_removes_expired_owned_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_TTL_SECONDS", "60")
    digest = hashlib.sha256(b"abandoned").hexdigest()
    stale_files = [
        tmp_path / f"{digest}.json",
        tmp_path / f".{digest}.json.{("a" * 32)}.tmp",
        tmp_path / f".{digest}.json.{("b" * 32)}.claim",
    ]
    for path in stale_files:
        path.write_text("[]", encoding="utf-8")
        os.utime(path, (time.time() - 61, time.time() - 61))
    unrelated = tmp_path / "keep-me.json"
    unrelated.write_text("[]", encoding="utf-8")

    store_result_data("fresh", [{"text": "available"}])

    assert all(not path.exists() for path in stale_files)
    assert unrelated.exists()
    assert consume_result_data("fresh") == [{"text": "available"}]


def test_expiry_sweep_does_not_delete_concurrently_replaced_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = worker_result_store._result_path(tmp_path, "replaced")
    target.write_text("[1]", encoding="utf-8")
    os.utime(target, (time.time() - 61, time.time() - 61))
    replacement = tmp_path / "replacement"
    replacement.write_text("[2]", encoding="utf-8")
    original_replace = os.replace

    def replace_during_claim(source: Path, destination: Path) -> None:
        if source == target and str(destination).endswith(".cleanup"):
            original_replace(replacement, target)
        original_replace(source, destination)

    with monkeypatch.context() as context:
        context.setattr(worker_result_store.os, "replace", replace_during_claim)
        removed = worker_result_store._remove_expired_result(target, cutoff=time.time() - 60)

    assert not removed
    assert target.read_text(encoding="utf-8") == "[2]"


def test_expiry_sweep_leaves_result_when_hard_links_are_unsupported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = worker_result_store._result_path(tmp_path, "unsupported-hard-links")
    target.write_text("[1]", encoding="utf-8")
    os.utime(target, (time.time() - 61, time.time() - 61))

    def unsupported_link(*_: object) -> None:
        raise OSError(errno.EOPNOTSUPP, "Hard links are not supported")

    def unexpected_replace(*_: object) -> None:
        raise AssertionError("A result must not be claimed when hard links are unsupported")

    with monkeypatch.context() as context:
        context.setattr(worker_result_store.os, "link", unsupported_link)
        context.setattr(worker_result_store.os, "replace", unexpected_replace)
        removed = worker_result_store._remove_expired_result(target, cutoff=time.time() - 60)

    assert not removed
    assert target.read_text(encoding="utf-8") == "[1]"


def test_shared_result_store_keeps_unexpired_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_TTL_SECONDS", "3600")
    store_result_data("existing", [{"text": "existing"}])
    clear_for_tests()  # Make the next operation run an immediate sweep.

    store_result_data("new", [{"text": "new"}])

    assert consume_result_data("existing") == [{"text": "existing"}]
    assert consume_result_data("new") == [{"text": "new"}]


def test_worker_result_endpoint_returns_retryable_503(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from nemo_retriever.service.app import create_app
    from nemo_retriever.service.config import ServiceConfig
    from nemo_retriever.service.routers import ingest

    def unavailable(_: str) -> None:
        raise ResultStoreTemporarilyUnavailable("shared result store unavailable")

    monkeypatch.setattr(ingest, "consume_result_data", unavailable)

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

    monkeypatch.setattr(ingest, "consume_result_data", unavailable)

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


def test_gateway_falls_back_to_proxy_for_invalid_shared_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from nemo_retriever.service.routers import ingest

    monkeypatch.setenv("NEMO_RETRIEVER_RESULTS_DIR", str(tmp_path))
    target = worker_result_store._result_path(tmp_path, "doc-invalid")
    target.write_text("{", encoding="utf-8")
    rows = [{"text": "worker fallback"}]

    class _Response:
        status_code = 200

        def json(self) -> dict[str, list[dict[str, str]]]:
            return {"result_data": rows}

    class _Client:
        async def get(self, _: str) -> _Response:
            return _Response()

    class _Proxy:
        def _client_for(self, _: object) -> _Client:
            return _Client()

    monkeypatch.setattr(ingest, "get_proxy", lambda: _Proxy())

    assert asyncio.run(ingest._fetch_result_data_from_workers("doc-invalid")) == rows


def test_gateway_status_routes_consume_shared_results(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
        response = client.get("/v1/ingest/job/job-shared/document/doc-job-route")
        assert response.status_code == 200
        assert response.json()["result_data"] == rows

        tracker.register_document("doc-status-route", job_id="job-shared")
        tracker.mark_completed("doc-status-route", result_rows=1)
        store_result_data("doc-status-route", rows)
        response = client.get("/v1/ingest/status/doc-status-route")
        assert response.status_code == 200
        assert response.json()["result_data"] == rows
