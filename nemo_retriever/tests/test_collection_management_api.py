# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hashlib
from typing import assert_type
from unittest.mock import AsyncMock, patch

import httpx
import lancedb
import pyarrow as pa
import pytest
from fastapi.testclient import TestClient

from nemo_retriever import RetrieverServiceClient
from nemo_retriever.common.schemas.collections import CollectionCreateRequest
from nemo_retriever.common.schemas.requests import JobCreateRequest
from nemo_retriever.service.auth import ScopeAuthorizer
from nemo_retriever.service.app import create_app
from nemo_retriever.service.config import AuthConfig, ServiceConfig, VectorDbConfig
from nemo_retriever.service.query_schema import QueryRequest
from nemo_retriever.service.errors import RetrieverServiceError
from nemo_retriever.service.aiq_contract import AIQCompatibleClient
import nemo_retriever.service.client as client_module
from nemo_retriever.service.services.job_tracker import JobFullError, JobTracker
from nemo_retriever.service.vectordb_app import VectorDBState, create_vectordb_app


def _row(chunk_id: str, document_id: str, text: str, version: str = "v1") -> dict:
    return {
        "vector": [1.0, 0.0],
        "pdf_page": "report_1",
        "filename": "report.pdf",
        "pdf_basename": "report",
        "page_number": 1,
        "source": '{"source_id":"report.pdf"}',
        "source_id": "report.pdf",
        "path": "report.pdf",
        "text": text,
        "metadata": '{"page_number":1}',
        "stored_image_uri": "",
        "content_type": "text",
        "bbox_xyxy_norm": "",
        "chunk_id": chunk_id,
        "document_id": document_id,
        "document_version": version,
        "content_sha256": version,
        "created_at": "now",
    }


def test_collection_crud_scope_pagination_and_injection_rejection(tmp_path) -> None:
    app = create_vectordb_app(lancedb_uri=str(tmp_path), embed_endpoint="http://embed")
    with TestClient(app) as client:
        headers = {"X-NRL-Scope": "workspace-a"}
        assert (
            client.post(
                "/v1/collections", json={"name": "one"}, headers=headers
            ).status_code
            == 201
        )
        assert (
            client.post(
                "/v1/collections", json={"name": "two"}, headers=headers
            ).status_code
            == 201
        )
        page = client.get("/v1/collections?limit=1", headers=headers).json()
        assert len(page["items"]) == 1 and page["next_token"]
        other_scope = client.get(
            "/v1/collections/one", headers={"X-NRL-Scope": "workspace-b"}
        )
        assert other_scope.status_code == 404
        injected = client.post(
            "/v1/query",
            json={"query": "x", "collection_name": "one", "table_name": "secret"},
            headers=headers,
        )
        assert injected.status_code == 422
        health = client.get("/v1/health").json()
        assert "table" not in health and "workspace-a" not in str(health)
        metrics = client.get("/metrics").text
        assert (
            "workspace-a" not in metrics and "nrl_vectordb_cleanup_pending" in metrics
        )
        assert client.delete("/v1/collections/one", headers=headers).status_code == 200
        repeated = client.delete(
            "/v1/collections/one?if_exists=true", headers=headers
        ).json()
        assert repeated == {
            "name": "one",
            "scope": "workspace-a",
            "existed": False,
            "deleted": False,
            "status": "deleted",
            "cleanup_pending": False,
        }


def test_append_replace_and_document_delete_are_collection_scoped(tmp_path) -> None:
    state = VectorDBState(str(tmp_path), "legacy", "", "model", "")
    state.create_collection("scope", CollectionCreateRequest(name="research"))
    state.write_rows(
        [_row("a", "doc", "old"), _row("b", "doc", "obsolete")],
        scope="scope",
        collection_name="research",
        document_id="doc",
        filename="report.pdf",
        content_sha256="v1",
    )
    assert state.total_rows(scope="scope", collection_name="research") == 2
    state.write_rows(
        [_row("c", "doc", "new", "v2")],
        scope="scope",
        collection_name="research",
        document_id="doc",
        filename="report.pdf",
        content_sha256="v2",
        operation="replace",
    )
    assert state.total_rows(scope="scope", collection_name="research") == 1
    assert state.get_document("scope", "research", "doc").document_version == "v2"
    assert state.delete_document("scope", "research", "doc", False).deleted is True
    assert state.delete_document("scope", "research", "doc", True).deleted is False


def test_aiq_protocol_and_citation_ready_query(tmp_path) -> None:
    app = create_vectordb_app(lancedb_uri=str(tmp_path), embed_endpoint="http://embed")
    with patch.object(VectorDBState, "embed_queries", return_value=[[1.0, 0.0]]):
        with TestClient(app) as service:

            class InProcessClient(RetrieverServiceClient):
                def _request(self, method: str, path: str, **kwargs):
                    response = service.request(
                        method, path, headers=self._auth_headers, **kwargs
                    )
                    self._raise_for_response(response, f"{method} {path}")
                    return response.json() if response.content else None

                async def _arequest(self, method: str, path: str, **kwargs):
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=app),
                        base_url="http://test",
                        headers=self._auth_headers,
                    ) as client:
                        response = await client.request(method, path, **kwargs)
                    self._raise_for_response(response, f"{method} {path}")
                    return response.json() if response.content else None

            sdk = InProcessClient(scope="workspace")
            collection = sdk.create_collection("research")
            assert collection.name == "research"
            service.post(
                "/internal/vectordb/write",
                json={
                    "rows": [_row("chunk", "doc", "finding")],
                    "scope": "workspace",
                    "collection_name": "research",
                    "document_id": "doc",
                    "filename": "report.pdf",
                    "content_sha256": "v1",
                },
            )
            hits = asyncio.run(sdk.aquery("finding", collection_name="research"))
            assert hits[0].chunk_id == "chunk"
            assert hits[0].text == "finding"
            assert 0.0 <= hits[0].score <= 1.0
            assert hits[0].filename == "report.pdf"
            assert sdk.list_documents("research").items[0].document_id == "doc"

            compatible: AIQCompatibleClient = sdk
            assert_type(compatible, AIQCompatibleClient)


def test_idempotency_replay_and_conflict() -> None:
    tracker = JobTracker()
    original = tracker.register_job(
        "job",
        expected_documents=1,
        scope="workspace",
        idempotency_key="request",
        idempotency_fingerprint="same",
    )
    replay = tracker.get_idempotent_job("workspace", "request", "same")
    assert replay is not None and replay.job_id == original.job_id

    try:
        tracker.get_idempotent_job("workspace", "request", "different")
    except JobFullError:
        pass
    else:
        raise AssertionError("conflicting idempotency payload must fail")


def test_manifest_entry_replay_is_capacity_neutral_and_identity_stable() -> None:
    tracker = JobTracker()
    tracker.register_job(
        "job",
        expected_documents=1,
        document_manifest=[
            {
                "manifest_entry_id": "a" * 64,
                "filename": "report.pdf",
                "content_sha256": "b" * 64,
            }
        ],
    )
    accepted, created = tracker.register_document_idempotent(
        "attempt-1",
        job_id="job",
        stable_document_id="document-1",
        filename="report.pdf",
        content_sha256="b" * 64,
        manifest_entry_id="a" * 64,
    )
    assert created is True
    tracker.mark_completed("attempt-1")
    replay, created = tracker.register_document_idempotent(
        "attempt-2",
        job_id="job",
        stable_document_id="document-2",
        filename="report.pdf",
        content_sha256="b" * 64,
        manifest_entry_id="a" * 64,
    )
    assert created is False
    assert replay.id == accepted.id == "attempt-1"
    assert replay.stable_document_id == accepted.stable_document_id == "document-1"
    assert len(tracker.job_documents("job")) == 1

    try:
        tracker.register_document_idempotent(
            "attempt-3",
            job_id="job",
            filename="report.pdf",
            content_sha256="c" * 64,
            manifest_entry_id="a" * 64,
        )
    except JobFullError:
        pass
    else:
        raise AssertionError("conflicting manifest replay must fail")


def test_raw_storage_selection_is_rejected_for_legacy_and_collection_requests() -> None:
    for key in ("table_name", "lancedb_uri", "uri", "physical_table"):
        try:
            JobCreateRequest.model_validate({"expected_documents": 1, key: "untrusted"})
        except ValueError:
            pass
        else:
            raise AssertionError(f"job storage key {key} must be rejected")
        try:
            QueryRequest.model_validate({"query": "x", key: "untrusted"})
        except ValueError:
            pass
        else:
            raise AssertionError(f"query storage key {key} must be rejected")


def test_scope_authorizer_secret_mapping_and_internal_vectordb_token(tmp_path) -> None:
    secret = tmp_path / "scope-tokens.json"
    secret.write_text(
        '{"tokens":[{"token":"alpha-token","scopes":["alpha"]}]}', encoding="utf-8"
    )
    authorizer = ScopeAuthorizer(
        AuthConfig(scope_token_file=str(secret), allow_unscoped_dev=False)
    )
    assert authorizer.authorize("alpha-token", "alpha") == ("alpha", None)
    assert authorizer.authorize("alpha-token", "beta") == (None, 404)
    assert authorizer.authorize("invalid", "alpha") == (None, 401)

    app = create_vectordb_app(
        lancedb_uri=str(tmp_path / "db"),
        embed_endpoint="http://embed",
        internal_api_token="internal-secret",
    )
    with TestClient(app) as client:
        assert client.get("/v1/health").status_code == 200
        assert client.get("/v1/collections").status_code == 401
        assert (
            client.get(
                "/v1/collections", headers={"X-NRL-Internal-Token": "wrong"}
            ).status_code
            == 401
        )
        assert (
            client.get(
                "/v1/collections", headers={"X-NRL-Internal-Token": "internal-secret"}
            ).status_code
            == 200
        )


def test_service_routes_use_authorized_scope_not_raw_header() -> None:
    app = create_app(
        ServiceConfig(
            mode="gateway",
            auth=AuthConfig(
                api_token="alpha-token",
                default_scope="alpha",
                allow_unscoped_dev=False,
            ),
            vectordb=VectorDbConfig(internal_api_token="internal-secret"),
        )
    )
    with TestClient(app) as client:
        assert (
            client.post("/v1/ingest/job", json={"expected_documents": 1}).status_code
            == 401
        )
        assert (
            client.post(
                "/v1/ingest/job",
                json={"expected_documents": 1},
                headers={"Authorization": "Bearer alpha-token", "X-NRL-Scope": "beta"},
            ).status_code
            == 404
        )
        created = client.post(
            "/v1/ingest/job",
            json={"expected_documents": 1},
            headers={"Authorization": "Bearer alpha-token", "X-NRL-Scope": "alpha"},
        )
        assert created.status_code == 201
        job_id = created.json()["job_id"]
        assert (
            client.get(
                f"/v1/ingest/job/{job_id}",
                headers={"Authorization": "Bearer alpha-token", "X-NRL-Scope": "alpha"},
            ).status_code
            == 200
        )
        assert (
            client.get(
                "/v1/internal/document-result/missing",
                headers={"Authorization": "Bearer alpha-token"},
            ).status_code
            == 401
        )
        assert (
            client.get(
                "/v1/internal/document-result/missing",
                headers={"X-NRL-Internal-Token": "internal-secret"},
            ).status_code
            == 404
        )


def test_sdk_replays_every_manifest_entry_after_idempotent_job_replay(
    tmp_path, monkeypatch
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")

    class FakeResponse:
        status_code = 200
        content = b"{}"
        text = ""

        def json(self):
            return {
                "job_id": "job",
                "expected_documents": 2,
                "status": "completed",
                "created_at": "now",
                "counts": {"completed": 2},
            }

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, _url):
            return FakeResponse()

    monkeypatch.setattr(client_module.httpx, "AsyncClient", FakeAsyncClient)
    sdk = RetrieverServiceClient()
    sdk._create_job = AsyncMock(
        return_value=client_module._CreatedJob("job", replayed=True)
    )
    sdk._upload_one = AsyncMock(return_value={"status": "accepted"})

    result = asyncio.run(
        sdk.asubmit_documents("research", [first, second], idempotency_key="key")
    )
    assert result.job_id == "job"
    assert sdk._upload_one.await_count == 2
    entry_ids = [
        call.kwargs["manifest_entry_id"] for call in sdk._upload_one.await_args_list
    ]
    expected = []
    for position, path in enumerate((first, second)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected.append(
            hashlib.sha256(
                f"{position}\0{path.name}\0{digest}".encode("utf-8")
            ).hexdigest()
        )
    assert entry_ids == expected


def test_sdk_wraps_malformed_sync_and_async_lifecycle_responses() -> None:
    class MalformedClient(RetrieverServiceClient):
        def _request(self, *_args, **_kwargs):
            return {}

        async def _arequest(self, *_args, **_kwargs):
            return {}

    sdk = MalformedClient()
    with pytest.raises(RetrieverServiceError, match="invalid response"):
        sdk.create_collection("research")
    with pytest.raises(RetrieverServiceError, match="invalid response"):
        asyncio.run(sdk.acreate_collection("research"))


def test_expiration_is_timezone_aware_and_normalized() -> None:
    with pytest.raises(ValueError, match="timezone"):
        CollectionCreateRequest(name="bad", expires_at="2030-01-01T00:00:00")
    request = CollectionCreateRequest(
        name="good", expires_at="2030-01-01T01:00:00+01:00"
    )
    assert request.expires_at == "2030-01-01T00:00:00+00:00"


def test_keyset_cursors_are_stable_and_context_bound(tmp_path) -> None:
    state = VectorDBState(str(tmp_path), "legacy", "", "model", "")
    state.create_collection("scope", CollectionCreateRequest(name="a"))
    state.create_collection("scope", CollectionCreateRequest(name="c"))
    first = state.list_collections("scope", 1, None)
    assert [item.name for item in first.items] == ["a"]
    state.create_collection("scope", CollectionCreateRequest(name="b"))
    second = state.list_collections("scope", 2, first.next_token)
    assert [item.name for item in second.items] == ["b", "c"]
    with pytest.raises(Exception, match="context"):
        state.list_collections("other", 1, first.next_token)

    state.write_rows(
        [_row("one", "doc-1", "one")],
        scope="scope",
        collection_name="a",
        document_id="doc-1",
        filename="one.pdf",
        content_sha256="v1",
    )
    state.write_rows(
        [_row("two", "doc-2", "two")],
        scope="scope",
        collection_name="a",
        document_id="doc-2",
        filename="two.pdf",
        content_sha256="v1",
    )
    documents = state.list_documents("scope", "a", 1, None)
    assert documents.next_token
    with pytest.raises(Exception, match="context"):
        state.list_documents("scope", "b", 1, documents.next_token)


def test_replacement_marker_recovers_after_catalog_finalize_failure(
    tmp_path, monkeypatch
) -> None:
    state = VectorDBState(str(tmp_path), "legacy", "", "model", "")
    state.create_collection("scope", CollectionCreateRequest(name="research"))
    state.write_rows(
        [_row("old", "doc", "old", "v1")],
        scope="scope",
        collection_name="research",
        document_id="doc",
        filename="report.pdf",
        content_sha256="v1",
    )
    original_persist = state._persist_document_row

    def fail_finalize(row):
        if row.get("current_document_version") == "v2":
            raise RuntimeError("injected finalize failure")
        return original_persist(row)

    monkeypatch.setattr(state, "_persist_document_row", fail_finalize)
    with pytest.raises(RuntimeError, match="injected"):
        state.write_rows(
            [_row("new", "doc", "new", "v2")],
            scope="scope",
            collection_name="research",
            document_id="doc",
            filename="report.pdf",
            content_sha256="v2",
            operation="replace",
        )
    monkeypatch.setattr(state, "_persist_document_row", original_persist)
    assert state.get_document("scope", "research", "doc").status == "replacing"
    result = state.reconcile()
    assert result["successes"] == 1
    assert state.get_document("scope", "research", "doc").document_version == "v2"
    table = state._open_table(state._resolved_table("scope", "research"))
    versions = {row["document_version"] for row in table.search().to_list()}
    assert versions == {"v2"}


def test_retryable_cleanup_removes_owned_artifacts_but_not_external_uris(
    tmp_path, monkeypatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    owned = artifact_root / "scope" / "research" / "doc" / "v1"
    owned.mkdir(parents=True)
    (owned / "image.png").write_bytes(b"image")
    external = tmp_path / "external"
    external.mkdir()
    (external / "keep.txt").write_text("keep", encoding="utf-8")

    state = VectorDBState(
        str(tmp_path / "db"),
        "legacy",
        "",
        "model",
        "",
        collection_artifact_root=str(artifact_root),
    )
    state.create_collection("scope", CollectionCreateRequest(name="research"))
    state.write_rows(
        [_row("owned", "doc", "owned")],
        scope="scope",
        collection_name="research",
        document_id="doc",
        filename="report.pdf",
        content_sha256="v1",
        artifact_prefix=str(owned),
    )
    original_cleanup = state._delete_owned_artifacts
    monkeypatch.setattr(
        state,
        "_delete_owned_artifacts",
        lambda _prefix: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    pending = state.delete_document("scope", "research", "doc", False)
    assert pending.cleanup_pending and pending.status == "deleting"
    monkeypatch.setattr(state, "_delete_owned_artifacts", original_cleanup)
    completed = state.delete_document("scope", "research", "doc", False)
    assert completed.deleted and not owned.exists()

    state.write_rows(
        [_row("external", "external-doc", "external")],
        scope="scope",
        collection_name="research",
        document_id="external-doc",
        filename="external.pdf",
        content_sha256="v1",
        artifact_prefix=str(external),
    )
    assert state.delete_document("scope", "research", "external-doc", False).deleted
    assert (external / "keep.txt").exists()


def test_collection_cleanup_retries_from_persisted_phase(tmp_path, monkeypatch) -> None:
    root = tmp_path / "artifacts"
    prefix = root / "scope" / "research"
    prefix.mkdir(parents=True)
    (prefix / "artifact").write_text("data", encoding="utf-8")
    state = VectorDBState(
        str(tmp_path / "db"),
        "legacy",
        "",
        "model",
        "",
        collection_artifact_root=str(root),
    )
    state.create_collection("scope", CollectionCreateRequest(name="research"))
    original_cleanup = state._delete_owned_artifacts
    monkeypatch.setattr(
        state,
        "_delete_owned_artifacts",
        lambda _prefix: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    pending = state.delete_collection("scope", "research", False)
    assert pending.status == "deleting" and pending.cleanup_pending
    row = state._collection_row("scope", "research")
    assert (
        row and row["deletion_phase"] == "delete_artifacts" and row["retry_count"] == 1
    )
    monkeypatch.setattr(state, "_delete_owned_artifacts", original_cleanup)
    completed = state.delete_collection("scope", "research", False)
    assert completed.status == "deleted" and not prefix.exists()


def test_expired_collection_uses_retryable_deletion_and_health_is_aggregate_only(
    tmp_path,
) -> None:
    state = VectorDBState(str(tmp_path), "secret-legacy-table", "", "model", "")
    state.create_collection(
        "tenant-secret",
        CollectionCreateRequest(name="expired", expires_at="2000-01-01T00:00:00Z"),
    )
    assert state.reconcile()["successes"] == 1
    with pytest.raises(Exception):
        state.get_collection("tenant-secret", "expired")
    health = state.operational_health()
    assert health["catalog"]["schema_version"] == 2
    assert "tenant-secret" not in str(health)
    assert "secret-legacy-table" not in str(health)


def test_catalog_migration_and_scalar_indexes_are_idempotent(tmp_path) -> None:
    db = lancedb.connect(str(tmp_path))
    db.create_table(
        "_nrl_collections",
        schema=pa.schema(
            [
                pa.field("scope", pa.string()),
                pa.field("name", pa.string()),
                pa.field("physical_table", pa.string()),
                pa.field("status", pa.string()),
                pa.field("description", pa.string()),
                pa.field("metadata_json", pa.string()),
                pa.field("created_at", pa.string()),
                pa.field("updated_at", pa.string()),
                pa.field("expires_at", pa.string()),
            ]
        ),
    )
    state = VectorDBState(str(tmp_path), "legacy", "", "model", "")
    assert "deletion_phase" in state._db.open_table("_nrl_collections").schema.names
    assert state._db.open_table("_nrl_collections").list_indices()
    VectorDBState(str(tmp_path), "legacy", "", "model", "")


def test_catalog_startup_fails_fast_on_incompatible_schema(tmp_path) -> None:
    db = lancedb.connect(str(tmp_path))
    db.create_table(
        "_nrl_collections",
        schema=pa.schema(
            [
                pa.field("scope", pa.int64()),
                pa.field("name", pa.string()),
                pa.field("physical_table", pa.string()),
                pa.field("status", pa.string()),
                pa.field("description", pa.string()),
                pa.field("metadata_json", pa.string()),
                pa.field("created_at", pa.string()),
                pa.field("updated_at", pa.string()),
                pa.field("expires_at", pa.string()),
            ]
        ),
    )
    with pytest.raises(RuntimeError, match="Incompatible"):
        VectorDBState(str(tmp_path), "legacy", "", "model", "")
