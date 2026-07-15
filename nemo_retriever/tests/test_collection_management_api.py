# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import assert_type
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from nemo_retriever import RetrieverServiceClient
from nemo_retriever.common.schemas.collections import CollectionCreateRequest
from nemo_retriever.service.aiq_contract import AIQCompatibleClient
from nemo_retriever.service.services.job_tracker import JobFullError, JobTracker
from nemo_retriever.service.vectordb_app import VectorDBState, create_vectordb_app


def _row(chunk_id: str, document_id: str, text: str, version: str = "v1") -> dict:
    return {
        "vector": [1.0, 0.0], "pdf_page": "report_1", "filename": "report.pdf",
        "pdf_basename": "report", "page_number": 1, "source": '{"source_id":"report.pdf"}',
        "source_id": "report.pdf", "path": "report.pdf", "text": text,
        "metadata": '{"page_number":1}', "stored_image_uri": "", "content_type": "text",
        "bbox_xyxy_norm": "", "chunk_id": chunk_id, "document_id": document_id,
        "document_version": version, "content_sha256": version, "created_at": "now",
    }


def test_collection_crud_scope_pagination_and_injection_rejection(tmp_path) -> None:
    app = create_vectordb_app(lancedb_uri=str(tmp_path), embed_endpoint="http://embed")
    with TestClient(app) as client:
        headers = {"X-NRL-Scope": "workspace-a"}
        assert client.post("/v1/collections", json={"name": "one"}, headers=headers).status_code == 201
        assert client.post("/v1/collections", json={"name": "two"}, headers=headers).status_code == 201
        page = client.get("/v1/collections?limit=1", headers=headers).json()
        assert len(page["items"]) == 1 and page["next_token"]
        other_scope = client.get("/v1/collections/one", headers={"X-NRL-Scope": "workspace-b"})
        assert other_scope.status_code == 404
        injected = client.post(
            "/v1/query", json={"query": "x", "collection_name": "one", "table_name": "secret"},
            headers=headers,
        )
        assert injected.status_code == 422
        assert client.delete("/v1/collections/one", headers=headers).status_code == 200
        assert client.delete("/v1/collections/one?if_exists=true", headers=headers).json() is None


def test_append_replace_and_document_delete_are_collection_scoped(tmp_path) -> None:
    state = VectorDBState(str(tmp_path), "legacy", "", "model", "")
    state.create_collection("scope", CollectionCreateRequest(name="research"))
    state.write_rows(
        [_row("a", "doc", "old"), _row("b", "doc", "obsolete")], scope="scope",
        collection_name="research", document_id="doc", filename="report.pdf", content_sha256="v1",
    )
    assert state.total_rows(scope="scope", collection_name="research") == 2
    state.write_rows(
        [_row("c", "doc", "new", "v2")], scope="scope", collection_name="research",
        document_id="doc", filename="report.pdf", content_sha256="v2", operation="replace",
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
                    response = service.request(method, path, headers=self._auth_headers, **kwargs)
                    self._raise_for_response(response, f"{method} {path}")
                    return response.json() if response.content else None

                async def _arequest(self, method: str, path: str, **kwargs):
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=app), base_url="http://test",
                        headers=self._auth_headers,
                    ) as client:
                        response = await client.request(method, path, **kwargs)
                    self._raise_for_response(response, f"{method} {path}")
                    return response.json() if response.content else None

            sdk = InProcessClient(scope="workspace")
            collection = sdk.create_collection("research")
            assert collection.name == "research"
            service.post("/internal/vectordb/write", json={
                "rows": [_row("chunk", "doc", "finding")], "scope": "workspace",
                "collection_name": "research", "document_id": "doc", "filename": "report.pdf",
                "content_sha256": "v1",
            })
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
        "job", expected_documents=1, scope="workspace", idempotency_key="request",
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
