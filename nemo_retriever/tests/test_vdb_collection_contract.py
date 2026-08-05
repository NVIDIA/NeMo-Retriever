# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest
from pydantic import ValidationError

from nemo_retriever.common.schemas.collections import IngestOperation
from nemo_retriever.common.schemas.requests import JobCreateRequest
from nemo_retriever.common.vdb.adt_vdb import (
    CollectionWriteContext,
    CollectionWriteResult,
    VDB,
)
from nemo_retriever.service.services.pipeline_pool import DocumentWriteContext, WorkItem


class IncompleteVDB(VDB):
    """VDB missing the required collection-management contract."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def create_index(self, **kwargs: Any) -> None:
        return None

    def write_to_index(self, records: list, **kwargs: Any) -> None:
        return None

    def retrieval(self, queries: list, **kwargs: Any) -> list[list[dict[str, Any]]]:
        return [[] for _ in queries]

    def run(self, records: list) -> None:
        return None


def test_vdb_requires_collection_management_implementations() -> None:
    with pytest.raises(TypeError) as exc_info:
        IncompleteVDB()

    for method in (
        "create_collection",
        "get_collection",
        "list_collections",
        "update_collection",
        "delete_collection",
        "get_document",
        "list_documents",
        "delete_document",
        "write_collection",
        "retrieve_collection",
    ):
        assert method in str(exc_info.value)


def test_optional_collection_maintenance_has_safe_defaults() -> None:
    assert VDB.reconcile_collections(None) == {"successes": 0, "failures": 0}  # type: ignore[arg-type]
    assert VDB.health(None) == {}  # type: ignore[arg-type]


def test_collection_write_contract_is_immutable_and_reports_counts() -> None:
    context = CollectionWriteContext(
        scope="workspace-a",
        collection_name="collection-a",
        document_id="document-a",
        document_version="version-a",
        content_sha256="sha256-a",
        filename="document.txt",
        job_id=None,
        operation="replace",
    )

    assert context.operation == "replace"
    assert CollectionWriteResult(written=3, total_rows=7) == CollectionWriteResult(
        written=3,
        total_rows=7,
    )
    with pytest.raises(FrozenInstanceError):
        context.collection_name = "other"  # type: ignore[misc]


def _write_context(**overrides: Any) -> CollectionWriteContext:
    fields: dict[str, Any] = {
        "scope": "workspace-a",
        "collection_name": "collection-a",
        "document_id": "document-a",
        "document_version": "version-a",
        "content_sha256": "sha256-a",
        "filename": "document.txt",
    }
    fields.update(overrides)
    return CollectionWriteContext(**fields)


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        ("append", IngestOperation.APPEND),
        ("replace", IngestOperation.REPLACE),
        (IngestOperation.REPLACE, IngestOperation.REPLACE),
    ],
)
def test_collection_write_context_coerces_operation_to_enum(supplied: Any, expected: IngestOperation) -> None:
    """Backends compare with ``is``, so a wire string must become the member."""
    context = _write_context(operation=supplied)

    assert context.operation is expected
    assert replace(context, content_sha256="other").operation is expected


def test_collection_write_context_rejects_unknown_operation() -> None:
    with pytest.raises(ValueError):
        _write_context(operation="upsert")


def test_collection_write_context_defaults_to_append() -> None:
    assert _write_context().operation is IngestOperation.APPEND


@pytest.mark.parametrize(
    ("wire_value", "target_document_id"),
    [("append", None), ("replace", "document-1")],
)
def test_ingest_operation_round_trips_as_a_plain_wire_string(
    wire_value: str,
    target_document_id: str | None,
) -> None:
    """The enum is an internal type only; the REST contract stays strings."""
    request = JobCreateRequest(
        expected_documents=1,
        collection_name="c",
        operation=wire_value,
        target_document_id=target_document_id,
    )

    assert isinstance(request.operation, IngestOperation)
    assert request.model_dump(mode="json")["operation"] == wire_value
    assert json.dumps({"operation": request.operation}) == json.dumps({"operation": wire_value})


def test_ingest_operation_rejects_unknown_wire_values() -> None:
    with pytest.raises(ValidationError):
        JobCreateRequest(expected_documents=1, collection_name="c", operation="upsert")


def test_job_idempotency_fingerprint_is_unchanged_by_the_enum() -> None:
    """The stored fingerprint must not shift, or retries would stop matching."""
    fingerprint_input = {"operation": IngestOperation.APPEND, "expected_documents": 1}
    literal_input = {"operation": "append", "expected_documents": 1}

    def _digest(payload: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    assert _digest(fingerprint_input) == _digest(literal_input)


def test_work_item_rebuilds_the_write_context_from_a_broker_claim() -> None:
    """``RichModel`` ignores unknown keys, so the nested key must survive intact."""
    original = DocumentWriteContext(
        scope="workspace",
        collection_name="research",
        operation=IngestOperation.REPLACE,
        content_sha256="a" * 64,
        storage_document_id="document-1",
    )

    claim_extra = {"write": original.model_dump(mode="json")}
    rebuilt = WorkItem(id="attempt-1", **claim_extra)

    assert rebuilt.write == original
    assert rebuilt.write.operation is IngestOperation.REPLACE
    assert rebuilt.write.storage_document_id == "document-1"


def test_work_item_write_context_falls_back_to_the_attempt_id() -> None:
    item = WorkItem(id="attempt-1")

    assert item.write.resolved(fallback_document_id=item.id).storage_document_id == "attempt-1"
    assert item.write.resolved(fallback_document_id=item.id).operation is IngestOperation.APPEND
