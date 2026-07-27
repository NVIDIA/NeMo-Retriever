# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, Callable

import pytest

from nemo_retriever.common.schemas.collections import (
    CollectionCreateRequest,
    CollectionUpdateRequest,
)
from nemo_retriever.common.vdb.adt_vdb import (
    CollectionWriteContext,
    CollectionWriteResult,
    UnsupportedVDBOperation,
    VDB,
)


class LegacyVDB(VDB):
    """Minimal pre-collection implementation of the original VDB contract."""

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


def _unsupported_calls(vdb: VDB) -> list[Callable[[], Any]]:
    return [
        lambda: vdb.create_collection(
            scope="workspace-a",
            request=CollectionCreateRequest(name="collection-a"),
        ),
        lambda: vdb.get_collection(
            scope="workspace-a",
            collection_name="collection-a",
        ),
        lambda: vdb.list_collections(
            scope="workspace-a",
            limit=100,
            continuation_token=None,
        ),
        lambda: vdb.update_collection(
            scope="workspace-a",
            collection_name="collection-a",
            request=CollectionUpdateRequest(description="updated"),
        ),
        lambda: vdb.delete_collection(
            scope="workspace-a",
            collection_name="collection-a",
            if_exists=False,
        ),
        lambda: vdb.get_document(
            scope="workspace-a",
            collection_name="collection-a",
            document_id="document-a",
        ),
        lambda: vdb.list_documents(
            scope="workspace-a",
            collection_name="collection-a",
            limit=100,
            continuation_token=None,
        ),
        lambda: vdb.delete_document(
            scope="workspace-a",
            collection_name="collection-a",
            document_id="document-a",
            if_exists=False,
        ),
        lambda: vdb.write_collection(
            [[]],
            context=CollectionWriteContext(
                scope="workspace-a",
                collection_name="collection-a",
                document_id="document-a",
                document_version="version-a",
                content_sha256="sha256-a",
                filename="document.txt",
                job_id="job-a",
            ),
        ),
        lambda: vdb.retrieve_collection(
            [[0.1, 0.2]],
            scope="workspace-a",
            collection_name="collection-a",
            query_texts=["query"],
            top_k=5,
        ),
    ]


def test_legacy_vdb_stays_instantiable_and_fails_closed() -> None:
    vdb = LegacyVDB()

    for call in _unsupported_calls(vdb):
        with pytest.raises(UnsupportedVDBOperation, match="collection management is not supported"):
            call()


def test_optional_collection_maintenance_has_safe_defaults() -> None:
    vdb = LegacyVDB()

    assert vdb.reconcile_collections() == {"successes": 0, "failures": 0}
    assert vdb.health() == {}


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
