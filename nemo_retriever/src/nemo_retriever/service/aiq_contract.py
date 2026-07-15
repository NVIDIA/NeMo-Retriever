# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural contract for a future AIQ knowledge-layer adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from nemo_retriever.common.schemas.collections import (
    CollectionDeleteResult,
    CollectionInfo,
    CollectionPage,
    DocumentDeleteResult,
    DocumentInfo,
    DocumentPage,
    QueryHit,
)
from nemo_retriever.common.schemas.responses import JobAggregateResponse


class AIQCompatibleClient(Protocol):
    def create_collection(self, name: str, **kwargs: Any) -> CollectionInfo: ...
    def delete_collection(self, name: str, **kwargs: Any) -> CollectionDeleteResult: ...
    def list_collections(self, **kwargs: Any) -> CollectionPage: ...
    def submit_documents(
        self, collection_name: str, files: list[str | Path], **kwargs: Any,
    ) -> JobAggregateResponse: ...
    def get_job(self, job_id: str) -> JobAggregateResponse: ...
    def list_documents(self, collection_name: str, **kwargs: Any) -> DocumentPage: ...
    def get_document(self, collection_name: str, document_id: str) -> DocumentInfo: ...
    def delete_document(
        self, collection_name: str, document_id: str, **kwargs: Any,
    ) -> DocumentDeleteResult: ...
    async def aquery(
        self, query: str, *, collection_name: str, top_k: int = 10,
    ) -> list[QueryHit]: ...
