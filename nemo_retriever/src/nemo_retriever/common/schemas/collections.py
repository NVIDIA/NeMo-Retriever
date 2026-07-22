# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public collection-management wire models.

These models are shared by the service and Python SDK.  Keeping the contract
here prevents agent adapters from depending on service implementation details
or LanceDB-specific names.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, field_validator

from nemo_retriever.common.api.util.converters.datetools import (
    normalize_timezone_aware_iso8601_to_utc,
)
from nemo_retriever.common.schemas.base import RichModel

CollectionStatus = Literal["active", "deleting"]
IngestOperation = Literal["append", "replace"]
DeleteStatus = Literal["deleting", "deleted"]
DocumentId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
]


def _normalize_expires_at(value: str | None) -> str | None:
    if value is None:
        return None
    return normalize_timezone_aware_iso8601_to_utc(value)


class CollectionCreateRequest(RichModel):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    description: str | None = Field(default=None, max_length=4096)
    metadata: dict[str, Any] = Field(default_factory=dict)
    expires_at: str | None = None

    @field_validator("expires_at", mode="before")
    @classmethod
    def _validate_expiry(cls, value: str | None) -> str | None:
        return _normalize_expires_at(value)


class CollectionUpdateRequest(RichModel):
    description: str | None = Field(default=None, max_length=4096)
    metadata: dict[str, Any] | None = None
    expires_at: str | None = None

    @field_validator("expires_at", mode="before")
    @classmethod
    def _validate_expiry(cls, value: str | None) -> str | None:
        return _normalize_expires_at(value)


class CollectionInfo(RichModel):
    name: str
    scope: str
    status: CollectionStatus = "active"
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    expires_at: str | None = None


class CollectionPage(RichModel):
    items: list[CollectionInfo] = Field(default_factory=list)
    next_token: str | None = None


class DocumentInfo(RichModel):
    document_id: DocumentId
    collection_name: str
    scope: str
    filename: str
    content_sha256: str
    document_version: str
    status: str
    chunk_count: int = 0
    job_id: str | None = None
    created_at: str
    updated_at: str
    error: str | None = None


class DocumentPage(RichModel):
    items: list[DocumentInfo] = Field(default_factory=list)
    next_token: str | None = None


class DocumentDeleteResult(RichModel):
    document_id: DocumentId
    collection_name: str
    scope: str
    existed: bool
    deleted: bool
    status: DeleteStatus
    cleanup_pending: bool = False


class CollectionDeleteResult(RichModel):
    name: str
    scope: str
    existed: bool
    deleted: bool
    status: DeleteStatus
    cleanup_pending: bool = False


class QueryHit(RichModel):
    """Citation-ready hit returned to agentic applications."""

    chunk_id: str
    document_id: DocumentId
    text: str
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Query-relative relevance; not a probability or comparable across queries.",
    )
    filename: str
    page_number: int | None = Field(
        default=None,
        ge=1,
        description="One-based human-facing document page, or null when not applicable.",
    )
    content_type: str | None = None
    source: Any = None
    source_id: str | None = None
    stored_image_uri: str | None = None
    bbox: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
