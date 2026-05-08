# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AgentMcpErrorCode(str, Enum):
    PATH_OUTSIDE_ALLOWED_ROOT = "PATH_OUTSIDE_ALLOWED_ROOT"
    PATH_NOT_FOUND = "PATH_NOT_FOUND"
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    COLLECTION_NOT_FOUND = "COLLECTION_NOT_FOUND"
    COLLECTION_ALREADY_EXISTS = "COLLECTION_ALREADY_EXISTS"
    COLLECTION_NOT_QUERYABLE = "COLLECTION_NOT_QUERYABLE"
    INGEST_JOB_NOT_FOUND = "INGEST_JOB_NOT_FOUND"
    INGEST_JOB_FAILED = "INGEST_JOB_FAILED"
    HYBRID_NOT_AVAILABLE = "HYBRID_NOT_AVAILABLE"
    EMBEDDING_CONFIG_MISMATCH = "EMBEDDING_CONFIG_MISMATCH"
    VDB_NOT_FOUND = "VDB_NOT_FOUND"
    BACKEND_ERROR = "BACKEND_ERROR"


class AgentMcpError(Exception):
    def __init__(
        self,
        code: AgentMcpErrorCode,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = bool(retryable)
        self.details = dict(details or {})

    def __str__(self) -> str:
        return f"{self.code.value}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }


class CollectionStatus(str, Enum):
    EMPTY = "empty"
    INGESTING = "ingesting"
    QUERYABLE = "queryable"
    ERROR = "error"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    PARTIAL = "partial"


class Locator(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int | None = None
    timestamp_start_s: float | None = None
    timestamp_end_s: float | None = None
    frame_index: int | None = None
    bbox_xyxy_norm: list[float] | None = None
    chunk_id: str | None = None


class EvidenceArtifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stored_image_uri: str | None = None
    thumbnail_uri: str | None = None


class EvidenceHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    score: float | None = None
    source_path: str = ""
    media_type: str = "unknown"
    content_type: str = "unknown"
    locator: Locator = Field(default_factory=Locator)
    artifacts: EvidenceArtifacts = Field(default_factory=EvidenceArtifacts)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CollectionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    root_path: str
    temporary: bool = False
    backend: Literal["inprocess", "service"] = "inprocess"
    vdb_backend: str = "lancedb"
    vdb_uri: str | None = None
    vdb_table: str = "nv-ingest"
    artifact_root: str | None = None
    embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2"
    embedding_endpoint: str | None = None
    hybrid: bool = False
    queryable: bool = False
    status: CollectionStatus = CollectionStatus.EMPTY
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wait: bool = True
    timeout_s: float | None = None
    run_mode: Literal["inprocess", "batch"] = "inprocess"
    hybrid: bool = False
    store_artifacts: bool = True
    max_files: int = 1000
    max_total_bytes: int | None = None


class IngestJobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    collection: str
    status: JobStatus = JobStatus.QUEUED
    source_count: int = 0
    accepted_count: int = 0
    skipped_count: int = 0
    row_count: int | None = None
    artifact_count: int | None = None
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class QueryOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=10, ge=1, le=1000)
    hybrid: bool = False
    rerank: bool = False
    filters: dict[str, Any] | None = None
