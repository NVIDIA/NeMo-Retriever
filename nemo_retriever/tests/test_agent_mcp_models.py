# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.agent_mcp.models import (
    AgentMcpError,
    AgentMcpErrorCode,
    CollectionRecord,
    CollectionStatus,
    EvidenceHit,
    IngestJobRecord,
    JobStatus,
    Locator,
)


def test_agent_mcp_error_serializes_actionable_payload() -> None:
    err = AgentMcpError(
        AgentMcpErrorCode.COLLECTION_NOT_FOUND,
        "Collection 'docs' was not found.",
        retryable=False,
        details={"collection": "docs"},
    )

    assert err.to_dict() == {
        "code": "COLLECTION_NOT_FOUND",
        "message": "Collection 'docs' was not found.",
        "retryable": False,
        "details": {"collection": "docs"},
    }
    assert "COLLECTION_NOT_FOUND" in str(err)


def test_collection_record_defaults_to_unqueryable_lancedb_collection() -> None:
    record = CollectionRecord(name="default", root_path="/tmp/mcp/collections/default")

    assert record.name == "default"
    assert record.backend == "inprocess"
    assert record.vdb_backend == "lancedb"
    assert record.vdb_table == "nv-ingest"
    assert record.status is CollectionStatus.EMPTY
    assert record.queryable is False


def test_ingest_job_record_tracks_partial_success() -> None:
    job = IngestJobRecord(
        job_id="job-1",
        collection="default",
        status=JobStatus.PARTIAL,
        source_count=3,
        accepted_count=2,
        skipped_count=1,
        errors=[{"path": "/tmp/bad.xyz", "code": "UNSUPPORTED_MEDIA_TYPE"}],
    )

    assert job.status is JobStatus.PARTIAL
    assert job.accepted_count == 2
    assert job.errors[0]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_evidence_hit_uses_sparse_locator_fields() -> None:
    hit = EvidenceHit(
        text="transcript text",
        score=0.12,
        source_path="/data/video.mp4",
        media_type="video",
        content_type="transcript",
        locator=Locator(timestamp_start_s=4.0, timestamp_end_s=7.5),
    )

    payload = hit.model_dump(exclude_none=True)
    assert payload["locator"] == {"timestamp_start_s": 4.0, "timestamp_end_s": 7.5}
    assert "page_number" not in payload["locator"]
