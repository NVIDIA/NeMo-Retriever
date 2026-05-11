# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from nemo_retriever.agent_mcp.models import AgentMcpError, AgentMcpErrorCode, CollectionStatus, JobStatus
from nemo_retriever.agent_mcp.registry import CollectionRegistry


def test_create_and_load_collection(tmp_path: Path) -> None:
    registry = CollectionRegistry(tmp_path / "registry.sqlite", data_root=tmp_path)

    record = registry.create_collection("docs", temporary=False, hybrid=True)
    loaded = registry.get_collection("docs")

    assert loaded.name == "docs"
    assert loaded.root_path == record.root_path
    assert loaded.vdb_uri.endswith("/collections/docs/lancedb")
    assert loaded.artifact_root.endswith("/collections/docs/artifacts")
    assert loaded.hybrid is True
    assert loaded.queryable is False
    assert loaded.status is CollectionStatus.EMPTY


def test_default_collection_is_lazily_created(tmp_path: Path) -> None:
    registry = CollectionRegistry(tmp_path / "registry.sqlite", data_root=tmp_path)

    record = registry.get_or_create_collection("default")

    assert record.name == "default"
    assert registry.list_collections()[0].name == "default"


def test_duplicate_collection_raises_structured_error(tmp_path: Path) -> None:
    registry = CollectionRegistry(tmp_path / "registry.sqlite", data_root=tmp_path)
    registry.create_collection("docs")

    with pytest.raises(AgentMcpError) as exc:
        registry.create_collection("docs")

    assert exc.value.code is AgentMcpErrorCode.COLLECTION_ALREADY_EXISTS


def test_path_like_collection_names_are_rejected_before_directories_are_created(tmp_path: Path) -> None:
    registry = CollectionRegistry(tmp_path / "registry.sqlite", data_root=tmp_path)

    invalid_names = [
        ("../escape", tmp_path / "escape"),
        (str(tmp_path / "absolute_escape"), tmp_path / "absolute_escape"),
    ]

    for collection_name, escaped_path in invalid_names:
        with pytest.raises(AgentMcpError) as exc:
            registry.create_collection(collection_name)

        assert exc.value.code is AgentMcpErrorCode.PATH_OUTSIDE_ALLOWED_ROOT
        assert not escaped_path.exists()

    assert registry.list_collections() == []


def test_parent_directory_collection_name_is_rejected_without_creating_data_root_artifacts(tmp_path: Path) -> None:
    registry = CollectionRegistry(tmp_path / "registry.sqlite", data_root=tmp_path)

    with pytest.raises(AgentMcpError) as exc:
        registry.create_collection("..")

    assert exc.value.code is AgentMcpErrorCode.PATH_OUTSIDE_ALLOWED_ROOT
    assert not (tmp_path / "lancedb").exists()
    assert not (tmp_path / "artifacts").exists()
    assert registry.list_collections() == []


def test_update_collection_queryable_state_persists(tmp_path: Path) -> None:
    registry = CollectionRegistry(tmp_path / "registry.sqlite", data_root=tmp_path)
    registry.create_collection("docs")

    registry.mark_collection_queryable("docs", row_count=12)
    loaded = registry.get_collection("docs")

    assert loaded.queryable is True
    assert loaded.status is CollectionStatus.QUERYABLE
    assert loaded.metadata["row_count"] == 12


def test_job_lifecycle_persists(tmp_path: Path) -> None:
    registry = CollectionRegistry(tmp_path / "registry.sqlite", data_root=tmp_path)
    registry.create_collection("docs")

    job = registry.create_job("docs", source_count=3)
    registry.update_job(job.job_id, status=JobStatus.RUNNING, accepted_count=1)
    registry.update_job(job.job_id, status=JobStatus.PARTIAL, accepted_count=2, skipped_count=1)

    loaded = registry.get_job(job.job_id)
    jobs = registry.list_jobs("docs")

    assert loaded.status is JobStatus.PARTIAL
    assert loaded.accepted_count == 2
    assert loaded.skipped_count == 1
    assert jobs[0].job_id == job.job_id


def test_update_job_collection_keeps_listing_index_consistent(tmp_path: Path) -> None:
    registry = CollectionRegistry(tmp_path / "registry.sqlite", data_root=tmp_path)
    registry.create_collection("docs")
    registry.create_collection("other")

    job = registry.create_job("docs")
    registry.update_job(job.job_id, collection="other")

    loaded = registry.get_job(job.job_id)
    other_jobs = registry.list_jobs("other")
    docs_jobs = registry.list_jobs("docs")

    assert loaded.collection == "other"
    assert [listed.job_id for listed in other_jobs] == [job.job_id]
    assert job.job_id not in {listed.job_id for listed in docs_jobs}


def test_update_job_rejects_unknown_collection_move(tmp_path: Path) -> None:
    registry = CollectionRegistry(tmp_path / "registry.sqlite", data_root=tmp_path)
    registry.create_collection("docs")

    job = registry.create_job("docs")

    with pytest.raises(AgentMcpError) as exc:
        registry.update_job(job.job_id, collection="missing")

    loaded = registry.get_job(job.job_id)
    docs_jobs = registry.list_jobs("docs")

    assert exc.value.code is AgentMcpErrorCode.COLLECTION_NOT_FOUND
    assert loaded.collection == "docs"
    assert [listed.job_id for listed in docs_jobs] == [job.job_id]


def test_registry_records_reload_from_existing_database(tmp_path: Path) -> None:
    db_path = tmp_path / "registry.sqlite"
    registry = CollectionRegistry(db_path, data_root=tmp_path)
    registry.create_collection("docs", hybrid=True)
    registry.mark_collection_queryable("docs", row_count=4)
    job = registry.create_job("docs", source_count=2)
    registry.update_job(job.job_id, status=JobStatus.COMPLETE, accepted_count=2, row_count=4)

    reloaded = CollectionRegistry(db_path, data_root=tmp_path)

    collection = reloaded.get_collection("docs")
    loaded_job = reloaded.get_job(job.job_id)

    assert collection.hybrid is True
    assert collection.queryable is True
    assert collection.metadata["row_count"] == 4
    assert loaded_job.status is JobStatus.COMPLETE
    assert loaded_job.accepted_count == 2
    assert reloaded.list_jobs("docs")[0].job_id == job.job_id
