# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from nemo_retriever.agent_mcp.backend import InProcessRetrieverBackend
from nemo_retriever.agent_mcp.models import (
    AgentMcpError,
    AgentMcpErrorCode,
    EvidenceArtifacts,
    EvidenceHit,
    JobStatus,
    Locator,
)
from nemo_retriever.agent_mcp.paths import PathPolicy
from nemo_retriever.agent_mcp.registry import CollectionRegistry


class FakeRetriever:
    constructor_kwargs: list[dict[str, Any]] = []
    query_calls: list[tuple[str, dict[str, Any]]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.constructor_kwargs.append(kwargs)

    def query(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.query_calls.append((query, kwargs))
        return [
            {
                "text": "A normalized answer.",
                "_distance": "0.42",
                "source": json.dumps({"source_id": "/tmp/source.pdf"}),
                "metadata": json.dumps({"page_number": "3", "content_type": "text"}),
            }
        ]


class HybridUnsupportedRetriever(FakeRetriever):
    def query(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        raise NotImplementedError("hybrid is not supported")


class FakeIngestor:
    calls: list[tuple[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.constructor_kwargs = kwargs
        self.current_files: list[str] = []
        self.calls.append(("init", kwargs))

    def files(self, files: list[str]) -> "FakeIngestor":
        self.current_files = files
        self.calls.append(("files", files))
        return self

    def extract(self) -> "FakeIngestor":
        self.calls.append(("extract", list(self.current_files)))
        return self

    def extract_image_files(self) -> "FakeIngestor":
        self.calls.append(("extract_image_files", list(self.current_files)))
        return self

    def extract_txt(self) -> "FakeIngestor":
        self.calls.append(("extract_txt", list(self.current_files)))
        return self

    def extract_html(self) -> "FakeIngestor":
        self.calls.append(("extract_html", list(self.current_files)))
        return self

    def extract_audio(self) -> "FakeIngestor":
        self.calls.append(("extract_audio", list(self.current_files)))
        return self

    def extract_video(self) -> "FakeIngestor":
        self.calls.append(("extract_video", list(self.current_files)))
        return self

    def embed(self, **kwargs: Any) -> "FakeIngestor":
        self.calls.append(("embed", kwargs))
        return self

    def store(self, **kwargs: Any) -> "FakeIngestor":
        self.calls.append(("store", kwargs))
        return self

    def ingest(self) -> list[dict[str, Any]]:
        result = [{"path": path} for path in self.current_files]
        self.calls.append(("ingest", result))
        return result


class FailingTextIngestor(FakeIngestor):
    def extract_txt(self) -> "FakeIngestor":
        self.calls.append(("extract_txt", list(self.current_files)))
        raise RuntimeError("text extraction failed")


def _backend(tmp_path: Path, **kwargs: Any) -> InProcessRetrieverBackend:
    FakeRetriever.constructor_kwargs = []
    FakeRetriever.query_calls = []
    registry = CollectionRegistry(tmp_path / "registry.sqlite", data_root=tmp_path)
    retriever_factory = kwargs.pop("retriever_factory", FakeRetriever)
    return InProcessRetrieverBackend(
        registry,
        path_policy=PathPolicy(allowed_roots=[tmp_path]),
        retriever_factory=retriever_factory,
        **kwargs,
    )


def _ingest_backend(tmp_path: Path, **kwargs: Any) -> InProcessRetrieverBackend:
    FakeIngestor.calls = []
    backend = _backend(tmp_path, ingestor_factory=FakeIngestor, **kwargs)
    backend._upload_records = lambda record, result, **kwargs: None
    return backend


def test_query_collection_uses_collection_embedding_and_vdb_config(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    record = backend.create_collection("docs")
    record = backend.registry.mark_collection_queryable(record.name, row_count=1)

    hits = backend.query_collection(collection="docs", query="what is here?", top_k=3)

    assert hits[0].text == "A normalized answer."
    assert hits[0].score == 0.42
    assert hits[0].locator.page_number == 3

    kwargs = FakeRetriever.constructor_kwargs[0]
    assert kwargs["vdb_kwargs"]["vdb_op"] == record.vdb_backend
    assert kwargs["vdb_kwargs"]["vdb_kwargs"]["uri"] == record.vdb_uri
    assert kwargs["vdb_kwargs"]["vdb_kwargs"]["table_name"] == record.vdb_table
    assert kwargs["embed_kwargs"]["model_name"] == record.embedding_model
    assert kwargs["embed_kwargs"]["embed_model_name"] == record.embedding_model
    assert FakeRetriever.query_calls == [("what is here?", {"top_k": 3, "vdb_kwargs": None})]


def test_ingest_local_paths_starts_job_and_marks_collection_queryable(tmp_path: Path) -> None:
    backend = _ingest_backend(tmp_path)
    doc = tmp_path / "manual.pdf"
    doc.write_text("pdf-ish")

    job = backend.ingest_local_paths("docs", [doc], wait=True)

    assert job.status is JobStatus.COMPLETE
    assert job.source_count == 1
    assert job.accepted_count == 1
    assert job.skipped_count == 0
    assert backend.registry.get_collection("docs").queryable is True
    assert ("extract", [str(doc.resolve())]) in FakeIngestor.calls


def test_ingest_groups_media_types(tmp_path: Path) -> None:
    backend = _ingest_backend(tmp_path)
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    txt = docs_dir / "notes.txt"
    pdf = docs_dir / "paper.pdf"
    mp4 = docs_dir / "clip.mp4"
    for path in (mp4, txt, pdf):
        path.write_text("content")

    job = backend.ingest_local_paths("docs", [docs_dir], wait=True)

    assert job.status is JobStatus.COMPLETE
    assert job.accepted_count == 3
    extraction_calls = [(name, value) for name, value in FakeIngestor.calls if name.startswith("extract")]
    assert extraction_calls == [
        ("extract", [str(pdf.resolve())]),
        ("extract_txt", [str(txt.resolve())]),
        ("extract_video", [str(mp4.resolve())]),
    ]


def test_ingest_uploads_first_media_group_with_overwrite_then_appends(tmp_path: Path) -> None:
    backend = _ingest_backend(tmp_path)
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    for filename in ("paper.pdf", "notes.txt", "clip.mp4"):
        (docs_dir / filename).write_text("content")
    overwrite_flags: list[bool] = []

    def fake_upload(record: Any, result: Any, *, overwrite: bool) -> None:
        overwrite_flags.append(overwrite)

    backend._upload_records = fake_upload

    job = backend.ingest_local_paths("docs", [docs_dir], wait=True)

    assert job.status is JobStatus.COMPLETE
    assert overwrite_flags == [True, False, False]


def test_ingest_appends_from_first_group_for_queryable_collection(tmp_path: Path) -> None:
    backend = _ingest_backend(tmp_path)
    backend.create_collection("docs")
    backend.registry.mark_collection_queryable("docs", row_count=3)
    doc = tmp_path / "manual.pdf"
    doc.write_text("pdf-ish")
    overwrite_flags: list[bool] = []

    def fake_upload(record: Any, result: Any, *, overwrite: bool) -> None:
        overwrite_flags.append(overwrite)

    backend._upload_records = fake_upload

    job = backend.ingest_local_paths("docs", [doc], wait=True)

    assert job.status is JobStatus.COMPLETE
    assert overwrite_flags == [False]


def test_concurrent_ingests_append_after_first_empty_collection_write(tmp_path: Path) -> None:
    backend = _ingest_backend(tmp_path, max_workers=2)
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_text("first")
    second.write_text("second")
    upload_started = Event()
    release_first_upload = Event()
    overwrite_flags: list[bool] = []

    def fake_upload(record: Any, result: Any, *, overwrite: bool) -> None:
        overwrite_flags.append(overwrite)
        if len(overwrite_flags) == 1:
            upload_started.set()
            release_first_upload.wait(timeout=5)

    backend._upload_records = fake_upload

    first_job = backend.start_ingestion("docs", [first])
    assert upload_started.wait(timeout=5)
    second_job = backend.start_ingestion("docs", [second])
    release_first_upload.set()

    backend._futures[first_job.job_id].result(timeout=5)
    backend._futures[second_job.job_id].result(timeout=5)

    assert overwrite_flags == [True, False]


def test_ingest_with_skipped_files_finishes_partial_with_warnings(tmp_path: Path) -> None:
    backend = _ingest_backend(tmp_path)
    doc = tmp_path / "manual.pdf"
    unsupported = tmp_path / "notes.bin"
    doc.write_text("pdf-ish")
    unsupported.write_text("unsupported")

    job = backend.ingest_local_paths("docs", [doc, unsupported], wait=True)

    assert job.status is JobStatus.PARTIAL
    assert job.accepted_count == 1
    assert job.skipped_count == 1
    assert job.warnings[0]["path"] == str(unsupported.resolve())
    assert job.warnings[0]["code"] == AgentMcpErrorCode.UNSUPPORTED_MEDIA_TYPE.value
    assert backend.registry.get_collection("docs").queryable is True


def test_ingest_skipped_only_finishes_partial_without_marking_queryable(tmp_path: Path) -> None:
    backend = _ingest_backend(tmp_path)
    unsupported = tmp_path / "notes.bin"
    unsupported.write_text("unsupported")

    job = backend.ingest_local_paths("docs", [unsupported], wait=True)

    assert job.status is JobStatus.PARTIAL
    assert job.accepted_count == 0
    assert job.skipped_count == 1
    assert backend.registry.get_collection("docs").queryable is False
    assert [name for name, _value in FakeIngestor.calls if name.startswith("extract")] == []


def test_ingest_failure_after_success_returns_partial_job_status(tmp_path: Path) -> None:
    FakeIngestor.calls = []
    backend = _backend(tmp_path, ingestor_factory=FailingTextIngestor)
    backend._upload_records = lambda record, result, **kwargs: None
    pdf = tmp_path / "paper.pdf"
    txt = tmp_path / "notes.txt"
    pdf.write_text("pdf-ish")
    txt.write_text("text")

    job = backend.ingest_local_paths("docs", [pdf, txt], wait=True)

    assert job.status is JobStatus.PARTIAL
    assert job.accepted_count == 1
    assert job.row_count == 1
    assert job.errors[0]["code"] == AgentMcpErrorCode.BACKEND_ERROR.value
    assert backend.registry.get_collection("docs").queryable is True


def test_ingest_wait_timeout_returns_current_job_status(tmp_path: Path) -> None:
    backend = _ingest_backend(tmp_path)
    doc = tmp_path / "manual.pdf"
    doc.write_text("pdf-ish")
    release_upload = Event()

    def fake_upload(record: Any, result: Any, *, overwrite: bool) -> None:
        release_upload.wait(timeout=5)

    backend._upload_records = fake_upload

    job = backend.ingest_local_paths("docs", [doc], wait=True, timeout_s=0.01)

    assert job.status in {JobStatus.QUEUED, JobStatus.RUNNING}
    release_upload.set()
    backend._futures[job.job_id].result(timeout=5)
    assert backend.get_ingestion_status(job.job_id).status is JobStatus.COMPLETE


def test_query_rejects_unqueryable_collection(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    backend.create_collection("docs")

    with pytest.raises(AgentMcpError) as exc:
        backend.query_collection(collection="docs", query="not yet")

    assert exc.value.code is AgentMcpErrorCode.COLLECTION_NOT_QUERYABLE


def test_query_rejects_hybrid_when_collection_is_not_hybrid(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    backend.create_collection("docs", hybrid=False)
    backend.registry.mark_collection_queryable("docs")

    with pytest.raises(AgentMcpError) as exc:
        backend.query_collection(collection="docs", query="hybrid please", hybrid=True)

    assert exc.value.code is AgentMcpErrorCode.HYBRID_NOT_AVAILABLE


def test_query_sets_reranker_flag_when_requested(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    backend.create_collection("docs")
    backend.registry.mark_collection_queryable("docs")

    backend.query_collection(collection="docs", query="rerank it", rerank=True)

    assert FakeRetriever.constructor_kwargs[0]["rerank"] is True


def test_query_forwards_hybrid_flag_for_hybrid_collection(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    backend.create_collection("docs", hybrid=True)
    backend.registry.mark_collection_queryable("docs")

    backend.query_collection(collection="docs", query="hybrid please", hybrid=True)

    assert FakeRetriever.query_calls == [("hybrid please", {"top_k": 10, "vdb_kwargs": {"hybrid": True}})]


def test_query_translates_hybrid_not_implemented(tmp_path: Path) -> None:
    backend = _backend(tmp_path, retriever_factory=HybridUnsupportedRetriever)
    backend.create_collection("docs", hybrid=True)
    backend.registry.mark_collection_queryable("docs")

    with pytest.raises(AgentMcpError) as exc:
        backend.query_collection(collection="docs", query="hybrid please", hybrid=True)

    assert exc.value.code is AgentMcpErrorCode.HYBRID_NOT_AVAILABLE


def test_query_forwards_filters(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    backend.create_collection("docs")
    backend.registry.mark_collection_queryable("docs")

    backend.query_collection(collection="docs", query="filtered", filters={"tenant": "a"})

    assert FakeRetriever.query_calls == [("filtered", {"top_k": 10, "vdb_kwargs": {"tenant": "a"}})]


def test_standalone_rerank_results_uses_injected_reranker(tmp_path: Path) -> None:
    rerank_calls = []

    def fake_rerank(query: str, hits: list[dict[str, Any]], *, top_n: int | None = None) -> list[dict[str, Any]]:
        rerank_calls.append((query, hits, top_n))
        return [{**hits[0], "_rerank_score": 0.99}]

    backend = _backend(tmp_path, rerank_fn=fake_rerank)
    hit = EvidenceHit(text="candidate", score=0.1, source_path="/tmp/source.pdf")

    reranked = backend.rerank_results("best one?", [hit], top_n=1)

    assert rerank_calls[0][0] == "best one?"
    assert rerank_calls[0][1][0]["text"] == "candidate"
    assert rerank_calls[0][1][0]["source_path"] == "/tmp/source.pdf"
    assert rerank_calls[0][1][0]["path"] == "/tmp/source.pdf"
    assert rerank_calls[0][2] == 1
    assert reranked[0].text == "candidate"
    assert reranked[0].score == 0.99


def test_default_standalone_rerank_uses_supplied_reranker_model(tmp_path: Path) -> None:
    class FakeRerankerModel:
        model_name = "nvidia/llama-nemotron-rerank-1b-v2"

        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def score(
            self,
            query: str,
            documents: list[str],
            *,
            max_length: int,
            batch_size: int,
        ) -> list[float]:
            self.calls.append(
                {
                    "query": query,
                    "documents": documents,
                    "max_length": max_length,
                    "batch_size": batch_size,
                }
            )
            return [0.77]

    model = FakeRerankerModel()
    backend = _backend(tmp_path, reranker_model=model, reranker_max_length=128, reranker_batch_size=2)

    reranked = backend.rerank_results("best one?", [EvidenceHit(text="candidate")], top_n=1)

    assert model.calls == [
        {
            "query": "best one?",
            "documents": ["candidate"],
            "max_length": 128,
            "batch_size": 2,
        }
    ]
    assert reranked[0].score == 0.77


def test_standalone_rerank_preserves_evidence_fields(tmp_path: Path) -> None:
    def fake_rerank(query: str, hits: list[dict[str, Any]], *, top_n: int | None = None) -> list[dict[str, Any]]:
        return [{**hits[0], "_rerank_score": 0.99}]

    backend = _backend(tmp_path, rerank_fn=fake_rerank)
    hit = EvidenceHit(
        text="candidate",
        score=0.1,
        source_path="/tmp/source.pdf",
        content_type="text",
        locator=Locator(
            page_number=7,
            timestamp_start_s=1.25,
            timestamp_end_s=2.5,
            frame_index=3,
            bbox_xyxy_norm=[0.1, 0.2, 0.8, 0.9],
        ),
        artifacts=EvidenceArtifacts(
            stored_image_uri="file:///tmp/page.png",
            thumbnail_uri="file:///tmp/thumb.png",
        ),
        metadata={"section": "intro"},
    )

    reranked = backend.rerank_results("best one?", [hit], top_n=1)

    assert reranked[0].score == 0.99
    assert reranked[0].source_path == "/tmp/source.pdf"
    assert reranked[0].content_type == "text"
    assert reranked[0].locator.page_number == 7
    assert reranked[0].locator.timestamp_start_s == 1.25
    assert reranked[0].locator.timestamp_end_s == 2.5
    assert reranked[0].locator.frame_index == 3
    assert reranked[0].locator.bbox_xyxy_norm == [0.1, 0.2, 0.8, 0.9]
    assert reranked[0].artifacts.stored_image_uri == "file:///tmp/page.png"
    assert reranked[0].artifacts.thumbnail_uri == "file:///tmp/thumb.png"
    assert reranked[0].metadata["section"] == "intro"
