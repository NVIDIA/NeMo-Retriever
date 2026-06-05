# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :meth:`ServiceIngestor.ingest_async` artifact-flag wiring.

The async wrapper accepts ``return_failures`` / ``return_traces`` from
the public ingestor interface; these tests pin the contract that the
resolved future returns the same tuple/list shape the synchronous
:meth:`ServiceIngestor.ingest` produces with the same flags.

The flags themselves are also re-checked at the sync layer so that
regressions show up on the closer surface (``ingest()``) before they
hit the ``ingest_async()`` glue.
"""

from __future__ import annotations

from typing import Any, Iterator
from unittest.mock import patch

import pytest

from nemo_retriever.params import IngestExecuteParams
from nemo_retriever.service_ingestor import ServiceIngestor, ServiceIngestResult


# ----------------------------------------------------------------------
# Fixture: a stub ingest_stream that exercises both success and failure
# event paths without touching the HTTP transport.
# ----------------------------------------------------------------------


def _stub_event_sequence() -> list[dict[str, Any]]:
    return [
        {"event": "job_created", "job_id": "JOB-1", "expected_documents": 2},
        {"event": "upload_complete", "filename": "a.pdf", "document_id": "doc-a"},
        {"event": "upload_complete", "filename": "b.pdf", "document_id": "doc-b"},
        {
            "event": "document_complete",
            "document_id": "doc-a",
            "status": "completed",
        },
        {
            "event": "document_complete",
            "document_id": "doc-b",
            "status": "failed",
            "error": "boom",
        },
        {"event": "job_partial", "job_id": "JOB-1"},
    ]


def _fake_materialize_completed_document(
    self: ServiceIngestor,
    document_id: str,
    *,
    return_results: bool,
) -> list[dict[str, Any]] | None:
    if not return_results and self._save_to_disk_dir is None:
        return None
    rows = [
        {
            "path": f"/uploads/{document_id}.pdf",
            "page_number": 1,
            "text": f"content-{document_id}",
            "metadata": {"source_id": document_id},
        }
    ]
    if self._save_to_disk_dir is not None:
        self._write_result_data_to_disk(document_id, rows)
    return rows if return_results else None


@pytest.fixture
def stub_ingestor() -> Iterator[ServiceIngestor]:
    """A ``ServiceIngestor`` whose stream yields a fixed event sequence."""
    ing = ServiceIngestor(base_url="http://example:7670")
    events = _stub_event_sequence()

    def _fake_stream(self: ServiceIngestor, *, retain_results: bool = False) -> Iterator[dict[str, Any]]:
        _ = retain_results
        return iter(events)

    with (
        patch.object(ServiceIngestor, "ingest_stream", _fake_stream),
        patch.object(ServiceIngestor, "_materialize_completed_document", _fake_materialize_completed_document),
    ):
        yield ing


# ----------------------------------------------------------------------
# Sync surface
# ----------------------------------------------------------------------


def test_ingest_default_returns_service_ingest_result(stub_ingestor: ServiceIngestor) -> None:
    """Default flags ⇒ ServiceIngestResult with fetched row payloads."""
    result = stub_ingestor.ingest()
    assert isinstance(result, ServiceIngestResult)
    assert not isinstance(result, tuple)
    assert result.job_id == "JOB-1"
    assert result.job_status == "partial_success"
    # Both completed and failed document_complete events land in the
    # ServiceIngestResult list; only the failed ones are mirrored on
    # ``.failures``.
    assert len(result) == 2
    assert result.failures == [("doc-b", "boom")]
    assert result.document_filenames == {"doc-a": "a.pdf", "doc-b": "b.pdf"}
    assert result.dataframe is not None
    assert len(result.dataframe) == 1
    assert "document_id" not in result.dataframe.columns
    assert result.dataframe.iloc[0]["text"] == "content-doc-a"


def test_ingest_return_failures_returns_tuple(stub_ingestor: ServiceIngestor) -> None:
    result, failures = stub_ingestor.ingest(return_failures=True)
    assert isinstance(result, ServiceIngestResult)
    assert isinstance(failures, list)
    assert failures == [("doc-b", "boom")]
    assert failures == list(result.failures)


def test_ingest_return_traces_returns_tuple_with_all_events(stub_ingestor: ServiceIngestor) -> None:
    result, traces = stub_ingestor.ingest(return_traces=True)
    assert isinstance(result, ServiceIngestResult)
    assert isinstance(traces, list)
    assert traces == _stub_event_sequence()


def test_ingest_both_flags_returns_three_tuple(stub_ingestor: ServiceIngestor) -> None:
    out = stub_ingestor.ingest(return_failures=True, return_traces=True)
    assert isinstance(out, tuple)
    assert len(out) == 3
    result, failures, traces = out
    assert isinstance(result, ServiceIngestResult)
    assert failures == [("doc-b", "boom")]
    assert traces == _stub_event_sequence()


def test_ingest_reads_flags_from_params_model(stub_ingestor: ServiceIngestor) -> None:
    """``IngestExecuteParams`` is the public params object for this method."""
    params = IngestExecuteParams(return_failures=True, return_traces=True)
    out = stub_ingestor.ingest(params=params)
    assert isinstance(out, tuple)
    assert len(out) == 3


def test_ingest_kwargs_take_precedence_over_params(stub_ingestor: ServiceIngestor) -> None:
    """Explicit kwargs win over fields on the ``params`` model."""
    params = IngestExecuteParams(return_failures=True, return_traces=True)
    out = stub_ingestor.ingest(params=params, return_failures=False, return_traces=False)
    assert isinstance(out, ServiceIngestResult)
    assert not isinstance(out, tuple)


def test_ingest_ignores_unrelated_kwargs(stub_ingestor: ServiceIngestor) -> None:
    """Service run_mode silently drops execute-time knobs it cannot honour."""
    out = stub_ingestor.ingest(show_progress=True, parallel=True, max_workers=4)
    assert isinstance(out, ServiceIngestResult)


def test_ingest_return_results_false_skips_dataframe(stub_ingestor: ServiceIngestor) -> None:
    result = stub_ingestor.ingest(return_results=False)
    assert isinstance(result, ServiceIngestResult)
    assert result.dataframe is None


def test_ingest_return_results_reads_from_params_model(stub_ingestor: ServiceIngestor) -> None:
    params = IngestExecuteParams(return_results=False)
    result = stub_ingestor.ingest(params=params)
    assert result.dataframe is None


def test_ingest_return_results_kwargs_override_params(stub_ingestor: ServiceIngestor) -> None:
    params = IngestExecuteParams(return_results=False)
    result = stub_ingestor.ingest(params=params, return_results=True)
    assert result.dataframe is not None
    assert len(result.dataframe) == 1


# ----------------------------------------------------------------------
# Async-future surface (the originally reported defect)
# ----------------------------------------------------------------------


def test_ingest_async_forwards_return_failures(stub_ingestor: ServiceIngestor) -> None:
    future = stub_ingestor.ingest_async(return_failures=True)
    out = future.result(timeout=5.0)
    assert isinstance(out, tuple)
    result, failures = out
    assert isinstance(result, ServiceIngestResult)
    assert failures == [("doc-b", "boom")]


def test_ingest_async_forwards_return_traces(stub_ingestor: ServiceIngestor) -> None:
    future = stub_ingestor.ingest_async(return_traces=True)
    out = future.result(timeout=5.0)
    assert isinstance(out, tuple)
    result, traces = out
    assert isinstance(result, ServiceIngestResult)
    assert traces == _stub_event_sequence()


def test_ingest_async_forwards_both_flags(stub_ingestor: ServiceIngestor) -> None:
    """Regression for the reported F105 defect.

    Prior to the fix, ``ingest_async`` deleted the flags before
    submitting ``self.ingest``, so the future always resolved to a
    plain :class:`ServiceIngestResult`.
    """
    future = stub_ingestor.ingest_async(return_failures=True, return_traces=True)
    out = future.result(timeout=5.0)
    assert isinstance(out, tuple)
    assert len(out) == 3
    result, failures, traces = out
    assert isinstance(result, ServiceIngestResult)
    assert failures == [("doc-b", "boom")]
    assert traces == _stub_event_sequence()


def test_ingest_async_default_matches_ingest_default(stub_ingestor: ServiceIngestor) -> None:
    """No flags ⇒ future resolves to a plain :class:`ServiceIngestResult`."""
    future = stub_ingestor.ingest_async()
    out = future.result(timeout=5.0)
    assert isinstance(out, ServiceIngestResult)
    assert not isinstance(out, tuple)


def test_ingest_async_forwards_return_results(stub_ingestor: ServiceIngestor) -> None:
    future = stub_ingestor.ingest_async(return_results=False)
    out = future.result(timeout=5.0)
    assert isinstance(out, ServiceIngestResult)
    assert out.dataframe is None
