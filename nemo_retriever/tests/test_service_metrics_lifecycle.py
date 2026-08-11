# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle projection tests for tracker-owned ingest metrics."""

from nemo_retriever.service.services.job_tracker import JobTracker, MarkOutcome
from nemo_retriever.service.services import metrics as metrics_module
from nemo_retriever.service.services.metrics import IngestMetrics


def _wired_tracker() -> tuple[JobTracker, IngestMetrics]:
    tracker = JobTracker()
    metrics = IngestMetrics()
    tracker.add_terminal_observer(metrics.record_terminal_transition)
    return tracker, metrics


def test_terminal_transitions_reconcile_document_and_job_metrics() -> None:
    tracker, metrics = _wired_tracker()
    tracker.register_job("job", expected_documents=2)
    metrics.record_job_created("job")
    for document_id in ("ok", "bad"):
        tracker.register_document(document_id, job_id="job")
        metrics.record_document_accepted(document_id=document_id, job_id="job")
        tracker.mark_processing(document_id)

    assert tracker.mark_completed("ok", elapsed_s=1.25) is MarkOutcome.TRANSITIONED
    assert tracker.mark_failed("bad", "pipeline failed", elapsed_s=2.5) is MarkOutcome.TRANSITIONED

    job = metrics.get_job("job")
    assert job is not None
    assert (job.documents_completed, job.documents_failed, job.status) == (
        1,
        1,
        "partial_success",
    )
    assert job.completed_at is not None
    assert job.wall_duration_s is not None
    completed = metrics.get_document("ok")
    failed = metrics.get_document("bad")
    assert completed is not None and completed.status == "completed"
    assert completed.completed_at is not None and completed.processing_duration_s == 1.25
    assert failed is not None and failed.status == "failed"
    assert failed.error == "pipeline failed" and failed.processing_duration_s == 2.5


def test_duplicate_and_unknown_terminal_transitions_do_not_change_metrics() -> None:
    tracker, metrics = _wired_tracker()
    tracker.register_job("job", expected_documents=1)
    metrics.record_job_created("job")
    tracker.register_document("doc", job_id="job")
    metrics.record_document_accepted(document_id="doc", job_id="job")

    assert tracker.mark_completed("doc", elapsed_s=1) is MarkOutcome.TRANSITIONED
    before = metrics.get_job("job")
    assert tracker.mark_completed("doc", elapsed_s=99) is MarkOutcome.IDEMPOTENT
    assert tracker.mark_failed("unknown", "missing") is MarkOutcome.UNKNOWN_DOCUMENT
    assert metrics.get_job("job") == before


def test_explicit_page_terminal_transitions_reconcile_page_counts() -> None:
    tracker, metrics = _wired_tracker()
    tracker.register_job("job", expected_documents=2)
    metrics.record_job_created("job")
    for page_id in ("page-1", "page-2"):
        tracker.register_document(page_id, job_id="job")
        metrics.record_page_accepted(page_id=page_id, document_id="source", job_id="job")
        tracker.mark_processing(page_id)

    tracker.mark_completed("page-1", elapsed_s=0.5)
    tracker.mark_failed("page-2", "OCR failed", elapsed_s=0.75)

    job = metrics.get_job("job")
    assert job is not None
    assert (job.pages_total, job.pages_completed, job.pages_failed) == (2, 1, 1)
    page = metrics.get_page("page-2")
    assert page is not None
    assert page.job_id == "job" and page.status == "failed" and page.error == "OCR failed"


def test_page_terminal_counts_survive_recent_page_eviction(monkeypatch) -> None:
    monkeypatch.setattr(metrics_module, "MAX_RECENT_PAGES", 1)
    tracker, metrics = _wired_tracker()
    tracker.register_job("job", expected_documents=2)
    metrics.record_job_created("job")
    for page_id in ("page-1", "page-2"):
        tracker.register_document(page_id, job_id="job")
        metrics.record_page_accepted(page_id=page_id, document_id="source", job_id="job")
        tracker.mark_processing(page_id)

    tracker.mark_completed("page-1")
    tracker.mark_failed("page-2", "OCR failed")

    job = metrics.get_job("job")
    assert job is not None
    assert (job.pages_total, job.pages_completed, job.pages_failed) == (2, 1, 1)
