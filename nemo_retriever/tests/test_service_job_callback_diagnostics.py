# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic logging for the worker → gateway completion callback.

Hung ``run_mode="service"`` jobs are notoriously hard to triage from
gateway logs alone: the original ``Gateway callback: id=… status=…
rows=… subscribers=…`` line carried no ``job_id`` and no indication of
whether the underlying ``tracker.mark_completed()`` /
``tracker.mark_failed()`` call actually transitioned the document (vs.
silently no-op'd because the doc was unknown, or because a duplicate
callback raced an earlier one).

These tests pin down the enriched callback log line so a future
refactor cannot regress diagnosability of hangs:

* every successful callback logs ``outcome=transitioned`` and the real
  ``job_id`` of the doc the worker just finished,
* a duplicate callback for a doc already in a terminal state logs
  ``outcome=idempotent`` (so duplicate-callback storms are visible),
* a callback for a doc the gateway has never seen (the classic
  symptom of a gateway pod restart that strands the client) logs
  ``outcome=unknown_document`` AT WARNING SEVERITY plus an explicit
  hint about gateway restarts.

The ``JobTracker``-level warning emitted by ``_mark_terminal`` is also
covered: it must fire exactly once and name the missing document so
operators can grep for it without re-walking the upload route.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from fastapi.testclient import TestClient

from nemo_retriever.service.app import create_app
from nemo_retriever.service.config import (
    PipelineOverridesConfig,
    PipelinePoolConfig,
    ServiceConfig,
)
from nemo_retriever.service.services.event_bus import EventBus
from nemo_retriever.service.services.job_tracker import (
    DocumentStatus,
    JobTracker,
    MarkOutcome,
)
from nemo_retriever.service.services.pipeline_pool import WorkItem

from .conftest import create_test_job


# ----------------------------------------------------------------------
# JobTracker.mark_* return value contract
# ----------------------------------------------------------------------


def test_mark_completed_returns_transitioned_for_pending_doc() -> None:
    tracker = JobTracker()
    tracker.register_job("j", expected_documents=1)
    tracker.register_document("a", job_id="j")

    outcome = tracker.mark_completed("a", result_rows=42)

    assert outcome == MarkOutcome.TRANSITIONED
    rec = tracker.get_document("a")
    assert rec is not None
    assert rec.status == DocumentStatus.COMPLETED


def test_mark_failed_returns_transitioned_for_pending_doc() -> None:
    tracker = JobTracker()
    tracker.register_job("j", expected_documents=1)
    tracker.register_document("a", job_id="j")

    outcome = tracker.mark_failed("a", "boom")

    assert outcome == MarkOutcome.TRANSITIONED
    rec = tracker.get_document("a")
    assert rec is not None
    assert rec.status == DocumentStatus.FAILED


def test_mark_completed_returns_idempotent_when_already_terminal() -> None:
    tracker = JobTracker()
    tracker.register_job("j", expected_documents=1)
    tracker.register_document("a", job_id="j")
    tracker.mark_completed("a")

    outcome = tracker.mark_completed("a", result_rows=99)

    assert outcome == MarkOutcome.IDEMPOTENT


def test_mark_failed_after_completed_returns_idempotent() -> None:
    """Duplicate callbacks of mixed status types are still no-ops."""
    tracker = JobTracker()
    tracker.register_job("j", expected_documents=1)
    tracker.register_document("a", job_id="j")
    tracker.mark_completed("a")

    outcome = tracker.mark_failed("a", "late failure")

    assert outcome == MarkOutcome.IDEMPOTENT
    rec = tracker.get_document("a")
    assert rec is not None
    assert rec.status == DocumentStatus.COMPLETED


def test_mark_completed_returns_unknown_document_for_missing_id() -> None:
    tracker = JobTracker()

    outcome = tracker.mark_completed("never-registered", result_rows=0)

    assert outcome == MarkOutcome.UNKNOWN_DOCUMENT


def test_mark_failed_returns_unknown_document_for_missing_id() -> None:
    tracker = JobTracker()

    outcome = tracker.mark_failed("never-registered", "boom")

    assert outcome == MarkOutcome.UNKNOWN_DOCUMENT


def test_mark_terminal_unknown_document_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Hang-triage signal: the tracker must shout when a callback arrives
    for a doc it has no memory of (the classic "gateway restarted" hang).
    """
    tracker = JobTracker()

    with caplog.at_level(logging.WARNING, logger="nemo_retriever.service.services.job_tracker"):
        tracker.mark_failed("ghost-doc-id", "worker error")

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "expected a WARNING when callback fires for unknown doc"
    msg = warnings[-1].getMessage()
    assert "ghost-doc-id" in msg
    assert "mark_failed" in msg
    assert "gateway-pod restart" in msg.lower()
    assert "hang" in msg.lower()


def test_mark_terminal_transition_does_not_log_warning(caplog: pytest.LogCaptureFixture) -> None:
    tracker = JobTracker()
    tracker.register_job("j", expected_documents=1)
    tracker.register_document("a", job_id="j")

    with caplog.at_level(logging.WARNING, logger="nemo_retriever.service.services.job_tracker"):
        outcome = tracker.mark_completed("a")

    assert outcome == MarkOutcome.TRANSITIONED
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == [], f"unexpected warnings on successful transition: {[w.getMessage() for w in warnings]}"


# ----------------------------------------------------------------------
# /v1/internal/job-callback log line shape
# ----------------------------------------------------------------------


@pytest.fixture
def app_with_stub_pool(monkeypatch: pytest.MonkeyPatch):
    """Build a standalone-mode app whose pools record items instead of running pipelines.

    The pools are stubbed so we can drive the lifecycle (register →
    processing → callback) deterministically from the test body.
    """
    captured: list[WorkItem] = []

    async def _stub_work(item: WorkItem) -> tuple[int, list[dict[str, Any]]]:
        captured.append(item)
        return 1, [{"id": item.id, "stub": True}]

    def _stub_factory(_config: ServiceConfig):
        return _stub_work

    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_realtime_work_fn",
        _stub_factory,
    )
    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_batch_work_fn",
        _stub_factory,
    )

    cfg = ServiceConfig(
        mode="standalone",
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        pipeline_overrides=PipelineOverridesConfig(),
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        yield client


def _post_callback(client: TestClient, body: dict[str, Any]) -> None:
    resp = client.post("/v1/internal/job-callback", json=body)
    assert resp.status_code == 200, resp.text


def _callback_log_lines(records: list[logging.LogRecord]) -> list[logging.LogRecord]:
    return [r for r in records if "Gateway callback:" in r.getMessage()]


def test_callback_log_includes_job_id_and_outcome_for_completed(
    app_with_stub_pool: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    job_id = create_test_job(app_with_stub_pool)
    # Register a document directly via the tracker so we can drive the
    # callback in isolation without going through the upload route.
    from nemo_retriever.service.services.job_tracker import get_job_tracker

    tracker = get_job_tracker()
    assert tracker is not None
    tracker.register_document("doc-OK", job_id=job_id)
    tracker.mark_processing("doc-OK")

    with caplog.at_level(logging.INFO, logger="nemo_retriever.service.routers.ingest"):
        _post_callback(
            app_with_stub_pool,
            {"id": "doc-OK", "status": "completed", "result_rows": 16},
        )

    lines = _callback_log_lines(caplog.records)
    assert lines, "expected at least one 'Gateway callback:' log line"
    msg = lines[-1].getMessage()
    assert f"job_id={job_id}" in msg
    assert "status=completed" in msg
    assert "outcome=transitioned" in msg
    assert "rows=16" in msg
    assert "id=doc-OK" in msg
    assert lines[-1].levelno == logging.INFO


def test_callback_log_includes_job_id_and_outcome_for_failed(
    app_with_stub_pool: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    job_id = create_test_job(app_with_stub_pool)
    from nemo_retriever.service.services.job_tracker import get_job_tracker

    tracker = get_job_tracker()
    assert tracker is not None
    tracker.register_document("doc-BAD", job_id=job_id)
    tracker.mark_processing("doc-BAD")

    with caplog.at_level(logging.INFO, logger="nemo_retriever.service.routers.ingest"):
        _post_callback(
            app_with_stub_pool,
            {"id": "doc-BAD", "status": "failed", "result_rows": 0, "error": "boom"},
        )

    lines = _callback_log_lines(caplog.records)
    assert lines
    msg = lines[-1].getMessage()
    assert f"job_id={job_id}" in msg
    assert "status=failed" in msg
    assert "outcome=transitioned" in msg
    assert "rows=0" in msg
    assert lines[-1].levelno == logging.INFO


def test_callback_log_for_unknown_doc_is_warning(
    app_with_stub_pool: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """The hung-job smoking-gun: callback arrived but the tracker has no
    record of the doc. Must log a WARNING with ``outcome=unknown_document``
    and ``job_id=?`` so operators can grep gateway logs for stranded docs.
    """
    with caplog.at_level(
        logging.INFO,
        logger="nemo_retriever.service.routers.ingest",
    ):
        with caplog.at_level(
            logging.WARNING,
            logger="nemo_retriever.service.services.job_tracker",
        ):
            _post_callback(
                app_with_stub_pool,
                {"id": "ghost-doc", "status": "failed", "result_rows": 0, "error": "boom"},
            )

    callback_lines = _callback_log_lines(caplog.records)
    assert callback_lines
    msg = callback_lines[-1].getMessage()
    assert callback_lines[-1].levelno >= logging.WARNING
    assert "id=ghost-doc" in msg
    assert "outcome=unknown_document" in msg
    assert "job_id=?" in msg


def test_callback_log_for_duplicate_completion_is_idempotent(
    app_with_stub_pool: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Duplicate worker callbacks (retry or SSE/poll race) must surface
    as ``outcome=idempotent`` so operators can quantify retry storms."""
    job_id = create_test_job(app_with_stub_pool)
    from nemo_retriever.service.services.job_tracker import get_job_tracker

    tracker = get_job_tracker()
    assert tracker is not None
    tracker.register_document("doc-DUP", job_id=job_id)
    tracker.mark_processing("doc-DUP")
    # First callback drives the transition.
    _post_callback(
        app_with_stub_pool,
        {"id": "doc-DUP", "status": "completed", "result_rows": 1},
    )

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="nemo_retriever.service.routers.ingest"):
        _post_callback(
            app_with_stub_pool,
            {"id": "doc-DUP", "status": "completed", "result_rows": 1},
        )

    lines = _callback_log_lines(caplog.records)
    assert lines
    msg = lines[-1].getMessage()
    assert "outcome=idempotent" in msg
    assert f"job_id={job_id}" in msg


def test_callback_subscriber_count_is_per_job(app_with_stub_pool: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    """Subscribers count must reflect listeners for THIS job, not the
    global firehose-plus-everything total, so a busy dashboard doesn't
    mislead hang-triage into thinking the client is connected when it
    isn't.
    """
    job_id = create_test_job(app_with_stub_pool)
    from nemo_retriever.service.services.event_bus import get_event_bus
    from nemo_retriever.service.services.job_tracker import get_job_tracker

    tracker = get_job_tracker()
    bus = get_event_bus()
    assert tracker is not None
    assert bus is not None
    tracker.register_document("doc-COUNT", job_id=job_id)
    tracker.mark_processing("doc-COUNT")

    # One subscriber for a DIFFERENT job — must not be counted.
    other_id, _other_q = bus.subscribe(job_id="some-other-job")
    try:
        with caplog.at_level(logging.INFO, logger="nemo_retriever.service.routers.ingest"):
            _post_callback(
                app_with_stub_pool,
                {"id": "doc-COUNT", "status": "completed", "result_rows": 3},
            )
    finally:
        bus.unsubscribe(other_id)

    lines = _callback_log_lines(caplog.records)
    assert lines
    msg = lines[-1].getMessage()
    # Per-job count is 0 (no SSE consumer subscribed to THIS job).
    assert "subscribers=0" in msg


def test_callback_log_warning_carries_actionable_hint(
    app_with_stub_pool: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Cross-check that the JobTracker-level WARNING carries the
    actionable "gateway-pod restart" hint and that it is emitted
    alongside the router-level WARNING line (single triage anchor).
    """
    with caplog.at_level(
        logging.WARNING,
        logger="nemo_retriever.service.services.job_tracker",
    ):
        _post_callback(
            app_with_stub_pool,
            {"id": "stranded-doc", "status": "completed", "result_rows": 0},
        )

    tracker_warnings = [
        r
        for r in caplog.records
        if r.name == "nemo_retriever.service.services.job_tracker" and r.levelno >= logging.WARNING
    ]
    assert tracker_warnings, "expected JobTracker-level WARNING for unknown doc"
    msg = tracker_warnings[-1].getMessage()
    assert "stranded-doc" in msg
    assert "client may hang" in msg.lower()


# ----------------------------------------------------------------------
# Per-job subscriber counting helper
# ----------------------------------------------------------------------


def test_event_bus_subscribers_for_counts_firehose_and_job_subscribers() -> None:
    """``subscribers_for`` underpins the per-job count in the callback log;
    pin its semantics so the log line cannot regress to the global total.
    """
    bus = EventBus()
    a_id, _a = bus.subscribe(job_id="job-A")
    b_id, _b = bus.subscribe(job_id="job-B")
    firehose_id, _f = bus.subscribe(job_id=None)
    try:
        # Per-job count includes the firehose subscriber but not the
        # subscriber listening for a different job.
        assert bus.subscribers_for("job-A") == 2
        assert bus.subscribers_for("job-B") == 2
        assert bus.subscribers_for("job-missing") == 1  # only firehose
        # Total is unaffected.
        assert bus.subscriber_count == 3
    finally:
        for sub in (a_id, b_id, firehose_id):
            bus.unsubscribe(sub)
