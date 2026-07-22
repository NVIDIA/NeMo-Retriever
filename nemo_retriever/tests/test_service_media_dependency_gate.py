# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the upload-time ffmpeg/ffprobe availability gate.

When the retriever service container is deployed without FFmpeg (the
default in the Helm chart, where ``service.installFfmpeg=false``), the
old behaviour was to accept audio / video uploads, route them to the
batch worker pool, and then crash the worker with::

    RuntimeError: MediaChunkActor requires media dependencies; missing:
    ffmpeg, ffprobe.

The fix gates uploads at request time with an HTTP 501 response that
points at the Helm value and ``apt-get`` command needed to make media
ingestion work, and logs a startup-time WARNING so cluster operators
see the problem before any traffic arrives.

These tests exercise the gate without requiring real FFmpeg binaries:
:func:`is_media_available` is monkey-patched to return ``False`` so the
behaviour is reproducible on CI runners that have FFmpeg installed.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from nemo_retriever.service.app import _check_media_dependencies, create_app
from nemo_retriever.service.config import (
    PipelineOverridesConfig,
    PipelinePoolConfig,
    ServiceConfig,
)
from nemo_retriever.service.services.pipeline_pool import WorkItem
from nemo_retriever.service.utils.file_type import (
    FileCategory,
    FileClassification,
    category_requires_media_deps,
    enforce_media_dependencies,
)

from .conftest import create_test_job


def _classification(category: FileCategory, *, filename: str = "sample") -> FileClassification:
    return FileClassification(
        filename=filename,
        suffix=".bin",
        category=category,
        content_type="application/octet-stream",
    )


def test_category_requires_media_deps_only_audio_and_video() -> None:
    """Only AUDIO / VIDEO categories trigger the FFmpeg gate."""
    assert category_requires_media_deps(FileCategory.AUDIO)
    assert category_requires_media_deps(FileCategory.VIDEO)
    for c in (
        FileCategory.DOCUMENT,
        FileCategory.TEXT,
        FileCategory.HTML,
        FileCategory.IMAGE,
    ):
        assert not category_requires_media_deps(c)


def test_enforce_media_dependencies_passes_through_non_media() -> None:
    """Non-media uploads never invoke the FFmpeg probe."""
    with patch(
        "nemo_retriever.common.modality.audio.media_interface.is_media_available",
        return_value=False,
    ) as is_avail:
        for category in (
            FileCategory.DOCUMENT,
            FileCategory.TEXT,
            FileCategory.HTML,
            FileCategory.IMAGE,
        ):
            enforce_media_dependencies(_classification(category))
    assert not is_avail.called, "FFmpeg probe must not run for non-media uploads"


def test_enforce_media_dependencies_passes_when_ffmpeg_available() -> None:
    """Media uploads pass through when ffmpeg/ffprobe are installed."""
    with patch(
        "nemo_retriever.common.modality.audio.media_interface.is_media_available",
        return_value=True,
    ):
        enforce_media_dependencies(_classification(FileCategory.AUDIO))
        enforce_media_dependencies(_classification(FileCategory.VIDEO))


def test_enforce_media_dependencies_raises_501_with_actionable_detail() -> None:
    """Missing FFmpeg → HTTP 501 with Helm value + apt-get command."""
    with (
        patch(
            "nemo_retriever.common.modality.audio.media_interface.is_media_available",
            return_value=False,
        ),
        patch(
            "nemo_retriever.common.modality.audio.media_interface.missing_media_dependencies",
            return_value=["ffmpeg", "ffprobe"],
        ),
    ):
        with pytest.raises(HTTPException) as excinfo:
            enforce_media_dependencies(_classification(FileCategory.AUDIO, filename="clip.mp3"))

    err = excinfo.value
    assert err.status_code == 501
    detail = str(err.detail)
    assert "ffmpeg" in detail and "ffprobe" in detail
    assert "service.installFfmpeg=true" in detail
    assert "apt-get update && apt-get install -y --no-install-recommends ffmpeg" in detail
    assert "clip.mp3" in detail


@pytest.fixture
def app_with_stub_pool_no_ffmpeg(monkeypatch: pytest.MonkeyPatch):
    """Standalone-mode app with FFmpeg masked as unavailable.

    The work-fn is stubbed so non-media uploads (PDF) succeed without
    touching Ray / GPU, isolating the gate's behaviour from the rest
    of the pipeline.
    """

    async def _stub_work(item: WorkItem) -> tuple[int, list[dict[str, Any]]]:
        return 1, [{"id": item.id, "stub": True}]

    def _stub_realtime(_config: ServiceConfig):
        return _stub_work

    def _stub_batch(_config: ServiceConfig):
        return _stub_work

    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_realtime_work_fn",
        _stub_realtime,
    )
    monkeypatch.setattr(
        "nemo_retriever.service.services.pipeline_executor.create_batch_work_fn",
        _stub_batch,
    )
    monkeypatch.setattr(
        "nemo_retriever.common.modality.audio.media_interface.is_media_available",
        lambda: False,
    )
    monkeypatch.setattr(
        "nemo_retriever.common.modality.audio.media_interface.missing_media_dependencies",
        lambda *_, **__: ["ffmpeg", "ffprobe"],
    )

    cfg = ServiceConfig(
        mode="standalone",
        pipeline=PipelinePoolConfig(realtime_workers=1, batch_workers=1),
        pipeline_overrides=PipelineOverridesConfig(),
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize(
    "filename,content_type",
    [
        ("clip.mp3", "audio/mpeg"),
        ("voice.wav", "audio/wav"),
        ("movie.mp4", "video/mp4"),
        ("clip.mkv", "video/x-matroska"),
    ],
)
def test_audio_video_upload_rejected_with_501_when_ffmpeg_missing(
    app_with_stub_pool_no_ffmpeg: TestClient,
    filename: str,
    content_type: str,
) -> None:
    """Audio / video uploads get a clean HTTP 501 instead of a worker crash."""
    job_id = create_test_job(app_with_stub_pool_no_ffmpeg)
    resp = app_with_stub_pool_no_ffmpeg.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": (filename, b"\x00\x00\x00\x00", content_type)},
        data={"metadata": "{}"},
    )
    assert resp.status_code == 501, resp.text
    detail = resp.json()["detail"]
    assert "ffmpeg" in detail and "ffprobe" in detail
    assert "service.installFfmpeg=true" in detail
    assert filename in detail


def test_pdf_upload_is_unaffected_when_ffmpeg_missing(
    app_with_stub_pool_no_ffmpeg: TestClient,
) -> None:
    """PDF uploads must not be impacted by missing FFmpeg."""
    job_id = create_test_job(app_with_stub_pool_no_ffmpeg)
    resp = app_with_stub_pool_no_ffmpeg.post(
        f"/v1/ingest/job/{job_id}/document",
        files={"file": ("doc.pdf", b"%PDF-1.4\n%stub\n", "application/pdf")},
        data={"metadata": "{}"},
    )
    assert resp.status_code == 202, resp.text


def test_whole_endpoint_also_rejects_video_when_ffmpeg_missing(
    app_with_stub_pool_no_ffmpeg: TestClient,
) -> None:
    """The /ingest/job/{id}/whole endpoint enforces the gate too."""
    job_id = create_test_job(app_with_stub_pool_no_ffmpeg)
    resp = app_with_stub_pool_no_ffmpeg.post(
        f"/v1/ingest/job/{job_id}/whole",
        files={"file": ("movie.mp4", b"\x00", "video/mp4")},
        data={"metadata": "{}"},
    )
    assert resp.status_code == 501, resp.text
    assert "ffmpeg" in resp.json()["detail"]


def test_startup_logs_warning_when_ffmpeg_missing(caplog: pytest.LogCaptureFixture) -> None:
    """The lifespan startup logs a clear WARNING when FFmpeg is missing."""
    with (
        patch(
            "nemo_retriever.common.modality.audio.media_interface.is_media_available",
            return_value=False,
        ),
        patch(
            "nemo_retriever.common.modality.audio.media_interface.missing_media_dependencies",
            return_value=["ffmpeg", "ffprobe"],
        ),
    ):
        with caplog.at_level(logging.WARNING, logger="nemo_retriever.service.app"):
            _check_media_dependencies(mode="standalone")

    records = [r for r in caplog.records if "Media dependencies missing" in r.getMessage()]
    assert records, "Expected a WARNING about missing media dependencies"
    msg = records[-1].getMessage()
    assert "ffmpeg" in msg and "ffprobe" in msg
    assert "service.installFfmpeg=true" in msg
    assert "HTTP 501" in msg


def test_startup_logs_info_when_ffmpeg_present(caplog: pytest.LogCaptureFixture) -> None:
    """The lifespan startup confirms FFmpeg availability with an INFO log."""
    with patch(
        "nemo_retriever.common.modality.audio.media_interface.is_media_available",
        return_value=True,
    ):
        with caplog.at_level(logging.INFO, logger="nemo_retriever.service.app"):
            _check_media_dependencies(mode="batch")

    records = [r for r in caplog.records if "Media dependencies (ffmpeg, ffprobe) detected" in r.getMessage()]
    assert records, "Expected an INFO log when FFmpeg is available"
    assert "mode=batch" in records[-1].getMessage()
