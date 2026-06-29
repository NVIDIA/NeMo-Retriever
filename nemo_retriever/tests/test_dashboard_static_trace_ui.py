# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC_VIEWS = ROOT / "src" / "nemo_retriever" / "service" / "dashboard" / "static" / "views"


def _source(name: str) -> str:
    return (STATIC_VIEWS / name).read_text(encoding="utf-8")


def test_jobs_table_renders_and_retains_trace_id() -> None:
    source = _source("jobs.jsx")

    assert "'Trace ID'" in source
    assert "trace_id: ev.trace_id != null ? ev.trace_id" in source
    assert "j.trace_id" in source


def test_job_detail_renders_and_retains_trace_id() -> None:
    source = _source("job_detail.jsx")

    assert "'Trace ID'" in source
    assert "trace_id: data.trace_id != null ? data.trace_id" in source
    assert "job.trace_id" in source
