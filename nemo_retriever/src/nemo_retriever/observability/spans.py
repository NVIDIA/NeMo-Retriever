# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable span and metric context managers for the pipeline seam.

Centralizing the span and metric schema here keeps attribute keys, span
names, and metric labels identical across the three call sites that
exercise the seam (the inprocess executor, the service-mode worker
subprocess, and the public ingest entry point), so a single
OpenTelemetry collector view works for both library and service
deployments.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator, Mapping

from opentelemetry import trace as _trace_api

from nemo_retriever.observability import attributes as A
from nemo_retriever.observability.tracer import (
    documents_failed_counter,
    documents_processed_counter,
    get_tracer,
    operator_duration_histogram,
    safe_add,
)
from nemo_retriever.params.models import RunMode


def tag_current_span(**attrs: Any) -> None:
    """Set attributes on the active span, skipping any that are None.

    Safe to call when no span is recording; the call is a no-op.
    """
    span = _trace_api.get_current_span()
    if not span.is_recording():
        return
    filtered = {k: v for k, v in attrs.items() if v is not None}
    if not filtered:
        return
    try:
        span.set_attributes(filtered)
    except Exception:  # noqa: BLE001 — telemetry must never raise
        pass


def tag_job(job_id: str | None) -> None:
    """Tag the active span with the job id."""
    if job_id is not None:
        tag_current_span(**{A.JOB_ID: job_id})


def tag_document(document_id: str | None) -> None:
    """Tag the active span with the document id."""
    if document_id is not None:
        tag_current_span(**{A.DOCUMENT_ID: document_id})


def tag_upload(*, filename: str | None = None, total_pages: int | None = None) -> None:
    """Tag the active span with the upload's filename and page count."""
    tag_current_span(**{A.FILENAME: filename, A.TOTAL_PAGES: total_pages})


# Operator class name substring to canonical NIM kind. Lets traces be
# filtered by role rather than URL or port, which matters when the same
# NIM image is reachable on different ports across deployments. Order
# matters: more specific substrings come first.
_NIM_KIND_RULES: tuple[tuple[str, str], ...] = (
    ("PageElement", "page_elements"),
    ("GraphicElements", "graphic_elements"),
    ("TableStructure", "table_structure"),
    ("OCR", "ocr"),
    ("BatchEmbed", "embed"),
    ("Embed", "embed"),
    ("Caption", "caption"),
    ("NemotronParse", "nemotron_parse"),
    ("ASR", "asr"),
    ("Rerank", "rerank"),
)


@lru_cache(maxsize=128)
def nim_kind_for_operator(class_name: str) -> str | None:
    """Return the canonical NIM kind for an operator class name, or None.

    Cached because operator_span calls this once per operator per batch
    on the hot path, and the operator class set is small and stable.
    """
    for needle, kind in _NIM_KIND_RULES:
        if needle in class_name:
            return kind
    return None


@contextmanager
def pipeline_span(
    run_mode: RunMode,
    *,
    extraction_mode: str | None = None,
    stages: Mapping[str, Any] | tuple[str, ...] | list[str] | None = None,
    extra_attrs: Mapping[str, Any] | None = None,
) -> Iterator[_trace_api.Span]:
    """Open a root span around an end-to-end pipeline execution.

    Wraps the ingest call so a single trace view shows the full request,
    operator chain, and outbound NIM call tree. Safe to use whether or
    not a global tracer provider is set; with no provider the span is a
    no-op.
    """
    tracer = get_tracer()
    attrs: dict[str, Any] = {A.RUN_MODE: run_mode}
    if extraction_mode is not None:
        attrs[A.EXTRACTION_MODE] = extraction_mode
    if stages is not None:
        if isinstance(stages, (list, tuple)):
            attrs[A.PIPELINE_STAGES] = list(stages)
        else:
            attrs[A.PIPELINE_STAGES] = list(stages.keys())
    if extra_attrs:
        attrs.update(extra_attrs)

    with tracer.start_as_current_span(
        "ingestor.ingest",
        attributes=attrs,
        record_exception=True,
        set_status_on_exception=True,
    ) as span:
        yield span


@contextmanager
def operator_span(
    name: str,
    run_mode: RunMode,
    *,
    operator_class: str | None = None,
    operator_index: int | None = None,
    batch_size: int | None = None,
    extra_attrs: Mapping[str, Any] | None = None,
    parent_context: Any = None,
) -> Iterator[_trace_api.Span]:
    """Open a span and record duration and throughput metrics for one operator call.

    On normal exit, records a histogram observation for the operator's
    wall-clock latency and increments the documents-processed counter by
    the batch size when one is supplied. On exception, records the
    exception on the span, flips the duration label to failed, and
    increments the documents-failed counter by the batch size (or one
    when no batch size is given), tagged with the exception class name.

    The parent_context argument attaches the span to an explicit parent,
    used in Ray batch mode where the actor process has no active context
    and needs to inherit the driver's pipeline span.
    """
    tracer = get_tracer()
    span_attrs: dict[str, Any] = {
        A.OPERATOR_NAME: name,
        A.RUN_MODE: run_mode,
    }
    if operator_class is not None:
        span_attrs[A.OPERATOR_CLASS] = operator_class
        kind = nim_kind_for_operator(operator_class)
        if kind is not None:
            span_attrs[A.NIM_KIND] = kind
    if operator_index is not None:
        span_attrs[A.OPERATOR_INDEX] = operator_index
    if batch_size is not None:
        span_attrs[A.BATCH_SIZE] = batch_size
    if extra_attrs:
        span_attrs.update(extra_attrs)

    metric_attrs = {A.OPERATOR_NAME: name, A.RUN_MODE: run_mode}

    t0 = time.monotonic()
    error_class: str | None = None
    try:
        with tracer.start_as_current_span(
            f"operator.{name}",
            context=parent_context,
            attributes=span_attrs,
            record_exception=True,
            set_status_on_exception=True,
        ) as span:
            yield span
    except BaseException as exc:  # noqa: BLE001 — re-raised immediately; we only tag the metric
        error_class = type(exc).__name__
        raise
    finally:
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        status = "ok" if error_class is None else "failed"
        try:
            operator_duration_histogram().record(elapsed_ms, attributes={**metric_attrs, A.STATUS: status})
        except Exception:  # noqa: BLE001 — instrumentation must never raise
            pass

        if error_class is None:
            if batch_size:
                safe_add(
                    documents_processed_counter(),
                    batch_size,
                    {**metric_attrs, A.STATUS: "ok"},
                )
        else:
            safe_add(
                documents_failed_counter(),
                batch_size or 1,
                {**metric_attrs, A.ERROR_CLASS: error_class},
            )
