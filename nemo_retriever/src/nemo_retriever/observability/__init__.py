# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry observability primitives for NeMo Retriever.

The library exposes a thin, API-only wrapper over OpenTelemetry so it
can be imported safely whether or not the user has installed the
``[otel]`` extra.  When no ``TracerProvider`` / ``MeterProvider`` is
registered globally — the default — every instrumentation point is a
near-zero-cost no-op.

Two usage patterns:

* **Embedded library mode**: the host application configures
  OpenTelemetry once at startup; NeMo Retriever's spans and metrics
  inherit that provider automatically.  No code change required on
  our side.
* **Service mode**: call :func:`configure` (or pass an :class:`OTELConfig`
  to the relevant factory) to install global providers wired to an OTLP
  collector.
"""

from __future__ import annotations

from nemo_retriever.observability import attributes
from nemo_retriever.observability.configure import OTELConfig, apply_otel_env_defaults, configure
from nemo_retriever.observability.propagate import extract_context, inject_current_context
from nemo_retriever.observability.ray_integration import RayOperatorSpanWrapper, collect_otel_env
from nemo_retriever.observability.spans import (
    nim_kind_for_operator,
    operator_span,
    pipeline_span,
    tag_current_span,
    tag_document,
    tag_job,
    tag_upload,
)
from nemo_retriever.observability.tracer import (
    documents_failed_counter,
    documents_processed_counter,
    get_meter,
    get_tracer,
    jobs_completed_counter,
    jobs_failed_counter,
    jobs_submitted_counter,
    operator_duration_histogram,
    pages_completed_counter,
    pages_failed_counter,
    reset_instrument_cache,
    safe_add,
)

__all__ = [
    "OTELConfig",
    "RayOperatorSpanWrapper",
    "apply_otel_env_defaults",
    "attributes",
    "collect_otel_env",
    "configure",
    "documents_failed_counter",
    "documents_processed_counter",
    "extract_context",
    "get_meter",
    "get_tracer",
    "inject_current_context",
    "jobs_completed_counter",
    "jobs_failed_counter",
    "jobs_submitted_counter",
    "nim_kind_for_operator",
    "operator_duration_histogram",
    "operator_span",
    "pages_completed_counter",
    "pages_failed_counter",
    "pipeline_span",
    "reset_instrument_cache",
    "safe_add",
    "tag_current_span",
    "tag_document",
    "tag_job",
    "tag_upload",
]
