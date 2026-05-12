# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Span and metric attribute name constants used by NeMo Retriever instrumentation."""

from __future__ import annotations

# Run mode: "inprocess", "batch", or "service".
RUN_MODE = "nemo_retriever.run_mode"

# Operator metadata.
OPERATOR_NAME = "nemo_retriever.operator.name"
OPERATOR_CLASS = "nemo_retriever.operator.class"
OPERATOR_INDEX = "nemo_retriever.operator.index"

# Batch metadata.
BATCH_SIZE = "nemo_retriever.batch.size"
BATCH_OUTPUT_ROWS = "nemo_retriever.batch.output_rows"
BATCH_JOB_COUNT = "nemo_retriever.batch.job_count"
BATCH_DOCUMENT_COUNT = "nemo_retriever.batch.document_count"

# Pipeline metadata.
PIPELINE_STAGES = "nemo_retriever.pipeline.stages"
EXTRACTION_MODE = "nemo_retriever.extraction_mode"

# Per-request correlation in service mode.
REQUEST_ID = "nemo_retriever.request_id"

# Document/job correlation.
DOCUMENT_ID = "nemo_retriever.document_id"
JOB_ID = "nemo_retriever.job_id"
FILENAME = "nemo_retriever.filename"
TOTAL_PAGES = "nemo_retriever.total_pages"

# Worker process identity (service mode cross-process spans).
WORKER_PID = "nemo_retriever.worker.pid"

# Error classification on failed operator spans.
ERROR_CLASS = "nemo_retriever.error.class"

# Status label used on document counters: "ok", "failed", "cancelled".
STATUS = "nemo_retriever.status"

# NIM endpoint metadata.
NIM_KIND = "nemo_retriever.nim.kind"
NIM_NAME = "nemo_retriever.nim.name"
NIM_URL = "nemo_retriever.nim.url"
NIM_PROBE_STATUS = "nemo_retriever.nim.probe_status"

# Instrumentation scope name (passed to ``get_tracer`` / ``get_meter``).
INSTRUMENTATION_SCOPE = "nemo_retriever"
