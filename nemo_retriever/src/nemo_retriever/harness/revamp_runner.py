# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.harness.contracts import (
    EXIT_ARTIFACT_WRITE_FAILURE,
    EXIT_EVALUATION_FAILURE,
    EXIT_INGEST_FAILURE,
    EXIT_INTERNAL_ERROR,
    EXIT_INVALID,
    EXIT_METRIC_GATE_FAILURE,
    EXIT_MISSING_INPUT,
    EXIT_QUERY_FAILURE,
    EXIT_SUCCESS,
    FailurePayload,
    HarnessRunError,
    RunOutcome,
)
from nemo_retriever.harness.execution import run_benchmark
from nemo_retriever.harness.resolution import show_benchmark_payload

__all__ = [
    "EXIT_ARTIFACT_WRITE_FAILURE",
    "EXIT_EVALUATION_FAILURE",
    "EXIT_INGEST_FAILURE",
    "EXIT_INTERNAL_ERROR",
    "EXIT_INVALID",
    "EXIT_METRIC_GATE_FAILURE",
    "EXIT_MISSING_INPUT",
    "EXIT_QUERY_FAILURE",
    "EXIT_SUCCESS",
    "FailurePayload",
    "HarnessRunError",
    "RunOutcome",
    "run_benchmark",
    "show_benchmark_payload",
]
