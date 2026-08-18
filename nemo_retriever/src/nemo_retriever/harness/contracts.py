# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

STATUS_VALUES = {"planned", "running", "complete", "failed"}
PHASE_VALUES = {
    "resolve",
    "ingest_plan",
    "ingest",
    "query_plan",
    "query",
    "evaluate",
    "write_artifacts",
}
EXIT_SUCCESS = 0
EXIT_INVALID = 2
EXIT_MISSING_INPUT = 3
EXIT_INGEST_FAILURE = 10
EXIT_QUERY_FAILURE = 11
EXIT_EVALUATION_FAILURE = 12
EXIT_METRIC_GATE_FAILURE = 20
EXIT_ARTIFACT_WRITE_FAILURE = 30
EXIT_INTERNAL_ERROR = 70
MODE_TO_RUN_MODE = {"local": "inprocess", "batch": "batch", "service": "service"}


@dataclass(frozen=True)
class FailurePayload:
    failed_phase: str
    failure_reason: str
    retryable: bool
    message: str
    debug_artifacts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunOutcome:
    exit_code: int
    artifact_dir: Path
    results: dict[str, Any]
    results_path: Path | None


class HarnessRunError(Exception):
    def __init__(self, exit_code: int, failure: FailurePayload) -> None:
        super().__init__(failure.message)
        self.exit_code = exit_code
        self.failure = failure
