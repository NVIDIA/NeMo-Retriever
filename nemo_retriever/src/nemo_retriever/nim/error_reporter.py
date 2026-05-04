# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight error collector for operator-internal exceptions.

Operators that catch exceptions internally (to allow the pipeline to
continue processing remaining rows) call :func:`report_error` so the
service layer can later drain the collected errors and persist them to
the provenance event log.

This uses a simple module-level list.  Each worker process has its own
address space, so there is no cross-process contention.  The service
layer calls :func:`drain_errors` after each pipeline stage completes.
"""

from __future__ import annotations

import dataclasses
import traceback as _traceback
from typing import Optional


@dataclasses.dataclass(frozen=True)
class OperatorError:
    """A single error caught internally by an operator."""

    stage: str
    exc_type: str
    message: str
    traceback: str
    row_index: Optional[int] = None


_errors: list[OperatorError] = []


def report_error(stage: str, exc: BaseException, *, row_index: Optional[int] = None) -> None:
    """Record an operator-internal exception for later persistence.

    Call this alongside existing error-in-metadata handling so the
    pipeline continues but the service layer still captures the event.
    """
    _errors.append(
        OperatorError(
            stage=stage,
            exc_type=type(exc).__name__,
            message=str(exc),
            traceback="".join(_traceback.format_exception(type(exc), exc, exc.__traceback__)),
            row_index=row_index,
        )
    )


def drain_errors() -> list[OperatorError]:
    """Pop and return all collected errors since last drain."""
    results = list(_errors)
    _errors.clear()
    return results
