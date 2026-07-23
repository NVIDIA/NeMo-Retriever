# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed errors raised by the public retriever service client."""


class RetrieverServiceError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RetrieverServiceNotFoundError(RetrieverServiceError):
    """The requested scoped resource does not exist."""


class RetrieverServiceConflictError(RetrieverServiceError):
    """The request conflicts with current state or idempotency history."""


class RetrieverServiceValidationError(RetrieverServiceError):
    """The service rejected invalid input."""
