# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Mapping

from opentelemetry import context as otel_context
from opentelemetry import trace as _trace_api
from opentelemetry.propagate import extract, inject


def inject_current_context() -> dict[str, str]:
    """Serialise the active span context to a carrier dict; ``{}`` when none."""
    if not _trace_api.get_current_span().get_span_context().is_valid:
        return {}
    carrier: dict[str, str] = {}
    inject(carrier)
    return carrier


def extract_context(carrier: Mapping[str, Any] | None) -> Any:
    """Rehydrate an OTEL ``Context`` from a carrier; empty ``Context`` when missing."""
    if not carrier:
        return otel_context.Context()
    return extract(carrier)
