# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared validation helpers for agentic retrieval options."""

from __future__ import annotations

import math
from urllib.parse import urlparse

AGENTIC_DEFAULT_BEIR_KS = (1, 3, 5, 10)
AGENTIC_AUDIO_RECALL_KS = (1, 5, 10)
AGENTIC_TEMPERATURE_MIN = 0.0
AGENTIC_OPENAI_COMPATIBLE_TEMPERATURE_MAX = 2.0
AGENTIC_NVIDIA_TEMPERATURE_MAX = 1.0


def _parse_integer(value: object, *, field_name: str) -> tuple[int | None, str | None]:
    if isinstance(value, bool):
        return None, f"{field_name} must be an integer"
    if isinstance(value, int):
        return value, None
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None, f"{field_name} must be an integer"
        return int(value), None

    text = str(value).strip()
    if not text:
        return None, f"{field_name} must be an integer"
    try:
        return int(text), None
    except ValueError:
        try:
            as_float = float(text)
        except ValueError:
            return None, f"{field_name} must be an integer"
        if not math.isfinite(as_float) or not as_float.is_integer():
            return None, f"{field_name} must be an integer"
        return int(as_float), None


def agentic_int_value(value: object, *, field_name: str) -> int:
    parsed, error = _parse_integer(value, field_name=field_name)
    if error or parsed is None:
        raise ValueError(error or f"{field_name} must be an integer")
    return parsed


def agentic_int_min_error(value: object, *, field_name: str, min_value: int) -> str | None:
    parsed, error = _parse_integer(value, field_name=field_name)
    if error or parsed is None:
        return error or f"{field_name} must be an integer"
    if parsed < int(min_value):
        return f"{field_name} must be >= {int(min_value)}"
    return None


def is_nvidia_agentic_endpoint(invoke_url: str | None) -> bool:
    """Return true for the default/hosted NVIDIA chat-completions endpoint."""

    if invoke_url is None or not str(invoke_url).strip():
        return True
    host = urlparse(str(invoke_url).strip()).netloc.lower()
    return host == "api.nvidia.com" or host.endswith(".api.nvidia.com")


def agentic_temperature_max(invoke_url: str | None) -> float:
    """Return the documented max temperature for the configured chat API."""

    if is_nvidia_agentic_endpoint(invoke_url):
        return AGENTIC_NVIDIA_TEMPERATURE_MAX
    return AGENTIC_OPENAI_COMPATIBLE_TEMPERATURE_MAX


def agentic_temperature_error(
    temperature: object,
    *,
    invoke_url: str | None,
    field_name: str = "agentic_temperature",
) -> str | None:
    try:
        value = float(temperature)
    except (TypeError, ValueError):
        return f"{field_name} must be a number"
    if not math.isfinite(value):
        return f"{field_name} must be finite"

    max_value = agentic_temperature_max(invoke_url)
    if value < AGENTIC_TEMPERATURE_MIN or value > max_value:
        endpoint_label = "NVIDIA NIM" if is_nvidia_agentic_endpoint(invoke_url) else "OpenAI-compatible"
        return f"{field_name} must be between {AGENTIC_TEMPERATURE_MIN:.1f} and {max_value:.1f} for {endpoint_label} endpoints"
    return None


def agentic_target_top_k(evaluation_mode: str, beir_k: list[int] | tuple[int, ...] | None = None) -> int:
    """Resolve the final document count required by the selected evaluation."""

    if str(evaluation_mode) == "beir":
        raw_ks = tuple(beir_k or AGENTIC_DEFAULT_BEIR_KS)
    elif str(evaluation_mode) == "audio_recall":
        raw_ks = AGENTIC_AUDIO_RECALL_KS
    else:
        raise ValueError(f"Unsupported agentic evaluation mode: {evaluation_mode!r}")

    positive_ks: list[int] = []
    for raw_k in raw_ks:
        parsed_k, error = _parse_integer(raw_k, field_name="k")
        if error or parsed_k is None:
            raise ValueError("agentic evaluation k values must be integers")
        if parsed_k > 0:
            positive_ks.append(parsed_k)
    if not positive_ks:
        raise ValueError("agentic evaluation requires at least one positive k")
    return max(positive_ks)


def agentic_backend_top_k_error(
    backend_top_k: object,
    *,
    target_top_k: int,
    field_name: str = "agentic_backend_top_k",
) -> str | None:
    parsed, error = _parse_integer(backend_top_k, field_name=field_name)
    if error or parsed is None:
        return error or f"{field_name} must be an integer"
    value = parsed

    if value < 1:
        return f"{field_name} must be >= 1"
    if value < int(target_top_k):
        return f"{field_name} must be >= target top_k ({int(target_top_k)})"
    return None
