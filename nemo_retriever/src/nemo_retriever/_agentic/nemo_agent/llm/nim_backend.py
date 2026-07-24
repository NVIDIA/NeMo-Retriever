# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Placeholder for the library's default NIM-backed (litellm-free) LLM backend."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from .base_backend import BaseLLMBackend, BaseLLMConfig
from .result import CompletionResult


class NIMLLMConfig(BaseLLMConfig):
    """Configuration for :class:`NIMLLMBackend`.

    Stub — NIM-specific fields land here when the backend is implemented.
    ``backend`` is pinned by type: a ``NIMLLMConfig`` cannot be constructed
    with (and therefore never routed to) a different backend.
    """

    backend: Literal["nim"] = "nim"


class NIMLLMBackend(BaseLLMBackend):
    """NVIDIA NIM LLM backend — NOT IMPLEMENTED YET.

    Once implemented, this should satisfy the full :class:`BaseLLMBackend`
    contract (see its docstring and the contract test suite) without depending
    on litellm. Until then, use ``LiteLLMConfig(backend="litellm", ...)``.
    """

    config_cls = NIMLLMConfig

    def __init__(self, config: NIMLLMConfig) -> None:
        raise NotImplementedError(
            "NIMLLMBackend is not implemented yet. "
            "Use LiteLLMConfig(backend='litellm', ...) with the litellm extra installed."
        )

    def _completion_impl(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **overrides: Any,
    ) -> CompletionResult:
        raise NotImplementedError
