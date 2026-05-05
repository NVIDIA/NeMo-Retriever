# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Concrete LLM client implementations and a lightweight client registry.

The ``llm.clients`` package hosts every concrete :class:`LLMClient`
implementation in its own submodule (``litellm.py``, ``judge.py``, ...)
so that adding a new client means adding a new file rather than
extending a monolithic module.  To keep the public import path stable,
the registered client classes plus the internal prompt-helper
(``_build_rag_prompt``) and response-parser (``_parse_judge_response``)
are re-exported from this package's namespace.  Any caller that imports
``from nemo_retriever.llm.clients import LiteLLMClient`` will therefore
continue to work unchanged after the module-to-package refactor.
"""

from __future__ import annotations

from nemo_retriever.llm.clients.judge import LLMJudge, _parse_judge_response
from nemo_retriever.llm.clients.litellm import LiteLLMClient, _build_rag_prompt

_REGISTRY: dict[str, type] = {}


def register_client(name: str, cls: type) -> None:
    """Register a client class under a human-readable name.

    The registry is optional: every concrete client remains addressable
    by its import path.  Registration is offered as a convenience for
    configuration-driven instantiation (e.g. reading a ``type`` key from
    a YAML file and looking up the matching class).

    Args:
        name: Stable lookup name (e.g. ``"litellm"``).
        cls: The client class.  Must expose a ``from_kwargs``
            classmethod to be useful to configuration-driven callers.
    """

    _REGISTRY[name] = cls


def get_client(name: str) -> type:
    """Return a registered client class by name.

    Raises:
        KeyError: When ``name`` has not been registered.
    """

    return _REGISTRY[name]


register_client("litellm", LiteLLMClient)


__all__ = [
    "LLMJudge",
    "LiteLLMClient",
    "_build_rag_prompt",
    "_parse_judge_response",
    "get_client",
    "register_client",
]
