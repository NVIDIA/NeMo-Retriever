# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LLM primitives: Protocols, result dataclasses, and concrete clients.

Types, Protocols, and result dataclasses are always available (zero
external deps).  ``LiteLLMClient`` and ``LLMJudge`` are lazy-loaded so
that lightweight consumers can use the type contracts without
installing ``litellm``::

    from nemo_retriever.llm import RetrieverStrategy, RetrievalResult  # cheap
    from nemo_retriever.llm import LiteLLMClient  # imports litellm on first use

Credentials
-----------
Per-component API keys (``api_key``) and base URLs (``api_base``) are
passed directly on ``LiteLLMClient.from_kwargs`` / ``LLMJudge.from_kwargs``
or on ``Retriever(embedding_api_key=..., embedding_endpoint=...)``.  When
``api_key`` is left ``None`` the shared ``_ParamsModel`` validator
resolves ``NVIDIA_API_KEY`` / ``NGC_API_KEY`` from the environment.  This
keeps the common single-provider path a one-liner while still allowing
multiple independent endpoints to coexist -- each component takes its
own ``(api_base, api_key, model)`` triple.

Public surface contract
-----------------------
The names in ``__all__`` below are the frozen public API of this
module.  External callers should import from ``nemo_retriever.llm``
rather than reaching into submodules (``llm.clients.litellm``,
``llm.text_utils``) directly -- those submodule paths are implementation
details and may be reorganised in future releases without notice.  The
Protocols + result dataclasses + concrete clients + re-exported params
models listed here are the supported integration points.
"""

from nemo_retriever.llm.types import (
    AnswerJudge,
    AnswerResult,
    GenerationResult,
    JudgeResult,
    LLMClient,
    RetrievalResult,
    RetrieverStrategy,
)
from nemo_retriever.params.models import LLMInferenceParams, LLMRemoteClientParams

_LAZY_IMPORTS = {
    "LiteLLMClient": "nemo_retriever.llm.clients.litellm",
    "LLMJudge": "nemo_retriever.llm.clients.judge",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Protocols
    "AnswerJudge",
    "LLMClient",
    "RetrieverStrategy",
    # Result dataclasses
    "AnswerResult",
    "GenerationResult",
    "JudgeResult",
    "RetrievalResult",
    # Concrete clients (lazy-loaded)
    "LLMJudge",
    "LiteLLMClient",
    # Transport / sampling params (re-exported for ergonomics)
    "LLMInferenceParams",
    "LLMRemoteClientParams",
]
