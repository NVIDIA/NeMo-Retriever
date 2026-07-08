# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LLM primitives: Protocols, result dataclasses, and concrete clients.

Types, Protocols, and result dataclasses are always available (zero
external deps).  ``LiteLLMClient`` and ``LLMJudge`` are lazy-loaded so
that lightweight consumers can use the type contracts without
installing ``litellm``::

    from nemo_retriever.models.llm import RetrieverStrategy, RetrievalResult  # cheap
    from nemo_retriever.models.llm import LiteLLMClient  # imports litellm on first use

Credentials
-----------
Per-component API keys (``api_key``) and base URLs (``api_base``) are
passed directly on ``LiteLLMClient.from_kwargs`` / ``LLMJudge.from_kwargs``
or via ``Retriever(embed_kwargs={"api_key": ..., "embedding_endpoint": ...})``.  When
``api_key`` is left ``None``, LiteLLM performs provider-native environment
lookup. An explicit ``os.environ/VARIABLE_NAME`` value resolves that variable
immediately before the provider call. Literal keys are supported for local,
non-persisted execution but graph persistence rejects them.

Public surface contract
-----------------------
The one-shot text generation names in ``__all__`` below are a provisional v1
API. They support synchronous, single-turn text only; tools, streaming,
multiple choices, and structured domain responses are not supported. Other
established names retain their existing compatibility commitments. External
callers should import from ``nemo_retriever.models.llm``
rather than reaching into submodules (``models.llm.clients.litellm``,
``models.llm.text_utils``) directly -- those submodule paths are implementation
details and may be reorganised in future releases without notice.
"""

from nemo_retriever.models.llm.types import (
    AnswerJudge,
    AnswerRequest,
    AnswerResult,
    GeneratedTextResult,
    GenerationRequest,
    GenerationResult,
    JudgeResult,
    LLMClient,
    RetrievalResult,
    RetrieverStrategy,
    TextCompletionClient,
)
from nemo_retriever.common.params.models import (
    LLMInferenceParams,
    LLMRemoteClientParams,
    LLMSamplingOverrides,
    TextGenerationParams,
)
from nemo_retriever.models.llm.tasks import (
    TextGenerationTask,
    GenerationTaskError,
    GenericPromptTask,
    RagAnswerTask,
    SummarizeTask,
)

_LAZY_IMPORTS = {
    "LiteLLMClient": "nemo_retriever.models.llm.clients.litellm",
    "LLMJudge": "nemo_retriever.models.llm.clients.judge",
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
    "TextGenerationTask",
    "GenerationTaskError",
    "LLMClient",
    "RetrieverStrategy",
    "TextCompletionClient",
    # Request/result models
    "AnswerRequest",
    "AnswerResult",
    "GeneratedTextResult",
    "GenerationRequest",
    "GenerationResult",
    "JudgeResult",
    "RetrievalResult",
    # Tasks
    "GenericPromptTask",
    "RagAnswerTask",
    "SummarizeTask",
    # Concrete clients (lazy-loaded)
    "LLMJudge",
    "LiteLLMClient",
    # Transport / sampling params (re-exported for ergonomics)
    "LLMInferenceParams",
    "LLMRemoteClientParams",
    "LLMSamplingOverrides",
    "TextGenerationParams",
]
