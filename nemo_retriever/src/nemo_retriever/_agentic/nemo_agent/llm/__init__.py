# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal LLM backend components for the private agent implementation.

Internal integration surface:

- :class:`BaseLLMBackend` — the ABC to implement for a custom backend
  (one required method: ``_completion_impl``).
- :class:`CompletionResult` — the envelope every completion call returns.
- :class:`BaseLLMConfig` / :class:`CallableLLMConfig` / :class:`LiteLLMConfig` —
  configuration models, each colocated with its backend class.
- :func:`create_llm` — config-driven backend factory.
- :func:`create_llm_config` — kwargs-filtering config factory: builds a
  backend's config, dropping (with a warning) any field that config does not
  support.
- :func:`get_available_backends` — the registered backend names; the source of
  truth callers should read.
- ``errors`` — the :class:`LLMCallError` hierarchy callers branch on.
- ``error_classification`` — :func:`classify_call_exception`, which maps a
  completion callable's failure onto that hierarchy without depending on any
  HTTP library.
- ``usage`` binders — :func:`bind_query_id` / :func:`bind_stage` for
  per-(query, stage) token-usage attribution.
- ``helpers`` — reusable pieces for backend authors
  (:func:`extract_reasoning_from_message`, :func:`extract_text_content`,
  :func:`normalize_messages_for_api`, :func:`strip_private_message_keys`,
  :func:`redact_url`, :func:`excerpt`, :func:`resolve_api_key`).
"""

import logging
from typing import Any

from .base_backend import BaseLLMBackend, BaseLLMConfig
from .callable_backend import CallableLLMBackend, CallableLLMConfig
from .error_classification import classify_call_exception
from .errors import ContentPolicyError, ContextLimitError, LLMCallError, RateLimitError
from .helpers import (
    excerpt,
    extract_reasoning_from_message,
    extract_text_content,
    normalize_messages_for_api,
    redact_url,
    resolve_api_key,
    strip_private_message_keys,
)
from .litellm_backend import LiteLLMBackend, LiteLLMConfig
from .result import CompletionResult
from .usage import (
    UNSET_QUERY,
    UNSET_STAGE,
    bind_query_id,
    bind_stage,
    coerce_usage_to_dict,
    deep_merge_usage,
    deep_merge_usage_breakdown,
    get_query_id,
    get_stage,
    sum_usage_breakdown,
)

logger = logging.getLogger(__name__)

_BACKEND_REGISTRY = {
    "litellm": LiteLLMBackend,
    "callable": CallableLLMBackend,
}


def get_available_backends() -> tuple[str, ...]:
    """Return the names of all registered LLM backends, sorted."""
    return tuple(sorted(_BACKEND_REGISTRY))


def create_llm(config: BaseLLMConfig, **kwargs: Any) -> BaseLLMBackend:
    """Instantiate the LLM backend selected by ``config.backend``.

    Each backend validates that it received its own config subclass (e.g.
    ``backend="litellm"`` requires a :class:`LiteLLMConfig`).

    Extra keyword arguments are forwarded verbatim to the backend constructor —
    this is how injection-only backends receive their runtime dependency (e.g.
    ``create_llm(config, completion_fn=fn)`` for :class:`CallableLLMBackend`). The
    factory does not inspect or filter them: passing a kwarg a backend does not
    accept is a caller error and surfaces as a ``TypeError`` from the constructor.
    """
    try:
        backend_cls = _BACKEND_REGISTRY[config.backend]
    except KeyError:
        raise ValueError(
            f"Unrecognized LLM backend {config.backend!r}. " f"Available backends: {sorted(_BACKEND_REGISTRY)}."
        ) from None
    return backend_cls(config, **kwargs)


def create_llm_config(backend: str, **kwargs: Any) -> BaseLLMConfig:
    """Build the configuration for ``backend``, dropping unsupported fields.

    ``backend`` selects the config subclass (via the same registry as
    :func:`create_llm`). Any keyword not declared on that config subclass is
    dropped with a ``WARNING`` naming the dropped keys (values are never logged);
    required-field and per-field type validation still apply to the kept fields.

    ``backend`` is the selector and is never forwarded into the config — the
    chosen subclass's ``Literal`` default sets ``config.backend``. Building a
    config never instantiates the backend.
    """
    try:
        backend_cls = _BACKEND_REGISTRY[backend]
    except KeyError:
        raise ValueError(
            f"Unrecognized LLM backend {backend!r}. " f"Available backends: {sorted(_BACKEND_REGISTRY)}."
        ) from None
    config_cls = backend_cls.config_cls
    supported = set(config_cls.model_fields)
    kept = {k: v for k, v in kwargs.items() if k in supported}
    dropped = sorted(k for k in kwargs if k not in supported)
    if dropped:
        logger.warning(
            "create_llm_config: backend %r (%s) does not support config field(s) " "%s; ignoring them.",
            backend,
            config_cls.__name__,
            dropped,
        )
    return config_cls(**kept)


__all__ = [
    "BaseLLMBackend",
    "BaseLLMConfig",
    "CallableLLMBackend",
    "CallableLLMConfig",
    "CompletionResult",
    "ContentPolicyError",
    "ContextLimitError",
    "LLMCallError",
    "LiteLLMBackend",
    "LiteLLMConfig",
    "RateLimitError",
    "UNSET_QUERY",
    "UNSET_STAGE",
    "bind_query_id",
    "bind_stage",
    "classify_call_exception",
    "coerce_usage_to_dict",
    "create_llm",
    "create_llm_config",
    "deep_merge_usage",
    "deep_merge_usage_breakdown",
    "excerpt",
    "extract_reasoning_from_message",
    "extract_text_content",
    "get_available_backends",
    "get_query_id",
    "get_stage",
    "normalize_messages_for_api",
    "redact_url",
    "resolve_api_key",
    "strip_private_message_keys",
    "sum_usage_breakdown",
]
