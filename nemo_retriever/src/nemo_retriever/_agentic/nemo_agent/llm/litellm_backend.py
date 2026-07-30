# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LiteLLM-backed implementation of :class:`BaseLLMBackend`.

litellm is an OPTIONAL dependency: this module must never import it at top
level. The import happens in ``LiteLLMBackend.__init__`` so missing litellm
fails fast at construction with a clear error, while ``import nemo_agent.llm``
stays safe for users of other backends.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

from pydantic import field_validator

from .base_backend import BaseLLMBackend, BaseLLMConfig
from .errors import ContentPolicyError, ContextLimitError, LLMCallError, RateLimitError
from .helpers import (
    extract_reasoning_from_message,
    extract_text_content,
    normalize_messages_for_api,
    strip_private_message_keys,
)
from .result import CompletionResult
from .usage import coerce_usage_to_dict

if TYPE_CHECKING:  # pragma: no cover - typing only, litellm may be absent
    from litellm.types.utils import ModelResponse

logger = logging.getLogger(__name__)

_REDACTED = "***REDACTED***"
_SENSITIVE_REQUEST_KEYS = frozenset({"api_key"})
_HEADER_CONTAINER_KEYS = frozenset({"headers", "extra_headers", "default_headers"})
_SENSITIVE_HEADER_MARKERS = ("authorization", "api-key", "api_key", "token", "secret")

_TAGGABLE_BLOCK_TYPES = frozenset({"text", "tool_result", "tool_use", "image", "document"})


def _patch_litellm_nvidia_nim_cache_passthrough() -> bool:
    """Stop LiteLLM's ``nvidia_nim`` provider from stripping ``cache_control``.

    LiteLLM's ``OpenAIGPTConfig.(async_)transform_request`` calls
    ``remove_cache_control_flag_from_messages_and_tools`` on every outgoing
    request, which deletes any ``cache_control`` field from ``messages`` and
    ``tools``. Because ``NvidiaNimConfig`` extends ``OpenAIGPTConfig`` without
    overriding that method, every Anthropic-style cache-control marker is
    silently scrubbed before the request leaves the process, even though NIM
    itself relays the markers to Bedrock/Anthropic correctly. The Databricks
    provider has the same problem and ships an explicit no-op override; we
    mirror that fix by replacing the method with a no-op on the NIM config
    class.

    Applied at backend construction (idempotent). Disable by exporting
    ``RB_PATCH_LITELLM_CACHE_CONTROL=0`` (e.g. to A/B against an upstream
    LiteLLM release that ships this fix natively).

    Returns ``True`` if the patch was applied this call, ``False`` otherwise
    (disabled via env, already applied, or the import failed on an
    incompatible LiteLLM version).
    """
    if os.environ.get("RB_PATCH_LITELLM_CACHE_CONTROL", "1").lower() in ("0", "false", "no", "off"):
        logger.info("RB_PATCH_LITELLM_CACHE_CONTROL=0 — skipping cache_control passthrough patch.")
        return False

    try:
        from litellm.llms.nvidia_nim.chat.transformation import NvidiaNimConfig
    except Exception as e:  # pragma: no cover - depends on installed litellm
        logger.warning(
            "Could not import NvidiaNimConfig to patch cache_control passthrough: %s. "
            "Anthropic prompt caching may be silently disabled for nvidia_nim/* models.",
            e,
        )
        return False

    if getattr(NvidiaNimConfig, "_agent_core_cache_control_patched", False):
        return False

    def _noop_remove_cache_control_flag(self, model, messages, tools=None):
        # Same signature as OpenAIGPTConfig.remove_cache_control_flag_from_messages_and_tools
        # but a no-op so cache_control survives all the way to NIM.
        return messages, tools

    NvidiaNimConfig.remove_cache_control_flag_from_messages_and_tools = (  # type: ignore[assignment]
        _noop_remove_cache_control_flag
    )
    NvidiaNimConfig._agent_core_cache_control_patched = True  # type: ignore[attr-defined]
    logger.info(
        "Patched litellm NvidiaNimConfig.remove_cache_control_flag_from_messages_and_tools "
        "to no-op so Anthropic cache_control markers reach NIM (set "
        "RB_PATCH_LITELLM_CACHE_CONTROL=0 to disable)."
    )
    return True


def _block_has_cache_control(block: Any) -> bool:
    """True if ``block`` is a dict carrying an Anthropic ``cache_control`` marker."""
    return isinstance(block, dict) and isinstance(block.get("cache_control"), dict)


def _is_taggable_block(block: Any) -> bool:
    """True if ``block`` is a content block that Anthropic accepts a marker on."""
    return isinstance(block, dict) and block.get("type") in _TAGGABLE_BLOCK_TYPES


def _find_last_markable_msg_idx(messages: List[Dict[str, Any]]) -> Optional[int]:
    """Return the index of the most-recent message we can safely add a marker to.

    Returns ``None`` when either no message is markable OR the most-recent
    markable message already carries a deliberate ``cache_control`` marker
    (see :meth:`LiteLLMBackend._with_last_message_cache_control_marker` for the
    dedup rationale).
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            if content.strip() == "":
                # Empty assistant tool-call turn: skip, look further back.
                continue
            return i  # plain-string content cannot carry a pre-existing marker
        if isinstance(content, list) and any(_is_taggable_block(p) for p in content):
            if any(_block_has_cache_control(p) for p in content):
                return None  # caller already marked this message; preserve their intent
            return i
    return None


def _apply_cache_control_to_last_block(
    message: Dict[str, Any], cache_control: Dict[str, Any]
) -> Dict[str, Any]:
    """Return a copy of ``message`` with ``cache_control`` on its last taggable block.

    Handles both content shapes:
    - ``content: str``: wrap into a single-block text list carrying the marker.
    - ``content: [blocks]``: copy the list, attach the marker to the last
      taggable block (text / tool_result / tool_use / image / document); leave
      the rest untouched. If no taggable block exists the message is returned
      verbatim.
    """
    out = dict(message)
    content = out.get("content")
    cc = dict(cache_control)
    if isinstance(content, str):
        out["content"] = [{"type": "text", "text": content, "cache_control": cc}]
        return out
    if not isinstance(content, list):
        return out
    last_taggable_idx = max(
        (j for j, p in enumerate(content) if _is_taggable_block(p)), default=-1
    )
    if last_taggable_idx < 0:
        return out
    new_content: List[Any] = []
    for j, part in enumerate(content):
        if j == last_taggable_idx and isinstance(part, dict):
            marked = dict(part)
            marked["cache_control"] = cc
            new_content.append(marked)
        else:
            new_content.append(part)
    out["content"] = new_content
    return out


def _redacted_request(request_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copied, credential-redacted snapshot of the request for ``raw_request``."""
    out: Dict[str, Any] = {}
    for k, v in request_kwargs.items():
        if k in _SENSITIVE_REQUEST_KEYS:
            out[k] = _REDACTED
            continue
        if k in _HEADER_CONTAINER_KEYS and isinstance(v, dict):
            out[k] = {
                hk: (_REDACTED if any(m in str(hk).lower() for m in _SENSITIVE_HEADER_MARKERS) else deepcopy(hv))
                for hk, hv in v.items()
            }
            continue
        out[k] = deepcopy(v)
    return out


class LiteLLMConfig(BaseLLMConfig):
    """Configuration for :class:`LiteLLMBackend`.

    Attributes
    ----------
    thinking:
        Provider extended-thinking knob (e.g. Anthropic-style
        ``{"type": "enabled", "budget_tokens": ...}``); routed through litellm's
        supported-params check and forwarded via ``extra_body`` when the provider
        adapter does not recognize it (so ``drop_params=True`` cannot silently
        strip it). ``reasoning_effort`` (inherited from ``BaseLLMConfig``) is
        routed the same way.
    api_version:
        Provider API version (e.g. Azure-style endpoints).
    num_retries:
        litellm-internal retry count for transient errors. Distinct from the
        base class's rate-limit pause loop. Overridable per call
        (e.g. ``num_retries=0`` for a fail-fast preflight probe).
    drop_params / allowed_openai_params:
        litellm parameter-filtering controls, passed through as-is.
        ``drop_params`` defaults to ``True`` so an OpenAI-compatible endpoint that
        rejects a param litellm's adapter emits does not 400 the whole run; set it
        ``False`` to surface those rejections instead. Note this is exactly why
        provider-specific params (``thinking`` / ``reasoning_effort`` /
        ``cache_control``) are routed via ``extra_body`` below — top-level params
        litellm does not recognize are silently stripped when this is on.
    cache_control:
        Prompt-cache marker (e.g. ``{"type": "ephemeral"}``). For Anthropic
        models this becomes per-message/tool ``cache_control`` block markers;
        for other providers it is sent as a request param.
    prompt_cache_key / prompt_cache_retention:
        OpenAI-style prompt-cache controls; ignored (with a warning) for
        Anthropic models, which use ``cache_control`` markers instead.

    ``backend`` is pinned by type: a ``LiteLLMConfig`` cannot be constructed
    with (and therefore never routed to) a different backend.

    ``base_url`` (inherited) is normalized on construction: a trailing
    ``/chat/completions`` is stripped.
    """

    backend: Literal["litellm"] = "litellm"
    thinking: Optional[Dict[str, Any]] = None
    api_version: Optional[str] = None
    num_retries: Optional[int] = 4
    drop_params: bool = True
    allowed_openai_params: Optional[List[str]] = None
    cache_control: Optional[Dict[str, Any]] = None
    prompt_cache_key: Optional[str] = None
    prompt_cache_retention: Optional[str] = None

    @field_validator("base_url", mode="after")
    @classmethod
    def _strip_chat_completions_suffix(cls, base_url: Optional[str]) -> Optional[str]:
        """Normalize ``base_url`` to the endpoint base litellm expects.

        litellm appends ``/chat/completions`` itself, so a caller-supplied
        endpoint that already ends in ``/chat/completions`` must
        have that suffix removed.
        """
        if base_url is None:
            return base_url
        cleaned = base_url.rstrip("/").removesuffix("/chat/completions")
        return cleaned or base_url


class LiteLLMBackend(BaseLLMBackend):
    """LLM backend that makes chat-completion calls through litellm.

    Handles the provider-specific details: param routing via ``extra_body``
    for params the provider adapter doesn't recognize, Anthropic prompt-cache
    block markers (system / last message / last tool), the NIM cache-control
    passthrough patch, and ``os.environ/`` api-key resolution.
    """

    config_cls = LiteLLMConfig

    def __init__(self, config: LiteLLMConfig) -> None:
        super().__init__(config)
        self.config: LiteLLMConfig

        try:
            import litellm
        except ImportError as e:
            raise RuntimeError(
                "litellm is not installed but LiteLLMBackend was requested. "
                "Install it (`pip install litellm`) or use a different backend."
            ) from e
        self._litellm = litellm

        # Reduce noisy provider/help banners on handled API errors.
        if hasattr(litellm, "suppress_debug_info"):
            litellm.suppress_debug_info = True
        _patch_litellm_nvidia_nim_cache_passthrough()

        self._cache_control_message_marker: Optional[Dict[str, Any]] = None

        self.completion_kwargs: Dict[str, Any] = dict(
            model=config.model,
            tool_choice=config.tool_choice,
            base_url=config.base_url,
            api_version=config.api_version,
            num_retries=config.num_retries,
            max_completion_tokens=config.max_completion_tokens,
        )
        if config.drop_params:
            self.completion_kwargs["drop_params"] = config.drop_params
        if config.allowed_openai_params:
            self.completion_kwargs["allowed_openai_params"] = config.allowed_openai_params
        if config.temperature is not None:
            self.completion_kwargs["temperature"] = config.temperature
        if config.parallel_tool_calls is not None:
            self.completion_kwargs["parallel_tool_calls"] = config.parallel_tool_calls

        supported_params = None
        get_supported_openai_params = getattr(litellm, "get_supported_openai_params", None)
        if callable(get_supported_openai_params):
            try:
                supported_params = get_supported_openai_params(model=config.model)
            except Exception:
                supported_params = None

        def _param_supported(name: str) -> bool:
            return isinstance(supported_params, list) and name in supported_params

        def _set_param_or_extra_body(name: str, value: Any) -> None:
            if value is None:
                return
            if _param_supported(name):
                self.completion_kwargs[name] = value
                return
            # Some OpenAI-compatible endpoints only accept provider-specific
            # parameters via `extra_body`; keep top-level kwargs clean.
            extra_body = self.completion_kwargs.get("extra_body")
            if not isinstance(extra_body, dict):
                extra_body = {}
            extra_body[name] = value
            self.completion_kwargs["extra_body"] = extra_body
            logger.info("Forwarding `%s` via extra_body for model=%s.", name, config.model)

        model_lower = str(config.model).lower()
        use_anthropic_message_cache_markers = "anthropic" in model_lower

        # `reasoning_effort` must go through the same supported/extra_body
        # router as `thinking` / `cache_control`: several provider adapters
        # (e.g. nvidia_nim) ship static supported-params lists that predate
        # reasoning_effort, and with drop_params=True litellm silently strips
        # unsupported top-level params before the wire. extra_body keys pass
        # through opaquely.
        if isinstance(config.reasoning_effort, str) and config.reasoning_effort.strip():
            _set_param_or_extra_body("reasoning_effort", config.reasoning_effort.strip())

        if isinstance(config.thinking, dict) and len(config.thinking) != 0:
            _set_param_or_extra_body("thinking", dict(config.thinking))
        if isinstance(config.cache_control, dict) and len(config.cache_control) != 0:
            effective_cache_control = dict(config.cache_control)
            if use_anthropic_message_cache_markers:
                # Anthropic prompt caching is exclusively driven by `cache_control`
                # markers on content blocks (system/messages/tools). There is no
                # top-level `cache_control` API field; sending one via extra_body
                # makes Bedrock-proxied routes reject the request.
                self._cache_control_message_marker = dict(effective_cache_control)
                logger.info(
                    "Applying `cache_control` as message-block marker for model=%s "
                    "(no request-level cache_control).",
                    config.model,
                )
            else:
                _set_param_or_extra_body("cache_control", effective_cache_control)
        if isinstance(config.prompt_cache_key, str) and config.prompt_cache_key.strip():
            if use_anthropic_message_cache_markers:
                logger.warning(
                    "Ignoring `prompt_cache_key` for anthropic model=%s; using cache_control markers instead.",
                    config.model,
                )
            else:
                _set_param_or_extra_body("prompt_cache_key", config.prompt_cache_key.strip())
        if isinstance(config.prompt_cache_retention, str) and config.prompt_cache_retention.strip():
            if use_anthropic_message_cache_markers:
                logger.warning(
                    "Ignoring `prompt_cache_retention` for anthropic model=%s; use cache_control ttl instead.",
                    config.model,
                )
            else:
                _set_param_or_extra_body("prompt_cache_retention", config.prompt_cache_retention.strip())

        if config.api_key is not None:
            self.completion_kwargs["api_key"] = config.api_key
        self._resolved_api_key: Optional[str] = None
        if config.api_key is not None and config.api_key.strip().startswith("os.environ/"):
            self._resolved_api_key = os.environ[config.api_key.strip().removeprefix("os.environ/")]

    # ------------------------------------------------------------------
    # Impls.
    # ------------------------------------------------------------------

    def _completion_impl(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **overrides: Any,
    ) -> CompletionResult:
        request_kwargs = self._prepare_request(messages, tools=tools, **overrides)
        try:
            response = self._litellm.completion(**request_kwargs)
        except Exception as e:
            translated = self._translate_provider_error(e)
            if translated is None:
                raise
            raise translated from e
        return self._build_result(response, request_kwargs)

    async def _acompletion_impl(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **overrides: Any,
    ) -> CompletionResult:
        request_kwargs = self._prepare_request(messages, tools=tools, **overrides)
        try:
            response = await self._litellm.acompletion(**request_kwargs)
        except Exception as e:
            translated = self._translate_provider_error(e)
            if translated is None:
                raise
            raise translated from e
        return self._build_result(response, request_kwargs)

    # ------------------------------------------------------------------
    # Request preparation (shared by both impls).
    # ------------------------------------------------------------------

    def _prepare_request(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **overrides: Any,
    ) -> Dict[str, Any]:
        """Build litellm request kwargs. Never mutates ``messages`` / ``tools``.

        Order matters: normalization collapses text-only content-block lists to
        strings (discarding block-level metadata), so it must run BEFORE cache
        markers are attached.
        """
        request_messages = strip_private_message_keys(messages)
        request_messages = normalize_messages_for_api(request_messages)
        request_tools = tools
        if isinstance(self._cache_control_message_marker, dict):
            # Multi-turn caching strategy (Anthropic allows up to 4 breakpoints
            # per request; longest prefix match wins):
            #   1. Mark the SYSTEM message so caching works from the first turn
            #      even if a provider does not forward per-tool `cache_control`.
            #   2. Mark the LAST message so the entire prior history is in the
            #      cache prefix; next turn the breakpoint moves forward and
            #      Anthropic's cascading window picks up the longest matching
            #      prefix from the prior breakpoint.
            #   3. Mark the LAST tools entry so the stable `tools + system`
            #      prefix is always cached.
            request_messages = self._with_cache_control_marker(
                messages=request_messages,
                cache_control=self._cache_control_message_marker,
            )
            request_messages = self._with_last_message_cache_control_marker(
                messages=request_messages,
                cache_control=self._cache_control_message_marker,
            )
            if isinstance(request_tools, list) and len(request_tools) > 0:
                request_tools = self._with_tool_cache_control_marker(
                    tools=request_tools,
                    cache_control=self._cache_control_message_marker,
                )

        # Merge in steps so per-call overrides can intentionally win over
        # config defaults without raising `TypeError` on duplicate keys.
        # completion_kwargs holds shared mutables (extra_body, allowed lists):
        # deep-copy per request so downstream mutation of one request can never
        # corrupt the template for subsequent calls.
        request_kwargs: Dict[str, Any] = {"messages": request_messages}
        request_kwargs.update(deepcopy(self.completion_kwargs))
        request_kwargs.update(overrides)
        if request_tools is not None:
            request_kwargs["tools"] = request_tools
        else:
            # No tools this call: drop tool-only params so providers that reject
            # them without a `tools` array don't 400.
            request_kwargs.pop("tool_choice", None)
            request_kwargs.pop("parallel_tool_calls", None)
        if self._resolved_api_key is not None:
            request_kwargs["api_key"] = self._resolved_api_key
        return request_kwargs

    # ------------------------------------------------------------------
    # Error translation (client-call failures only; see errors module).
    # ------------------------------------------------------------------

    def _translate_provider_error(self, e: Exception) -> Optional[LLMCallError]:
        """Map a client-call exception to a library error, or None to re-raise raw."""
        exceptions = self._litellm.exceptions
        if isinstance(e, exceptions.RateLimitError):
            return RateLimitError(str(e))
        # NOTE: ContextWindowExceededError and ContentPolicyViolationError are
        # BadRequestError subclasses in litellm — check them first.
        if isinstance(e, exceptions.ContextWindowExceededError):
            return ContextLimitError(str(e))
        if isinstance(e, exceptions.ContentPolicyViolationError):
            return ContentPolicyError(str(e))
        if isinstance(e, exceptions.BadRequestError):
            err_str = str(e).lower()
            if "contentpolicyviolationerror" in err_str:
                return ContentPolicyError(str(e))
            if (
                "contextwindowexceedederror" in err_str
                or ("context" in err_str and "window" in err_str)
                or ("context" in err_str and "reduce" in err_str)
                # Prompt so long the completion budget went negative — a
                # context-overflow symptom.
                or ("max_tokens must be at least 1" in err_str and "got -" in err_str)
            ):
                return ContextLimitError(str(e))
            return LLMCallError(str(e))
        # Any other litellm/openai exception type is still an API-call failure
        # (auth, timeout, connection, 5xx, ...). Exceptions from other modules
        # (KeyError, TypeError, ...) are bugs and propagate raw.
        module = (type(e).__module__ or "").split(".")[0]
        if module in ("litellm", "openai"):
            return LLMCallError(str(e))
        return None

    # ------------------------------------------------------------------
    # Result assembly.
    # ------------------------------------------------------------------

    def _build_result(self, response: "ModelResponse", request_kwargs: Dict[str, Any]) -> CompletionResult:
        choices = getattr(response, "choices", None) or []
        if len(choices) != 1:
            raise LLMCallError(f"Expected exactly 1 choice in the API response, got {len(choices)}.")
        choice = choices[0]
        message_obj = choice.message

        message: Dict[str, Any] = {"role": "assistant", "content": extract_text_content(message_obj.content)}
        tool_calls = getattr(message_obj, "tool_calls", None)
        if tool_calls:
            message["tool_calls"] = [
                tc.model_dump() if hasattr(tc, "model_dump") else dict(tc) for tc in tool_calls
            ]

        raw_request: Optional[Dict[str, Any]] = None
        raw_response: Optional[Dict[str, Any]] = None
        if self.config.capture_raw_io:
            try:
                raw_request = _redacted_request(request_kwargs)
                raw_response = response.model_dump()
            except Exception:
                logger.warning("Failed to capture raw LLM IO; continuing without it.", exc_info=True)

        return CompletionResult(
            message=message,
            finish_reason=choice.finish_reason or "unknown",
            reasoning=extract_reasoning_from_message(message_obj),
            usage=coerce_usage_to_dict(getattr(response, "usage", None)),
            extra_response_info=self._build_extra_response_info(response),
            raw_request=raw_request,
            raw_response=raw_response,
        )

    def _build_extra_response_info(self, response: "ModelResponse") -> Dict[str, Any]:
        """Everything the backend knows about the response except the message itself.

        Schema-unstable, log-only (see :class:`CompletionResult`). Best-effort:
        degrades to ``{}`` rather than failing a successful call.
        """
        info: Dict[str, Any] = {}
        try:
            info = response.model_dump(exclude={"choices"})
        except Exception:
            info = {}
        try:
            additional_headers = getattr(response, "_hidden_params", {}).get("additional_headers", {}) or {}
            info["ratelimit"] = {
                "TPM": additional_headers.get("llm_provider-x-ratelimit-remaining-tokens"),
                "RQ": additional_headers.get("llm_provider-x-ratelimit-remaining-requests"),
            }
        except Exception:
            pass
        return info

    # ------------------------------------------------------------------
    # Anthropic cache-control markers.
    # ------------------------------------------------------------------

    @staticmethod
    def _with_cache_control_marker(
        messages: List[Dict[str, Any]], cache_control: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Return a copy of messages with one explicit cache marker block."""
        if not isinstance(messages, list) or len(messages) == 0:
            return messages

        out_messages: List[Dict[str, Any]] = []
        for msg in messages:
            out_messages.append(dict(msg))

        if isinstance(out_messages[0].get("role"), str) and out_messages[0].get("role") == "system":
            target_idx = 0
        else:
            target_idx = len(out_messages) - 1

        target = dict(out_messages[target_idx])
        content = target.get("content")

        cc = dict(cache_control)
        if isinstance(content, str):
            target["content"] = [{"type": "text", "text": content, "cache_control": cc}]
        elif isinstance(content, list):
            replaced = False
            new_parts = []
            for part in content:
                if isinstance(part, dict):
                    p = dict(part)
                    if not replaced and p.get("type") == "text":
                        p["cache_control"] = cc
                        replaced = True
                    new_parts.append(p)
                else:
                    new_parts.append(part)
            if not replaced:
                new_parts.append({"type": "text", "text": "", "cache_control": cc})
            target["content"] = new_parts
        else:
            return out_messages

        out_messages[target_idx] = target
        return out_messages

    @staticmethod
    def _with_last_message_cache_control_marker(
        messages: List[Dict[str, Any]], cache_control: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Attach a ``cache_control`` marker on the LAST eligible message.

        This caches the entire conversation history up to and including the
        most recent assistant/tool turn. Combined with the static system +
        tools breakpoint, multi-turn agents reuse the maximum possible prefix
        on each call.

        Content-shape behaviour
        -----------------------
        - ``content: str`` → wrap into ``[{type: "text", text, cache_control}]``.
        - ``content: [blocks]`` → set ``cache_control`` on the LAST taggable
          block (text / tool_result / tool_use / image / document).
        - ``content: None`` or empty string (typical of assistant tool-call
          turns) → skip and fall back to the previous non-empty message.

        Pre-existing breakpoint dedup
        -----------------------------
        If the first eligible message (scanning from the tail) already carries
        a ``cache_control`` marker on at least one block, this function returns
        the input unchanged: the caller has placed an explicit breakpoint at a
        specific prefix boundary we must not shadow. Two breakpoints in one
        message empirically break cache_read on Bedrock-via-NIM — only the
        byte-identical explicit one ever matches on subsequent requests.
        """
        if not isinstance(messages, list) or len(messages) == 0:
            return messages
        target_idx = _find_last_markable_msg_idx(messages)
        if target_idx is None:
            return messages
        out_messages: List[Dict[str, Any]] = [dict(m) if isinstance(m, dict) else m for m in messages]
        out_messages[target_idx] = _apply_cache_control_to_last_block(
            message=out_messages[target_idx], cache_control=cache_control
        )
        return out_messages

    @staticmethod
    def _with_tool_cache_control_marker(
        tools: List[Dict[str, Any]], cache_control: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Return a copy of ``tools`` with a cache marker on the last tool definition.

        For Anthropic models, ``cache_control`` may be attached either at the
        top of a tool entry or on its nested ``function`` object; we attach it
        to the nested ``function`` since LiteLLM's anthropic adapter also reads
        ``tool["function"]["cache_control"]`` and Bedrock's tool-cache markers
        are passed through similarly.
        """
        if not isinstance(tools, list) or len(tools) == 0:
            return tools
        out_tools: List[Dict[str, Any]] = []
        for t in tools:
            out_tools.append(dict(t) if isinstance(t, dict) else t)
        last_idx = len(out_tools) - 1
        last = out_tools[last_idx]
        if not isinstance(last, dict):
            return out_tools
        last = dict(last)
        func = last.get("function")
        if isinstance(func, dict):
            func = dict(func)
            func["cache_control"] = dict(cache_control)
            last["function"] = func
        else:
            last["cache_control"] = dict(cache_control)
        out_tools[last_idx] = last
        return out_tools
