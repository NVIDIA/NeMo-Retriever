# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``nemo_agent.llm.create_llm_config`` — the kwargs-filtering config factory.

These tests intentionally avoid instantiating a real ``LiteLLMBackend`` with a
valid config (which would import the optional ``litellm`` dependency). They only
exercise config *building* and the wrong-config-type failure path, both of which
run without ``litellm`` installed (the failure path raises before the lazy
``import litellm`` in ``LiteLLMBackend.__init__``).
"""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from nemo_retriever._agentic.nemo_agent.llm import (
    BaseLLMConfig,
    LiteLLMBackend,
    LiteLLMConfig,
    NIMLLMBackend,
    NIMLLMConfig,
    create_llm_config,
    get_available_backends,
)


def _factory_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Rendered warning messages emitted by the factory itself."""
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("create_llm_config:")]


class TestBackendSelection:
    def test_litellm_selected(self):
        config = create_llm_config("litellm", model="m")
        assert isinstance(config, LiteLLMConfig)
        assert config.backend == "litellm"
        assert config.model == "m"

    def test_nim_not_registered(self):
        # nim is a stub backend (NIMLLMBackend raises on construction), so it is
        # deliberately absent from the selection registry: choosing it must fail
        # fast at config-build time rather than crash later at agent-build time.
        assert "nim" not in get_available_backends()
        with pytest.raises(ValueError, match="nim"):
            create_llm_config("nim", model="m")

    def test_unknown_backend_raises_valueerror(self):
        with pytest.raises(ValueError) as exc:
            create_llm_config("bogus", model="m")
        message = str(exc.value)
        assert "bogus" in message
        # Message lists the available backends.
        assert "litellm" in message

    def test_get_available_backends_reflects_registry(self):
        # Public accessor over the registry: the wider library reads this instead
        # of hard-coding a backend list, so new registrations propagate for free.
        backends = get_available_backends()
        assert isinstance(backends, tuple)
        assert set(backends) == {"callable", "litellm"}
        # Sorted, so callers can build stable, deterministic messages/choices.
        assert list(backends) == sorted(backends)
        # Every advertised name is actually selectable by the config factory.
        for name in backends:
            assert create_llm_config(name, model="m").backend == name


class TestDropAndWarn:
    def test_unsupported_field_dropped_with_warning(self, caplog):
        # drop_params is a LiteLLM-only field; callable (base-only) does not have it.
        with caplog.at_level(logging.WARNING):
            config = create_llm_config("callable", model="m", drop_params=True)
        assert not hasattr(config, "drop_params")
        assert any("drop_params" in w for w in _factory_warnings(caplog))

    def test_warns_even_when_dropped_value_is_none(self, caplog):
        # None is a valid value a caller may intend to set; a drop is still a drop.
        with caplog.at_level(logging.WARNING):
            create_llm_config("callable", model="m", cache_control=None)
        assert any("cache_control" in w for w in _factory_warnings(caplog))

    def test_supported_field_kept_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            config = create_llm_config("litellm", model="m", drop_params=True)
        assert config.drop_params is True
        assert _factory_warnings(caplog) == []

    def test_warning_lists_keys_not_values(self, caplog):
        # Values (potential secrets) must never be logged — only field names.
        with caplog.at_level(logging.WARNING):
            create_llm_config("callable", model="m", prompt_cache_key="SENSITIVE-VALUE")
        joined = " ".join(_factory_warnings(caplog))
        assert "prompt_cache_key" in joined
        assert "SENSITIVE-VALUE" not in joined


class TestValidationPreserved:
    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            create_llm_config("litellm")  # no model

    def test_bad_type_for_known_field_raises(self):
        with pytest.raises(ValidationError):
            create_llm_config("litellm", model="m", max_completion_tokens="not-an-int")


class TestReasoningEffortHoisted:
    @pytest.mark.parametrize("backend", ["litellm", "callable"])
    def test_reasoning_effort_supported_on_every_backend(self, backend, caplog):
        with caplog.at_level(logging.WARNING):
            config = create_llm_config(backend, model="m", reasoning_effort="high")
        assert config.reasoning_effort == "high"
        # It is a base field now, so it is never dropped and never warned about.
        assert not any("reasoning_effort" in w for w in _factory_warnings(caplog))

    def test_reasoning_effort_declared_on_base_config(self):
        assert "reasoning_effort" in BaseLLMConfig.model_fields


class TestConfigClsPairing:
    def test_litellm_pairing(self):
        assert LiteLLMBackend.config_cls is LiteLLMConfig

    def test_nim_pairing(self):
        assert NIMLLMBackend.config_cls is NIMLLMConfig

    def test_wrong_config_type_rejected_before_litellm_import(self):
        # Passing a NIMLLMConfig to LiteLLMBackend must raise TypeError from the
        # centralized base check, which runs before the lazy `import litellm`.
        with pytest.raises(TypeError):
            LiteLLMBackend(NIMLLMConfig(model="m"))


class TestLiteLLMCompletionKwargs:
    """``temperature`` / ``parallel_tool_calls`` reach ``completion_kwargs`` only when set.

    Constructs a real ``LiteLLMBackend``, which imports litellm; skipped when
    litellm is not installed.
    """

    def test_forwarded_when_set(self):
        pytest.importorskip("litellm")
        backend = LiteLLMBackend(
            LiteLLMConfig(model="gpt-4o-mini", temperature=0.3, parallel_tool_calls=True)
        )
        assert backend.completion_kwargs["temperature"] == 0.3
        assert backend.completion_kwargs["parallel_tool_calls"] is True

    def test_omitted_when_none(self):
        pytest.importorskip("litellm")
        backend = LiteLLMBackend(LiteLLMConfig(model="gpt-4o-mini"))
        assert "temperature" not in backend.completion_kwargs
        assert "parallel_tool_calls" not in backend.completion_kwargs

    def test_parallel_tool_calls_dropped_when_no_tools(self):
        pytest.importorskip("litellm")
        backend = LiteLLMBackend(LiteLLMConfig(model="gpt-4o-mini", parallel_tool_calls=True))
        messages = [{"role": "user", "content": "hi"}]
        no_tools = backend._prepare_request(messages, tools=None)
        assert "parallel_tool_calls" not in no_tools
        assert "tool_choice" not in no_tools
        with_tools = backend._prepare_request(
            messages,
            tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
        )
        assert with_tools["parallel_tool_calls"] is True


class TestFalsyValuesPreserved:
    """0.0 / False are real settings, not "unset" — the factory must keep them."""

    def test_zero_temperature_and_false_parallel_tool_calls_kept(self):
        config = create_llm_config("litellm", model="m", temperature=0.0, parallel_tool_calls=False)
        assert config.temperature == 0.0
        assert config.parallel_tool_calls is False


class TestLiteLLMBaseUrlNormalization:
    """LiteLLMConfig strips a trailing /chat/completions (litellm appends it itself)."""

    def test_strips_chat_completions_suffix(self):
        c = LiteLLMConfig(model="m", base_url="https://integrate.api.nvidia.com/v1/chat/completions")
        assert c.base_url == "https://integrate.api.nvidia.com/v1"

    def test_leaves_clean_base_untouched(self):
        c = LiteLLMConfig(model="m", base_url="https://integrate.api.nvidia.com/v1")
        assert c.base_url == "https://integrate.api.nvidia.com/v1"

    def test_none_is_preserved(self):
        assert LiteLLMConfig(model="m", base_url=None).base_url is None

    def test_tolerates_trailing_slash(self):
        assert LiteLLMConfig(model="m", base_url="https://x/v1/chat/completions/").base_url == "https://x/v1"

    def test_idempotent(self):
        once = LiteLLMConfig(model="m", base_url="https://x/v1/chat/completions").base_url
        twice = LiteLLMConfig(model="m", base_url=once).base_url
        assert once == twice == "https://x/v1"

    def test_applied_via_create_llm_config(self):
        c = create_llm_config("litellm", model="m", base_url="https://x/v1/chat/completions")
        assert c.base_url == "https://x/v1"

    def test_callable_base_url_not_normalized(self):
        # Normalization is litellm-specific; other backends keep the URL verbatim.
        c = create_llm_config("callable", model="m", base_url="https://x/v1/chat/completions")
        assert c.base_url == "https://x/v1/chat/completions"
