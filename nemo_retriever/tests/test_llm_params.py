# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified LLM params layer and client / judge composition.

Covers:
    * LLMRemoteClientParams validation and api_key auto-resolution
    * LiteLLMClient(transport, sampling) and .from_kwargs(...) parity
    * top_p omission from litellm call kwargs when unset
    * LLMJudge default sampling and .from_kwargs(...) back-compat
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _fake_litellm_response(text: str = "ok") -> SimpleNamespace:
    """Mimic the litellm.completion() response shape used by LiteLLMClient."""
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


class TestLLMRemoteClientParams:
    """Validate LLMRemoteClientParams validators, defaults, and api_key auto-resolution."""

    def test_defaults(self):
        from nemo_retriever.params.models import LLMRemoteClientParams

        p = LLMRemoteClientParams(model="nvidia_nim/meta/llama-3.1-70b-instruct")
        assert p.model == "nvidia_nim/meta/llama-3.1-70b-instruct"
        assert p.api_base is None
        assert p.num_retries == 3
        assert p.timeout == 120.0
        assert p.extra_params == {}

    def test_model_is_required(self):
        from nemo_retriever.params.models import LLMRemoteClientParams

        with pytest.raises(ValueError):
            LLMRemoteClientParams()  # type: ignore[call-arg]

    def test_negative_num_retries_rejected(self):
        from nemo_retriever.params.models import LLMRemoteClientParams

        with pytest.raises(ValueError, match="num_retries must be >= 0"):
            LLMRemoteClientParams(model="m", num_retries=-1)

    def test_zero_timeout_rejected(self):
        from nemo_retriever.params.models import LLMRemoteClientParams

        with pytest.raises(ValueError, match="timeout must be > 0"):
            LLMRemoteClientParams(model="m", timeout=0.0)

    def test_negative_timeout_rejected(self):
        from nemo_retriever.params.models import LLMRemoteClientParams

        with pytest.raises(ValueError, match="timeout must be > 0"):
            LLMRemoteClientParams(model="m", timeout=-1.0)

    def test_extra_forbid(self):
        """Unknown kwargs should be rejected by _ParamsModel(extra='forbid')."""
        from nemo_retriever.params.models import LLMRemoteClientParams

        with pytest.raises(ValueError):
            LLMRemoteClientParams(model="m", unknown_field=123)  # type: ignore[call-arg]

    def test_api_key_auto_resolved_from_env(self, monkeypatch):
        """api_key=None should resolve from the remote-auth helper."""
        from nemo_retriever.params import models as params_models

        monkeypatch.setattr(params_models, "resolve_remote_api_key", lambda: "resolved-secret")
        p = params_models.LLMRemoteClientParams(model="m")
        assert p.api_key == "resolved-secret"

    def test_api_key_no_api_key_sentinel_yields_none(self):
        """Explicit NO_API_KEY sentinel suppresses auto-resolution."""
        from nemo_retriever.params.models import NO_API_KEY, LLMRemoteClientParams

        p = LLMRemoteClientParams(model="m", api_key=NO_API_KEY)
        assert p.api_key is None


class TestLiteLLMClientConstruction:
    """LiteLLMClient should accept structured params and expose .model for back-compat."""

    def test_structured_construction(self):
        from nemo_retriever.llm.clients import LiteLLMClient
        from nemo_retriever.params.models import LLMInferenceParams, LLMRemoteClientParams

        transport = LLMRemoteClientParams(model="openai/gpt-4o-mini", api_key="k")
        sampling = LLMInferenceParams(temperature=0.2, top_p=0.9, max_tokens=512)
        client = LiteLLMClient(transport=transport, sampling=sampling)

        assert client.transport is transport
        assert client.sampling is sampling
        assert client.model == "openai/gpt-4o-mini"

    def test_default_sampling_matches_from_kwargs_for_rag_determinism(self):
        """``LiteLLMClient`` is a RAG-eval client and must default to
        deterministic sampling regardless of which constructor path the
        caller picks.  The structured constructor therefore overrides
        ``LLMInferenceParams``'s general-purpose ``temperature=1.0`` with
        ``0.0`` so it agrees with :meth:`LiteLLMClient.from_kwargs`.
        ``top_p`` / ``max_tokens`` still come from ``LLMInferenceParams``
        (they already match what ``from_kwargs`` builds).
        """
        from nemo_retriever.llm.clients import LiteLLMClient
        from nemo_retriever.params.models import LLMInferenceParams, LLMRemoteClientParams

        client = LiteLLMClient(transport=LLMRemoteClientParams(model="m"))
        assert isinstance(client.sampling, LLMInferenceParams)
        assert client.sampling.temperature == 0.0
        assert client.sampling.top_p is None
        assert client.sampling.max_tokens == 1024

    def test_from_kwargs_matches_explicit(self):
        from nemo_retriever.llm.clients import LiteLLMClient

        flat = LiteLLMClient.from_kwargs(
            model="openai/gpt-4o-mini",
            api_key="k",
            temperature=0.3,
            top_p=0.8,
            max_tokens=256,
            num_retries=5,
            timeout=30.0,
            extra_params={"user": "tester"},
        )
        assert flat.transport.model == "openai/gpt-4o-mini"
        assert flat.transport.api_key == "k"
        assert flat.transport.num_retries == 5
        assert flat.transport.timeout == 30.0
        assert flat.transport.extra_params == {"user": "tester"}
        assert flat.sampling.temperature == 0.3
        assert flat.sampling.top_p == 0.8
        assert flat.sampling.max_tokens == 256

    def test_from_kwargs_defaults_top_p_to_none(self):
        """The old flat default of top_p=1.0 is now top_p=None (behavior fix)."""
        from nemo_retriever.llm.clients import LiteLLMClient

        client = LiteLLMClient.from_kwargs(model="m")
        assert client.sampling.top_p is None


class TestLiteLLMCompleteCallKwargs:
    """Inspect the exact kwargs LiteLLMClient.complete() forwards to litellm."""

    @patch("litellm.completion")
    def test_top_p_omitted_when_none(self, mock_completion):
        from nemo_retriever.llm.clients import LiteLLMClient

        mock_completion.return_value = _fake_litellm_response("hi")
        client = LiteLLMClient.from_kwargs(model="openai/gpt-4o-mini", temperature=0.5)
        client.complete([{"role": "user", "content": "hi"}])

        kwargs = mock_completion.call_args.kwargs
        assert kwargs["model"] == "openai/gpt-4o-mini"
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 4096
        assert "top_p" not in kwargs
        assert kwargs["num_retries"] == 3
        assert kwargs["timeout"] == 120.0

    @patch("litellm.completion")
    def test_top_p_forwarded_when_set(self, mock_completion):
        from nemo_retriever.llm.clients import LiteLLMClient

        mock_completion.return_value = _fake_litellm_response("hi")
        client = LiteLLMClient.from_kwargs(model="m", top_p=0.9)
        client.complete([{"role": "user", "content": "hi"}])

        kwargs = mock_completion.call_args.kwargs
        assert kwargs["top_p"] == 0.9

    @patch("litellm.completion")
    def test_max_tokens_override(self, mock_completion):
        from nemo_retriever.llm.clients import LiteLLMClient

        mock_completion.return_value = _fake_litellm_response("hi")
        client = LiteLLMClient.from_kwargs(model="m", max_tokens=4096)
        client.complete([{"role": "user", "content": "hi"}], max_tokens=128)

        kwargs = mock_completion.call_args.kwargs
        assert kwargs["max_tokens"] == 128

    @patch("litellm.completion")
    def test_api_key_and_api_base_forwarded(self, mock_completion):
        from nemo_retriever.llm.clients import LiteLLMClient

        mock_completion.return_value = _fake_litellm_response("hi")
        client = LiteLLMClient.from_kwargs(
            model="openai/gpt-4o-mini",
            api_base="http://local-vllm:8000/v1",
            api_key="secret",
        )
        client.complete([{"role": "user", "content": "hi"}])

        kwargs = mock_completion.call_args.kwargs
        assert kwargs["api_base"] == "http://local-vllm:8000/v1"
        assert kwargs["api_key"] == "secret"

    @patch("litellm.completion")
    def test_extra_params_merged_last(self, mock_completion):
        """extra_params should win over keys it overlaps with."""
        from nemo_retriever.llm.clients import LiteLLMClient

        mock_completion.return_value = _fake_litellm_response("hi")
        client = LiteLLMClient.from_kwargs(
            model="m",
            extra_params={"user": "tester", "num_retries": 99},
        )
        client.complete([{"role": "user", "content": "hi"}])

        kwargs = mock_completion.call_args.kwargs
        assert kwargs["user"] == "tester"
        assert kwargs["num_retries"] == 99


class TestLLMJudgeConstruction:
    """LLMJudge should default to deterministic sampling and expose .model."""

    def test_structured_construction_uses_defaults(self):
        from nemo_retriever.llm.clients import LLMJudge
        from nemo_retriever.params.models import LLMRemoteClientParams

        transport = LLMRemoteClientParams(model="nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1")
        judge = LLMJudge(transport=transport)
        assert judge.model == "nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1"
        assert judge._client.sampling.temperature == 0.0
        assert judge._client.sampling.max_tokens == 256

    def test_custom_sampling_override(self):
        from nemo_retriever.llm.clients import LLMJudge
        from nemo_retriever.params.models import LLMInferenceParams, LLMRemoteClientParams

        transport = LLMRemoteClientParams(model="m")
        sampling = LLMInferenceParams(temperature=0.4, max_tokens=1024)
        judge = LLMJudge(transport=transport, sampling=sampling)
        assert judge._client.sampling.temperature == 0.4
        assert judge._client.sampling.max_tokens == 1024

    def test_from_kwargs_matches_structured(self):
        from nemo_retriever.llm.clients import LLMJudge

        judge = LLMJudge.from_kwargs(
            model="m",
            api_key="k",
            num_retries=2,
            timeout=60.0,
            extra_params={"user": "t"},
        )
        assert judge._client.transport.model == "m"
        assert judge._client.transport.api_key == "k"
        assert judge._client.transport.num_retries == 2
        assert judge._client.transport.timeout == 60.0
        assert judge._client.transport.extra_params == {"user": "t"}
        # Sampling stays at judge defaults even when using flat constructor.
        assert judge._client.sampling.temperature == 0.0
        assert judge._client.sampling.max_tokens == 256

    def test_from_kwargs_uses_default_model(self):
        from nemo_retriever.llm.clients import LLMJudge

        judge = LLMJudge.from_kwargs()
        assert judge.model == LLMJudge._DEFAULT_MODEL

    @patch("litellm.completion")
    def test_judge_returns_parsed_result(self, mock_completion):
        from nemo_retriever.llm.clients import LLMJudge

        mock_completion.return_value = _fake_litellm_response(
            '{"score": 4, "reasoning": "mostly correct"}',
        )
        judge = LLMJudge.from_kwargs(model="m")
        verdict = judge.judge(query="q", reference="ref", candidate="cand")
        assert verdict.score == 4
        assert verdict.reasoning == "mostly correct"
        assert verdict.error is None

    def test_judge_empty_candidate_short_circuits(self):
        """Empty candidate is handled locally with no LLM call."""
        from nemo_retriever.llm.clients import LLMJudge

        with patch("litellm.completion") as mock_completion:
            judge = LLMJudge.from_kwargs(model="m")
            verdict = judge.judge(query="q", reference="r", candidate="   ")
            mock_completion.assert_not_called()

        assert verdict.score is None
        assert verdict.error == "empty_candidate"


class TestBackCompatCallSites:
    """The four migrated call sites all use .from_kwargs, so they must still work."""

    @patch("litellm.completion")
    def test_qa_generation_operator_constructs_cleanly(self, mock_completion):
        from nemo_retriever.evaluation.generation import QAGenerationOperator

        mock_completion.return_value = _fake_litellm_response("answer")
        op = QAGenerationOperator(model="m", temperature=0.0, max_tokens=128)
        assert op._client.transport.model == "m"
        assert op._client.sampling.temperature == 0.0
        assert op._client.sampling.max_tokens == 128

    def test_judging_operator_constructs_cleanly(self):
        from nemo_retriever.evaluation.judging import JudgingOperator

        op = JudgingOperator(model="nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1")
        assert op._judge.model == "nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1"
        assert op._judge._client.sampling.temperature == 0.0


class TestApiKeyRedaction:
    """Guard the repr/str of every transport params object against key leakage.

    The ``_ParamsModel`` base redacts ``api_key`` + ``*_api_key`` fields in
    ``__repr__`` / ``__str__`` so that logging a transport object (or
    letting Pydantic's default error formatter echo one back) never
    prints a bearer token.  Consumers still read the plain ``str`` via
    attribute access, so no downstream litellm/NIM call is affected.
    """

    def test_api_key_masked_in_repr(self):
        from nemo_retriever.params.models import LLMRemoteClientParams

        p = LLMRemoteClientParams(model="m", api_key="nvapi-SECRET-TOKEN")
        rendered = repr(p)
        assert "nvapi-SECRET-TOKEN" not in rendered
        assert "api_key=***" in rendered

    def test_api_key_masked_in_str(self):
        from nemo_retriever.params.models import LLMRemoteClientParams

        p = LLMRemoteClientParams(model="m", api_key="nvapi-SECRET-TOKEN")
        assert "nvapi-SECRET-TOKEN" not in str(p)

    def test_api_key_attribute_is_plain_str(self):
        """Redaction is display-only -- attribute access still yields the raw string."""
        from nemo_retriever.params.models import LLMRemoteClientParams

        p = LLMRemoteClientParams(model="m", api_key="nvapi-SECRET-TOKEN")
        assert p.api_key == "nvapi-SECRET-TOKEN"
        assert isinstance(p.api_key, str)

    def test_empty_api_key_not_masked(self):
        """Redaction only fires when a key is actually present."""
        from nemo_retriever.params.models import NO_API_KEY, LLMRemoteClientParams

        p = LLMRemoteClientParams(model="m", api_key=NO_API_KEY)
        assert p.api_key is None
        assert "api_key=***" not in repr(p)
        assert "api_key=None" in repr(p)

    @patch("litellm.completion")
    def test_plain_str_reaches_litellm_call_site(self, mock_completion):
        """The redacted __repr__ must not break the wire-format contract."""
        from nemo_retriever.llm.clients import LiteLLMClient

        mock_completion.return_value = _fake_litellm_response("ok")
        client = LiteLLMClient.from_kwargs(model="m", api_key="nvapi-SECRET-TOKEN")
        client.generate(query="q", chunks=[])

        _, call_kwargs = mock_completion.call_args
        assert call_kwargs["api_key"] == "nvapi-SECRET-TOKEN"
        assert isinstance(call_kwargs["api_key"], str)

    def test_nested_api_key_fields_also_masked(self):
        """Fields matching *_api_key (not only bare api_key) get redacted."""
        from nemo_retriever.params.models import ExtractParams

        p = ExtractParams(
            page_elements_api_key="nvapi-PAGE-ELEM-TOKEN",
            ocr_api_key="nvapi-OCR-TOKEN",
        )
        rendered = repr(p)
        assert "nvapi-PAGE-ELEM-TOKEN" not in rendered
        assert "nvapi-OCR-TOKEN" not in rendered
        assert "page_elements_api_key=***" in rendered
        assert "ocr_api_key=***" in rendered


class TestLiteLLMDefaultModel:
    """Mirror of LLMJudge._DEFAULT_MODEL coverage for LiteLLMClient."""

    def test_from_kwargs_uses_default_model(self):
        from nemo_retriever.llm.clients import LiteLLMClient

        client = LiteLLMClient.from_kwargs()
        assert client.model == LiteLLMClient._DEFAULT_MODEL

    def test_default_model_is_a_non_empty_string(self):
        from nemo_retriever.llm.clients import LiteLLMClient

        assert isinstance(LiteLLMClient._DEFAULT_MODEL, str)
        assert LiteLLMClient._DEFAULT_MODEL


class TestLiteLLMDefaultSamplingAlignment:
    """Both constructor paths must default to the same deterministic sampling.

    Regression test for the Greptile P1 finding that
    ``LiteLLMClient(transport=...)`` with ``sampling=None`` silently
    fell through to ``LLMInferenceParams()`` (``temperature=1.0``) while
    ``LiteLLMClient.from_kwargs(...)`` explicitly defaulted to
    ``temperature=0.0``.  For RAG-eval reproducibility the two paths
    must converge on the same default.
    """

    def test_structured_constructor_defaults_to_zero_temperature(self):
        from nemo_retriever.llm.clients import LiteLLMClient
        from nemo_retriever.params import LLMRemoteClientParams

        client = LiteLLMClient(transport=LLMRemoteClientParams(model="m"))
        assert client.sampling.temperature == 0.0

    def test_structured_and_flat_paths_agree_on_defaults(self):
        from nemo_retriever.llm.clients import LiteLLMClient
        from nemo_retriever.params import LLMRemoteClientParams

        structured = LiteLLMClient(transport=LLMRemoteClientParams(model="m"))
        flat = LiteLLMClient.from_kwargs(model="m")
        assert structured.sampling.temperature == flat.sampling.temperature
        assert structured.sampling.max_tokens == flat.sampling.max_tokens
        assert structured.sampling.top_p == flat.sampling.top_p

    def test_explicit_sampling_is_not_overridden(self):
        """Passing an explicit ``LLMInferenceParams`` must win over the default."""
        from nemo_retriever.llm.clients import LiteLLMClient
        from nemo_retriever.params import LLMInferenceParams, LLMRemoteClientParams

        client = LiteLLMClient(
            transport=LLMRemoteClientParams(model="m"),
            sampling=LLMInferenceParams(temperature=0.7),
        )
        assert client.sampling.temperature == 0.7
