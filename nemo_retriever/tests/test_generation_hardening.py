# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security and strict-contract tests for provisional text generation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import time
from unittest.mock import patch

import pytest


class _Client:
    model = "test/model"

    def __init__(self, handler):
        self._handler = handler
        self.calls = []

    def complete(self, messages, max_tokens=None, extra_params=None):
        self.calls.append((messages, max_tokens, extra_params))
        return self._handler(messages, max_tokens, extra_params)


def _response(
    content="text",
    *,
    tool_calls=None,
    function_call=None,
    refusal=None,
    audio=None,
    finish_reason="stop",
):
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        function_call=function_call,
        refusal=refusal,
        audio=audio,
        images=None,
        videos=None,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


class TestStrictGenerationTask:
    def test_invoke_raises_typed_sanitized_transport_error_and_execute_collects(self):
        from nemo_retriever.models.llm.tasks import (
            GenerationTaskError,
            SummarizeTask,
        )

        secret = "sk-TRANSPORT-MUST-NOT-LEAK"

        class RetryableFailure(RuntimeError):
            retryable = True

        def fail(messages, max_tokens, extra_params):
            time.sleep(0.001)
            raise RetryableFailure(secret)

        client = _Client(fail)
        task = SummarizeTask()

        with pytest.raises(GenerationTaskError) as exc_info:
            task.invoke(client, text="source")
        error = exc_info.value
        assert error.code == "transport_error"
        assert error.phase == "transport"
        assert error.retryable is True
        assert error.latency_s > 0.0
        assert secret not in str(error)
        assert error.__cause__ is None
        assert error.__context__ is None

        collected = task.execute(client, text="source")
        assert collected.text == ""
        assert collected.error == "transport_error"
        assert collected.latency_s > 0.0
        assert secret not in repr(collected)

    def test_request_build_failures_are_request_errors(self):
        from nemo_retriever.models.llm.tasks import GenerationTask, GenerationTaskError

        class BrokenTask(GenerationTask):
            def build_request(self, **inputs):
                raise RuntimeError("sk-REQUEST-MUST-NOT-LEAK")

        class WrongTypeTask(GenerationTask):
            def build_request(self, **inputs):
                return {"messages": []}

        client = _Client(lambda *args: ("unused", 0.0))
        for task in (BrokenTask(), WrongTypeTask()):
            with pytest.raises(GenerationTaskError) as exc_info:
                task.invoke(client)
            assert exc_info.value.code == "request_error"
            assert exc_info.value.phase == "request"
            assert "sk-REQUEST-MUST-NOT-LEAK" not in str(exc_info.value)
        assert client.calls == []

    @pytest.mark.parametrize("parsed", [{"structured": True}, ["not", "text"], 7])
    def test_non_text_parse_results_are_parse_errors(self, parsed):
        from nemo_retriever.models.llm.tasks import (
            GenerationTask,
            GenerationTaskError,
        )
        from nemo_retriever.models.llm.types import GenerationRequest

        class NonTextTask(GenerationTask):
            def build_request(self, **inputs):
                return GenerationRequest(messages=[{"role": "user", "content": "hello"}])

            def parse(self, raw_text):
                return parsed

        with pytest.raises(GenerationTaskError) as exc_info:
            NonTextTask().invoke(_Client(lambda *args: ("raw", 0.1)))
        assert exc_info.value.code == "parse_error"
        assert exc_info.value.phase == "parse"

    def test_parser_exception_is_sanitized(self):
        from nemo_retriever.models.llm.tasks import (
            GenerationTask,
            GenerationTaskError,
        )
        from nemo_retriever.models.llm.types import GenerationRequest

        class BrokenParser(GenerationTask):
            def build_request(self, **inputs):
                return GenerationRequest(messages=[{"role": "user", "content": "hello"}])

            def parse(self, raw_text):
                raise ValueError("sk-PARSE-MUST-NOT-LEAK")

        with pytest.raises(GenerationTaskError) as exc_info:
            BrokenParser().invoke(_Client(lambda *args: ("raw", 0.1)))
        assert exc_info.value.code == "parse_error"
        assert exc_info.value.phase == "parse"
        assert "sk-PARSE-MUST-NOT-LEAK" not in str(exc_info.value)


class TestImmutableTextContracts:
    def test_request_snapshots_inputs_and_rejects_non_text_messages(self):
        from nemo_retriever.models.llm.types import GenerationRequest

        messages = [{"role": "user", "content": "original"}]
        extras = {"provider": {"seed": 1}}
        request = GenerationRequest(messages=messages, extra_params=extras)

        messages[0]["content"] = "mutated"
        extras["provider"]["seed"] = 2
        assert request.messages == [{"role": "user", "content": "original"}]
        assert request.extra_params == {"provider": {"seed": 1}}

        with pytest.raises(FrozenInstanceError):
            request.max_tokens = 10
        with pytest.raises(TypeError, match="string role and content"):
            GenerationRequest(messages=[{"role": "user", "content": [{"type": "image"}]}])
        with pytest.raises(ValueError, match="tool"):
            GenerationRequest(
                messages=[
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "call-1"}],
                    }
                ]
            )

    def test_builtin_tasks_and_results_are_frozen(self):
        from nemo_retriever.models.llm.tasks import (
            GenericPromptTask,
            RagAnswerTask,
            SummarizeTask,
        )
        from nemo_retriever.models.llm.types import GeneratedTextResult

        values = [
            GenericPromptTask(prompt="{value}", input_names=("value",)),
            RagAnswerTask(),
            SummarizeTask(),
            GeneratedTextResult(text="ok", latency_s=0.1, model="m"),
        ]
        for value in values:
            with pytest.raises(FrozenInstanceError):
                value.unexpected = True


class TestLiteLLMTextOnlyResponse:
    @pytest.mark.parametrize(
        "response",
        [
            _response(content=None),
            _response(content=[{"type": "text", "text": "hello"}]),
            _response(content="ignored", tool_calls=[{"id": "call-1"}]),
            _response(content="ignored", function_call={"name": "legacy"}),
            _response(content="ignored", refusal="policy refusal"),
            _response(content="ignored", audio={"id": "audio-1"}),
            _response(content="ignored", finish_reason="tool_calls"),
            SimpleNamespace(choices=[]),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="one")),
                    SimpleNamespace(message=SimpleNamespace(content="two")),
                ]
            ),
        ],
    )
    @patch("litellm.completion")
    def test_unsupported_provider_shapes_collect_stable_code(
        self,
        mock_completion,
        response,
    ):
        from nemo_retriever.models.llm.clients import LiteLLMClient
        from nemo_retriever.models.llm.tasks import SummarizeTask

        mock_completion.return_value = response
        result = SummarizeTask().execute(
            LiteLLMClient.from_kwargs(model="openai/mock"),
            text="source",
        )

        assert result.text == ""
        assert result.error == "unsupported_response"
        assert result.latency_s >= 0.0

    @patch("litellm.completion")
    def test_plain_text_and_empty_text_remain_distinct(self, mock_completion):
        from nemo_retriever.models.llm.clients import LiteLLMClient
        from nemo_retriever.models.llm.tasks import SummarizeTask

        client = LiteLLMClient.from_kwargs(model="openai/mock")
        task = SummarizeTask()

        mock_completion.return_value = _response(content="  plain text  ")
        success = task.execute(client, text="source")
        assert success.text == "plain text"
        assert success.error is None

        mock_completion.return_value = _response(content="   ")
        empty = task.execute(client, text="source")
        assert empty.text == ""
        assert empty.error == "empty_output"


def _reconstruct_summary_operator(constructor_kwargs):
    """Importable Ray target proving constructor-only worker reconstruction."""
    from nemo_retriever.operators.generation import SummarizationOperator

    operator = SummarizationOperator(**constructor_kwargs)
    return {
        "operator": type(operator).__name__,
        "task": type(operator._task).__name__,
        "client": type(operator._client).__name__,
        "model": operator._params.transport.model,
        "constructor_keys": sorted(operator.get_constructor_kwargs()),
    }


class TestWorkerReconstruction:
    def test_real_constructor_reconstructs_in_process_and_on_ray_when_available(self):
        import importlib.util

        from nemo_retriever.common.params import TextGenerationParams
        from nemo_retriever.operators.generation import SummarizationOperator

        params = TextGenerationParams.from_kwargs(
            model="openai/mock",
            api_key="",
            prompt="Summarize: {text}",
        )
        source = SummarizationOperator(params)
        constructor_kwargs = source.get_constructor_kwargs()
        expected = {
            "operator": "SummarizationOperator",
            "task": "SummarizeTask",
            "client": "LiteLLMClient",
            "model": "openai/mock",
            "constructor_keys": sorted(constructor_kwargs),
        }

        assert _reconstruct_summary_operator(constructor_kwargs) == expected
        if importlib.util.find_spec("ray") is None:
            return

        import ray

        def reconstruct_on_worker(values):
            from nemo_retriever.operators.generation import SummarizationOperator

            operator = SummarizationOperator(**values)
            return {
                "operator": type(operator).__name__,
                "task": type(operator._task).__name__,
                "client": type(operator._client).__name__,
                "model": operator._params.transport.model,
                "constructor_keys": sorted(operator.get_constructor_kwargs()),
            }

        owned_runtime = not ray.is_initialized()
        if owned_runtime:
            ray.init(
                address="local",
                num_cpus=1,
                include_dashboard=False,
                log_to_driver=False,
            )
        try:
            remote_reconstruct = ray.remote(reconstruct_on_worker)
            assert (
                ray.get(
                    remote_reconstruct.remote(constructor_kwargs),
                    timeout=60,
                )
                == expected
            )
        finally:
            if owned_runtime:
                ray.shutdown()
