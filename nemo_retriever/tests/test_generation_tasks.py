# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused offline tests for the reusable text-generation task layer."""

from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

import pandas as pd
import pytest


class FakeCompletionClient:
    """Small thread-safe completion client used by task and operator tests."""

    def __init__(
        self,
        handler: Callable[[list[dict], int | None, dict[str, Any] | None], tuple[str, float]] | None = None,
        *,
        model: str = "fake/model",
    ) -> None:
        self._model = model
        self._handler = handler or (lambda messages, max_tokens, extra_params: ("generated", 0.25))
        self._lock = threading.Lock()
        self.calls: list[tuple[list[dict], int | None, dict[str, Any] | None]] = []

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> tuple[str, float]:
        with self._lock:
            self.calls.append((messages, max_tokens, extra_params))
        return self._handler(messages, max_tokens, extra_params)


def _params(**kwargs: Any):
    from nemo_retriever.common.params import TextGenerationParams

    kwargs.setdefault("api_key", "")
    return TextGenerationParams.from_kwargs(model="fake/model", **kwargs)


class TestTextGenerationParams:
    def test_flat_constructor_composes_transport_and_partial_sampling(self):
        params = _params(
            api_base="http://llm.test/v1",
            api_key="secret",
            temperature=0.4,
            top_p=0.8,
            max_tokens=77,
            num_retries=5,
            timeout=12.0,
            extra_params={"seed": 3},
            prompt="Prompt {text}",
            system_prompt="System",
            reasoning_enabled=False,
            max_workers=3,
        )

        assert params.transport.model == "fake/model"
        assert params.transport.api_base == "http://llm.test/v1"
        assert params.transport.api_key == "secret"
        assert params.transport.num_retries == 5
        assert params.transport.timeout == 12.0
        assert params.transport.extra_params == {"seed": 3}
        assert params.sampling.temperature == 0.4
        assert params.sampling.top_p == 0.8
        assert params.sampling.max_tokens == 77
        assert params.prompt == "Prompt {text}"
        assert params.system_prompt == "System"
        assert params.reasoning_enabled is False
        assert params.max_workers == 3

    def test_sampling_overrides_only_explicit_fields(self):
        from nemo_retriever.common.params import LLMInferenceParams

        task_defaults = LLMInferenceParams(temperature=0.0, top_p=0.35, max_tokens=4096)

        inherited = _params().resolve_sampling(task_defaults)
        overridden = _params(temperature=0.7, max_tokens=123).resolve_sampling(task_defaults)

        assert inherited == task_defaults
        assert overridden.temperature == 0.7
        assert overridden.top_p == 0.35
        assert overridden.max_tokens == 123

    def test_generic_api_key_resolution_is_deferred_and_explicit_keys_are_redacted(self, monkeypatch):
        from nemo_retriever.common.params import models as params_models

        monkeypatch.setattr(params_models, "resolve_remote_api_key", lambda: "resolved-secret")
        params = params_models.TextGenerationParams.from_kwargs(model="m")
        explicit = params_models.TextGenerationParams.from_kwargs(model="m", api_key="explicit-secret")

        assert params.transport.api_key is None
        assert "explicit-secret" not in repr(explicit)
        assert "explicit-secret" not in str(explicit)

    def test_no_api_key_survives_nested_validation(self, monkeypatch):
        from nemo_retriever.common.params import models as params_models

        monkeypatch.setattr(
            params_models,
            "resolve_remote_api_key",
            lambda: "environment-secret",
        )
        params = params_models.TextGenerationParams.from_kwargs(model="m", api_key="")

        assert params.transport.api_key is None

    def test_max_workers_must_be_positive(self):
        with pytest.raises(ValueError, match="max_workers"):
            _params(max_workers=0)


class TestGenerationTasks:
    def test_summary_builds_request_and_executes(self):
        from nemo_retriever.models.llm.tasks import SummarizeTask

        client = FakeCompletionClient(lambda messages, max_tokens, extra_params: ("  concise summary  ", 0.4))
        task = SummarizeTask()

        request = task.build_request(text="Source text")
        result = task.execute(client, text="Source text")

        assert [message["role"] for message in request.messages] == ["system", "user"]
        assert request.messages[-1]["content"] == "Summarize the following text:\n\nSource text"
        assert result.text == "concise summary"
        assert result.latency_s == 0.4
        assert result.model == "fake/model"
        assert result.error is None
        assert client.calls == [(request.messages, request.max_tokens, request.extra_params)]

    def test_summary_empty_input_short_circuits(self):
        from nemo_retriever.models.llm.tasks import SummarizeTask

        client = FakeCompletionClient()
        result = SummarizeTask().execute(client, text="  \n")

        assert result.text == ""
        assert result.error == "empty_input"
        assert result.model == "fake/model"
        assert client.calls == []

    def test_task_converts_transport_exception_to_result(self):
        from nemo_retriever.models.llm.tasks import SummarizeTask

        def fail(messages, max_tokens, extra_params):
            raise RuntimeError("service unavailable")

        result = SummarizeTask().execute(FakeCompletionClient(fail), text="source")

        assert result.text == ""
        assert result.latency_s > 0.0
        assert result.model == "fake/model"
        assert result.error == "transport_error"

    def test_rag_request_applies_no_reasoning_controls_and_think_cleanup(self):
        from nemo_retriever.models.llm.tasks import RagAnswerTask

        client = FakeCompletionClient(
            lambda messages, max_tokens, extra_params: (
                "<think>private</think> final answer ",
                0.2,
            )
        )
        task = RagAnswerTask(reasoning_enabled=False)

        request = task.build_request(query="What?", chunks=["First", "Second"])
        result = task.execute(client, query="What?", chunks=["First", "Second"])

        assert request.messages[0]["content"].startswith("/no_think\n")
        assert "First\n\n---\n\nSecond" in request.messages[-1]["content"]
        assert request.extra_params == {"chat_template_kwargs": {"enable_thinking": False}}
        assert result.text == "final answer"
        assert result.error is None

    def test_rag_think_only_output_uses_compatibility_error(self):
        from nemo_retriever.models.llm.tasks import RagAnswerTask

        client = FakeCompletionClient(
            lambda messages, max_tokens, extra_params: (
                "<think>unfinished</think>",
                0.3,
            )
        )
        result = RagAnswerTask().execute(client, query="q", chunks=[])

        assert result.text == ""
        assert result.latency_s == 0.3
        assert result.error == "thinking_truncated"

    def test_generic_template_supports_declared_fields_and_escaped_braces(self):
        from nemo_retriever.models.llm.tasks import GenericPromptTask

        task = GenericPromptTask(
            prompt="Keep {{literal}} and greet {name}",
            input_names=("name",),
            system_prompt="Be brief.",
        )
        request = task.build_request(name="Ada")

        assert request.messages == [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Keep {literal} and greet Ada"},
        ]

    @pytest.mark.parametrize(
        ("prompt", "input_names"),
        [
            ("constant text", ("name",)),
            ("hello {name}", ("name", "unused")),
            ("hello {other}", ("name",)),
            ("hello {user.name}", ("user",)),
            ("hello {items[0]}", ("items",)),
            ("hello {name", ("name",)),
        ],
    )
    def test_generic_template_rejects_invalid_field_contracts(self, prompt, input_names):
        from nemo_retriever.models.llm.tasks import GenericPromptTask

        with pytest.raises(ValueError):
            GenericPromptTask(prompt=prompt, input_names=input_names)

    def test_generic_missing_runtime_input_is_an_error_result(self):
        from nemo_retriever.models.llm.tasks import GenericPromptTask

        client = FakeCompletionClient()
        result = GenericPromptTask(prompt="hello {name}", input_names=("name",)).execute(client)

        assert result.text == ""
        assert result.error
        assert client.calls == []


class TestTextGenerationOperators:
    def test_base_is_exported_from_canonical_operator_package(self):
        from nemo_retriever.operators import TextGenerationOperator
        from nemo_retriever.operators.generation import (
            TextGenerationOperator as DirectExport,
        )

        assert TextGenerationOperator is DirectExport

    def test_summary_happy_path_uses_namespaced_metadata(self):
        from nemo_retriever.operators.generation import SummarizationOperator

        client = FakeCompletionClient(lambda messages, max_tokens, extra_params: ("short", 0.1))
        source = pd.DataFrame({"text": ["long source"]})

        out = SummarizationOperator(_params(), client=client).run(source)

        assert out.to_dict(orient="records") == [
            {
                "text": "long source",
                "summary": "short",
                "summary_latency_s": 0.1,
                "summary_model": "fake/model",
                "summary_error": None,
            }
        ]
        assert list(source.columns) == ["text"]

    def test_missing_and_colliding_columns_are_batch_errors(self):
        from nemo_retriever.operators.generation import SummarizationOperator

        op = SummarizationOperator(_params(), client=FakeCompletionClient())

        with pytest.raises(ValueError, match="missing columns"):
            op.run(pd.DataFrame({"body": ["source"]}))
        with pytest.raises(ValueError, match="already exist"):
            op.run(pd.DataFrame({"text": ["source"], "summary": ["old"]}))

    def test_overwrite_allows_existing_output_column(self):
        from nemo_retriever.operators.generation import SummarizationOperator

        client = FakeCompletionClient(lambda messages, max_tokens, extra_params: ("new", 0.1))
        op = SummarizationOperator(_params(), overwrite=True, client=client)

        out = op.run(pd.DataFrame({"text": ["source"], "summary": ["old"]}))

        assert out["summary"].tolist() == ["new"]

    def test_duplicate_indices_order_and_mixed_failures(self):
        from nemo_retriever.operators.generation import SummarizationOperator

        def respond(messages, max_tokens, extra_params):
            content = messages[-1]["content"]
            if "boom" in content:
                raise RuntimeError("offline for boom")
            if "slow" in content:
                time.sleep(0.03)
                return "SLOW", 0.3
            return "FAST", 0.1

        source = pd.DataFrame({"text": ["slow", "boom", "fast"]}, index=[7, 7, 2])
        out = SummarizationOperator(_params(max_workers=3), client=FakeCompletionClient(respond)).run(source)

        assert out.index.tolist() == [7, 7, 2]
        assert out["text"].tolist() == ["slow", "boom", "fast"]
        assert out["summary"].tolist() == ["SLOW", "", "FAST"]
        assert out["summary_error"].tolist()[0] is None
        assert out["summary_error"].tolist()[1] == "transport_error"
        assert out["summary_latency_s"].tolist()[1] > 0.0
        assert out["summary_error"].tolist()[2] is None
        assert out["summary_model"].tolist() == ["fake/model"] * 3

    def test_empty_batch_adds_schema_without_calling_client(self):
        from nemo_retriever.operators.generation import SummarizationOperator

        client = FakeCompletionClient()
        out = SummarizationOperator(_params(), client=client).run(pd.DataFrame({"text": pd.Series(dtype=str)}))

        assert list(out.columns) == [
            "text",
            "summary",
            "summary_latency_s",
            "summary_model",
            "summary_error",
        ]
        assert out.empty
        assert client.calls == []
        assert out.dtypes.astype(str).to_dict() == {
            "text": "object",
            "summary": "object",
            "summary_latency_s": "float64",
            "summary_model": "object",
            "summary_error": "object",
        }

    def test_generic_operator_maps_multiple_logical_inputs(self):
        from nemo_retriever.operators.generation import GenericGenerationOperator

        client = FakeCompletionClient(lambda messages, max_tokens, extra_params: ("bonjour", 0.2))
        params = _params(prompt="Translate {text} to {language}.", system_prompt="Translator")
        op = GenericGenerationOperator(
            params,
            input_columns={"text": "body", "language": "target"},
            client=client,
        )

        out = op.run(pd.DataFrame({"body": ["hello"], "target": ["French"]}))

        assert out["generated_text"].tolist() == ["bonjour"]
        assert client.calls[0][0] == [
            {"role": "system", "content": "Translator"},
            {"role": "user", "content": "Translate hello to French."},
        ]

    def test_injected_and_built_clients_are_not_graph_constructor_state(self):
        from nemo_retriever.graph.pipeline_graph import Node
        from nemo_retriever.operators.generation import SummarizationOperator

        client = FakeCompletionClient()
        injected = SummarizationOperator(_params(), client=client)
        built = SummarizationOperator(_params())

        assert injected._client is client
        for operator in (injected, built):
            kwargs = Node(operator).operator_kwargs
            assert "client" not in kwargs
            assert "task" not in kwargs
            assert client not in kwargs.values()

    def test_params_and_graph_kwargs_are_defensive_snapshots(self):
        from nemo_retriever.operators.generation import SummarizationOperator

        client = FakeCompletionClient()
        params = _params(
            prompt="Original {text}",
            extra_params={"provider": {"seed": 1}},
        )
        operator = SummarizationOperator(params, client=client)

        params.prompt = "Changed {text}"
        params.transport.model = "changed/model"
        params.transport.extra_params["provider"]["seed"] = 99

        first = operator.get_constructor_kwargs()
        first["params"].prompt = "Mutated graph copy {text}"
        first["params"].transport.extra_params["provider"]["seed"] = 7
        second = operator.get_constructor_kwargs()

        assert second["params"].prompt == "Original {text}"
        assert second["params"].transport.model == "fake/model"
        assert second["params"].transport.extra_params == {"provider": {"seed": 1}}

        operator.run(pd.DataFrame({"text": ["source"]}))
        assert client.calls[0][0][-1]["content"] == "Original source"

    def test_client_factory_hook_receives_task_and_copied_params(self):
        from nemo_retriever.models.llm.tasks import SummarizeTask
        from nemo_retriever.operators.generation import SummarizationOperator

        created_client = FakeCompletionClient()

        class FactoryOperator(SummarizationOperator):
            def _create_client(self, params, sampling):
                self.factory_params = params
                self.factory_sampling = sampling
                self.task_at_factory = self._task
                return created_client

        params = _params(temperature=0.3)
        operator = FactoryOperator(params)

        assert operator._client is created_client
        assert isinstance(operator.task_at_factory, SummarizeTask)
        assert operator.factory_params is operator._params
        assert operator.factory_params is not params
        assert operator.factory_sampling.temperature == 0.3

    def test_unknown_clients_serialize_and_opted_in_clients_run_concurrently(self):
        from nemo_retriever.operators.generation import SummarizationOperator

        def peak_for(client):
            state = {"active": 0, "peak": 0}
            lock = threading.Lock()

            def respond(messages, max_tokens, extra_params):
                with lock:
                    state["active"] += 1
                    state["peak"] = max(state["peak"], state["active"])
                time.sleep(0.02)
                with lock:
                    state["active"] -= 1
                return "done", 0.02

            client._handler = respond
            SummarizationOperator(_params(max_workers=4), client=client).run(
                pd.DataFrame({"text": ["a", "b", "c", "d"]})
            )
            return state["peak"]

        unknown_client = FakeCompletionClient()
        concurrent_client = FakeCompletionClient()
        concurrent_client.supports_concurrent_calls = True

        assert peak_for(unknown_client) == 1
        assert peak_for(concurrent_client) > 1

    def test_nonempty_batch_uses_the_same_output_dtypes_as_empty_batch(self):
        from nemo_retriever.operators.generation import SummarizationOperator

        out = SummarizationOperator(_params(), client=FakeCompletionClient()).run(pd.DataFrame({"text": ["source"]}))

        assert out.dtypes.astype(str).to_dict() == {
            "text": "object",
            "summary": "object",
            "summary_latency_s": "float64",
            "summary_model": "object",
            "summary_error": "object",
        }

    def test_failure_model_fallback_logs_only_sanitized_exception_type(self, caplog):
        from nemo_retriever.operators.generation import SummarizationOperator

        class BrokenModelClient(FakeCompletionClient):
            @property
            def model(self):
                raise RuntimeError("must-not-appear: secret-provider-message")

        operator = SummarizationOperator(_params(), client=BrokenModelClient())

        with caplog.at_level("DEBUG", logger="nemo_retriever.operators.generation.base"):
            assert operator._failure_model() == "fake/model"

        assert "builtins.RuntimeError" in caplog.text
        assert "secret-provider-message" not in caplog.text

    def test_result_position_mismatch_is_an_operator_failure(self):
        from nemo_retriever.operators.generation import SummarizationOperator

        class MisroutedResultOperator(SummarizationOperator):
            def _execute_row(self, position, inputs):
                _, result = super()._execute_row(position, inputs)
                return position + 1, result

        operator = MisroutedResultOperator(_params(), client=FakeCompletionClient())

        with pytest.raises(RuntimeError, match="result position 1 does not match submitted position 0"):
            operator.run(pd.DataFrame({"text": ["source"]}))


class TestQACompatibility:
    def test_qa_run_preserves_schema_order_and_overwrites_legacy_outputs(self):
        from nemo_retriever.tools.evaluation.generation import QAGenerationOperator

        client = FakeCompletionClient(lambda messages, max_tokens, extra_params: ("grounded answer", 0.6))
        operator = QAGenerationOperator(model="fake/model", api_key="")
        operator._client = client

        source = pd.DataFrame({"query": ["question"], "context": [["context"]]})
        out = operator.run(source)

        assert list(out.columns) == [
            "query",
            "context",
            "answer",
            "latency_s",
            "model",
            "gen_error",
        ]
        assert out.loc[0, ["answer", "latency_s", "model", "gen_error"]].tolist() == [
            "grounded answer",
            0.6,
            "fake/model",
            None,
        ]

        existing = out.assign(
            answer="stale",
            latency_s=-1.0,
            model="stale/model",
            gen_error="stale error",
        )
        overwritten = operator.run(existing)

        assert list(overwritten.columns) == list(existing.columns)
        assert overwritten.loc[0, ["answer", "latency_s", "model", "gen_error"]].tolist() == [
            "grounded answer",
            0.6,
            "fake/model",
            None,
        ]

    def test_qa_graph_kwargs_remain_flat_and_reconstructible(self):
        from nemo_retriever.graph.pipeline_graph import Node
        from nemo_retriever.tools.evaluation.generation import QAGenerationOperator

        operator = QAGenerationOperator(
            "fake/model",
            api_base="http://llm.test/v1",
            api_key="",
            temperature=0.2,
            top_p=0.7,
            max_tokens=321,
            extra_params={"seed": 4},
            num_retries=5,
            timeout=11.0,
            max_workers=2,
            rag_system_prompt="Use only context.",
            rag_system_prompt_prefix="Prefix",
            reasoning_enabled=False,
        )
        kwargs = Node(operator).operator_kwargs

        assert set(kwargs) == {
            "model",
            "api_base",
            "api_key",
            "temperature",
            "top_p",
            "max_tokens",
            "extra_params",
            "num_retries",
            "timeout",
            "max_workers",
            "rag_system_prompt",
            "rag_system_prompt_prefix",
            "reasoning_enabled",
        }
        assert not ({"params", "input_columns", "output_column", "task", "client"} & kwargs.keys())

        kwargs["extra_params"]["seed"] = 99
        assert operator.get_constructor_kwargs()["extra_params"] == {"seed": 4}
        kwargs["extra_params"]["seed"] = 4

        reconstructed = QAGenerationOperator(**kwargs)
        assert reconstructed._client.transport.model == "fake/model"
        assert reconstructed._client.sampling.temperature == 0.2
        assert reconstructed._client.sampling.top_p == 0.7
        assert reconstructed._client.sampling.max_tokens == 321
        assert reconstructed.required_columns == ("query", "context")
        assert reconstructed.output_columns == (
            "answer",
            "latency_s",
            "model",
            "gen_error",
        )

    def test_litellm_generate_preserves_legacy_result_sentinels(self):
        from nemo_retriever.models.llm.clients import LiteLLMClient
        from nemo_retriever.models.llm.types import GenerationResult

        client = LiteLLMClient.from_kwargs(model="fake/model", api_key="")
        client.complete = lambda messages, max_tokens=None, extra_params=None: (
            "<think>reasoning only</think>",
            0.45,
        )
        truncated = client.generate("question", ["context"])

        assert isinstance(truncated, GenerationResult)
        assert truncated == GenerationResult(
            answer="",
            latency_s=0.45,
            model="fake/model",
            error="thinking_truncated",
        )

        def fail(messages, max_tokens=None, extra_params=None):
            raise RuntimeError("transport unavailable")

        client.complete = fail
        failed = client.generate("question", ["context"])
        assert failed.answer == ""
        assert failed.latency_s > 0.0
        assert failed.model == "fake/model"
        assert failed.error == "transport_error"

    def test_rag_prompt_helper_reexport_preserves_exact_messages(self):
        from nemo_retriever.models.llm.clients import _build_rag_prompt

        empty = _build_rag_prompt("Where?", [], formatted_rag_system_prompt="System")
        populated = _build_rag_prompt(
            "Where?",
            ["first", "second"],
            formatted_rag_system_prompt="System",
        )

        assert empty == [
            {"role": "system", "content": "System"},
            {
                "role": "user",
                "content": "Context:\n(no context retrieved)\n\nQuestion: Where?\n\nAnswer:",
            },
        ]
        assert populated[-1]["content"] == ("Context:\nfirst\n\n---\n\nsecond\n\nQuestion: Where?\n\nAnswer:")


class TestSamplingRemediation:
    def test_omitted_and_explicit_null_overrides_survive_round_trip(self):
        from nemo_retriever.common.params import (
            LLMInferenceParams,
            LLMSamplingOverrides,
        )

        defaults = LLMInferenceParams(temperature=0.4, top_p=0.7, max_tokens=99)
        omitted = LLMSamplingOverrides()
        cleared = LLMSamplingOverrides(temperature=None, top_p=None)

        omitted_restored = LLMSamplingOverrides.model_validate_json(omitted.model_dump_json())
        cleared_restored = LLMSamplingOverrides.model_validate_json(cleared.model_dump_json())

        assert omitted.model_dump() == {}
        assert omitted_restored.resolve(defaults) == defaults
        assert omitted != cleared
        assert _params() != _params(temperature=None)
        assert cleared.model_dump() == {"temperature": None, "top_p": None}
        resolved = cleared_restored.resolve(defaults)
        assert resolved.temperature is None
        assert resolved.top_p is None
        assert resolved.to_sampling_kwargs() == {"max_tokens": 99}

    def test_nested_text_params_preserve_partial_sampling_state(self):
        from nemo_retriever.common.params import (
            LLMInferenceParams,
            TextGenerationParams,
        )

        params = _params(top_p=None)
        restored = TextGenerationParams.model_validate(params.model_dump())
        defaults = LLMInferenceParams(temperature=0.3, top_p=0.7, max_tokens=10)

        assert restored.sampling.model_fields_set == {"top_p"}
        assert restored.resolve_sampling(defaults).top_p is None
        assert restored.resolve_sampling(defaults).temperature == 0.3

    def test_explicit_null_max_tokens_is_invalid(self):
        from nemo_retriever.common.params import LLMSamplingOverrides

        with pytest.raises(ValueError, match="omit it"):
            LLMSamplingOverrides(max_tokens=None)


class TestOperatorRemediation:
    def test_mixed_numeric_columns_preserve_exact_integer_prompt_value(self):
        from nemo_retriever.operators.generation import GenericGenerationOperator

        large = 2**60 + 1
        seen: list[str] = []

        def respond(messages, max_tokens, extra_params):
            seen.append(messages[-1]["content"])
            return "ok", 0.1

        source = pd.DataFrame(
            {
                "large": pd.Series([large], dtype="int64"),
                "ratio": pd.Series([0.5], dtype="float64"),
            }
        )
        operator = GenericGenerationOperator(
            _params(prompt="{large}|{ratio}"),
            input_columns={"large": "large", "ratio": "ratio"},
            client=FakeCompletionClient(respond),
        )

        out = operator.run(source)

        assert seen == [f"{large}|0.5"]
        assert out["generated_text"].tolist() == ["ok"]

    def test_ambiguous_mapped_input_label_is_rejected(self):
        from nemo_retriever.operators.generation import SummarizationOperator

        source = pd.DataFrame([["first", "second"]], columns=["text", "text"])
        operator = SummarizationOperator(_params(), client=FakeCompletionClient())

        with pytest.raises(ValueError, match="mapped input columns are ambiguous"):
            operator.run(source)

    def test_ambiguous_output_label_is_rejected_when_overwriting(self):
        from nemo_retriever.operators.generation import SummarizationOperator

        source = pd.DataFrame(
            [["source", "old one", "old two"]],
            columns=["text", "summary", "summary"],
        )
        operator = SummarizationOperator(_params(), overwrite=True, client=FakeCompletionClient())

        with pytest.raises(ValueError, match="ambiguous duplicate output"):
            operator.run(source)

    def test_reconstruction_hook_rejects_nested_runtime_client(self):
        from nemo_retriever.graph.pipeline_graph import Node
        from nemo_retriever.operators.generation import SummarizationOperator

        class InvalidOperator(SummarizationOperator):
            def _get_generation_constructor_kwargs(self):
                kwargs = super()._get_generation_constructor_kwargs()
                kwargs["input_column"] = {
                    "nested": [self._client],
                }
                return kwargs

        operator = InvalidOperator(
            _params(),
            client=FakeCompletionClient(),
        )

        with pytest.raises(TypeError, match="captured a live client or task"):
            Node(operator)


class TestQALegacyClientRemediation:
    def test_generate_only_client_without_model_is_called(self):
        from nemo_retriever.models.llm.types import GenerationResult
        from nemo_retriever.tools.evaluation.generation import QAGenerationOperator

        class LegacyClient:
            def __init__(self):
                self.calls: list[tuple[str, list[str]]] = []

            def generate(self, query, chunks, *, reasoning_enabled=None):
                self.calls.append((query, chunks))
                return GenerationResult("legacy answer", 0.4, "legacy/model")

        client = LegacyClient()
        operator = QAGenerationOperator(model="configured/model", api_key="")
        operator._client = client

        assert operator._effective_max_workers(4) == 1
        out = operator.run(pd.DataFrame({"query": ["q"], "context": [["c"]]}))

        assert client.calls == [("q", ["c"])]
        assert out.loc[0, "answer"] == "legacy answer"
        assert out.loc[0, "model"] == "legacy/model"
        assert out.loc[0, "gen_error"] is None

    def test_generate_only_client_receives_operator_reasoning_setting(self):
        from nemo_retriever.models.llm.types import GenerationResult
        from nemo_retriever.tools.evaluation.generation import QAGenerationOperator

        class LegacyClient:
            def __init__(self):
                self.reasoning_values: list[bool | None] = []

            def generate(self, query, chunks, *, reasoning_enabled=None):
                self.reasoning_values.append(reasoning_enabled)
                return GenerationResult("legacy answer", 0.2, "legacy/model")

        client = LegacyClient()
        operator = QAGenerationOperator(
            model="configured/model",
            api_key="",
            reasoning_enabled=False,
        )
        operator._client = client

        operator.run(pd.DataFrame({"query": ["q"], "context": [["c"]]}))

        assert client.reasoning_values == [False]

    def test_generate_only_failure_uses_configured_model(self):
        from nemo_retriever.tools.evaluation.generation import QAGenerationOperator

        class FailingLegacyClient:
            def generate(self, query, chunks, *, reasoning_enabled=None):
                raise RuntimeError("legacy unavailable")

        operator = QAGenerationOperator(model="configured/model", api_key="")
        operator._client = FailingLegacyClient()

        out = operator.run(pd.DataFrame({"query": ["q"], "context": [["c"]]}))

        assert out.loc[0, "answer"] == ""
        assert out.loc[0, "model"] == "configured/model"
        assert out.loc[0, "gen_error"] == "transport_error"
        assert out.loc[0, "latency_s"] > 0.0

    def test_generate_only_result_preserves_legacy_error_sentinel(self):
        from nemo_retriever.models.llm.types import GenerationResult
        from nemo_retriever.tools.evaluation.generation import QAGenerationOperator

        class LegacyClient:
            def generate(self, query, chunks, *, reasoning_enabled=None):
                return GenerationResult(
                    answer="",
                    latency_s=0.4,
                    model="legacy/model",
                    error="legacy_sentinel",
                )

        operator = QAGenerationOperator(model="configured/model", api_key="")
        operator._client = LegacyClient()

        out = operator.run(pd.DataFrame({"query": ["q"], "context": [["c"]]}))

        assert out.loc[0, "answer"] == ""
        assert out.loc[0, "latency_s"] == 0.4
        assert out.loc[0, "model"] == "legacy/model"
        assert out.loc[0, "gen_error"] == "legacy_sentinel"

    def test_dual_protocol_client_prefers_completion_contract(self):
        from nemo_retriever.models.llm.types import GenerationResult
        from nemo_retriever.tools.evaluation.generation import QAGenerationOperator

        class DualClient(FakeCompletionClient):
            def __init__(self):
                super().__init__(
                    lambda messages, max_tokens, extra_params: (
                        "completion answer",
                        0.2,
                    )
                )
                self.generate_calls = 0

            def generate(self, query, chunks, *, reasoning_enabled=None):
                self.generate_calls += 1
                return GenerationResult("legacy answer", 0.3, self.model)

        client = DualClient()
        operator = QAGenerationOperator(model="configured/model", api_key="")
        operator._client = client

        out = operator.run(pd.DataFrame({"query": ["q"], "context": [["c"]]}))

        assert out.loc[0, "answer"] == "completion answer"
        assert len(client.calls) == 1
        assert client.generate_calls == 0


class TestPublicLLMExports:
    def test_direct_and_star_imports_resolve_canonical_clients(self):
        import nemo_retriever.models.llm as llm_module
        from nemo_retriever.models.llm import (
            LLMJudge,
            LiteLLMClient,
            TextCompletionClient,
        )

        namespace: dict[str, Any] = {}
        exec("from nemo_retriever.models.llm import *", namespace)

        assert LiteLLMClient.__module__ == "nemo_retriever.models.llm.clients.litellm"
        assert LLMJudge.__module__ == "nemo_retriever.models.llm.clients.judge"
        assert namespace["LiteLLMClient"] is LiteLLMClient
        assert namespace["LLMJudge"] is LLMJudge
        assert namespace["TextCompletionClient"] is TextCompletionClient
        assert "CompletionClient" not in namespace
        assert not hasattr(llm_module, "CompletionClient")

    def test_type_contract_import_does_not_eagerly_load_clients(self):
        import subprocess
        import sys

        code = (
            "import sys; "
            "from nemo_retriever.models.llm import TextCompletionClient; "
            "assert 'nemo_retriever.models.llm.clients.litellm' not in sys.modules; "
            "assert 'litellm' not in sys.modules; "
            "from nemo_retriever.models.llm import LiteLLMClient; "
            "assert 'nemo_retriever.models.llm.clients.litellm' in sys.modules; "
            "assert 'litellm' not in sys.modules"
        )

        source_path = str(Path(__file__).resolve().parents[1] / "src")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([source_path, env.get("PYTHONPATH", "")])

        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            env=env,
        )


class TestGenerationGraphPersistence:
    @staticmethod
    def _single_root_graph(operator):
        from nemo_retriever.graph.pipeline_graph import Graph

        graph = Graph()
        graph.add_root(operator)
        return graph

    def test_typed_params_are_secret_free_and_rehydrate_from_environment(
        self,
        monkeypatch,
        tmp_path,
    ):
        from nemo_retriever.common.params import TextGenerationParams
        from nemo_retriever.graph.graph_pipeline_registry import (
            deserialize_graph,
            load_graph,
            save_graph,
            serialize_graph,
        )
        from nemo_retriever.operators.generation import SummarizationOperator

        api_key_reference = "os.environ/OPENAI_API_KEY"
        params = TextGenerationParams.from_kwargs(
            model="openai/fake-model",
            api_key=api_key_reference,
            temperature=None,
            top_p=0.7,
        )
        graph = self._single_root_graph(SummarizationOperator(params))

        payload = serialize_graph(graph)
        encoded = json.dumps(payload)

        assert payload["format_version"] == 2
        assert "__pydantic_model__" in encoded
        assert api_key_reference in encoded

        monkeypatch.setenv("NVIDIA_API_KEY", "wrong-provider-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "worker-secret")
        restored = deserialize_graph(payload)
        restored_operator = restored.roots[0].operator

        assert isinstance(restored_operator, SummarizationOperator)
        assert restored_operator._params.transport.api_key == api_key_reference
        assert restored_operator._params.sampling.model_fields_set == {
            "temperature",
            "top_p",
        }
        assert restored_operator._client.sampling.temperature is None
        assert restored_operator._client.sampling.top_p == 0.7

        path = tmp_path / "generation.json"
        save_graph(graph, path)
        assert "worker-secret" not in path.read_text()
        loaded_operator = load_graph(path).roots[0].operator
        assert isinstance(loaded_operator, SummarizationOperator)
        assert loaded_operator._params.transport.api_key == api_key_reference

    def test_no_auth_survives_clone_with_environment_key_present(self, monkeypatch):
        from nemo_retriever.graph.graph_pipeline_registry import clone_graph
        from nemo_retriever.operators.generation import GenericGenerationOperator

        monkeypatch.setenv("NVIDIA_API_KEY", "must-not-be-used")
        operator = GenericGenerationOperator(
            _params(prompt="Hello {name}", api_key=""),
            input_columns={"name": "name"},
        )

        cloned = clone_graph(self._single_root_graph(operator))
        cloned_operator = cloned.roots[0].operator

        assert isinstance(cloned_operator, GenericGenerationOperator)
        assert cloned_operator._params.transport.api_key is None
        assert cloned_operator._params.transport._uses_no_api_key("api_key")

    def test_qa_and_registry_round_trip_to_concrete_operators(
        self,
        monkeypatch,
        tmp_path,
    ):
        from nemo_retriever.graph.graph_pipeline_registry import (
            GraphPipelineRegistry,
            deserialize_graph,
            serialize_graph,
        )
        from nemo_retriever.tools.evaluation.generation import QAGenerationOperator

        api_key_reference = "os.environ/OPENAI_API_KEY"
        operator = QAGenerationOperator(
            model="openai/fake-model",
            api_key=api_key_reference,
        )
        graph = self._single_root_graph(operator)
        payload = serialize_graph(graph)

        monkeypatch.setenv("NVIDIA_API_KEY", "wrong-provider-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "qa-worker-secret")
        restored = deserialize_graph(payload)
        assert isinstance(restored.roots[0].operator, QAGenerationOperator)
        assert restored.roots[0].operator._client.transport.api_key == api_key_reference

        registry = GraphPipelineRegistry()
        registry.register_graph(
            "qa",
            lambda: graph,
            description="QA graph",
            tags=["generation"],
        )
        path = tmp_path / "registry.json"
        registry.save_all(path)
        raw = json.loads(path.read_text())

        assert raw["format_version"] == 2
        assert set(raw["graphs"]) == {"qa"}
        assert api_key_reference in path.read_text()
        assert "qa-worker-secret" not in path.read_text()
        assert "wrong-provider-secret" not in path.read_text()

        loaded_registry = GraphPipelineRegistry()
        assert loaded_registry.load_all(path) == ["qa"]
        rebuilt = loaded_registry.build("qa")
        assert isinstance(rebuilt.roots[0].operator, QAGenerationOperator)

    def test_opaque_nested_api_key_is_rejected_with_context(self):
        from nemo_retriever.graph.graph_pipeline_registry import (
            GraphSerializationError,
            serialize_graph,
        )
        from nemo_retriever.operators.generation import SummarizationOperator

        graph = self._single_root_graph(SummarizationOperator(_params()))
        graph.roots[0].operator_kwargs["opaque"] = {
            "api_key": "must-not-leak",
        }

        with pytest.raises(
            GraphSerializationError,
            match="opaque mapping",
        ):
            serialize_graph(graph)

    def test_graph_diagnostics_redact_flat_and_nested_credentials(self):
        from nemo_retriever.graph.graph_pipeline_registry import (
            diff_graphs,
            format_full_report,
            format_graph_tree,
            format_node_details,
        )
        from nemo_retriever.tools.evaluation.generation import QAGenerationOperator

        first = QAGenerationOperator(
            model="fake/model",
            api_key="FIRST-SECRET",
            extra_params={
                "headers": {
                    "Authorization": "Bearer INNER-FIRST",
                },
            },
        )
        second = QAGenerationOperator(
            model="fake/model",
            api_key="SECOND-SECRET",
            extra_params={
                "headers": {
                    "Authorization": "Bearer INNER-SECOND",
                },
            },
        )
        first_graph = self._single_root_graph(first)
        second_graph = self._single_root_graph(second)

        rendered = [
            format_graph_tree(first_graph, show_kwargs=True),
            format_node_details(first_graph.roots[0]),
            format_full_report(first_graph),
            diff_graphs(first_graph, second_graph).format(),
        ]

        for report in rendered:
            assert "FIRST-SECRET" not in report
            assert "SECOND-SECRET" not in report
            assert "Bearer INNER-FIRST" not in report
            assert "Bearer INNER-SECOND" not in report
            assert "***" in report

    def test_non_rehydratable_token_fields_are_rejected(self):
        from nemo_retriever.common.params.models import ASRParams
        from nemo_retriever.graph.graph_pipeline_registry import (
            GraphSerializationError,
            _safe_serialize_value,
            serialize_graph,
        )
        from nemo_retriever.operators.generation import SummarizationOperator

        with pytest.raises(
            GraphSerializationError,
            match="non-rehydratable secret field",
        ):
            _safe_serialize_value(ASRParams(auth_token="typed-secret"))

        graph = self._single_root_graph(SummarizationOperator(_params()))
        graph.roots[0].operator_kwargs["auth_token"] = "flat-secret"

        with pytest.raises(
            GraphSerializationError,
            match="non-rehydratable secret field",
        ):
            serialize_graph(graph)

    def test_v2_constructor_failures_raise_instead_of_using_placeholder(self):
        from nemo_retriever.graph.graph_pipeline_registry import (
            GraphSerializationError,
            deserialize_graph,
            serialize_graph,
        )
        from nemo_retriever.operators.generation import SummarizationOperator

        graph = self._single_root_graph(SummarizationOperator(_params()))
        payload = serialize_graph(graph)
        root_id = payload["roots"][0]
        payload["nodes"][root_id]["operator_kwargs"]["unexpected"] = True

        with pytest.raises(
            GraphSerializationError,
            match="failed to construct operator",
        ):
            deserialize_graph(payload)
