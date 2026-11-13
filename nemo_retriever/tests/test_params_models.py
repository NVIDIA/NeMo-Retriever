# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _ParamsModel._resolve_api_keys model validator."""

import pytest
from pydantic import ValidationError

from nemo_retriever.common.params.models import EmbedParams, ExtractParams, NO_API_KEY, StoreParams, VideoFrameParams


class TestVideoFrameParams:
    def test_fps_zero_rejected(self) -> None:
        """``fps=0`` would div-by-zero in ``_extract_one``; reject at the model boundary."""
        with pytest.raises(ValidationError):
            VideoFrameParams(fps=0)


class TestExtractParams:
    @pytest.mark.parametrize("method", ["pdfium", "pdfium_hybrid", "ocr", "nemotron_parse", "audio"])
    def test_supported_extraction_methods_are_valid(self, method: str) -> None:
        assert ExtractParams(method=method).method == method

    def test_unsupported_extraction_method_is_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ExtractParams(method="unsupported")

        error = str(exc_info.value)
        for method in ("pdfium", "pdfium_hybrid", "ocr", "nemotron_parse", "audio"):
            assert method in error

    def test_nested_image_extraction_requires_image_extraction(self) -> None:
        with pytest.raises(ValidationError, match="extract_nested_images=True requires extract_images=True"):
            ExtractParams(extract_images=False, extract_nested_images=True)

        assert ExtractParams(extract_images=True, extract_nested_images=True).extract_nested_images is True

    def test_extraction_method_schema_describes_supported_and_legacy_values(self) -> None:
        schema = ExtractParams.model_json_schema()["properties"]["method"]

        assert "PDF extraction supports" in schema["description"]
        assert "legacy params-driven audio path" in schema["description"]

    @pytest.mark.parametrize("tokens", ["auto", None, 1, 16, 32])
    def test_local_parse_dflash_length(self, tokens):
        params = ExtractParams(method="nemotron_parse", nemotron_parse_dflash_tokens=tokens, invoke_url=" ")
        assert params.nemotron_parse_dflash_tokens == tokens

    @pytest.mark.parametrize("tokens", [0, -1, 33, True, 1.5, "32"])
    def test_invalid_parse_dflash_length(self, tokens):
        with pytest.raises(ValidationError):
            ExtractParams(method="nemotron_parse", nemotron_parse_dflash_tokens=tokens)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"nemotron_parse_model": "nvidia/NVIDIA-Nemotron-Parse-v1.2"},
            {"invoke_url": "http://parse.example/v1/chat/completions"},
            {"nemotron_parse_invoke_url": "http://parse.example/v1/chat/completions"},
        ],
    )
    def test_dflash_rejects_remote_and_legacy_parse(self, overrides):
        with pytest.raises(ValidationError, match="requires local Nemotron Parse 3.5"):
            ExtractParams(method="nemotron_parse", nemotron_parse_dflash_tokens=32, **overrides)

    @pytest.mark.parametrize("tokens", ["auto", None, 1, 32])
    def test_dflash_validation_after_cpu_only_resolution(self, tokens: str | int | None) -> None:
        from unittest.mock import patch

        from nemo_retriever.common.ray_resource_hueristics import Resources
        from nemo_retriever.graph.graph_pipeline_registry import get_node_kwargs
        from nemo_retriever.graph.ingestor_runtime import build_graph
        from nemo_retriever.operators.extract.parse.nemotron_parse import NemotronParseActor, NemotronParseCPUActor

        params = ExtractParams(method="nemotron_parse", nemotron_parse_dflash_tokens=tokens)
        graph = build_graph(extraction_mode="pdf", extract_params=params)
        kwargs = get_node_kwargs(graph, "NemotronParseActor")
        actor_class = NemotronParseActor.resolve_operator_class(
            Resources(cpu_count=8, gpu_count=0), operator_kwargs=kwargs
        )
        assert actor_class is NemotronParseCPUActor
        with patch("nemo_retriever.operators.extract.parse.nemotron_parse.NIMClient") as client:
            if isinstance(tokens, int):
                with pytest.raises(ValueError, match="requires local Nemotron Parse 3.5") as exc_info:
                    actor_class(**kwargs)
                assert "'auto' or None" in str(exc_info.value)
                client.assert_not_called()
            else:
                actor = actor_class(**kwargs)
                assert actor._invoke_url == NemotronParseCPUActor.DEFAULT_INVOKE_URL
                assert actor._nemotron_parse_dflash_tokens is None
                rebuilt = actor_class(**actor.get_constructor_kwargs())
                assert rebuilt._invoke_url == NemotronParseCPUActor.DEFAULT_INVOKE_URL
                assert rebuilt._nemotron_parse_dflash_tokens is None
                assert client.call_count == 2

    def test_parse_specific_configuration_requires_parse_method(self) -> None:
        for field, value in (
            ("nemotron_parse_invoke_url", "http://parse:8000/v1/chat/completions"),
            ("nemotron_parse_model", "nvidia/nemotron-parse"),
            ("nemotron_parse_dflash_tokens", 32),
        ):
            with pytest.raises(ValidationError, match="method='nemotron_parse'"):
                ExtractParams(**{field: value})

    def test_normal_and_selected_parse_configurations_are_valid(self) -> None:
        assert ExtractParams().method == "pdfium_hybrid"
        assert ExtractParams(invoke_url="http://generic").method == "pdfium_hybrid"
        params = ExtractParams(
            method="nemotron_parse",
            nemotron_parse_invoke_url="https://integrate.api.nvidia.com/v1/chat/completions",
            nemotron_parse_model="nvidia/nemotron-parse",
        )
        assert params.method == "nemotron_parse"

    @pytest.mark.parametrize("endpoint_field", ["nemotron_parse_invoke_url", "invoke_url"])
    def test_remote_parse_model_is_accepted(self, endpoint_field: str) -> None:
        params = ExtractParams(
            method="nemotron_parse",
            nemotron_parse_model="nvidia/NVIDIA-Nemotron-Parse-2.0",
            **{endpoint_field: "http://parse:8000/v1/chat/completions"},
        )

        assert params.nemotron_parse_model == "nvidia/NVIDIA-Nemotron-Parse-2.0"

    def test_unsupported_local_parse_model_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Local Nemotron Parse supports only") as exc_info:
            ExtractParams(
                method="nemotron_parse",
                nemotron_parse_model="nvidia/NVIDIA-Nemotron-Parse-2.0",
            )

        error = str(exc_info.value)
        assert "nvidia/NVIDIA-Nemotron-Parse-v1.2" in error
        assert "nvidia/NVIDIA-Nemotron-Parse-2.0" in error
        assert "nemotron_parse_invoke_url" in error

    @pytest.mark.parametrize("model_id", ["nvidia/NVIDIA-Nemotron-Parse-v1.2", "nvidia/Nemotron-3.5-Parse"])
    def test_supported_local_parse_models_are_accepted(self, model_id: str) -> None:
        params = ExtractParams(method="nemotron_parse", nemotron_parse_model=model_id)
        assert params.nemotron_parse_model == model_id

    @pytest.mark.parametrize(
        "model",
        [None, "nvidia/nemotron-parse", "nvidia/nemotron-parse-v1.2"],
    )
    def test_mixed_build_and_self_hosted_parse_endpoints_are_rejected(self, model: str | None) -> None:
        endpoints = "https://integrate.api.nvidia.com/v1/chat/completions," "http://127.0.0.1:8018/v1/chat/completions"

        with pytest.raises(ValidationError, match="cannot mix NVIDIA Build and self-hosted"):
            ExtractParams(
                method="nemotron_parse",
                nemotron_parse_invoke_url=endpoints,
                nemotron_parse_model=model,
            )

    def test_graphic_elements_controls_are_removed(self) -> None:
        assert "use_graphic_elements" not in ExtractParams.model_fields
        assert "graphic_elements_invoke_url" not in ExtractParams.model_fields
        assert "extract_charts" in ExtractParams.model_fields


class TestStoreParams:
    def test_storage_options_redacted_from_repr(self) -> None:
        params = StoreParams(storage_options={"key": "AKIA_TEST", "secret": "SECRET_TEST"})

        rendered = repr(params)

        assert "AKIA_TEST" not in rendered
        assert "SECRET_TEST" not in rendered
        assert "storage_options=***" in rendered
        assert params.storage_options == {"key": "AKIA_TEST", "secret": "SECRET_TEST"}


class TestResolveApiKeys:
    def test_nvidia_api_key_env_var(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        monkeypatch.delenv("NGC_API_KEY", raising=False)
        assert EmbedParams().api_key == "nvapi-test"

    def test_ngc_api_key_fallback(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.setenv("NGC_API_KEY", "ngc-test")
        assert EmbedParams().api_key == "ngc-test"

    def test_explicit_value_not_overwritten(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        assert EmbedParams(api_key="explicit-key").api_key == "explicit-key"

    def test_no_env_var_remains_none(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.delenv("NGC_API_KEY", raising=False)
        assert EmbedParams().api_key is None

    def test_no_api_key_sentinel_suppresses_resolution(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        assert EmbedParams(api_key=NO_API_KEY).api_key is None

    def test_all_api_key_fields_resolved_on_extract_params(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
        monkeypatch.delenv("NGC_API_KEY", raising=False)
        params = ExtractParams()
        assert params.api_key == "nvapi-test"
        assert params.page_elements_api_key == "nvapi-test"
        assert params.ocr_api_key == "nvapi-test"


@pytest.mark.parametrize("method", ["pdfium", "pdfium_hybrid", "ocr", "audio"])
def test_automatic_parse_decoder_does_not_change_other_methods(method):
    params = ExtractParams(method=method)
    assert params.nemotron_parse_dflash_tokens == "auto"


def test_parse_decoder_schema_exposes_auto_default_and_bounded_draft_length():
    schema = ExtractParams.model_json_schema()["properties"]["nemotron_parse_dflash_tokens"]
    assert schema["default"] == "auto"
    integer = next(item for item in schema["anyOf"] if item.get("type") == "integer")
    assert integer["minimum"] == 1 and integer["maximum"] == 32
