# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _ParamsModel._resolve_api_keys model validator."""

import pytest
from pydantic import ValidationError

from nemo_retriever.common.params.models import (
    EmbedParams,
    ExtractParams,
    NO_API_KEY,
    StoreParams,
    VdbSinkParams,
    VdbUploadParams,
    VideoFrameParams,
)
from nemo_retriever.graph.ingestor_runtime import build_post_extract_graph


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

    def test_extraction_method_schema_describes_supported_and_legacy_values(self) -> None:
        schema = ExtractParams.model_json_schema()["properties"]["method"]

        assert "PDF extraction supports" in schema["description"]
        assert "legacy params-driven audio path" in schema["description"]

    def test_parse_specific_configuration_requires_parse_method(self) -> None:
        for field, value in (
            ("nemotron_parse_invoke_url", "http://parse:8000/v1/chat/completions"),
            ("nemotron_parse_model", "nvidia/nemotron-parse"),
        ):
            with pytest.raises(ValidationError, match="method='nemotron_parse'"):
                ExtractParams(**{field: value})

    def test_normal_and_selected_parse_configurations_are_valid(self) -> None:
        assert ExtractParams().method == "pdfium"
        assert ExtractParams(invoke_url="http://generic").method == "pdfium"
        params = ExtractParams(
            method="nemotron_parse",
            nemotron_parse_invoke_url="https://integrate.api.nvidia.com/v1/chat/completions",
            nemotron_parse_model="nvidia/nemotron-parse",
        )
        assert params.method == "nemotron_parse"

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


class TestVdbSinkParams:
    def test_sink_configuration_reaches_the_operator_not_the_backend(self, tmp_path) -> None:
        params = VdbUploadParams(
            vdb_kwargs={
                "uri": str(tmp_path),
                "table_name": "chunks",
                "vector_dim": 2,
                "build_index": False,
            },
            sink=VdbSinkParams(
                max_batch_bytes=64 << 20,
                prefetch_batches=2,
                optimize=True,
                operation_id="qualification-run",
            ),
        )

        operator = build_post_extract_graph(vdb_upload_params=params, stage_order=()).roots[0].operator

        assert operator.sink_policy.max_batch_bytes == 64 << 20
        assert operator.sink_policy.prefetch_batches == 2
        assert operator.sink_policy.optimize is True
        assert operator.operation_id == "qualification-run"
        assert "sink" not in operator._vdb_kwargs
        assert "operation_id" not in operator._vdb_kwargs

    @pytest.mark.parametrize(
        ("field", "value"),
        [("max_batch_bytes", 0), ("prefetch_batches", -1), ("operation_id", "   ")],
    )
    def test_invalid_bounds_and_identity_are_rejected(self, field: str, value) -> None:
        with pytest.raises(ValueError):
            VdbSinkParams(**{field: value})


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
