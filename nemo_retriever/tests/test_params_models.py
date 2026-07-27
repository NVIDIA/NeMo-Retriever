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
