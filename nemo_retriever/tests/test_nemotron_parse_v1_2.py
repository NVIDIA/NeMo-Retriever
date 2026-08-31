# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the local vLLM-backed Nemotron Parse model."""

import os
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

from packaging.requirements import Requirement
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_core_dependencies_exclude_tokenizers_duplicate_pattern_regression():
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    tokenizers = next(Requirement(dependency) for dependency in dependencies if dependency.startswith("tokenizers"))

    assert tokenizers.specifier.contains("0.22.2")
    assert not tokenizers.specifier.contains("0.23.0rc0", prereleases=True)


def test_applies_vllm_startup_defaults_before_constructing_llm(monkeypatch):
    from nemo_retriever.models.local import nemotron_parse_v1_2 as mod

    monkeypatch.delenv("VLLM_DEEP_GEMM_WARMUP", raising=False)

    def assert_startup_defaults(**_kwargs):
        assert os.environ["VLLM_DEEP_GEMM_WARMUP"] == "skip"
        return MagicMock()

    with (
        patch.object(mod, "_patch_vllm_nemotron_parse_processor"),
        patch.object(mod, "configure_global_hf_cache_base"),
        patch.object(mod, "get_hf_revision", return_value="test-revision"),
        patch("vllm.LLM", side_effect=assert_startup_defaults),
        patch("vllm.SamplingParams"),
    ):
        mod.NemotronParseV12()


def test_parse_playground_accepts_only_the_supported_v1_2_model():
    from nemo_retriever.harness.portal.app import ParseTestRequest

    request = ParseTestRequest(image_b64="aW1hZ2U=")
    assert request.model_id == "nvidia/NVIDIA-Nemotron-Parse-v1.2"

    with pytest.raises(ValueError):
        ParseTestRequest(model_id="nvidia/NVIDIA-Nemotron-Parse-2.0", image_b64="aW1hZ2U=")
