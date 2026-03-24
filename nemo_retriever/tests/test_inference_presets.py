# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the inference preset feature and related create_ingestor() behaviour."""

import sys
import types

import pytest

from nemo_retriever.inference_presets import (
    _BUILD_NVIDIA_CAPTION_DEFAULTS,
    _BUILD_NVIDIA_EMBED_DEFAULTS,
    _BUILD_NVIDIA_EXTRACT_DEFAULTS,
    resolve_inference_preset,
)
from nemo_retriever.ingestor import create_ingestor
from nemo_retriever.params import IngestorCreateParams


# ---------------------------------------------------------------------------
# resolve_inference_preset()
# ---------------------------------------------------------------------------


class TestResolveInferencePreset:
    def test_none_returns_empty_dicts(self):
        extract, embed, caption = resolve_inference_preset(None)
        assert extract == {}
        assert embed == {}
        assert caption == {}

    def test_empty_string_returns_empty_dicts(self):
        extract, embed, caption = resolve_inference_preset("")
        assert extract == {}
        assert embed == {}
        assert caption == {}

    def test_build_nvidia_returns_expected_extract_defaults(self):
        extract, _, _ = resolve_inference_preset("build.nvidia.com")
        assert extract["page_elements_invoke_url"] == (
            "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3"
        )
        assert extract["graphic_elements_invoke_url"] == (
            "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-graphic-elements-v1"
        )
        assert extract["ocr_invoke_url"] == (
            "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr-v1"
        )
        assert extract["table_structure_invoke_url"] == (
            "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1"
        )

    def test_build_nvidia_returns_expected_embed_defaults(self):
        _, embed, _ = resolve_inference_preset("build.nvidia.com")
        assert embed["embed_invoke_url"] == "https://integrate.api.nvidia.com/v1/embeddings"
        assert embed["model_name"] == "nvidia/llama-nemotron-embed-1b-v2"
        assert embed["embed_modality"] == "text"

    def test_build_nvidia_returns_expected_caption_defaults(self):
        _, _, caption = resolve_inference_preset("build.nvidia.com")
        assert caption["endpoint_url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
        assert caption["model_name"] == "nvidia/nemotron-nano-12b-v2-vl"

    def test_returns_copies_not_originals(self):
        extract1, embed1, caption1 = resolve_inference_preset("build.nvidia.com")
        extract2, embed2, caption2 = resolve_inference_preset("build.nvidia.com")
        # Mutating one copy must not affect the other or the module-level constants.
        extract1["new_key"] = "new_value"
        caption1["another_key"] = "another_value"
        assert "new_key" not in extract2
        assert "another_key" not in caption2
        assert "new_key" not in _BUILD_NVIDIA_EXTRACT_DEFAULTS
        assert "another_key" not in _BUILD_NVIDIA_CAPTION_DEFAULTS

    def test_unknown_preset_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown inference preset"):
            resolve_inference_preset("unknown-preset")


# ---------------------------------------------------------------------------
# IngestorCreateParams accepts inference field
# ---------------------------------------------------------------------------


class TestIngestorCreateParamsInference:
    def test_default_is_none(self):
        p = IngestorCreateParams()
        assert p.inference is None

    def test_accepts_build_nvidia(self):
        p = IngestorCreateParams(inference="build.nvidia.com")
        assert p.inference == "build.nvidia.com"

    def test_rejects_unknown_value_at_ingestor_level(self):
        """Unknown preset names are caught when the ingestor is built, not at param construction."""
        p = IngestorCreateParams(inference="totally-unknown")
        assert p.inference == "totally-unknown"


# ---------------------------------------------------------------------------
# Helpers shared by factory tests below
# ---------------------------------------------------------------------------


def _make_capturing_inprocess_module(monkeypatch: pytest.MonkeyPatch) -> type:
    """Register a capturing DummyIngestor for nemo_retriever.ingest_modes.inprocess."""
    module = types.ModuleType("nemo_retriever.ingest_modes.inprocess")

    class CapturingIngestor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._default_extract_kwargs = kwargs.get("default_extract_kwargs", {})
            self._default_embed_kwargs = kwargs.get("default_embed_kwargs", {})
            self._default_caption_kwargs = kwargs.get("default_caption_kwargs", {})

    CapturingIngestor.__name__ = "InProcessIngestor"
    setattr(module, "InProcessIngestor", CapturingIngestor)
    monkeypatch.setitem(sys.modules, "nemo_retriever.ingest_modes.inprocess", module)
    return CapturingIngestor


# ---------------------------------------------------------------------------
# create_ingestor() with inference preset
# ---------------------------------------------------------------------------


class TestCreateIngestorWithPreset:
    def test_build_nvidia_preset_passes_extract_defaults(self, monkeypatch):
        """Factory passes preset extract defaults to InProcessIngestor."""
        dummy_cls = _make_capturing_inprocess_module(monkeypatch)

        ingestor = create_ingestor(run_mode="inprocess", inference="build.nvidia.com")

        assert isinstance(ingestor, dummy_cls)
        assert ingestor._default_extract_kwargs == _BUILD_NVIDIA_EXTRACT_DEFAULTS

    def test_build_nvidia_preset_passes_embed_defaults(self, monkeypatch):
        """Factory passes preset embed defaults to InProcessIngestor."""
        dummy_cls = _make_capturing_inprocess_module(monkeypatch)

        ingestor = create_ingestor(run_mode="inprocess", inference="build.nvidia.com")

        assert isinstance(ingestor, dummy_cls)
        assert ingestor._default_embed_kwargs == _BUILD_NVIDIA_EMBED_DEFAULTS

    def test_build_nvidia_preset_passes_caption_defaults(self, monkeypatch):
        """Factory passes preset caption defaults to InProcessIngestor."""
        dummy_cls = _make_capturing_inprocess_module(monkeypatch)

        ingestor = create_ingestor(run_mode="inprocess", inference="build.nvidia.com")

        assert isinstance(ingestor, dummy_cls)
        assert ingestor._default_caption_kwargs == _BUILD_NVIDIA_CAPTION_DEFAULTS

    def test_no_preset_leaves_defaults_empty(self, monkeypatch):
        """When inference is not provided, no defaults are passed to InProcessIngestor."""
        dummy_cls = _make_capturing_inprocess_module(monkeypatch)

        ingestor = create_ingestor(run_mode="inprocess")

        assert isinstance(ingestor, dummy_cls)
        assert ingestor._default_extract_kwargs == {}
        assert ingestor._default_embed_kwargs == {}
        assert ingestor._default_caption_kwargs == {}

    def test_unknown_preset_raises(self, monkeypatch):
        """Unknown preset name raises ValueError before the ingestor is constructed."""
        _make_capturing_inprocess_module(monkeypatch)
        with pytest.raises(ValueError, match="Unknown inference preset"):
            create_ingestor(run_mode="inprocess", inference="bad-preset")


# ---------------------------------------------------------------------------
# InProcessIngestor default kwargs merging (tested via factory layer)
# ---------------------------------------------------------------------------


class TestInProcessIngestorPresetMerging:
    """Verify that preset defaults are wired through correctly.

    These tests use a capturing DummyIngestor (same pattern as test_factory.py)
    to avoid pulling in the full batch/ray dependency stack.
    """

    def test_preset_defaults_stored_on_construction(self, monkeypatch):
        """_default_extract_kwargs and _default_embed_kwargs are stored via factory."""
        dummy_cls = _make_capturing_inprocess_module(monkeypatch)
        ingestor = create_ingestor(run_mode="inprocess", inference="build.nvidia.com")
        assert isinstance(ingestor, dummy_cls)
        assert ingestor._default_extract_kwargs == _BUILD_NVIDIA_EXTRACT_DEFAULTS
        assert ingestor._default_caption_kwargs == _BUILD_NVIDIA_CAPTION_DEFAULTS

    def test_empty_defaults_on_construction_without_preset(self, monkeypatch):
        """No defaults set when inference preset is absent."""
        dummy_cls = _make_capturing_inprocess_module(monkeypatch)
        ingestor = create_ingestor(run_mode="inprocess")
        assert isinstance(ingestor, dummy_cls)
        assert ingestor._default_extract_kwargs == {}
        assert ingestor._default_embed_kwargs == {}
        assert ingestor._default_caption_kwargs == {}
