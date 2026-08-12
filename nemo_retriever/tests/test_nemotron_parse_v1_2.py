# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the local vLLM-backed Nemotron Parse model."""

import os
from unittest.mock import MagicMock, patch


def test_applies_vllm_startup_defaults_before_constructing_llm(monkeypatch):
    from nemo_retriever.models.local import nemotron_parse_v1_2 as mod

    monkeypatch.delenv("VLLM_DEEP_GEMM_WARMUP", raising=False)

    def assert_startup_defaults(**_kwargs):
        assert os.environ["VLLM_DEEP_GEMM_WARMUP"] == "skip"
        return MagicMock()

    with (
        patch.object(mod, "_patch_vllm_nemotron_parse_processor"),
        patch.object(mod, "_patch_vllm_nemotron_parse_tied_lm_head"),
        patch.object(mod, "configure_global_hf_cache_base"),
        patch.object(mod, "get_hf_revision", return_value="test-revision"),
        patch("vllm.LLM", side_effect=assert_startup_defaults) as llm,
        patch("vllm.SamplingParams"),
    ):
        model = mod.NemotronParse()

    assert mod.NemotronParseV12 is mod.NemotronParse
    assert model.model_name == "NVIDIA-Nemotron-Parse-2.0"
    assert llm.call_args.kwargs["model"] == "nvidia/NVIDIA-Nemotron-Parse-2.0"
    assert llm.call_args.kwargs["revision"] == "test-revision"


def test_tied_lm_head_patch_restores_compact_checkpoint_weight(monkeypatch):
    from types import SimpleNamespace

    from nemo_retriever.models.local import nemotron_parse_v1_2 as mod
    from vllm.model_executor.models.nemotron_parse import NemotronParseForConditionalGeneration

    loaded = []

    def fake_load_weights(_self, weights):
        loaded.extend(weights)
        return {"decoder.embed_tokens.weight"}

    monkeypatch.setattr(NemotronParseForConditionalGeneration, "load_weights", fake_load_weights)
    monkeypatch.setattr(mod, "_VLLM_TIED_LM_HEAD_PATCHED", False)
    mod._patch_vllm_nemotron_parse_tied_lm_head()

    instance = SimpleNamespace(
        config=SimpleNamespace(decoder=SimpleNamespace(tie_word_embeddings=True)),
        decoder=SimpleNamespace(embed_tokens=SimpleNamespace(weight=object())),
        lm_head=SimpleNamespace(weight=object()),
    )
    result = NemotronParseForConditionalGeneration.load_weights(
        instance,
        iter([("decoder.embed_tokens.weight", "decoder-weight")]),
    )

    assert loaded == [("decoder.embed_tokens.weight", "decoder-weight")]
    assert result == {"decoder.embed_tokens.weight"}
    assert instance.lm_head.weight is instance.decoder.embed_tokens.weight


def test_tied_lm_head_patch_preserves_materialized_weight(monkeypatch):
    from types import SimpleNamespace

    from nemo_retriever.models.local import nemotron_parse_v1_2 as mod
    from vllm.model_executor.models.nemotron_parse import NemotronParseForConditionalGeneration

    def fake_load_weights(_self, _weights):
        return set()

    monkeypatch.setattr(NemotronParseForConditionalGeneration, "load_weights", fake_load_weights)
    monkeypatch.setattr(mod, "_VLLM_TIED_LM_HEAD_PATCHED", False)
    mod._patch_vllm_nemotron_parse_tied_lm_head()

    original_head = object()
    instance = SimpleNamespace(
        config=SimpleNamespace(decoder=SimpleNamespace(tie_word_embeddings=True)),
        decoder=SimpleNamespace(embed_tokens=SimpleNamespace(weight=object())),
        lm_head=SimpleNamespace(weight=original_head),
    )
    NemotronParseForConditionalGeneration.load_weights(instance, [("lm_head.weight", "head-weight")])

    assert instance.lm_head.weight is original_head
