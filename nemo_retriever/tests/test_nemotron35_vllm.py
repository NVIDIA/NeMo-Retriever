# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real vLLM attention selection hook without loading weights."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from nemo_retriever.models.local.nemotron35_vllm import (
    EncoderOnlyAttention,
    NRLNemotron35EmbeddingModel,
)


@pytest.mark.parametrize(
    "model_type,text_type,causal",
    [
        ("mistral3", "ministral3", True),
        ("mistral3", "ministral3", None),
        ("other", "ministral3", False),
        ("mistral3", "other", False),
    ],
)
def test_current_hook_rejects_unsupported_attention_profiles(model_type, text_type, causal):
    model = SimpleNamespace(
        config=SimpleNamespace(model_type=model_type),
        text_config=SimpleNamespace(model_type=text_type, is_causal=causal),
    )
    with pytest.raises(ValueError, match="bidirectional"):
        NRLNemotron35EmbeddingModel._get_attn_cls(model)


def test_current_hook_selects_encoder_attention_for_multimodal_checkpoint():
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="mistral3"),
        text_config=SimpleNamespace(model_type="ministral3", is_causal=False),
    )
    assert NRLNemotron35EmbeddingModel._get_attn_cls(model) is EncoderOnlyAttention


@pytest.mark.parametrize("legacy", [False, True])
def test_attention_diagnostics_handle_both_storage_layouts(legacy):
    model = NRLNemotron35EmbeddingModel.__new__(NRLNemotron35EmbeddingModel)
    torch.nn.Module.__init__(model)
    model.text_config = SimpleNamespace(is_causal=False)
    attention = EncoderOnlyAttention.__new__(EncoderOnlyAttention)
    torch.nn.Module.__init__(attention)
    if legacy:
        model.attention_instances = {0: attention}
    else:
        model.add_module("text_attention", attention)
    assert model.extra_repr() == "text_is_causal=False, attention=EncoderOnlyAttention"
