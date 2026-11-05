# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Reranker attention selection for the current vLLM construction hook."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

from nemo_retriever.models.local.nemotron35_rerank_vllm import (
    EncoderOnlyAttention,
    NRLNemotron35RerankModel,
)


def _profile():
    return SimpleNamespace(
        config=SimpleNamespace(model_type="mistral3"),
        text_config=SimpleNamespace(model_type="ministral3", is_causal=False, pooling="avg"),
    )


def test_current_hook_uses_bidirectional_attention():
    assert NRLNemotron35RerankModel._get_attn_cls(_profile()) is EncoderOnlyAttention


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("config", "model_type", "llama"),
        ("text_config", "model_type", "llama"),
        ("text_config", "is_causal", True),
        ("text_config", "pooling", "last"),
    ],
)
def test_current_hook_rejects_incompatible_checkpoint(target, field, value):
    model = _profile()
    setattr(getattr(model, target), field, value)
    with pytest.raises(ValueError):
        NRLNemotron35RerankModel._get_attn_cls(model)
