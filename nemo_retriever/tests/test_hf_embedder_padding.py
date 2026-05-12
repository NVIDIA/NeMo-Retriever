# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import torch

from nemo_retriever.model.local.llama_nemotron_embed_1b_v2_hf_embedder import LlamaNemotronEmbed1BV2HFEmbedder
from nemo_retriever.text_embed.runtime import _embed_group


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def encode(self, text, *, add_special_tokens=True, truncation=False):
        extra = 2 if add_special_tokens else 0
        return list(range(len(str(text).split()) + extra))

    def __call__(self, texts, *, padding, truncation, max_length, return_tensors):
        self.calls.append(
            {
                "texts": list(texts),
                "padding": padding,
                "truncation": truncation,
                "max_length": max_length,
                "return_tensors": return_tensors,
            }
        )
        seq_len = int(max_length) if padding == "max_length" else max(len(str(text).split()) for text in texts)
        batch_size = len(texts)
        return {
            "input_ids": torch.ones((batch_size, seq_len), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
        }


class _FakeModel:
    def __call__(self, **batch):
        input_ids = batch["input_ids"]
        hidden = torch.ones((input_ids.shape[0], input_ids.shape[1], 2), dtype=torch.float32)
        return SimpleNamespace(last_hidden_state=hidden)


def _make_loaded_embedder() -> tuple[LlamaNemotronEmbed1BV2HFEmbedder, _FakeTokenizer]:
    tokenizer = _FakeTokenizer()
    embedder = LlamaNemotronEmbed1BV2HFEmbedder()
    embedder._tokenizer = tokenizer
    embedder._model = _FakeModel()
    embedder._device = torch.device("cpu")
    embedder._ensure_loaded = lambda: None  # type: ignore[method-assign]
    return embedder, tokenizer


def test_hf_query_embedder_uses_fixed_padding() -> None:
    embedder, tokenizer = _make_loaded_embedder()

    embedder.embed_queries(["short", "a slightly longer query"], batch_size=1)

    assert [call["padding"] for call in tokenizer.calls] == ["max_length", "max_length"]
    assert [call["max_length"] for call in tokenizer.calls] == [128, 128]


def test_hf_query_embedder_warns_when_query_truncated(caplog) -> None:
    embedder, _tokenizer = _make_loaded_embedder()
    embedder.query_max_length = 3

    with caplog.at_level("WARNING", logger="nemo_retriever.model.local.llama_nemotron_embed_1b_v2_hf_embedder"):
        embedder.embed_queries(["one two three four"], batch_size=1)

    assert "Truncating 1/1 HF query embeddings to query_max_length=3 tokens" in caplog.text


def test_hf_passage_embedder_keeps_dynamic_padding() -> None:
    embedder, tokenizer = _make_loaded_embedder()

    embedder.embed(["short", "a slightly longer passage"], batch_size=1)

    assert [call["padding"] for call in tokenizer.calls] == [True, True]
    assert [call["max_length"] for call in tokenizer.calls] == [8192, 8192]


class _FakeTextModel:
    def __init__(self) -> None:
        self.embed_calls: list[list[str]] = []
        self.embed_query_calls: list[list[str]] = []

    def embed(self, texts, *, batch_size):
        self.embed_calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    def embed_queries(self, texts, *, batch_size):
        self.embed_query_calls.append(list(texts))
        return [[0.0, 1.0] for _ in texts]


def test_runtime_local_query_embedding_uses_embed_queries() -> None:
    model = _FakeTextModel()

    _embed_group(
        pd.DataFrame({"text": ["what is revenue?"]}),
        group_modality="text",
        model=model,
        endpoint=None,
        api_key=None,
        text_column="text",
        inference_batch_size=8,
        output_column="embedding",
        resolved_model_name="nvidia/llama-nemotron-embed-1b-v2",
        input_type="query",
    )

    assert model.embed_query_calls == [["what is revenue?"]]
    assert model.embed_calls == []


def test_runtime_local_passage_embedding_uses_embed() -> None:
    model = _FakeTextModel()

    _embed_group(
        pd.DataFrame({"text": ["annual revenue"]}),
        group_modality="text",
        model=model,
        endpoint=None,
        api_key=None,
        text_column="text",
        inference_batch_size=8,
        output_column="embedding",
        resolved_model_name="nvidia/llama-nemotron-embed-1b-v2",
        input_type="passage",
    )

    assert model.embed_calls == [["annual revenue"]]
    assert model.embed_query_calls == []
