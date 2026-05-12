# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd

from nemo_retriever.text_embed.runtime import embed_text_main_text_embed


class _FakeQueryAwareEmbedder:
    def __init__(self) -> None:
        self.embed_calls: list[list[str]] = []
        self.embed_query_calls: list[list[str]] = []

    def embed(self, texts, *, batch_size: int = 64):
        self.embed_calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    def embed_queries(self, texts, *, batch_size: int = 64):
        self.embed_query_calls.append(list(texts))
        return [[0.0, 1.0] for _ in texts]


def test_local_runtime_uses_query_embedder_method_for_query_input_type():
    model = _FakeQueryAwareEmbedder()

    result = embed_text_main_text_embed(
        pd.DataFrame({"text": ["what is the pressure altitude?"]}),
        model=model,
        input_type="query",
        inference_batch_size=4,
    )

    assert model.embed_calls == []
    assert model.embed_query_calls == [["what is the pressure altitude?"]]
    assert result.iloc[0]["metadata"]["embedding"] == [0.0, 1.0]
