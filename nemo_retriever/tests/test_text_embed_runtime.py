# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd
import pytest

from nemo_retriever.text_embed import runtime


@pytest.fixture(autouse=True)
def _clear_reported_errors():
    from nemo_retriever.nim.error_reporter import drain_errors

    drain_errors()
    yield
    drain_errors()


class _FailingEmbedder:
    def embed(self, texts, batch_size: int):
        raise RuntimeError("backend unavailable")


class _EmptyEmbedder:
    def embed(self, texts, batch_size: int):
        return [[] for _ in texts]


def _text_df() -> pd.DataFrame:
    return pd.DataFrame({"text": ["hello"], "metadata": [{}]})


def test_local_embed_failure_raises_instead_of_returning_empty_rows():
    with pytest.raises(RuntimeError, match="backend unavailable"):
        runtime.embed_text_main_text_embed(_text_df(), model=_FailingEmbedder())


def test_local_empty_embedding_result_is_reported_without_breaking_batch():
    result = runtime.embed_text_main_text_embed(_text_df(), model=_EmptyEmbedder())

    assert result.iloc[0]["text_embeddings_1b_v2_dim"] == 0
    assert not bool(result.iloc[0]["text_embeddings_1b_v2_has_embedding"])


def test_remote_embed_failure_preserves_error_payload(monkeypatch):
    def _raise_embed_group(*args, **kwargs):
        raise RuntimeError("remote unavailable")

    monkeypatch.setattr(runtime, "_embed_group", _raise_embed_group)

    result = runtime.embed_text_main_text_embed(_text_df(), embedding_endpoint="http://embed.example/v1")

    assert result.iloc[0]["text_embeddings_1b_v2"] == {
        "embedding": [],
        "error": "remote unavailable",
    }
    assert result.iloc[0]["text_embeddings_1b_v2_dim"] == 0
    assert not bool(result.iloc[0]["text_embeddings_1b_v2_has_embedding"])
