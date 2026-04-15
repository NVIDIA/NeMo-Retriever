# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_retriever.text_embed.vllm (no vLLM install required)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from nemo_retriever.text_embed.vllm import embed_with_vllm_llm


def _make_output(embedding):
    """Build a fake vLLM EmbeddingRequestOutput with out.outputs.embedding."""
    return SimpleNamespace(outputs=SimpleNamespace(embedding=embedding))


class TestEmbedWithVllmLlm:
    def test_well_formed_list_output(self):
        llm = MagicMock()
        llm.embed.return_value = [_make_output([0.1, 0.2, 0.3])]
        result = embed_with_vllm_llm(["hello"], llm)
        assert result == [[0.1, 0.2, 0.3]]

    def test_well_formed_tolist_output(self):
        """Embedding returned as a numpy-style object with .tolist()."""
        import array

        emb = array.array("f", [0.1, 0.2])
        llm = MagicMock()
        llm.embed.return_value = [_make_output(emb)]
        result = embed_with_vllm_llm(["hi"], llm)
        assert len(result) == 1
        assert len(result[0]) == 2

    def test_missing_embedding_returns_empty_list(self):
        llm = MagicMock()
        llm.embed.return_value = [SimpleNamespace(outputs=SimpleNamespace(embedding=None))]
        result = embed_with_vllm_llm(["oops"], llm)
        assert result == [[]]

    def test_prefix_prepended(self):
        llm = MagicMock()
        llm.embed.return_value = [_make_output([0.0])]
        embed_with_vllm_llm(["world"], llm, prefix="query: ")
        called_batch = llm.embed.call_args[0][0]
        assert called_batch == ["query: world"]

    def test_empty_prompts_early_return(self):
        llm = MagicMock()
        result = embed_with_vllm_llm([], llm)
        llm.embed.assert_not_called()
        assert result == []

    def test_batching(self):
        """Verifies batch_size splits calls correctly."""
        llm = MagicMock()
        llm.embed.side_effect = lambda batch: [_make_output([float(i)]) for i in range(len(batch))]
        result = embed_with_vllm_llm(["a", "b", "c"], llm, batch_size=2)
        assert llm.embed.call_count == 2
        assert len(result) == 3
