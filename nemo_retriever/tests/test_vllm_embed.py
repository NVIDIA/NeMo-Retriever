# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for nemo_retriever.text_embed.vllm (no vLLM install required)."""

import base64
import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

from nemo_retriever.text_embed.vllm import embed_multimodal_with_vllm_llm, embed_with_vllm_llm
from nemo_retriever.model.local.llama_nemotron_embed_1b_v2_embedder import LlamaNemotronEmbed1BV2Embedder


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


def _make_minimal_b64() -> str:
    """Return a minimal valid base64-encoded 1x1 pixel PNG."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (1, 1), color=(128, 128, 128)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_vllm_vl_embedder():
    """Instantiate LlamaNemotronEmbedVL1BV2VLLMEmbedder without GPU init."""
    from nemo_retriever.model.local.llama_nemotron_embed_vl_1b_v2_embedder import (
        LlamaNemotronEmbedVL1BV2VLLMEmbedder,
    )

    with patch.object(LlamaNemotronEmbedVL1BV2VLLMEmbedder, "__post_init__", lambda self: None):
        embedder = LlamaNemotronEmbedVL1BV2VLLMEmbedder()
    embedder._llm = MagicMock()
    return embedder


class TestEmbedMultimodalWithVllmLlm:
    def test_basic_prompt_dict(self):
        llm = MagicMock()
        llm.embed.return_value = [_make_output([0.1, 0.2, 0.3])]
        result = embed_multimodal_with_vllm_llm(
            [{"prompt": "passage: <image> ", "multi_modal_data": {"image": MagicMock()}}], llm
        )
        assert result == [[0.1, 0.2, 0.3]]

    def test_empty_list_early_return(self):
        llm = MagicMock()
        result = embed_multimodal_with_vllm_llm([], llm)
        llm.embed.assert_not_called()
        assert result == []

    def test_none_embedding_returns_empty_slot(self):
        llm = MagicMock()
        llm.embed.return_value = [SimpleNamespace(outputs=SimpleNamespace(embedding=None))]
        result = embed_multimodal_with_vllm_llm(
            [{"prompt": "passage: <image> ", "multi_modal_data": {"image": MagicMock()}}], llm
        )
        assert result == [[]]

    def test_batching_splits_calls(self):
        llm = MagicMock()
        llm.embed.side_effect = lambda batch: [_make_output([0.0]) for _ in batch]
        items = [{"prompt": "passage: <image> ", "multi_modal_data": {"image": MagicMock()}}] * 3
        embed_multimodal_with_vllm_llm(items, llm, batch_size=1)
        assert llm.embed.call_count == 3

    def test_tolist_output_path(self):
        import array

        emb = array.array("f", [0.5, 0.6])
        llm = MagicMock()
        llm.embed.return_value = [_make_output(emb)]
        result = embed_multimodal_with_vllm_llm(
            [{"prompt": "passage: <image> ", "multi_modal_data": {"image": MagicMock()}}], llm
        )
        assert len(result) == 1
        assert len(result[0]) == 2


class TestCreateVllmLlm:
    def setup_method(self):
        pytest.importorskip("vllm", reason="vLLM not installed")

    def test_limit_mm_per_prompt_absent_by_default(self):
        # LLM is imported inside create_vllm_llm's body, so patch at its source
        with patch("vllm.LLM") as mock_llm_cls:
            mock_llm_cls.return_value = MagicMock()
            from nemo_retriever.text_embed.vllm import create_vllm_llm

            create_vllm_llm("some-model")
        _, kwargs = mock_llm_cls.call_args
        assert "limit_mm_per_prompt" not in kwargs

    def test_limit_mm_per_prompt_forwarded_when_provided(self):
        with patch("vllm.LLM") as mock_llm_cls:
            mock_llm_cls.return_value = MagicMock()
            from nemo_retriever.text_embed.vllm import create_vllm_llm

            create_vllm_llm("some-model", limit_mm_per_prompt={"image": 1})
        _, kwargs = mock_llm_cls.call_args
        assert kwargs.get("limit_mm_per_prompt") == {"image": 1}


class TestVLLMEmbedderImages:
    def setup_method(self):
        self.embedder = _make_vllm_vl_embedder()

    def test_empty_input_returns_empty_tensor(self):
        result = self.embedder.embed_images([])
        assert isinstance(result, torch.Tensor)
        assert result.shape == (0, 2048)

    def test_all_blank_b64_returns_empty_tensor(self):
        result = self.embedder.embed_images(["", "   "])
        assert result.shape == (0, 2048)

    def test_calls_multimodal_helper(self):
        b64 = _make_minimal_b64()
        with patch("nemo_retriever.text_embed.vllm.embed_multimodal_with_vllm_llm") as mock_mm:
            mock_mm.return_value = [[0.6, 0.8]]
            self.embedder.embed_images([b64])
        mock_mm.assert_called_once()
        prompt_dicts = mock_mm.call_args[0][0]
        assert len(prompt_dicts) == 1
        assert "prompt" in prompt_dicts[0]
        assert "multi_modal_data" in prompt_dicts[0]
        assert "image" in prompt_dicts[0]["multi_modal_data"]

    def test_prompt_contains_image_token_and_prefix(self):
        b64 = _make_minimal_b64()
        captured = []
        with patch(
            "nemo_retriever.text_embed.vllm.embed_multimodal_with_vllm_llm",
            side_effect=lambda dicts, llm, **kw: captured.extend(dicts) or [[0.1, 0.2]],
        ):
            self.embedder.embed_images([b64])
        assert "passage:" in captured[0]["prompt"]
        assert "<image>" in captured[0]["prompt"]

    def test_output_is_l2_normalized(self):
        b64 = _make_minimal_b64()
        with patch(
            "nemo_retriever.text_embed.vllm.embed_multimodal_with_vllm_llm",
            return_value=[[3.0, 4.0]],
        ):
            result = self.embedder.embed_images([b64])
        assert result.shape == (1, 2)
        assert abs(float(torch.norm(result, dim=-1).item()) - 1.0) < 1e-5

    def test_no_valid_embeddings_returns_empty_tensor(self):
        b64 = _make_minimal_b64()
        with patch(
            "nemo_retriever.text_embed.vllm.embed_multimodal_with_vllm_llm",
            return_value=[[]],
        ):
            result = self.embedder.embed_images([b64])
        assert result.shape[0] == 0


def _make_text_embedder():
    with patch.object(LlamaNemotronEmbed1BV2Embedder, "__post_init__", lambda self: None):
        embedder = LlamaNemotronEmbed1BV2Embedder()
    embedder._llm = MagicMock()
    return embedder


class TestLlamaNemotronEmbed1BV2Embedder:
    def setup_method(self):
        self.embedder = _make_text_embedder()

    def test_finalize_vectors_all_empty_returns_empty_tensor(self):
        result = self.embedder._finalize_vectors([[], []])
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 0

    def test_finalize_vectors_zero_pads_missing(self):
        result = self.embedder._finalize_vectors([[1.0, 0.0], []])
        assert result.shape == (2, 2)
        assert result[1].tolist() == [0.0, 0.0]

    def test_embed_uses_passage_prefix_by_default(self):
        with patch("nemo_retriever.text_embed.vllm.embed_with_vllm_llm", return_value=[[0.6, 0.8]]) as mock_fn:
            self.embedder.embed(["hello"])
        assert mock_fn.call_args[1].get("prefix") == "passage: "

    def test_embed_queries_uses_query_prefix(self):
        with patch("nemo_retriever.text_embed.vllm.embed_with_vllm_llm", return_value=[[0.6, 0.8]]) as mock_fn:
            self.embedder.embed_queries(["hello"])
        assert mock_fn.call_args[1].get("prefix") == "query: "

    def test_embed_empty_input_returns_empty_tensor(self):
        result = self.embedder.embed(["", "  "])
        assert result.shape == (0, 0)

    def test_unload_clears_llm(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.embedder.unload()
        assert self.embedder._llm is None


class TestLlamaNemotronEmbed1BV2EmbedderNormalization:
    def test_output_is_l2_normalized_by_default(self):
        embedder = _make_text_embedder()
        with patch("nemo_retriever.text_embed.vllm.embed_with_vllm_llm", return_value=[[3.0, 4.0]]):
            result = embedder.embed(["text"])
        assert abs(float(torch.norm(result, dim=-1).item()) - 1.0) < 1e-5

    def test_output_unnormalized_when_normalize_false(self):
        embedder = _make_text_embedder()
        embedder.normalize = False
        with patch("nemo_retriever.text_embed.vllm.embed_with_vllm_llm", return_value=[[3.0, 4.0]]):
            result = embedder.embed(["text"])
        assert abs(float(result[0][0].item()) - 3.0) < 1e-5


class TestVLLMEmbedderTextImage:
    def setup_method(self):
        self.embedder = _make_vllm_vl_embedder()

    def test_empty_images_returns_empty_tensor(self):
        result = self.embedder.embed_text_image(["text"], [""])
        assert result.shape == (0, 2048)

    def test_all_blank_b64_returns_empty_tensor(self):
        result = self.embedder.embed_text_image(["a", "b"], ["", "   "])
        assert result.shape == (0, 2048)

    def test_calls_multimodal_helper(self):
        b64 = _make_minimal_b64()
        with patch(
            "nemo_retriever.text_embed.vllm.embed_multimodal_with_vllm_llm",
            return_value=[[0.6, 0.8]],
        ) as mock_mm:
            self.embedder.embed_text_image(["hello"], [b64])
        mock_mm.assert_called_once()

    def test_prompt_contains_image_token_and_text(self):
        b64 = _make_minimal_b64()
        captured = []
        with patch(
            "nemo_retriever.text_embed.vllm.embed_multimodal_with_vllm_llm",
            side_effect=lambda dicts, llm, **kw: captured.extend(dicts) or [[0.1, 0.2]],
        ):
            self.embedder.embed_text_image(["my document text"], [b64])
        assert "<image>" in captured[0]["prompt"]
        assert "my document text" in captured[0]["prompt"]

    def test_empty_b64_rows_filtered(self):
        b64 = _make_minimal_b64()
        captured = []
        with patch(
            "nemo_retriever.text_embed.vllm.embed_multimodal_with_vllm_llm",
            side_effect=lambda dicts, llm, **kw: captured.extend(dicts) or [[0.1, 0.2]],
        ):
            self.embedder.embed_text_image(["text a", "text b"], [b64, ""])
        assert len(captured) == 1

    def test_output_is_l2_normalized(self):
        b64 = _make_minimal_b64()
        with patch(
            "nemo_retriever.text_embed.vllm.embed_multimodal_with_vllm_llm",
            return_value=[[3.0, 4.0]],
        ):
            result = self.embedder.embed_text_image(["text"], [b64])
        assert result.shape == (1, 2)
        assert abs(float(torch.norm(result, dim=-1).item()) - 1.0) < 1e-5
