# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for NemotronRerankVLV2VLLM (vLLM backend).

Heavy dependencies (vllm, torch) are stubbed so no GPU or model download is
required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# NemotronRerankVLV2VLLM — construction and properties
# ---------------------------------------------------------------------------


class TestNemotronRerankVLV2VLLMProperties:
    def _make_instance(self, model_name="nvidia/llama-nemotron-rerank-vl-1b-v2"):
        from nemo_retriever.model.local import nemotron_rerank_vl_v2_vllm as mod

        with (
            patch.object(mod, "configure_global_hf_cache_base"),
            patch.object(mod, "get_hf_revision", return_value=None),
            patch("vllm.LLM") as MockLLM,
        ):
            MockLLM.return_value = MagicMock()
            obj = mod.NemotronRerankVLV2VLLM(model_name=model_name)
        return obj

    def test_model_name(self):
        obj = self._make_instance()
        assert obj.model_name == "nvidia/llama-nemotron-rerank-vl-1b-v2"

    def test_model_type(self):
        obj = self._make_instance()
        assert obj.model_type == "vl_reranker"

    def test_model_runmode(self):
        obj = self._make_instance()
        assert obj.model_runmode == "local"

    def test_input_batch_size(self):
        obj = self._make_instance()
        assert obj.input_batch_size == 32

    def test_llm_created_with_correct_kwargs(self):
        from nemo_retriever.model.local import nemotron_rerank_vl_v2_vllm as mod

        with (
            patch.object(mod, "configure_global_hf_cache_base"),
            patch.object(mod, "get_hf_revision", return_value=None),
            patch("vllm.LLM") as MockLLM,
        ):
            MockLLM.return_value = MagicMock()
            mod.NemotronRerankVLV2VLLM(
                model_name="nvidia/llama-nemotron-rerank-vl-1b-v2",
                gpu_memory_utilization=0.7,
            )
            MockLLM.assert_called_once_with(
                model="nvidia/llama-nemotron-rerank-vl-1b-v2",
                runner="pooling",
                max_model_len=10240,
                trust_remote_code=True,
                dtype="bfloat16",
                gpu_memory_utilization=0.7,
            )


# ---------------------------------------------------------------------------
# NemotronRerankVLV2VLLM — score()
# ---------------------------------------------------------------------------


class TestNemotronRerankVLV2VLLMScore:
    @pytest.fixture()
    def reranker(self):
        from nemo_retriever.model.local import nemotron_rerank_vl_v2_vllm as mod

        with (
            patch.object(mod, "configure_global_hf_cache_base"),
            patch.object(mod, "get_hf_revision", return_value=None),
            patch("vllm.LLM") as MockLLM,
        ):
            llm_inst = MockLLM.return_value
            obj = mod.NemotronRerankVLV2VLLM()
        obj._llm = llm_inst
        return obj

    def test_score_empty_documents_returns_empty(self, reranker):
        assert reranker.score("q", []) == []

    def test_score_pairs_empty_returns_empty(self, reranker):
        assert reranker.score_pairs([]) == []

    def test_score_text_only(self, reranker):
        out1 = MagicMock()
        out1.outputs.score = 3.5
        out2 = MagicMock()
        out2.outputs.score = -1.2

        reranker._llm.score.return_value = [out1, out2]

        scores = reranker.score("What is ML?", ["Machine learning is...", "Paris is..."])

        assert scores == [3.5, -1.2]
        reranker._llm.score.assert_called_once()
        call_args = reranker._llm.score.call_args
        # Query is first positional arg
        assert call_args[0][0] == "What is ML?"
        # Documents are second positional arg — plain strings for text-only
        docs = call_args[0][1]
        assert docs == ["Machine learning is...", "Paris is..."]

    def test_score_with_images(self, reranker):
        out1 = MagicMock()
        out1.outputs.score = 5.0
        out2 = MagicMock()
        out2.outputs.score = 2.0

        reranker._llm.score.return_value = [out1, out2]

        scores = reranker.score(
            "What is ML?",
            ["Machine learning is...", "Paris is..."],
            images_b64=["iVBOR_data", None],
        )

        assert scores == [5.0, 2.0]
        call_args = reranker._llm.score.call_args
        docs = call_args[0][1]
        # First doc has image → multimodal dict
        assert isinstance(docs[0], dict)
        assert docs[0]["content"][0]["type"] == "image_url"
        assert "data:image/png;base64,iVBOR_data" in docs[0]["content"][0]["image_url"]["url"]
        assert docs[0]["content"][1]["type"] == "text"
        assert docs[0]["content"][1]["text"] == "Machine learning is..."
        # Second doc has no image → plain string
        assert docs[1] == "Paris is..."

    def test_score_pairs_text_only(self, reranker):
        out1 = MagicMock()
        out1.outputs.score = 1.0
        out2 = MagicMock()
        out2.outputs.score = 2.0

        reranker._llm.score.side_effect = [[out1], [out2]]

        scores = reranker.score_pairs([("q1", "d1"), ("q2", "d2")])

        assert scores == [1.0, 2.0]
        assert reranker._llm.score.call_count == 2

    def test_score_chat_template_passed(self, reranker):
        from nemo_retriever.model.local.nemotron_rerank_vl_v2_vllm import SCORE_TEMPLATE

        out = MagicMock()
        out.outputs.score = 1.0
        reranker._llm.score.return_value = [out]

        reranker.score("q", ["doc"])

        call_kwargs = reranker._llm.score.call_args[1]
        assert call_kwargs["chat_template"] == SCORE_TEMPLATE


# ---------------------------------------------------------------------------
# _build_document helper
# ---------------------------------------------------------------------------


class TestBuildDocument:
    def test_text_only(self):
        from nemo_retriever.model.local.nemotron_rerank_vl_v2_vllm import NemotronRerankVLV2VLLM

        result = NemotronRerankVLV2VLLM._build_document("hello world")
        assert result == "hello world"

    def test_text_only_none_image(self):
        from nemo_retriever.model.local.nemotron_rerank_vl_v2_vllm import NemotronRerankVLV2VLLM

        result = NemotronRerankVLV2VLLM._build_document("hello", None)
        assert result == "hello"

    def test_text_only_empty_image(self):
        from nemo_retriever.model.local.nemotron_rerank_vl_v2_vllm import NemotronRerankVLV2VLLM

        result = NemotronRerankVLV2VLLM._build_document("hello", "")
        assert result == "hello"

    def test_with_image(self):
        from nemo_retriever.model.local.nemotron_rerank_vl_v2_vllm import NemotronRerankVLV2VLLM

        result = NemotronRerankVLV2VLLM._build_document("doc text", "abc123")
        assert isinstance(result, dict)
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "image_url"
        assert result["content"][0]["image_url"]["url"] == "data:image/png;base64,abc123"
        assert result["content"][1] == {"type": "text", "text": "doc text"}

    def test_image_only_no_text(self):
        from nemo_retriever.model.local.nemotron_rerank_vl_v2_vllm import NemotronRerankVLV2VLLM

        result = NemotronRerankVLV2VLLM._build_document("", "abc123")
        assert isinstance(result, dict)
        # Only image, no text entry
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "image_url"


# ---------------------------------------------------------------------------
# create_local_reranker dispatch
# ---------------------------------------------------------------------------


class TestCreateLocalRerankerBackend:
    def test_default_backend_dispatches_to_vllm(self):
        with patch("nemo_retriever.model.local.nemotron_rerank_vl_v2_vllm.NemotronRerankVLV2VLLM") as MockVLLM:
            MockVLLM.return_value = MagicMock()
            from nemo_retriever.model import create_local_reranker

            result = create_local_reranker("nvidia/llama-nemotron-rerank-vl-1b-v2")
            MockVLLM.assert_called_once()
            assert result is MockVLLM.return_value

    def test_vllm_backend_explicit(self):
        with patch("nemo_retriever.model.local.nemotron_rerank_vl_v2_vllm.NemotronRerankVLV2VLLM") as MockVLLM:
            MockVLLM.return_value = MagicMock()
            from nemo_retriever.model import create_local_reranker

            result = create_local_reranker("nvidia/llama-nemotron-rerank-vl-1b-v2", backend="vllm")
            MockVLLM.assert_called_once()
            assert result is MockVLLM.return_value

    def test_transformers_backend(self):
        with patch("nemo_retriever.model.local.nemotron_rerank_vl_v2.NemotronRerankVLV2") as MockTF:
            MockTF.return_value = MagicMock()
            from nemo_retriever.model import create_local_reranker

            result = create_local_reranker("nvidia/llama-nemotron-rerank-vl-1b-v2", backend="transformers")
            MockTF.assert_called_once()
            assert result is MockTF.return_value

    def test_text_only_model_ignores_backend(self):
        with patch("nemo_retriever.model.local.nemotron_rerank_v2.NemotronRerankV2") as MockText:
            MockText.return_value = MagicMock()
            from nemo_retriever.model import create_local_reranker

            result = create_local_reranker("nvidia/llama-nemotron-rerank-1b-v2", backend="vllm")
            MockText.assert_called_once()
            assert result is MockText.return_value

    def test_gpu_memory_utilization_passed_to_vllm(self):
        with patch("nemo_retriever.model.local.nemotron_rerank_vl_v2_vllm.NemotronRerankVLV2VLLM") as MockVLLM:
            MockVLLM.return_value = MagicMock()
            from nemo_retriever.model import create_local_reranker

            create_local_reranker(
                "nvidia/llama-nemotron-rerank-vl-1b-v2",
                backend="vllm",
                gpu_memory_utilization=0.8,
            )
            call_kwargs = MockVLLM.call_args[1]
            assert call_kwargs["gpu_memory_utilization"] == 0.8
