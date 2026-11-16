# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import io
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
from PIL import Image

from nemo_retriever.models import _create_local_embedder_from_spec
from nemo_retriever.models.embed_model_spec import EmbedModelSpec
from nemo_retriever.models.inference import vllm
from nemo_retriever.models.local.embedding_gemma2_embedder import EmbeddingGemma2Embedder


@pytest.fixture
def image_b64():
    with Image.new("RGB", (32, 32), "red") as image, io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def embedder():
    model = EmbeddingGemma2Embedder(
        model_id="google/embeddinggemma-2",
        revision="f9972ac22e696a105c482202646979e33e4b5a1e",
        query_prefix="task: search result | query: ",
        document_prefix="title: none | text: ",
    )
    model._llm = object()
    model._processor = MagicMock(return_value={"attention_mask": torch.ones(1, 300)})
    return model


@pytest.mark.parametrize(
    "modality,text,expected_prompt",
    [
        ("text_image", "Page text", "title: none | text: Page text <|image|>"),
        ("text_image", "", "title: none | text: <|image|>"),
        ("image", "", "title: none | text: <|image|>"),
    ],
)
def test_image_document_format(embedder, image_b64, monkeypatch, modality, text, expected_prompt):
    def embed(requests, llm, **kwargs):
        assert llm is embedder._llm
        assert kwargs == {"batch_size": 7, "normalize": True}
        assert len(requests) == 1
        assert requests[0]["prompt"] == expected_prompt
        image = requests[0]["multi_modal_data"]["image"]
        assert image.mode == "RGB"
        assert image.size == (32, 32)
        assert embedder._processor.call_args.kwargs["text"] == [requests[0]["prompt"]]
        assert embedder._processor.call_args.kwargs["truncation"] is False
        return [[3.0, 4.0]]

    monkeypatch.setattr(vllm, "embed_multimodal_with_vllm_llm", embed)
    if modality == "image":
        result = embedder.embed_images([image_b64], batch_size=7)
    else:
        result = embedder.embed_text_image([text], [image_b64], batch_size=7)
    assert torch.allclose(result, torch.tensor([[0.6, 0.8]]))


def test_text_and_query_formats_are_unchanged(embedder, monkeypatch):
    infer = MagicMock(return_value=[[1.0, 0.0]])
    monkeypatch.setattr(vllm, "embed_with_vllm_llm", infer)
    embedder.embed(["document"])
    assert infer.call_args.args[0] == ["document"]
    assert infer.call_args.kwargs["prefix"] == "title: none | text: "
    embedder.embed_queries(["question"])
    assert infer.call_args.args[0] == ["question"]
    assert infer.call_args.kwargs["prefix"] == "task: search result | query: "
    embedder._processor.assert_not_called()


def test_empty_image_inputs_do_not_load(embedder):
    embedder._llm = None
    assert embedder.embed_images([]).shape == (0, 768)
    assert embedder.embed_text_image([], []).shape == (0, 768)
    embedder._processor.assert_not_called()


def test_image_count_mismatch_is_rejected(embedder, image_b64):
    with pytest.raises(ValueError, match="counts must match"):
        embedder.embed_text_image([], [image_b64])


def test_invalid_image_is_rejected(embedder):
    with pytest.raises(ValueError, match="Failed to decode base64 image"):
        embedder.embed_images(["not an image"])


def test_expanded_image_budget_is_enforced(embedder, image_b64, monkeypatch):
    embedder._processor.return_value = {"attention_mask": torch.ones(1, 8193)}
    infer = MagicMock()
    monkeypatch.setattr(vllm, "embed_multimodal_with_vllm_llm", infer)
    with pytest.raises(ValueError, match="exceeds max_length after image expansion"):
        embedder.embed_text_image(["page"], [image_b64])
    infer.assert_not_called()


@pytest.mark.parametrize("vectors", [[], [[]]])
def test_missing_embeddings_are_rejected(embedder, image_b64, monkeypatch, vectors):
    monkeypatch.setattr(vllm, "embed_multimodal_with_vllm_llm", MagicMock(return_value=vectors))
    with pytest.raises(RuntimeError, match="embedding"):
        embedder.embed_images([image_b64])


def test_factory_selects_gemma2_adapter():
    spec = EmbedModelSpec(
        model_id="google/embeddinggemma-2",
        revision="a" * 40,
        family="vl",
        output_dimension=768,
        query_prefix="query",
        document_prefix="document",
        requires_vllm=True,
        max_input_tokens=8192,
        model_type="embedding_gemma2",
    )
    model = _create_local_embedder_from_spec(
        spec,
        backend="vllm",
        device=None,
        hf_cache_dir=None,
        gpu_memory_utilization=0.45,
        enforce_eager=False,
        dimensions=None,
        normalize=True,
        max_length=16384,
        query_max_length=128,
    )
    assert isinstance(model, EmbeddingGemma2Embedder)
    assert model.max_length == 8192
    assert model.document_prefix == "document"


def test_engine_uses_native_multimodal_configuration(embedder, monkeypatch):
    embedder._llm = None
    create = MagicMock()
    monkeypatch.setattr(vllm, "create_vllm_llm", create)
    embedder._ensure_loaded()
    assert create.call_args.kwargs["limit_mm_per_prompt"] == {"image": 1, "audio": 0, "video": 0}
    assert create.call_args.kwargs["max_model_len"] == 8192
    assert create.call_args.kwargs["attention_backend"] == "TRITON_ATTN"
    assert create.call_args.kwargs["revision"] == embedder.revision


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


def test_paired_admission_splits_losslessly(embedder, image_b64):
    embedder.max_length = 40
    embedder._processor.tokenizer = CharacterTokenizer()

    def process(text, **kwargs):
        assert text[0].startswith("title: none | text: ")
        assert text[0].endswith(" <|image|>")
        assert text[0].count("<|image|>") == 1
        return {"attention_mask": torch.ones(1, len(text[0]) + 5)}

    embedder._processor.side_effect = process
    text = "A long page of text with whitespace preserved. " * 2
    plan = embedder.plan_text_image(text, image_b64)
    assert plan.requires_split
    assert "".join(child.content for child in plan.children) == text
    assert all(len(embedder._image_prompt(child.content)) + 5 <= 40 for child in plan.children)


@pytest.mark.parametrize("text", ["", "ab", "abc"])
def test_paired_admission_counts_exact_suffix_prompt(embedder, image_b64, text):
    embedder.max_length = 38
    embedder._processor.tokenizer = CharacterTokenizer()
    embedder._processor.side_effect = lambda text, **kwargs: {"attention_mask": torch.ones(1, len(text[0]) + 5)}
    plan = embedder.plan_text_image(text, image_b64)
    assert not plan.requires_split
    assert plan.formatted_tokens == len(embedder._image_prompt(text)) + 5
    assert embedder._processor.call_args.kwargs["text"] == [embedder._image_prompt(text)]


def test_paired_admission_rejects_image_without_text_budget(embedder, image_b64):
    embedder.max_length = 300
    with pytest.raises(ValueError, match="leave no text budget"):
        embedder.plan_text_image("text", image_b64)


def test_image_queries_are_rejected(embedder, image_b64):
    with pytest.raises(ValueError, match="queries must be text-only"):
        embedder.plan_text_image("query", image_b64, query=True)


@pytest.mark.parametrize("modality", ["image", "text_image"])
def test_page_rows_use_multimodal_runtime(embedder, image_b64, monkeypatch, modality):
    import pandas as pd

    from nemo_retriever.models.inference.runtime import embed_text_main_text_embed

    infer = MagicMock(return_value=[[3.0, 4.0]])
    monkeypatch.setattr(vllm, "embed_multimodal_with_vllm_llm", infer)
    frame = pd.DataFrame([{"text": "Page text", "_image_b64": image_b64, "page_number": 1, "metadata": {}}])
    result = embed_text_main_text_embed(frame, model=embedder, model_name=embedder.model_id, embed_modality=modality)
    assert len(result) == 1
    assert result["embedding"].notna().all()
    assert result["page_number"].tolist() == [1]
    prompt = infer.call_args.args[0][0]["prompt"]
    assert prompt == embedder._image_prompt("Page text" if modality == "text_image" else "")
