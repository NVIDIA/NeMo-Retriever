# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import re
from typing import Sequence
from unittest.mock import Mock, patch

import httpx
import numpy as np
import pandas as pd
import pytest
from nemo_retriever.common.params import EmbedParams, ModelRuntimeParams
from nemo_retriever.common.vdb.records import to_client_vdb_records
from nemo_retriever.models.inference.embedding_input import (
    EmbeddingInputPolicy,
    prepare_embedding_inputs,
    resolve_embedding_input_policy,
)
from nemo_retriever.models.inference.runtime import embed_text_main_text_embed
from nemo_retriever.models.local.llama_nemotron_embed_1b_v2_embedder import LlamaNemotronEmbed1BV2Embedder


class _LengthEmbedder:
    def embed(self, texts: Sequence[str], *, batch_size: int):
        return [[float(len(text))] for text in texts]


class _AlwaysFailsEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str], *, batch_size: int):
        self.calls.append(list(texts))
        raise RuntimeError("echoed document: rejected input")


class _RecordingEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.call_batch_sizes: list[int] = []

    def embed(self, texts: Sequence[str], *, batch_size: int):
        self.calls.append(list(texts))
        self.call_batch_sizes.append(batch_size)
        return [[float(len(text))] for text in texts]


class _RecordingMultimodalEmbedder(_RecordingEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.image_calls: list[list[str]] = []

    def embed_queries(self, texts: Sequence[str], *, batch_size: int):
        return [[float(len(text))] for text in texts]

    def embed_images(self, images: Sequence[str], *, batch_size: int):
        self.image_calls.append(list(images))
        return [[float(len(image))] for image in images]


class _WhitespaceTokenizer:
    def __init__(self) -> None:
        self._tokens: dict[str, int] = {}
        self._ids: dict[int, str] = {}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for token in re.findall(r"\S+\s*|\s+", text):
            if token not in self._tokens:
                token_id = len(self._tokens) + 1
                self._tokens[token] = token_id
                self._ids[token_id] = token
            ids.append(self._tokens[token])
        return ([-1] + ids) if add_special_tokens else ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return "".join(self._ids[token_id] for token_id in token_ids if token_id != -1)


class _NormalizingWhitespaceTokenizer(_WhitespaceTokenizer):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return super().encode(" ".join(text.split()), add_special_tokens=add_special_tokens)


class _BatchWhitespaceTokenizer(_WhitespaceTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.batch_calls = 0

    def encode_batch(self, texts: list[str], *, add_special_tokens: bool = False) -> list[list[int]]:
        self.batch_calls += 1
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]


class _NeverCalledTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        raise AssertionError("non-text batches must not be tokenized")

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        raise AssertionError("non-text batches must not be decoded")


class _CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids = [ord(character) for character in text]
        return ([-1] + ids) if add_special_tokens else ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return "".join(chr(token_id) for token_id in token_ids if token_id != -1)


class _LiteralSpecialTokenizer:
    _literal = "<SPECIAL>"
    _literal_id = -2

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        while text:
            if text.startswith(self._literal):
                ids.append(self._literal_id)
                text = text[len(self._literal) :]
            else:
                ids.append(ord(text[0]))
                text = text[1:]
        return ([-1] + ids) if add_special_tokens else ids

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        parts = []
        for token_id in token_ids:
            if token_id == self._literal_id:
                if not skip_special_tokens:
                    parts.append(self._literal)
            elif token_id != -1:
                parts.append(chr(token_id))
        return "".join(parts)


def _write_local_text_policy_metadata(tmp_path, *, prompts=None, max_input_tokens=None) -> None:
    (tmp_path / "config.json").write_text(
        '{"model_type":"llama_bidirec","architectures":["LlamaBidirectionalModel"],'
        '"hidden_size":2048,"pooling":"avg"}',
        encoding="utf-8",
    )
    if prompts is not None:
        (tmp_path / "config_sentence_transformers.json").write_text(
            json.dumps({"prompts": prompts}),
            encoding="utf-8",
        )
    if max_input_tokens is not None:
        (tmp_path / "sentence_bert_config.json").write_text(
            json.dumps({"max_seq_length": max_input_tokens}),
            encoding="utf-8",
        )


def test_systemic_failure_is_not_replayed_or_exposed(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="nemo_retriever.models.inference.runtime")
    embedder = _AlwaysFailsEmbedder()

    result = embed_text_main_text_embed(
        pd.DataFrame({"text": ["one", "two"]}),
        model=embedder,
    )

    assert embedder.calls == [["passage: one", "passage: two"]]
    assert result["embedding_v1_counts_by_label"].tolist() == [
        {"unembedded": 1, "failed": 1},
        {"unembedded": 1, "failed": 1},
    ]
    assert result.loc[0, "text_embeddings_1b_v2"]["error"] == (
        "RuntimeError: embedding batch failed; inspect embed-stage logs for the cause"
    )
    assert "failed=2 embedded=0 unembedded=2" in caplog.text


def test_oversized_middle_row_is_split_below_the_formatted_model_limit(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="nemo_retriever.models.inference.runtime")
    source = pd.DataFrame(
        {
            "text": ["before", "one two three four five", "after"],
            "path": ["doc.pdf"] * 3,
            "page_number": [2, 7, 9],
            "_bbox_xyxy_norm": [None, np.array([0.1, 0.2, 0.8, 0.9]), None],
            "metadata": [
                {"chunk_index": 0},
                {"chunk_index": 4, "element": {"type": "table"}},
                {"chunk_index": 8},
            ],
        }
    )
    policy = EmbeddingInputPolicy(
        tokenizer=_WhitespaceTokenizer(),
        max_tokens=4,
        prefix="passage: ",
    )

    result = embed_text_main_text_embed(
        source,
        model=_LengthEmbedder(),
        inference_batch_size=3,
        embedding_input_policy=policy,
    )

    assert result["text"].tolist() == [
        "before",
        "one two ",
        "three four ",
        "five",
        "after",
    ]
    assert result["page_number"].tolist() == [2, 7, 7, 7, 9]
    assert result["text_embeddings_1b_v2_has_embedding"].tolist() == [True] * 5

    child_metadata = result.loc[1:3, "metadata"].tolist()
    assert [metadata["chunk_index"] for metadata in child_metadata] == [4, 4, 4]
    split_metadata = [metadata["embedding_split"] for metadata in child_metadata]
    assert [metadata["chunk_index"] for metadata in split_metadata] == [
        0,
        1,
        2,
    ]
    assert [metadata["chunk_count"] for metadata in split_metadata] == [
        3,
        3,
        3,
    ]
    assert [(metadata["start_token"], metadata["end_token"]) for metadata in split_metadata] == [
        (0, 2),
        (2, 4),
        (4, 5),
    ]
    assert [metadata["element"] for metadata in child_metadata] == [{"type": "table"}] * 3
    assert len({metadata["chunk_id"] for metadata in split_metadata}) == 3
    assert len({metadata["parent_id"] for metadata in split_metadata}) == 1
    assert [bbox.tolist() for bbox in result.loc[1:3, "_bbox_xyxy_norm"]] == [[0.1, 0.2, 0.8, 0.9]] * 3

    assert result["embedding_v1_counts_by_label"].tolist() == [
        {"embedded": 1},
        {"embedded": 1, "overlength": 1, "split": 1, "split_child": 1},
        {"embedded": 1, "split_child": 1},
        {"embedded": 1, "split_child": 1},
        {"embedded": 1},
    ]
    assert (
        "Embedding summary: input_rows=3 output_rows=5 overlength=1 split=1 truncated=0 "
        "failed=0 embedded=5 unembedded=0"
    ) in caplog.text
    assert any(record.levelno == logging.WARNING and "overlength=1" in record.message for record in caplog.records)


def test_policy_measures_and_preserves_leading_and_trailing_whitespace() -> None:
    policy = EmbeddingInputPolicy(tokenizer=_CharacterTokenizer(), max_tokens=5, prefix="p")

    result = prepare_embedding_inputs(pd.DataFrame({"text": [" ab "]}), policy=policy).frame

    assert result["text"].tolist() == [" ab", " "]
    assert "".join(result["text"]) == " ab "
    assert [
        (metadata["embedding_split"]["start_token"], metadata["embedding_split"]["end_token"])
        for metadata in result["metadata"]
    ] == [(0, 3), (3, 4)]


def test_split_children_reach_text_embedder_without_whitespace_normalization() -> None:
    policy = EmbeddingInputPolicy(tokenizer=_CharacterTokenizer(), max_tokens=5, prefix="p")
    embedder = _RecordingEmbedder()

    result = embed_text_main_text_embed(
        pd.DataFrame({"text": [" x ", "ab  cd"]}),
        model=embedder,
        embedding_input_policy=policy,
    )

    assert result["text"].tolist() == [" x ", "ab ", " cd"]
    assert embedder.calls == [["passage: x", "passage: ab ", "passage:  cd"]]
    assert result["text_embeddings_1b_v2_has_embedding"].tolist() == [True, True, True]


@pytest.mark.parametrize(("input_type", "prefix"), [("passage", "p"), ("query", "q")])
def test_whitespace_only_split_child_preserves_local_embedder_cardinality(
    monkeypatch: pytest.MonkeyPatch,
    input_type: str,
    prefix: str,
) -> None:
    policy = EmbeddingInputPolicy(tokenizer=_CharacterTokenizer(), max_tokens=5, prefix=prefix)
    embedder = LlamaNemotronEmbed1BV2Embedder()
    embedder._llm = object()
    model_inputs: list[str] = []

    def embed_with_vllm(texts, _model, *, batch_size, prefix, normalize):
        model_inputs.extend(texts)
        return [[3.0, 4.0] for _ in texts]

    monkeypatch.setattr(
        "nemo_retriever.models.inference.vllm.embed_with_vllm_llm",
        embed_with_vllm,
    )

    result = embed_text_main_text_embed(
        pd.DataFrame({"marker": ["before", "overflow", "after"], "text": ["x", " ab ", "y"]}),
        model=embedder,
        input_type=input_type,
        embedding_input_policy=policy,
    )

    assert result["text"].tolist() == ["x", " ab", " ", "y"]
    assert model_inputs == ["x", " ab", " ", "y"]
    assert result["text_embeddings_1b_v2_has_embedding"].tolist() == [True, True, True, True]


def test_policy_fails_closed_when_tokenizer_decode_changes_source_text() -> None:
    policy = EmbeddingInputPolicy(tokenizer=_NormalizingWhitespaceTokenizer(), max_tokens=4, prefix="passage: ")

    with pytest.raises(ValueError, match="without changing its (token sequence|source text)"):
        prepare_embedding_inputs(pd.DataFrame({"text": ["one  two three four five"]}), policy=policy)


def test_policy_batches_admission_and_preserves_non_overlength_rows_exactly() -> None:
    tokenizer = _BatchWhitespaceTokenizer()
    policy = EmbeddingInputPolicy(tokenizer=tokenizer, max_tokens=20, prefix="passage: ")
    source = pd.DataFrame(
        {
            "text": ["short input", "another short input"],
            "page_number": pd.Series([1, 2], dtype="int64"),
            "score": pd.Series([0.25, 0.5], dtype="float64"),
            "metadata": [{"chunk_index": 0}, {"chunk_index": 1}],
        }
    )
    source.index = pd.Index([17, 41], name="source_row")

    result = prepare_embedding_inputs(source, policy=policy).frame

    pd.testing.assert_frame_equal(result, source, check_exact=True)
    assert tokenizer.batch_calls == 1


def test_policy_preserves_non_text_batches_without_tokenizing() -> None:
    policy = EmbeddingInputPolicy(tokenizer=_NeverCalledTokenizer(), max_tokens=8192, prefix="passage: ")
    source = pd.DataFrame(
        {
            "text": ["image placeholder", "another placeholder"],
            "_embed_modality": ["image", "text_image"],
            "_image_b64": ["image-a", "image-b"],
            "metadata": [{"page_number": 1}, {"page_number": 2}],
        }
    )

    result = prepare_embedding_inputs(source, policy=policy).frame

    pd.testing.assert_frame_equal(result, source, check_exact=True)


def test_policy_preserves_mixed_short_batches_exactly() -> None:
    tokenizer = _BatchWhitespaceTokenizer()
    policy = EmbeddingInputPolicy(tokenizer=tokenizer, max_tokens=20, prefix="passage: ")
    source = pd.DataFrame(
        {
            "text": ["short text", "image placeholder"],
            "_embed_modality": ["text", "image"],
            "metadata": [{"page_number": 1}, {"page_number": 2}],
        }
    )

    result = prepare_embedding_inputs(source, policy=policy).frame

    pd.testing.assert_frame_equal(result, source, check_exact=True)
    assert tokenizer.batch_calls == 1


def test_text_image_without_image_is_split_on_the_actual_text_route() -> None:
    source_text = "ab  cd"
    source = pd.DataFrame(
        {
            "text": [source_text],
            "_embed_modality": ["text_image"],
            "page_number": [7],
            "metadata": [{}],
        }
    )
    policy = EmbeddingInputPolicy(tokenizer=_CharacterTokenizer(), max_tokens=4, prefix="p")
    model = _RecordingMultimodalEmbedder()

    result = embed_text_main_text_embed(source, model=model, embedding_input_policy=policy)

    assert result["text"].tolist() == ["ab", "  ", "cd"]
    assert "".join(result["text"]) == source_text
    assert model.calls == [["ab", "  ", "cd"]]
    assert model.call_batch_sizes == [8]
    assert result["page_number"].tolist() == [7, 7, 7]
    assert result["text_embeddings_1b_v2_has_embedding"].tolist() == [True, True, True]


def test_default_text_image_fallback_preserves_raw_text_and_multimodal_batch_cap() -> None:
    source = pd.DataFrame({"text": [" x "], "metadata": [{}]})
    policy = EmbeddingInputPolicy(tokenizer=_CharacterTokenizer(), max_tokens=16, prefix="p")
    model = _RecordingMultimodalEmbedder()

    result = embed_text_main_text_embed(
        source,
        model=model,
        embed_modality="text_image",
        inference_batch_size=32,
        embedding_input_policy=policy,
    )

    assert model.calls == [[" x "]]
    assert model.call_batch_sizes == [8]
    assert result["text"].tolist() == [" x "]


def test_nullable_default_modality_neighbor_is_not_lost_from_a_mixed_batch() -> None:
    source = pd.DataFrame(
        {
            "row_id": ["before", "long", "nullable-neighbor", "image"],
            "text": ["ok", "abcdef", "hi", ""],
            "_embed_modality": ["text", "text", pd.NA, "image"],
            "_image_b64": [None, None, None, "image-data"],
            "metadata": [{}, {}, {}, {}],
        }
    )
    policy = EmbeddingInputPolicy(tokenizer=_CharacterTokenizer(), max_tokens=4, prefix="p")
    model = _RecordingMultimodalEmbedder()

    result = embed_text_main_text_embed(source, model=model, embedding_input_policy=policy)

    assert result["row_id"].tolist() == ["before", "long", "long", "long", "nullable-neighbor", "image"]
    assert result["row_id"].tolist().count("nullable-neighbor") == 1
    assert result["text_embeddings_1b_v2_has_embedding"].tolist() == [True] * 6
    assert model.image_calls == [["image-data"]]


def test_policy_preserves_literal_special_token_text() -> None:
    text = "abc<SPECIAL>def"
    policy = EmbeddingInputPolicy(tokenizer=_LiteralSpecialTokenizer(), max_tokens=5, prefix="p")

    plan = policy.plan([text])[0]

    reconstructed = "".join(child.content for child in plan.children)
    assert reconstructed == text
    assert "<SPECIAL>" in reconstructed


def test_hf_text_policy_counts_an_existing_prefix_exactly_once() -> None:
    policy = EmbeddingInputPolicy(
        tokenizer=_CharacterTokenizer(),
        max_tokens=128,
        prefix="passage: ",
        prefix_if_missing=True,
    )

    assert policy.plan(["passage: source"])[0].formatted_tokens == len("passage: source") + 1


def test_policy_resolver_caps_runtime_length_at_checkpoint_support(monkeypatch, tmp_path) -> None:
    _write_local_text_policy_metadata(
        tmp_path,
        prompts={"query": "query: ", "document": "document: "},
        max_input_tokens=8192,
    )
    tokenizer = _WhitespaceTokenizer()
    monkeypatch.setattr(
        "nemo_retriever.models.inference.embedding_input.load_chunk_tokenizer",
        lambda *args, **kwargs: tokenizer,
    )

    policy = resolve_embedding_input_policy(
        str(tmp_path),
        configured_max_tokens=131_072,
        input_type="passage",
    )

    assert policy.max_tokens == 8192
    assert policy.prefix == "document: "
    assert policy.tokenizer is tokenizer


def test_policy_resolver_rejects_missing_checkpoint_input_limit(monkeypatch, tmp_path) -> None:
    _write_local_text_policy_metadata(tmp_path, prompts={"query": "query: ", "document": "document: "})
    monkeypatch.setattr(
        "nemo_retriever.models.inference.embedding_input.load_chunk_tokenizer",
        lambda *args, **kwargs: _WhitespaceTokenizer(),
    )

    with pytest.raises(ValueError, match="does not declare a supported input limit"):
        resolve_embedding_input_policy(
            str(tmp_path),
            configured_max_tokens=8192,
            input_type="passage",
        )


@pytest.mark.parametrize(
    ("input_type", "prompts"),
    [
        pytest.param("passage", None, id="passage"),
        pytest.param("query", {"document": "document: "}, id="query"),
    ],
)
def test_policy_resolver_rejects_missing_checkpoint_prompt(monkeypatch, tmp_path, input_type, prompts) -> None:
    _write_local_text_policy_metadata(tmp_path, prompts=prompts, max_input_tokens=8192)
    monkeypatch.setattr(
        "nemo_retriever.models.inference.embedding_input.load_chunk_tokenizer",
        lambda *args, **kwargs: _WhitespaceTokenizer(),
    )

    with pytest.raises(ValueError, match=f"does not declare a {input_type} prompt"):
        resolve_embedding_input_policy(
            str(tmp_path),
            configured_max_tokens=8192,
            input_type=input_type,
        )


def test_unpinned_model_fails_closed_before_embedding() -> None:
    with pytest.raises(ValueError, match="is not revision-pinned"):
        resolve_embedding_input_policy(
            "custom/unpinned-model",
            configured_max_tokens=8192,
            input_type="passage",
        )


def test_remote_actor_passes_the_resolved_input_policy_to_embedding(
    monkeypatch,
) -> None:
    from nemo_retriever.operators.embed import cpu_operator

    policy = object()
    embed = Mock(spec=cpu_operator.embed_text_main_text_embed, return_value=pd.DataFrame({"text": ["prepared"]}))

    def configure(kwargs, frame):
        kwargs["embedding_input_policy"] = policy
        return policy

    monkeypatch.setattr(cpu_operator, "ensure_embedding_input_policy_for_batch", configure)
    monkeypatch.setattr(cpu_operator, "embed_text_main_text_embed", embed)
    monkeypatch.setattr(cpu_operator, "probe_endpoint", Mock(spec=cpu_operator.probe_endpoint))

    actor = cpu_operator._BatchEmbedCPUActor(
        params=EmbedParams(
            model_name="nvidia/llama-nemotron-embed-1b-v2",
            embed_invoke_url="http://embedding.test/v1",
        )
    )
    actor.process(pd.DataFrame({"text": ["source"]}))

    assert embed.call_args.kwargs["embedding_input_policy"] is policy


def test_remote_image_only_actor_never_resolves_text_admission(monkeypatch) -> None:
    from nemo_retriever.models.inference import embedding_input
    from nemo_retriever.operators.embed import cpu_operator

    resolver = Mock(side_effect=AssertionError("image-only batches must not resolve a text policy"))
    embed = Mock(spec=cpu_operator.embed_text_main_text_embed, return_value=pd.DataFrame({"text": [""]}))
    monkeypatch.setattr(embedding_input, "configure_embedding_input_policy", resolver)
    monkeypatch.setattr(cpu_operator, "embed_text_main_text_embed", embed)

    actor = cpu_operator._BatchEmbedCPUActor(
        params=EmbedParams(
            model_name="nvidia/llama-nemotron-embed-vl-1b-v2",
            embed_invoke_url="http://embedding.test/v1",
            embed_modality="image",
        )
    )
    actor.process(pd.DataFrame({"_embed_modality": ["image"], "_image_b64": ["image-data"]}))

    resolver.assert_not_called()
    assert "embedding_input_policy" not in embed.call_args.kwargs


def test_local_image_only_actor_never_resolves_text_admission(monkeypatch) -> None:
    from nemo_retriever.models.inference import embedding_input
    from nemo_retriever.operators.embed import gpu_operator

    resolver = Mock(side_effect=AssertionError("image-only batches must not resolve a text policy"))
    embed = Mock(spec=gpu_operator.embed_text_main_text_embed, return_value=pd.DataFrame({"text": [""]}))
    monkeypatch.setattr(embedding_input, "configure_embedding_input_policy", resolver)
    monkeypatch.setattr(gpu_operator, "embed_text_main_text_embed", embed)

    actor = object.__new__(gpu_operator._BatchEmbedActor)
    actor._kwargs = {"embed_modality": "image"}
    actor._model = object()
    actor.process(pd.DataFrame({"_embed_modality": ["image"], "_image_b64": ["image-data"]}))

    resolver.assert_not_called()
    assert "embedding_input_policy" not in embed.call_args.kwargs


def test_query_actor_resolves_the_shared_policy_at_query_max_length(
    monkeypatch,
) -> None:
    from nemo_retriever.models.inference import embedding_input

    policy = object()
    resolver = Mock(spec=embedding_input.resolve_embedding_input_policy, return_value=policy)
    monkeypatch.setattr(embedding_input, "resolve_embedding_input_policy", resolver)

    kwargs = {
        "model_name": "nvidia/llama-nemotron-embed-1b-v2",
        "input_type": "query",
        "query_max_length": 257,
        "max_length": 8192,
    }
    result = embedding_input.configure_embedding_input_policy(kwargs)

    assert result is policy
    assert kwargs["embedding_input_policy"] is policy
    assert resolver.call_args.kwargs["configured_max_tokens"] == 257
    assert resolver.call_args.kwargs["input_type"] == "query"


def test_local_actor_does_not_raise_vllm_above_checkpoint_support(monkeypatch) -> None:
    from nemo_retriever.models import create_local_embedder as create_local_embedder_factory
    from nemo_retriever.models.embed_model_spec import EmbedModelSpec
    from nemo_retriever.operators.embed import gpu_operator

    create_local_embedder = Mock(spec=create_local_embedder_factory, return_value=object())
    checkpoint = EmbedModelSpec(
        model_id="nvidia/llama-nemotron-embed-1b-v2",
        revision="113abe4acafa848e77ead9c0623205e511932348",
        family="text",
        output_dimension=2048,
        query_prefix="query: ",
        document_prefix="passage: ",
        max_input_tokens=8192,
        query_prefix_declared=True,
        document_prefix_declared=True,
    )
    monkeypatch.setattr("nemo_retriever.models.create_local_embedder", create_local_embedder)
    monkeypatch.setattr(
        "nemo_retriever.models.embed_model_spec.resolve_embed_model_spec",
        Mock(return_value=checkpoint),
    )
    monkeypatch.setattr(
        "nemo_retriever.models.warmup_registry.get_warmed_model",
        lambda name, *, expected_identity=None: None,
    )

    gpu_operator._BatchEmbedActor(
        params=EmbedParams(
            model_name="nvidia/llama-nemotron-embed-1b-v2",
            runtime=ModelRuntimeParams(max_length=131_072),
        )
    )

    assert create_local_embedder.call_args.kwargs["max_length"] == 8192


def test_remote_default_text_image_fallback_preserves_raw_text_and_batch_cap() -> None:
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": index, "embedding": [float(len(text))]} for index, text in enumerate(payload["input"])
                ]
            },
        )

    source_texts = [" a ", " b ", " c ", " d ", " e "]
    source = pd.DataFrame({"text": source_texts, "metadata": [{} for _ in source_texts]})
    policy = EmbeddingInputPolicy(tokenizer=_CharacterTokenizer(), max_tokens=16, prefix="p")
    client_factory = httpx.Client
    with patch(
        "httpx.Client",
        side_effect=lambda **kwargs: client_factory(transport=httpx.MockTransport(handler)),
    ):
        result = embed_text_main_text_embed(
            source,
            embedding_endpoint="http://embedding.test/v1",
            model_name="test/model",
            embed_modality="text_image",
            inference_batch_size=32,
            embedding_input_policy=policy,
        )

    assert sorted(len(payload["input"]) for payload in payloads) == [1, 4]
    assert sorted(text for payload in payloads for text in payload["input"]) == sorted(source_texts)
    assert result["text_embeddings_1b_v2_has_embedding"].tolist() == [True] * 5


def test_split_identity_is_stable_across_batch_composition() -> None:
    policy = EmbeddingInputPolicy(tokenizer=_WhitespaceTokenizer(), max_tokens=4, prefix="passage: ")
    oversized = {
        "text": "one two three four five",
        "path": "doc.pdf",
        "page_number": 7,
        "metadata": {"chunk_index": 4},
    }
    alone = prepare_embedding_inputs(pd.DataFrame([oversized]), policy=policy).frame
    with_neighbors = prepare_embedding_inputs(
        pd.DataFrame([{"text": "before"}, oversized, {"text": "after"}]), policy=policy
    ).frame

    assert [metadata["embedding_split"]["chunk_id"] for metadata in alone["metadata"]] == [
        metadata["embedding_split"]["chunk_id"] for metadata in with_neighbors.loc[1:3, "metadata"]
    ]


def test_local_and_remote_adapters_apply_the_same_split_policy() -> None:
    source = pd.DataFrame({"text": ["before", "one two three four five", "after"]})
    policy = EmbeddingInputPolicy(tokenizer=_WhitespaceTokenizer(), max_tokens=4, prefix="passage: ")
    local = embed_text_main_text_embed(
        source,
        model=_LengthEmbedder(),
        inference_batch_size=3,
        embedding_input_policy=policy,
    )

    remote_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        remote_payloads.append(payload)
        prompts = payload["input"]
        return httpx.Response(
            200,
            json={"data": [{"index": index, "embedding": [float(len(text))]} for index, text in enumerate(prompts)]},
        )

    client_factory = httpx.Client
    with patch(
        "httpx.Client",
        side_effect=lambda **kwargs: client_factory(transport=httpx.MockTransport(handler)),
    ):
        remote = embed_text_main_text_embed(
            source,
            embedding_endpoint="http://embedding.test/v1",
            model_name="test/model",
            inference_batch_size=3,
            embedding_input_policy=policy,
        )

    assert (
        local["text"].tolist()
        == remote["text"].tolist()
        == [
            "before",
            "one two ",
            "three four ",
            "five",
            "after",
        ]
    )
    assert not any(column.startswith("_embedding_input_") for column in local.columns)
    assert not any(column.startswith("_embedding_input_") for column in remote.columns)
    assert local["text_embeddings_1b_v2_has_embedding"].tolist() == [True] * 5
    assert remote["text_embeddings_1b_v2_has_embedding"].tolist() == [True] * 5
    assert remote_payloads and all(payload["truncate"] == "NONE" for payload in remote_payloads)


def test_retrying_persisted_split_children_does_not_report_a_new_split() -> None:
    policy = EmbeddingInputPolicy(tokenizer=_CharacterTokenizer(), max_tokens=4, prefix="p")
    split_children = prepare_embedding_inputs(pd.DataFrame({"text": ["abcdef"], "metadata": [{}]}), policy=policy).frame

    result = embed_text_main_text_embed(
        split_children,
        model=_RecordingMultimodalEmbedder(),
        embedding_input_policy=policy,
    )

    assert result["embedding_v1_counts_by_label"].tolist() == [{"embedded": 1}] * 3


def test_retry_child_id_collision_does_not_change_current_run_split_telemetry() -> None:
    policy = EmbeddingInputPolicy(tokenizer=_CharacterTokenizer(), max_tokens=4, prefix="p")
    parent = pd.DataFrame({"text": ["abcdef"], "metadata": [{}]})
    persisted_first_child = prepare_embedding_inputs(parent, policy=policy).frame.iloc[[0]]
    mixed = pd.concat([persisted_first_child, parent], ignore_index=True)

    result = embed_text_main_text_embed(
        mixed,
        model=_RecordingMultimodalEmbedder(),
        embedding_input_policy=policy,
    )

    assert result["embedding_v1_counts_by_label"].tolist() == [
        {"embedded": 1},
        {"embedded": 1, "split_child": 1, "overlength": 1, "split": 1},
        {"embedded": 1, "split_child": 1},
        {"embedded": 1, "split_child": 1},
    ]


def test_persisted_split_child_with_nullable_source_field_uses_exact_metadata_content() -> None:
    row = pd.DataFrame(
        {
            "text": [pd.NA],
            "metadata": [{"embedding_split": {"chunk_id": "child", "content": "ab"}}],
        }
    )
    model = _RecordingMultimodalEmbedder()

    result = embed_text_main_text_embed(row, model=model)

    assert model.calls == [["ab"]]
    assert result["text_embeddings_1b_v2_has_embedding"].tolist() == [True]


def test_persisted_split_content_is_admitted_even_when_source_column_is_stale() -> None:
    policy = EmbeddingInputPolicy(tokenizer=_CharacterTokenizer(), max_tokens=4, prefix="p")
    row = pd.DataFrame(
        {
            "text": ["a"],
            "metadata": [{"embedding_split": {"chunk_id": "old-child", "content": "abcdef"}}],
        }
    )

    prepared = prepare_embedding_inputs(row, policy=policy)

    assert prepared.split_parent_positions == frozenset({0})
    assert "".join(prepared.frame["metadata"].map(lambda metadata: metadata["embedding_split"]["content"])) == "abcdef"
    assert len(prepared.frame.index) == 3


def test_split_rows_reach_vdb_conversion_with_complete_coverage() -> None:
    source = pd.DataFrame(
        {
            "text": ["before", "one two three four five", "after"],
            "path": ["doc.pdf"] * 3,
            "page_number": [1, 2, 3],
            "metadata": [{}, {}, {}],
        }
    )
    policy = EmbeddingInputPolicy(tokenizer=_WhitespaceTokenizer(), max_tokens=4, prefix="passage: ")

    embedded = embed_text_main_text_embed(
        source,
        model=_LengthEmbedder(),
        inference_batch_size=3,
        embedding_input_policy=policy,
    )
    records = to_client_vdb_records(embedded)

    assert len(records) == 1
    assert len(records[0]) == 5
    assert [record["metadata"]["content_metadata"]["page_number"] for record in records[0]] == [1, 2, 2, 2, 3]
    assert all(record["metadata"]["embedding"] for record in records[0])


def test_split_fallback_content_is_the_text_written_to_vdb() -> None:
    source = pd.DataFrame(
        {
            "text": [" "],
            "content": ["abcdefghi"],
            "path": ["doc.pdf"],
            "page_number": [7],
            "metadata": [{}],
        }
    )
    policy = EmbeddingInputPolicy(tokenizer=_CharacterTokenizer(), max_tokens=4, prefix="p")

    embedded = embed_text_main_text_embed(
        source,
        model=_RecordingEmbedder(),
        embedding_input_policy=policy,
    )
    records = to_client_vdb_records(embedded)

    assert embedded["content"].tolist() == ["ab", "cd", "ef", "gh", "i"]
    assert [record["metadata"]["content"] for record in records[0]] == ["ab", "cd", "ef", "gh", "i"]
