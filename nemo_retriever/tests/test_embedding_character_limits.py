# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless admission for remote text with independent character/token limits."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

from nemo_retriever.common.modality.txt.tokenizer_provider import _FastTokenizer
from nemo_retriever.models import NEMOTRON_3_EMBED_MODEL, NEMOTRON_35_EMBED_MODEL
from nemo_retriever.models.inference import embedding_input, main_text_embed, runtime
from nemo_retriever.models.inference.embedding_input import EmbeddingInputPolicy


@pytest.fixture
def tokenizer():
    alphabet = sorted(ByteLevel.alphabet())
    vocabulary = {character: index for index, character in enumerate(alphabet)}
    for piece in ("aa", "aaaa", "aaaaaaaa"):
        vocabulary[piece] = len(vocabulary)
    tokenizer = Tokenizer(BPE(vocabulary, [("a", "a"), ("aa", "aa"), ("aaaa", "aaaa")]))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = ByteLevelDecoder()
    return _FastTokenizer(tokenizer)


@pytest.mark.parametrize("length", [65535, 65536, 65537, 73514])
def test_character_admission_is_independent_of_token_budget(tokenizer, length):
    policy = EmbeddingInputPolicy(tokenizer, 32768, "passage: ", max_characters=65536)
    text = "a" * length

    plan = policy.plan([text])[0]

    assert plan.formatted_tokens < policy.max_tokens
    assert plan.requires_split is (length > 65536)
    if plan.requires_split:
        assert "".join(child.content for child in plan.children) == text
        assert all(len(child.content) <= 65536 for child in plan.children)
        assert all(policy._formatted_token_count(child.content) <= policy.max_tokens for child in plan.children)


@pytest.mark.parametrize("text", ["é🙂漢字" * 20, "  " + "a" * 80 + "\n\t"])
def test_character_splitting_preserves_unicode_whitespace_and_token_offsets(tokenizer, text):
    policy = EmbeddingInputPolicy(tokenizer, 32768, "", max_characters=15)
    original_tokens = tokenizer.encode(text, add_special_tokens=False)

    children = policy.plan([text])[0].children

    assert "".join(child.content for child in children) == text
    assert all(len(child.content) <= 15 for child in children)
    assert children[0].start_token == 0
    assert children[-1].end_token == len(original_tokens)
    for previous, following in zip(children, children[1:]):
        assert previous.end_token == following.start_token
    for child in children:
        assert tokenizer.encode(child.content) == original_tokens[child.start_token : child.end_token]


def test_split_children_obey_both_limits_including_formatted_token_budget(tokenizer):
    policy = EmbeddingInputPolicy(tokenizer, 17, "passage: ", max_characters=31)
    text = "a" * 250

    children = policy.plan([text])[0].children

    assert "".join(child.content for child in children) == text
    assert all(len(child.content) <= 31 for child in children)
    assert all(policy._formatted_token_count(child.content) <= 17 for child in children)


def test_character_limit_counts_raw_text_without_the_model_prefix(tokenizer):
    policy = EmbeddingInputPolicy(tokenizer, 32768, "passage: ", max_characters=16)
    assert not policy.plan(["a" * 16])[0].requires_split


def test_default_policy_has_no_character_limit(tokenizer):
    policy = EmbeddingInputPolicy(tokenizer, 32768, "")
    assert not policy.plan(["a" * 73514])[0].requires_split


@pytest.mark.parametrize("limit", [0, -1])
def test_character_limit_must_be_positive(tokenizer, limit):
    with pytest.raises(ValueError, match="max_characters must be positive"):
        EmbeddingInputPolicy(tokenizer, 32768, "", max_characters=limit)


@pytest.mark.parametrize("endpoint_key", ["embedding_endpoint", "embed_invoke_url"])
@pytest.mark.parametrize("model", [NEMOTRON_35_EMBED_MODEL, NEMOTRON_3_EMBED_MODEL, "custom/model"])
@pytest.mark.parametrize("remote", [False, True, "", "   "])
def test_automatic_character_cap_is_scoped_to_remote_nemotron35(tokenizer, monkeypatch, endpoint_key, model, remote):
    checkpoint = SimpleNamespace(
        model_id="test/tokenizer",
        revision="pinned",
        max_input_tokens=32768,
        model_type="bert",
        document_prefix="",
        document_prefix_declared=True,
    )
    monkeypatch.setattr(embedding_input, "load_chunk_tokenizer", lambda *args, **kwargs: tokenizer)
    kwargs = {"embed_model_name": model, "max_length": 32768, "_embedding_checkpoint": checkpoint}
    if remote is not False:
        kwargs[endpoint_key] = "http://embedding.test/v1/embeddings" if remote is True else remote

    policy = embedding_input.configure_embedding_input_policy(kwargs)

    assert policy.max_characters == (65536 if remote is True and model == NEMOTRON_35_EMBED_MODEL else None)
    assert kwargs["embedding_input_policy"] is policy


def test_character_split_reaches_http_losslessly_without_failing_neighbor_rows(tokenizer, monkeypatch):
    text = "a" * 73514
    frame = pd.DataFrame(
        {
            "text": ["before", text, "after"],
            "metadata": [{"source_id": "one"}, {"source_id": "two", "page_number": 7}, {"source_id": "three"}],
        },
        index=[4, 4, 7],
    )
    original_metadata = dict(frame.iloc[1].metadata)
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert all(len(prompt) <= 65536 for prompt in payload["input"])
        return httpx.Response(
            200,
            json={"data": [{"index": index, "embedding": [1.0, 0.0]} for index in range(len(payload["input"]))]},
        )

    policy = EmbeddingInputPolicy(tokenizer, 32768, "", max_characters=65536)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(main_text_embed, "_get_embed_http_client", lambda: client)
        result = runtime.embed_text_main_text_embed(
            frame,
            model_name=NEMOTRON_35_EMBED_MODEL,
            embedding_endpoint="http://embedding.test/v1/embeddings",
            embedding_input_policy=policy,
        )

    sent = [prompt for payload in requests for prompt in payload["input"]]
    assert sent == ["before", *result.text.tolist()[1:-1], "after"]
    assert "".join(sent[1:-1]) == text
    assert all(payload["truncate"] == "NONE" for payload in requests)
    assert result.embedding.notna().all()
    assert result.embedding_error.isna().all()
    assert result.index.tolist() == [0, 1, 2, 3]
    assert frame.index.tolist() == [4, 4, 7]
    split = [row["embedding_split"] for row in result.metadata.tolist()[1:-1]]
    assert len({child["parent_id"] for child in split}) == 1
    assert len({child["chunk_id"] for child in split}) == len(split)
    assert [child["chunk_index"] for child in split] == list(range(len(split)))
    assert all(child["chunk_count"] == len(split) for child in split)
    assert all(row["source_id"] == "two" and row["page_number"] == 7 for row in result.metadata.tolist()[1:-1])
    assert frame.iloc[1].metadata == original_metadata
    assert [row["source_id"] for row in (result.iloc[0].metadata, result.iloc[-1].metadata)] == ["one", "three"]
