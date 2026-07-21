# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for nemo_retriever.common.modality.txt.split: split_text_by_tokens and txt_file_to_chunks_df.
"""

import tempfile  # noqa: F401
from pathlib import Path

import pandas as pd
import pytest

from nemo_retriever.common.modality.txt.split import (
    TextChunkParams,
    split_text_by_tokens,
    text_to_chunks_df,
    txt_bytes_to_chunks_df,
    txt_file_to_chunks_df,
)


class _MockTokenizer:
    """Minimal tokenizer: encode = split on spaces, decode = join."""

    def encode(self, text: str, add_special_tokens: bool = False):
        return text.split()

    def decode(self, ids, skip_special_tokens: bool = True):
        if isinstance(ids, (list, range)):
            return " ".join(str(i) for i in ids)
        return str(ids)


def test_split_text_by_tokens_empty():
    tokenizer = _MockTokenizer()
    assert split_text_by_tokens("", tokenizer=tokenizer, max_tokens=10) == []
    assert split_text_by_tokens("   \n  ", tokenizer=tokenizer, max_tokens=10) == []


def test_split_text_by_tokens_no_overlap():
    tokenizer = _MockTokenizer()
    # "a b c d e f g h i j" -> 10 tokens, max_tokens=3 -> 4 chunks
    text = "a b c d e f g h i j"
    chunks = split_text_by_tokens(text, tokenizer=tokenizer, max_tokens=3, overlap_tokens=0)
    assert len(chunks) >= 1
    joined = " ".join(chunks)
    assert "a" in joined and "j" in joined


def test_split_text_by_tokens_single_chunk():
    tokenizer = _MockTokenizer()
    text = "one two three"
    chunks = split_text_by_tokens(text, tokenizer=tokenizer, max_tokens=10, overlap_tokens=0)
    assert len(chunks) == 1
    assert chunks[0] == "one two three"


def test_split_text_by_tokens_max_tokens_positive():
    tokenizer = _MockTokenizer()
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        split_text_by_tokens("hello", tokenizer=tokenizer, max_tokens=0)


def test_txt_file_to_chunks_df(tmp_path: Path, monkeypatch):
    pytest.importorskip("transformers")
    monkeypatch.setattr(
        "nemo_retriever.common.modality.txt.split._get_tokenizer", lambda model_id, cache_dir=None: _MockTokenizer()
    )
    f = tmp_path / "doc.txt"
    f.write_text("First paragraph here. Second paragraph there.", encoding="utf-8")
    df = txt_file_to_chunks_df(
        str(f),
        params=TextChunkParams(max_tokens=512, overlap_tokens=0),
    )
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["text", "content", "path", "page_number", "metadata"]
    assert len(df) >= 1
    assert df["path"].iloc[0] == str(f.resolve())
    assert df["page_number"].iloc[0] >= 1
    assert "source_path" in df["metadata"].iloc[0]
    assert "chunk_index" in df["metadata"].iloc[0]


def test_txt_file_to_chunks_df_empty_file(tmp_path: Path):
    pytest.importorskip("transformers")
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    df = txt_file_to_chunks_df(str(f), params=TextChunkParams(max_tokens=512))
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["text", "content", "path", "page_number", "metadata"]
    assert len(df) == 0


def test_text_to_chunks_df_preserves_logical_source_id(monkeypatch):
    monkeypatch.setattr(
        "nemo_retriever.common.modality.txt.split._get_tokenizer", lambda model_id, cache_dir=None: _MockTokenizer()
    )

    df = text_to_chunks_df(
        "one two three four",
        "inline://00000000",
        params=TextChunkParams(max_tokens=2),
    )

    assert df["text"].tolist() == ["one two", "three four"]
    assert df["path"].tolist() == ["inline://00000000", "inline://00000000"]
    assert [metadata["source_path"] for metadata in df["metadata"]] == [
        "inline://00000000",
        "inline://00000000",
    ]


def test_txt_bytes_preserves_service_inline_identity_and_utf8_transport(monkeypatch):
    monkeypatch.setattr(
        "nemo_retriever.common.modality.txt.split._get_tokenizer", lambda model_id, cache_dir=None: _MockTokenizer()
    )

    df = txt_bytes_to_chunks_df(
        "café document".encode("utf-8"),
        "inline://00000007",
        params=TextChunkParams(max_tokens=10, encoding="utf-16"),
    )

    assert df["text"].tolist() == ["café document"]
    assert df["path"].tolist() == ["inline://00000007"]
    assert df["metadata"].iloc[0]["source_path"] == "inline://00000007"


def test_file_and_decoded_text_helpers_produce_equivalent_chunks(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "nemo_retriever.common.modality.txt.split._get_tokenizer", lambda model_id, cache_dir=None: _MockTokenizer()
    )
    text = "one two three four five"
    path = tmp_path / "document.txt"
    path.write_text(text, encoding="utf-8")
    params = TextChunkParams(max_tokens=2)

    file_df = txt_file_to_chunks_df(str(path), params=params)
    inline_df = text_to_chunks_df(text, "inline://00000000", params=params)

    assert file_df["text"].tolist() == inline_df["text"].tolist()
    assert file_df["page_number"].tolist() == inline_df["page_number"].tolist()
    assert [metadata["chunk_index"] for metadata in file_df["metadata"]] == [
        metadata["chunk_index"] for metadata in inline_df["metadata"]
    ]
