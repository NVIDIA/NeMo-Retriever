# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dependency-light, revision-pinned chunk tokenizers."""

from __future__ import annotations

from pathlib import Path
import tomllib

import pandas as pd
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

import nemo_retriever.common.modality.txt.tokenizer_provider as provider
import nemo_retriever.operators.extract.html.ray_data as html_actor_module
import nemo_retriever.operators.extract.txt.ray_data as txt_actor_module
from nemo_retriever.common.modality.html.convert import html_bytes_to_chunks_df
from nemo_retriever.common.modality.txt.split import DEFAULT_TOKENIZER_MODEL_ID, txt_bytes_to_chunks_df
from nemo_retriever.models import resolve_embed_model


def _write_tokenizer(path: Path) -> None:
    tokenizer = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "hello": 1, "world": 2},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path))


def test_service_extra_declares_lightweight_tokenizer_dependencies() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((repo_root / "nemo_retriever" / "pyproject.toml").read_text(encoding="utf-8"))
    service_dependencies = pyproject["project"]["optional-dependencies"]["service"]
    assert any(item.startswith("tokenizers") for item in service_dependencies)
    assert any(item.startswith("huggingface-hub") for item in service_dependencies)
    assert not any(item.startswith("transformers") for item in service_dependencies)


def test_default_chunk_tokenizer_matches_default_embedding_model() -> None:
    assert DEFAULT_TOKENIZER_MODEL_ID == resolve_embed_model(None)


def test_load_chunk_tokenizer_uses_pinned_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    provider.load_chunk_tokenizer.cache_clear()
    monkeypatch.setattr(
        provider,
        "hf_hub_download_with_pinned_revision",
        lambda **_kwargs: str(tokenizer_path),
    )

    tokenizer = provider.load_chunk_tokenizer("nvidia/llama-nemotron-embed-vl-1b-v2")

    token_ids = tokenizer.encode("hello world")
    assert tokenizer.decode(token_ids) == "hello world"


def test_txt_and_html_chunk_without_transformers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    _write_tokenizer(tokenizer_path)
    provider.load_chunk_tokenizer.cache_clear()
    monkeypatch.setattr(
        provider,
        "hf_hub_download_with_pinned_revision",
        lambda **_kwargs: str(tokenizer_path),
    )

    txt = txt_bytes_to_chunks_df(b"hello world", "document.txt")
    html = html_bytes_to_chunks_df(
        b"<html><body><p>hello world</p></body></html>",
        "document.html",
    )

    assert txt["text"].tolist() == ["hello world"]
    assert html["text"].tolist() == ["hello world"]


def test_load_chunk_tokenizer_surfaces_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider.load_chunk_tokenizer.cache_clear()

    def fail_download(**_kwargs: object) -> str:
        raise RuntimeError("offline cache miss")

    monkeypatch.setattr(
        provider,
        "hf_hub_download_with_pinned_revision",
        fail_download,
    )

    with pytest.raises(
        provider.TokenizerUnavailableError,
        match="Pre-cache tokenizer.json",
    ):
        provider.load_chunk_tokenizer("nvidia/llama-nemotron-embed-vl-1b-v2")


def test_load_chunk_tokenizer_wraps_unregistered_model_error() -> None:
    provider.load_chunk_tokenizer.cache_clear()

    with pytest.raises(
        provider.TokenizerUnavailableError,
        match="unregistered/model",
    ) as exc_info:
        provider.load_chunk_tokenizer("unregistered/model")

    assert isinstance(exc_info.value.__cause__, ValueError)


@pytest.mark.parametrize(
    ("module", "actor_type", "splitter_name"),
    [
        (txt_actor_module, txt_actor_module.TxtSplitCPUActor, "txt_bytes_to_chunks_df"),
        (
            html_actor_module,
            html_actor_module.HtmlSplitCPUActor,
            "html_bytes_to_chunks_df",
        ),
    ],
)
def test_text_actors_propagate_tokenizer_failures(
    module: object,
    actor_type: type,
    splitter_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_split(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise provider.TokenizerUnavailableError("tokenizer unavailable")

    monkeypatch.setattr(module, splitter_name, fail_split)
    actor = actor_type()
    batch = pd.DataFrame([{"bytes": b"content", "path": "document.txt"}])

    with pytest.raises(provider.TokenizerUnavailableError, match="unavailable"):
        actor.process(batch)


@pytest.mark.parametrize(
    ("module", "actor_type", "splitter_name"),
    [
        (txt_actor_module, txt_actor_module.TxtSplitCPUActor, "txt_bytes_to_chunks_df"),
        (
            html_actor_module,
            html_actor_module.HtmlSplitCPUActor,
            "html_bytes_to_chunks_df",
        ),
    ],
)
def test_text_actors_isolate_non_tokenizer_document_failures(
    module: object,
    actor_type: type,
    splitter_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def split_document(_payload: object, path: str, **_kwargs: object) -> pd.DataFrame:
        if path == "bad.txt":
            raise ValueError("malformed document")
        return pd.DataFrame([{"text": "good", "path": path, "page_number": 1, "metadata": {}}])

    monkeypatch.setattr(module, splitter_name, split_document)
    actor = actor_type()
    batch = pd.DataFrame(
        [
            {"bytes": b"bad", "path": "bad.txt"},
            {"bytes": b"good", "path": "good.txt"},
        ]
    )

    result = actor.process(batch)

    assert result["path"].tolist() == ["good.txt"]
