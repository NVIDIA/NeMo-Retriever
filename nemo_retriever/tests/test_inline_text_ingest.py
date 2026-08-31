# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from nemo_retriever.common.params import TextChunkParams
from nemo_retriever.graph import Graph
from nemo_retriever.ingestor.graph_ingestor import GraphIngestor
from nemo_retriever.operators.abstract_operator import AbstractOperator
from nemo_retriever.operators.extract.txt.ray_data import TxtSplitActor


class _MockTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()

    def decode(self, ids: list[str], skip_special_tokens: bool = True) -> str:
        return " ".join(ids)


class _FakeEmbedOperator(AbstractOperator):
    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        result = data.copy()
        result["fake_embedding"] = [[1.0]] * len(result)
        return result

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


class _FakeVdbOperator(AbstractOperator):
    def preprocess(self, data: Any, **kwargs: Any) -> Any:
        return data

    def process(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        result = data.copy()
        result["stored"] = True
        return result

    def postprocess(self, data: Any, **kwargs: Any) -> Any:
        return data


def test_inline_text_runs_split_embed_and_vdb_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nemo_retriever.common.modality.txt.split._get_tokenizer", lambda *args, **kwargs: _MockTokenizer()
    )
    captured: dict[str, Any] = {}

    def fake_build_graph(**kwargs: Any) -> Graph:
        captured.update(kwargs)
        return (
            Graph()
            >> TxtSplitActor(params=kwargs["split_config"]["text"])
            >> _FakeEmbedOperator()
            >> _FakeVdbOperator()
        )

    monkeypatch.setattr("nemo_retriever.ingestor.graph_ingestor.build_graph", fake_build_graph)

    result = (
        GraphIngestor(run_mode="inprocess", show_progress=False)
        .texts(["one two three", "one two three"])
        .extract(split_config={"text": {"max_tokens": 2}})
        .embed()
        .vdb_upload()
        .ingest()
    )

    assert result["text"].tolist() == ["one two", "three", "one two", "three"]
    assert result["path"].tolist() == [
        "inline://00000000",
        "inline://00000000",
        "inline://00000001",
        "inline://00000001",
    ]
    assert result["stored"].tolist() == [True, True, True, True]
    assert captured["embed_params"] is not None
    assert captured["vdb_upload_params"] is not None


@pytest.mark.integration
def test_batch_inline_text_matches_text_file(tmp_path) -> None:
    ray = pytest.importorskip("ray")
    pytest.importorskip("transformers")
    text = "one two three four five"
    document = tmp_path / "document.txt"
    document.write_text(text, encoding="utf-8")
    params = TextChunkParams(max_tokens=2)

    try:
        file_result = (
            GraphIngestor(run_mode="batch", show_progress=False)
            .files([str(document)])
            .extract(split_config={"text": params})
            .ingest()
        )
        inline_result = (
            GraphIngestor(run_mode="batch", show_progress=False)
            .texts([text])
            .extract(split_config={"text": params})
            .ingest()
        )
    finally:
        ray.shutdown()

    assert file_result["text"].tolist() == inline_result["text"].tolist()
    assert file_result["page_number"].tolist() == inline_result["page_number"].tolist()


@pytest.mark.integration
def test_batch_inline_text_ingests_alongside_text_file(tmp_path) -> None:
    ray = pytest.importorskip("ray")
    pytest.importorskip("transformers")
    document = tmp_path / "document.txt"
    document.write_text("from file", encoding="utf-8")

    try:
        result = (
            GraphIngestor(run_mode="batch", show_progress=False)
            .files([str(document)])
            .texts(["from inline"])
            .extract(split_config={"text": {"max_tokens": 10}})
            .ingest()
        )
    finally:
        ray.shutdown()

    assert set(zip(result["path"], result["text"])) == {
        (str(document.resolve()), "from file"),
        ("inline://00000000", "from inline"),
    }
