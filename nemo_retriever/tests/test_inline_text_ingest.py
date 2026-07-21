# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from io import BytesIO
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


def test_texts_accepts_scalar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nemo_retriever.common.modality.txt.split._get_tokenizer", lambda *args, **kwargs: _MockTokenizer()
    )

    result = GraphIngestor(run_mode="inprocess", show_progress=False).texts("first").ingest()

    assert result["text"].tolist() == ["first"]
    assert result["path"].tolist() == ["inline://00000000"]


def test_texts_replaces_prior_inline_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nemo_retriever.common.modality.txt.split._get_tokenizer", lambda *args, **kwargs: _MockTokenizer()
    )

    result = GraphIngestor(run_mode="inprocess", show_progress=False).texts("first").texts(["second", "third"]).ingest()

    assert result["text"].tolist() == ["second", "third"]
    assert result["path"].tolist() == ["inline://00000000", "inline://00000001"]


@pytest.mark.parametrize("values", [["valid", None], ["valid", 3], [object()]])
def test_texts_rejects_non_string_values_with_index(values) -> None:
    bad_index = next(index for index, value in enumerate(values) if not isinstance(value, str))

    with pytest.raises(TypeError, match=rf"texts\[{bad_index}\] must be a string"):
        GraphIngestor(run_mode="inprocess").texts(values)


def test_texts_rejects_non_sequence_input() -> None:
    with pytest.raises(TypeError, match="string or sequence of strings"):
        GraphIngestor(run_mode="inprocess").texts(iter(["one", "two"]))


def test_texts_cannot_mix_with_files_or_buffers(tmp_path) -> None:
    document = tmp_path / "document.txt"
    document.write_text("document", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be combined"):
        GraphIngestor(run_mode="inprocess", documents=[str(document)]).texts(["inline"])
    with pytest.raises(ValueError, match="cannot be combined"):
        GraphIngestor(run_mode="inprocess").texts(["inline"]).files([str(document)])
    with pytest.raises(ValueError, match="cannot be combined"):
        GraphIngestor(run_mode="inprocess").texts(["inline"]).buffers(("document.txt", BytesIO(b"document")))


@pytest.mark.parametrize(
    "method_name", ["extract", "extract_image_files", "extract_html", "extract_audio", "extract_video"]
)
def test_texts_rejects_incompatible_extraction_methods(method_name: str) -> None:
    ingestor = GraphIngestor(run_mode="inprocess").texts(["inline"])

    with pytest.raises(ValueError, match="use extract_txt"):
        getattr(ingestor, method_name)()


def test_texts_rejects_preconfigured_non_text_extraction() -> None:
    ingestor = GraphIngestor(run_mode="inprocess").extract_html()

    with pytest.raises(ValueError, match="only supports text extraction"):
        ingestor.texts(["inline"])


def test_empty_and_blank_inline_corpus_short_circuits_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    ingestor = GraphIngestor(run_mode="inprocess").texts(["", "  \n"])
    monkeypatch.setattr(
        "nemo_retriever.ingestor.graph_ingestor.build_graph",
        lambda *args, **kwargs: pytest.fail("empty inline corpus should not execute the graph"),
    )

    result = ingestor.embed().vdb_upload().ingest()

    assert result.empty
    assert list(result.columns) == ["text", "content", "path", "page_number", "metadata"]


def test_empty_batch_inline_corpus_returns_ray_dataset_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, pd.DataFrame] = {}
    dataset = object()

    class _FakeRayData:
        @staticmethod
        def from_pandas(frame: pd.DataFrame) -> object:
            captured["frame"] = frame
            return dataset

    class _FakeRay:
        data = _FakeRayData()

    ingestor = GraphIngestor(run_mode="batch").texts(["", "  \n"])
    monkeypatch.setattr(ingestor, "_ensure_batch_runtime", lambda: (_FakeRay(), object()))
    monkeypatch.setattr(
        "nemo_retriever.ingestor.graph_ingestor.build_graph",
        lambda *args, **kwargs: pytest.fail("empty inline corpus should not execute the graph"),
    )

    result = ingestor.ingest()

    assert result is dataset
    assert ingestor.get_dataset() is dataset
    assert captured["frame"].empty
    assert list(captured["frame"].columns) == ["text", "content", "path", "page_number", "metadata"]


def test_batch_inline_text_constructs_ray_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    input_dataset = object()
    output_dataset = object()

    class _FakeRayData:
        @staticmethod
        def from_items(items: list[dict[str, str]]) -> object:
            captured["items"] = items
            return input_dataset

    class _FakeRay:
        data = _FakeRayData()

    class _FakeResources:
        @staticmethod
        def available_gpu_count() -> int:
            return 0

    class _FakeExecutor:
        def __init__(self, graph: object, **kwargs: object) -> None:
            captured["graph"] = graph

        def ingest(self, dataset: object) -> object:
            captured["dataset"] = dataset
            return output_dataset

    ingestor = GraphIngestor(run_mode="batch").texts(["first", "second"])
    monkeypatch.setattr(ingestor, "_ensure_batch_runtime", lambda: (_FakeRay(), _FakeResources()))
    monkeypatch.setattr("nemo_retriever.ingestor.graph_ingestor.build_graph", lambda **kwargs: object())
    monkeypatch.setattr(
        "nemo_retriever.ingestor.graph_ingestor.batch_tuning_to_node_overrides", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr("nemo_retriever.ingestor.graph_ingestor.RayDataExecutor", _FakeExecutor)

    result = ingestor.ingest()

    assert captured["items"] == [
        {"text": "first", "path": "inline://00000000"},
        {"text": "second", "path": "inline://00000001"},
    ]
    assert captured["dataset"] is input_dataset
    assert result is output_dataset
    assert ingestor.get_dataset() is output_dataset


def test_inline_text_runs_split_embed_and_vdb_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "nemo_retriever.common.modality.txt.split._get_tokenizer", lambda *args, **kwargs: _MockTokenizer()
    )
    captured: dict[str, Any] = {}

    def fake_build_graph(**kwargs: Any) -> Graph:
        captured.update(kwargs)
        return Graph() >> TxtSplitActor(params=kwargs["text_params"]) >> _FakeEmbedOperator() >> _FakeVdbOperator()

    monkeypatch.setattr("nemo_retriever.ingestor.graph_ingestor.build_graph", fake_build_graph)

    result = (
        GraphIngestor(run_mode="inprocess", show_progress=False)
        .texts(["one two three", "one two three"])
        .extract_txt(TextChunkParams(max_tokens=2))
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
            GraphIngestor(run_mode="batch", show_progress=False).files([str(document)]).extract_txt(params).ingest()
        )
        inline_result = GraphIngestor(run_mode="batch", show_progress=False).texts([text]).extract_txt(params).ingest()
    finally:
        ray.shutdown()

    assert file_result["text"].tolist() == inline_result["text"].tolist()
    assert file_result["page_number"].tolist() == inline_result["page_number"].tolist()
