# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
from typing import Any

from typer.testing import CliRunner

import nemo_retriever.query.workflow as query_core


RUNNER = CliRunner()
cli_main = importlib.import_module("nemo_retriever.cli.main")


def test_root_query_passes_query_options_and_prints_json(monkeypatch) -> None:
    retriever_calls: list[dict[str, Any]] = []
    query_calls: list[str] = []
    hits = [
        {
            "text": "passage",
            "source": "doc.pdf",
            "page_number": 1,
            "metadata": {"type": "text"},
            "_distance": 0.2,
        },
        {
            "text": "other",
            "source": "other.pdf",
            "page_number": 2,
            "metadata": {"type": "table"},
            "_distance": 0.4,
        },
    ]
    expected_output = [
        {"source": "doc.pdf", "page_number": 1, "text": "passage"},
        {"source": "other.pdf", "page_number": 2, "text": "other"},
    ]

    class FakeRetriever:
        def __init__(self, **kwargs: Any) -> None:
            retriever_calls.append(kwargs)

        def query(self, query: str, **_kwargs: Any) -> list[dict[str, Any]]:
            query_calls.append(query)
            return hits

    monkeypatch.setattr(query_core, "Retriever", FakeRetriever)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "query",
            "Which animal is responsible for typos?",
            "--top-k",
            "3",
            "--lancedb-uri",
            "/tmp/lancedb",
            "--table-name",
            "docs",
        ],
    )

    assert result.exit_code == 0
    # No rerank flag passed -> rerank is off (opt-in only).
    assert retriever_calls == [{"top_k": 3, "vdb_kwargs": {"uri": "/tmp/lancedb", "table_name": "docs"}}]
    assert query_calls == ["Which animal is responsible for typos?"]
    assert json.loads(result.output) == expected_output
    assert result.output == json.dumps(expected_output, indent=2, sort_keys=True, default=str) + "\n"


def test_root_query_passes_candidate_dedup_and_content_filters(monkeypatch) -> None:
    query_kwargs: list[dict[str, Any]] = []

    class FakeRetriever:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def query(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
            query_kwargs.append(kwargs)
            # query_documents returns results after Retriever.query has applied
            # candidate widening, page deduplication, filtering, and top-k.
            return [
                {"text": "text row", "metadata": {"type": "text"}, "page_number": 1, "source": "doc.pdf"},
            ]

    monkeypatch.setattr(query_core, "Retriever", FakeRetriever)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "query",
            "deployment?",
            "--top-k",
            "1",
            "--candidate-k",
            "3",
            "--page-dedup",
            "--content-types",
            "text,table",
        ],
    )

    assert result.exit_code == 0
    assert query_kwargs == [{"candidate_k": 3, "page_dedup": True, "content_types": "text,table"}]
    assert json.loads(result.output) == [
        {"page_number": 1, "source": "doc.pdf", "text": "text row"},
    ]


def test_root_query_passes_embed_options(monkeypatch) -> None:
    retriever_calls: list[dict[str, Any]] = []
    query_calls: list[str] = []

    class FakeRetriever:
        def __init__(self, **kwargs: Any) -> None:
            retriever_calls.append(kwargs)

        def query(self, query: str, **_kwargs: Any) -> list[dict[str, Any]]:
            query_calls.append(query)
            return []

    monkeypatch.setattr(query_core, "Retriever", FakeRetriever)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "query",
            "Which passages mention deployment?",
            "--embed-invoke-url",
            "http://embed:8000/v1/embeddings",
            "--embed-model-name",
            "nvidia/llama-nemotron-embed-1b-v2",
        ],
    )

    assert result.exit_code == 0
    # Embed options only -- no rerank-related arg, so rerank stays off.
    assert retriever_calls == [
        {
            "top_k": 10,
            "vdb_kwargs": {"uri": "lancedb", "table_name": "nemo-retriever"},
            "embed_kwargs": {
                "embed_invoke_url": "http://embed:8000/v1/embeddings",
                "embedding_endpoint": "http://embed:8000/v1/embeddings",
                "model_name": "nvidia/llama-nemotron-embed-1b-v2",
                "embed_model_name": "nvidia/llama-nemotron-embed-1b-v2",
            },
        }
    ]
    assert query_calls == ["Which passages mention deployment?"]
    assert json.loads(result.output) == []


def test_root_query_passes_reranker_url(monkeypatch) -> None:
    retriever_calls: list[dict[str, Any]] = []
    query_calls: list[str] = []
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")

    class FakeRetriever:
        def __init__(self, **kwargs: Any) -> None:
            retriever_calls.append(kwargs)

        def query(self, query: str, **_kwargs: Any) -> list[dict[str, Any]]:
            query_calls.append(query)
            return []

    monkeypatch.setattr(query_core, "Retriever", FakeRetriever)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "query",
            "Which passages mention deployment?",
            "--reranker-invoke-url",
            "http://rerank:8000/v1/ranking",
        ],
    )

    assert result.exit_code == 0
    assert retriever_calls == [
        {
            "top_k": 10,
            "vdb_kwargs": {"uri": "lancedb", "table_name": "nemo-retriever"},
            "rerank": True,
            "rerank_kwargs": {
                "rerank_invoke_url": "http://rerank:8000/v1/ranking",
                "api_key": "nvapi-test",
            },
        }
    ]
    assert query_calls == ["Which passages mention deployment?"]
    assert json.loads(result.output) == []


def test_root_query_rerank_flag_enables_local_rerank(monkeypatch) -> None:
    """``--rerank`` alone enables rerank with the local VL default model."""
    retriever_calls: list[dict[str, Any]] = []

    class FakeRetriever:
        def __init__(self, **kwargs: Any) -> None:
            retriever_calls.append(kwargs)

        def query(self, query: str, **_kwargs: Any) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(query_core, "Retriever", FakeRetriever)

    result = RUNNER.invoke(cli_main.app, ["query", "hello", "--rerank"])

    assert result.exit_code == 0
    assert retriever_calls == [
        {
            "top_k": 10,
            "vdb_kwargs": {"uri": "lancedb", "table_name": "nemo-retriever"},
            "rerank": True,
            "rerank_kwargs": {"model_name": "nvidia/llama-nemotron-rerank-vl-1b-v2"},
        }
    ]


def test_root_query_rerank_off_by_default(monkeypatch) -> None:
    """Without ``--rerank`` (or any rerank arg), rerank stays off."""
    retriever_calls: list[dict[str, Any]] = []

    class FakeRetriever:
        def __init__(self, **kwargs: Any) -> None:
            retriever_calls.append(kwargs)

        def query(self, query: str, **_kwargs: Any) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(query_core, "Retriever", FakeRetriever)

    result = RUNNER.invoke(cli_main.app, ["query", "hello"])

    assert result.exit_code == 0
    # No rerank fields set on the Retriever call.
    assert "rerank" not in retriever_calls[0]
    assert "rerank_kwargs" not in retriever_calls[0]


def test_root_query_reranker_model_name_override(monkeypatch) -> None:
    """`--reranker-model-name` mirrors `--embed-model-name`: it overrides the
    default model on the local path."""
    retriever_calls: list[dict[str, Any]] = []

    class FakeRetriever:
        def __init__(self, **kwargs: Any) -> None:
            retriever_calls.append(kwargs)

        def query(self, query: str, **_kwargs: Any) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(query_core, "Retriever", FakeRetriever)

    result = RUNNER.invoke(
        cli_main.app,
        ["query", "hello", "--reranker-model-name", "nvidia/llama-nemotron-rerank-1b-v2"],
    )

    assert result.exit_code == 0
    assert retriever_calls[0]["rerank_kwargs"] == {"model_name": "nvidia/llama-nemotron-rerank-1b-v2"}


def test_root_query_reports_os_errors(monkeypatch) -> None:
    def fail_query_documents(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise OSError("database unavailable")

    monkeypatch.setattr(cli_main, "query_documents", fail_query_documents)

    result = RUNNER.invoke(cli_main.app, ["query", "hello"])

    assert result.exit_code == 1
    assert "Error: database unavailable" in result.output
