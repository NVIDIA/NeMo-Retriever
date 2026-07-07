# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
from typing import Any

from typer.testing import CliRunner

import nemo_retriever.query.workflow as query_core
from nemo_retriever.models import VL_EMBED_MODEL, VL_RERANK_MODEL

RUNNER = CliRunner()
cli_main = importlib.import_module("nemo_retriever.cli.main")
query_cli_app = importlib.import_module("nemo_retriever.cli.query.app")


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
        {"source": "doc.pdf", "page_number": 1, "text": "passage", "modality": "text", "score": 0.2},
        {"source": "other.pdf", "page_number": 2, "text": "other", "modality": "table", "score": 0.4},
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
        {"page_number": 1, "source": "doc.pdf", "text": "text row", "modality": "text", "score": None},
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
            "--embed-model-provider-prefix",
            "nvidia",
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
                "model_name": "nvidia/nvidia/llama-nemotron-embed-1b-v2",
                "embed_model_name": "nvidia/nvidia/llama-nemotron-embed-1b-v2",
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


def test_root_query_passes_reranker_api_key_env(monkeypatch) -> None:
    retriever_calls: list[dict[str, Any]] = []
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    monkeypatch.setenv("NGC_INFERENCE_API_KEY", "inference-secret")

    class FakeRetriever:
        def __init__(self, **kwargs: Any) -> None:
            retriever_calls.append(kwargs)

        def query(self, query: str, **_kwargs: Any) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(query_core, "Retriever", FakeRetriever)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "query",
            "Which passages mention deployment?",
            "--reranker-invoke-url",
            "https://inference-api.nvidia.com/v1/rerank",
            "--reranker-model-name",
            "nvidia/nvidia/llama-3.2-nv-rerankqa-1b-v2",
            "--reranker-api-key-env",
            "NGC_INFERENCE_API_KEY",
        ],
    )

    assert result.exit_code == 0
    assert retriever_calls[0]["rerank_kwargs"] == {
        "rerank_invoke_url": "https://inference-api.nvidia.com/v1/rerank",
        "model_name": "nvidia/nvidia/llama-3.2-nv-rerankqa-1b-v2",
        "api_key": "inference-secret",
    }


def test_root_query_reports_missing_reranker_api_key_env() -> None:
    result = RUNNER.invoke(
        cli_main.app,
        [
            "query",
            "Which passages mention deployment?",
            "--reranker-invoke-url",
            "https://inference-api.nvidia.com/v1/rerank",
            "--reranker-api-key-env",
            "NGC_INFERENCE_API_KEY",
        ],
        env={"NVIDIA_API_KEY": "", "NGC_API_KEY": "", "NGC_INFERENCE_API_KEY": ""},
    )

    assert result.exit_code == 1
    assert "Error: NGC_INFERENCE_API_KEY is not set or is empty." in result.output


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

    monkeypatch.setattr(query_cli_app, "query_local_documents_with_metadata", fail_query_documents)

    result = RUNNER.invoke(cli_main.app, ["query", "hello"])

    assert result.exit_code == 1
    assert "Error: database unavailable" in result.output


def test_root_query_agentic_passes_config_and_prints_ranked(monkeypatch) -> None:
    """`--agentic` wires the LanceDB/embed/LLM options into AgenticRetrievalConfig
    and prints the agent's ranked doc_ids (sorted by rank, truncated to --top-k)."""
    import pandas as pd

    import nemo_retriever.query.agentic as agentic_retrieval

    config_calls: list[dict[str, Any]] = []
    retrieve_calls: list[tuple[Any, Any]] = []

    class FakeConfig:
        def __init__(self, **kwargs: Any) -> None:
            config_calls.append(kwargs)

    class FakeAgenticRetriever:
        def __init__(self, cfg: Any) -> None:
            self.cfg = cfg

        def retrieve(self, query_ids: Any, query_texts: Any) -> Any:
            retrieve_calls.append((query_ids, query_texts))
            # Deliberately out of rank order to exercise the sort.
            return pd.DataFrame(
                [
                    {"query_id": "0", "doc_id": "b.pdf", "rank": 2, "result_source": "rrf"},
                    {"query_id": "0", "doc_id": "a.pdf", "rank": 1, "result_source": "final_results"},
                    {"query_id": "0", "doc_id": "c.pdf", "rank": 3, "result_source": "rrf"},
                ]
            )

    monkeypatch.setattr(agentic_retrieval, "AgenticRetrievalConfig", FakeConfig)
    monkeypatch.setattr(agentic_retrieval, "AgenticRetriever", FakeAgenticRetriever)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "query",
            "how does ingest work?",
            "--agentic",
            "--agentic-llm-model",
            "nvidia/llama-3.3-nemotron-super-49b-v1.5",
            "--top-k",
            "2",
            "--lancedb-uri",
            "/tmp/lancedb",
            "--table-name",
            "docs",
        ],
    )

    assert result.exit_code == 0
    assert retrieve_calls == [(["0"], ["how does ingest work?"])]
    cfg = config_calls[0]
    assert cfg["vdb_op"] == "lancedb"
    assert cfg["vdb_kwargs"] == {"uri": "/tmp/lancedb", "table_name": "docs"}
    assert cfg["llm_model"] == "nvidia/llama-3.3-nemotron-super-49b-v1.5"
    # --top-k is honored end-to-end: plumbed into the agentic config (drives the
    # ReAct target / RRF / selection cut), not just applied as a post-filter.
    assert cfg["top_k"] == 2
    # Sorted by rank and truncated to --top-k=2.
    assert json.loads(result.output) == [
        {"rank": 1, "doc_id": "a.pdf", "result_source": "final_results"},
        {"rank": 2, "doc_id": "b.pdf", "result_source": "rrf"},
    ]


def test_root_query_agentic_requires_llm_model() -> None:
    """Agentic mode is inert without a chat model to drive the loop."""
    result = RUNNER.invoke(cli_main.app, ["query", "hello", "--agentic"])

    assert result.exit_code == 1
    assert "requires --agentic-llm-model" in result.output


def test_root_query_agentic_plumbs_rerank_into_config(monkeypatch) -> None:
    """`--rerank` with `--agentic` wires the reranker config into AgenticRetrievalConfig
    (reranker model + endpoint + backend), so the agent's retrieval backend reranks."""
    import pandas as pd

    import nemo_retriever.query.agentic as agentic_retrieval

    config_calls: list[dict[str, Any]] = []

    class FakeConfig:
        def __init__(self, **kwargs: Any) -> None:
            config_calls.append(kwargs)

    class FakeAgenticRetriever:
        def __init__(self, cfg: Any) -> None:
            self.cfg = cfg

        def retrieve(self, query_ids: Any, query_texts: Any) -> Any:
            return pd.DataFrame([{"query_id": "0", "doc_id": "a.pdf", "rank": 1, "result_source": "rrf"}])

    monkeypatch.setattr(agentic_retrieval, "AgenticRetrievalConfig", FakeConfig)
    monkeypatch.setattr(agentic_retrieval, "AgenticRetriever", FakeAgenticRetriever)

    base = ["query", "q", "--agentic", "--agentic-llm-model", "m"]

    # 1. Explicit reranker model + endpoint + backend flow through.
    result = RUNNER.invoke(
        cli_main.app,
        base
        + [
            "--reranker-model-name",
            "my-rerank",
            "--reranker-invoke-url",
            "http://rr/v1/rerank",
            "--reranker-backend",
            "hf",
        ],
    )
    assert result.exit_code == 0
    cfg = config_calls[-1]
    assert cfg["reranker"] == "my-rerank"
    assert cfg["reranker_endpoint"] == "http://rr/v1/rerank"
    assert cfg["local_reranker_backend"] == "hf"

    # 2. --rerank with no model name falls back to the default rerank model (gate stays on).
    config_calls.clear()
    result = RUNNER.invoke(cli_main.app, base + ["--rerank"])
    assert result.exit_code == 0
    assert config_calls[-1]["reranker"] == query_core._LOCAL_VL_RERANK_MODEL

    # 3. No rerank flags => no reranker key => backend rerank stays off (cfg default None).
    config_calls.clear()
    result = RUNNER.invoke(cli_main.app, base)
    assert result.exit_code == 0
    assert "reranker" not in config_calls[-1]


def test_root_query_passes_retrieval_mode_into_vdb_kwargs(monkeypatch) -> None:
    retriever_calls: list[dict[str, Any]] = []

    class FakeRetriever:
        def __init__(self, **kwargs: Any) -> None:
            retriever_calls.append(kwargs)

        def query(self, query: str, **_kwargs: Any) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(query_core, "Retriever", FakeRetriever)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "query",
            "q",
            "--top-k",
            "5",
            "--lancedb-uri",
            "/tmp/lancedb",
            "--table-name",
            "docs",
            "--retrieval-mode",
            "dense",
        ],
    )

    assert result.exit_code == 0
    assert retriever_calls == [
        {"top_k": 5, "vdb_kwargs": {"uri": "/tmp/lancedb", "table_name": "docs", "retrieval_mode": "dense"}}
    ]


def test_root_query_keeps_hidden_hybrid_alias(monkeypatch) -> None:
    retriever_calls: list[dict[str, Any]] = []

    class FakeRetriever:
        def __init__(self, **kwargs: Any) -> None:
            retriever_calls.append(kwargs)

        def query(self, query: str, **_kwargs: Any) -> list[dict[str, Any]]:
            return []

    monkeypatch.setattr(query_core, "Retriever", FakeRetriever)

    result = RUNNER.invoke(cli_main.app, ["query", "q", "--hybrid"])

    assert result.exit_code == 0
    assert retriever_calls == [
        {"top_k": 10, "vdb_kwargs": {"uri": "lancedb", "table_name": "nemo-retriever", "retrieval_mode": "hybrid"}}
    ]


def test_root_query_rejects_retrieval_mode_and_hybrid_alias_together() -> None:
    result = RUNNER.invoke(cli_main.app, ["query", "q", "--retrieval-mode", "dense", "--hybrid"])

    assert result.exit_code == 1
    assert "pass only one of --retrieval-mode or deprecated --hybrid" in result.output


def test_root_query_max_text_chars_truncates_and_omits(monkeypatch) -> None:
    hits = [{"text": "abcdefghij", "source": "d.pdf", "page_number": 1, "metadata": {"type": "text"}, "_distance": 0.1}]

    class FakeRetriever:
        def __init__(self, **_: Any) -> None:
            pass

        def query(self, query: str, **_kwargs: Any) -> list[dict[str, Any]]:
            return hits

    monkeypatch.setattr(query_core, "Retriever", FakeRetriever)

    snip = RUNNER.invoke(cli_main.app, ["query", "q", "--max-text-chars", "5"])
    assert snip.exit_code == 0
    snip_hit = json.loads(snip.output)[0]
    assert snip_hit["text"] == "abcde…"
    assert snip_hit["modality"] == "text"
    assert snip_hit["source"] == "d.pdf"

    meta = RUNNER.invoke(cli_main.app, ["query", "q", "--max-text-chars", "0"])
    meta_hit = json.loads(meta.output)[0]
    assert meta_hit["text"] == ""
    assert meta_hit["source"] == "d.pdf"
    assert meta_hit["page_number"] == 1


def test_root_query_help_lists_service_subcommand() -> None:
    result = RUNNER.invoke(cli_main.app, ["query", "--help"])

    assert result.exit_code == 0
    assert "service" in result.output
    assert "--run-mode" not in result.output
    assert "--lancedb-uri" not in result.output


def test_root_query_local_help_shows_retrieval_mode_not_hybrid() -> None:
    result = RUNNER.invoke(cli_main.app, ["query", "q", "--help"])

    assert result.exit_code == 0
    assert "--retrieval-mode" in result.output
    assert "--hybrid" not in result.output


def test_root_query_local_help_names_default_models() -> None:
    result = RUNNER.invoke(cli_main.app, ["query", "q", "--help"])

    assert result.exit_code == 0
    assert "Default embedding model" in result.output
    assert VL_EMBED_MODEL in result.output
    assert "Default local reranker model" in result.output
    assert VL_RERANK_MODEL in result.output


def test_root_query_service_help_hides_local_only_options() -> None:
    result = RUNNER.invoke(cli_main.app, ["query", "service", "--help"])

    assert result.exit_code == 0
    assert "--service-url" in result.output
    assert "--top-k" in result.output
    assert "--candidate-k" in result.output
    assert "--content-types" in result.output
    assert "--format" in result.output
    assert "--max-text-chars" in result.output
    assert "--run-mode" not in result.output
    assert "--lancedb-uri" not in result.output
    assert "--table-name" not in result.output
    assert "--embed-invoke" not in result.output
    assert "--reranker" not in result.output


def test_root_query_service_mode_uses_service_options_and_prints_json(monkeypatch) -> None:
    requests: list[Any] = []

    def fake_query_documents(request: Any) -> list[dict[str, Any]]:
        requests.append(request)
        return [
            {
                "text": "service passage",
                "source": "doc.pdf",
                "page_number": 3,
                "metadata": {"type": "text"},
                "_distance": 0.2,
            }
        ]

    monkeypatch.setattr(query_cli_app, "query_service_documents", fake_query_documents)

    result = RUNNER.invoke(
        cli_main.app,
        [
            "query",
            "service",
            "Which passages mention deployment?",
            "--service-url",
            "http://svc:7670",
            "--service-api-token",
            "secret",
            "--top-k",
            "2",
            "--candidate-k",
            "5",
            "--page-dedup",
            "--content-types",
            "text",
        ],
    )

    assert result.exit_code == 0
    assert len(requests) == 1
    request = requests[0]
    assert request.service.service_url == "http://svc:7670"
    assert request.service.service_api_token == "secret"
    assert request.retrieval.top_k == 2
    assert request.retrieval.candidate_k == 5
    assert request.retrieval.page_dedup is True
    assert request.retrieval.content_types == "text"
    assert json.loads(result.output) == [
        {"modality": "text", "page_number": 3, "score": 0.2, "source": "doc.pdf", "text": "service passage"},
    ]


def test_root_query_service_mode_rejects_local_storage_flags() -> None:
    result = RUNNER.invoke(
        cli_main.app,
        [
            "query",
            "service",
            "deployment?",
            "--lancedb-uri",
            "/tmp/lancedb",
        ],
    )

    assert result.exit_code != 0
    assert "No such option" in result.output
    assert "--lancedb-uri" in result.output


def test_root_query_service_evidence_format(monkeypatch) -> None:
    def fake_query_documents(request: Any) -> list[dict[str, Any]]:
        return [
            {
                "text": "service passage",
                "source": "doc.pdf",
                "page_number": 3,
                "metadata": {"type": "text"},
                "_distance": 0.2,
            }
        ]

    monkeypatch.setattr(query_cli_app, "query_service_documents", fake_query_documents)

    result = RUNNER.invoke(cli_main.app, ["query", "service", "deployment?", "--format", "evidence"])

    assert result.exit_code == 0
    body = json.loads(result.output)
    assert body["coverage"]["strategies_used"] == ["semantic"]
    assert body["evidence"][0]["text"] == "service passage"
    assert body["evidence"][0]["citation"] == "doc p.3"
