# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for ``retriever answer`` and ``retriever mcp serve`` wiring.

These tests intentionally do not hit any live LanceDB or LLM endpoints;
they exercise flag parsing, exit-code contracts, and the MCP tool schema
exposed by ``nemo_retriever.mcp_server`` so that regressions in the
front-end layer surface quickly.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import typer
from typer.testing import CliRunner

from nemo_retriever.adapters.cli.main import app
from nemo_retriever.answer_cli import (
    EXIT_BELOW_JUDGE_THRESHOLD,
    EXIT_EMPTY_RETRIEVAL,
    EXIT_GENERATION_FAILED,
    EXIT_OK,
    EXIT_USAGE,
)
from nemo_retriever.mcp_server import ANSWER_TOOL_INPUT_SCHEMA, ANSWER_TOOL_NAME

RUNNER = CliRunner()


def _make_answer_df(**overrides: Any) -> pd.DataFrame:
    """Build a one-row DataFrame in the generation schema with sensible defaults.

    Mirrors the column union produced by
    :func:`nemo_retriever.generation.eval` so the CLI's projection back
    onto the AnswerResult-shaped JSON payload is exercised end-to-end.
    The ``gen_error`` column maps to the historical ``error`` field.
    """
    row: dict[str, Any] = {
        "query": "q",
        "answer": "a",
        "chunks": ["chunk-1"],
        "metadata": [{"source": "doc"}],
        "model": "mock/llm",
        "latency_s": 0.01,
        "gen_error": None,
        "judge_score": None,
        "judge_reasoning": None,
        "judge_error": None,
        "token_f1": None,
        "exact_match": None,
        "answer_in_context": None,
        "failure_mode": None,
    }
    row.update(overrides)
    # Wrap every value in a single-element outer list so pandas treats
    # list-valued cells (chunks, metadata) as object-dtype per-row lists.
    return pd.DataFrame({k: [v] for k, v in row.items()})


# ---------------------------------------------------------------------------
# retriever answer --help / --model / --judge-model
# ---------------------------------------------------------------------------


def test_retriever_answer_help_exits_zero() -> None:
    """`retriever answer --help` must always succeed.

    Regression guard for import-time side effects in ``answer_cli`` that
    would otherwise break the whole CLI.
    """
    result = RUNNER.invoke(app, ["answer", "--help"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Single-query live RAG" in result.stdout


def test_retriever_answer_requires_model() -> None:
    """Missing ``--model`` exits 4 (usage error), not 2.

    2 is reserved for "answered but below judge threshold" in the
    documented exit-code contract, so conflating the two would break
    agents relying on the schema.
    """
    result = RUNNER.invoke(app, ["answer", "hello"])
    assert result.exit_code == EXIT_USAGE
    assert "--model is required" in result.stderr


def test_retriever_answer_judge_without_reference_exits_usage() -> None:
    """`--judge-model` without `--reference` is a usage error."""
    result = RUNNER.invoke(app, ["answer", "hi", "--model", "x", "--judge-model", "y"])
    assert result.exit_code == EXIT_USAGE
    assert "--judge-model requires --reference" in result.stderr


def test_retriever_answer_min_judge_score_without_judge_exits_usage() -> None:
    """`--min-judge-score` needs a judge."""
    result = RUNNER.invoke(app, ["answer", "hi", "--model", "x", "--min-judge-score", "4"])
    assert result.exit_code == EXIT_USAGE
    assert "--min-judge-score requires --judge-model" in result.stderr


def test_retriever_answer_missing_lancedb_uri_exits_usage() -> None:
    """Without ``--ingest``, ``--lancedb-uri`` is required."""
    result = RUNNER.invoke(app, ["answer", "hi", "--model", "x"])
    assert result.exit_code == EXIT_USAGE
    assert "--lancedb-uri is required" in result.stderr


# ---------------------------------------------------------------------------
# retriever answer: exit code plumbing
# ---------------------------------------------------------------------------


def _invoke_answer_with_patched_generation(result_df: pd.DataFrame, *args: str) -> tuple[int, str, str]:
    """Invoke the CLI with ``generation.answer`` / ``generation.eval`` patched.

    The command imports ``Retriever``, ``LiteLLMClient``, ``LLMJudge``,
    and the two ``generation.*`` entry points lazily inside the function
    body, so the patches must target the source modules -- not the
    ``answer_cli`` namespace where they would look like unbound
    attributes.  Both :func:`~nemo_retriever.generation.answer` and
    :func:`~nemo_retriever.generation.eval` are patched because the CLI
    dispatches to one or the other based on whether ``--reference`` is
    supplied.
    """
    with (
        patch("nemo_retriever.retriever.Retriever") as mock_retriever_cls,
        patch("nemo_retriever.llm.clients.LiteLLMClient") as mock_llm_cls,
        patch("nemo_retriever.llm.clients.LLMJudge") as mock_judge_cls,
        patch("nemo_retriever.generation.answer") as mock_answer_fn,
        patch("nemo_retriever.generation.eval") as mock_eval_fn,
    ):
        mock_retriever_cls.return_value = MagicMock()
        mock_llm_cls.from_kwargs.return_value = MagicMock()
        mock_judge_cls.from_kwargs.return_value = MagicMock()
        mock_answer_fn.return_value = result_df
        mock_eval_fn.return_value = result_df

        cli_result = RUNNER.invoke(app, ["answer", *args])
    return cli_result.exit_code, cli_result.stdout, cli_result.stderr


def test_retriever_answer_success_exit_zero(tmp_path: Any) -> None:
    """Happy path -- answer generated, chunks non-empty -> exit 0."""
    result = _make_answer_df()
    exit_code, _out, _err = _invoke_answer_with_patched_generation(
        result,
        "hello",
        "--model",
        "mock/llm",
        "--lancedb-uri",
        str(tmp_path),
        "--quiet",
    )
    assert exit_code == EXIT_OK


def test_retriever_answer_empty_retrieval_exits_three(tmp_path: Any) -> None:
    """Zero retrieved chunks -> exit 3."""
    result = _make_answer_df(chunks=[], metadata=[])
    exit_code, _out, _err = _invoke_answer_with_patched_generation(
        result,
        "hello",
        "--model",
        "mock/llm",
        "--lancedb-uri",
        str(tmp_path),
        "--quiet",
    )
    assert exit_code == EXIT_EMPTY_RETRIEVAL


def test_retriever_answer_generation_error_exits_five(tmp_path: Any) -> None:
    """Generation failure -> exit 5, checked before empty-retrieval."""
    result = _make_answer_df(gen_error="boom")
    exit_code, _out, _err = _invoke_answer_with_patched_generation(
        result,
        "hello",
        "--model",
        "mock/llm",
        "--lancedb-uri",
        str(tmp_path),
        "--quiet",
    )
    assert exit_code == EXIT_GENERATION_FAILED


def test_retriever_answer_below_threshold_exits_two(tmp_path: Any) -> None:
    """Judge score below ``--min-judge-score`` -> exit 2."""
    result = _make_answer_df(judge_score=2)
    exit_code, _out, _err = _invoke_answer_with_patched_generation(
        result,
        "hello",
        "--model",
        "mock/llm",
        "--lancedb-uri",
        str(tmp_path),
        "--reference",
        "ref",
        "--judge-model",
        "mock/judge",
        "--min-judge-score",
        "4",
        "--quiet",
    )
    assert exit_code == EXIT_BELOW_JUDGE_THRESHOLD


def test_retriever_answer_json_stdout_emits_valid_payload(tmp_path: Any) -> None:
    """``--json -`` sends a valid JSON AnswerResult payload to stdout."""
    result = _make_answer_df(
        judge_score=5,
        token_f1=0.8,
        exact_match=False,
        answer_in_context=True,
        failure_mode="correct",
    )
    exit_code, out, _err = _invoke_answer_with_patched_generation(
        result,
        "hello",
        "--model",
        "mock/llm",
        "--lancedb-uri",
        str(tmp_path),
        "--reference",
        "ref",
        "--judge-model",
        "mock/judge",
        "--json",
        "-",
        "--quiet",
    )
    assert exit_code == EXIT_OK
    payload = json.loads(out)
    assert payload["query"] == "q"
    assert payload["judge_score"] == 5
    assert payload["failure_mode"] == "correct"
    assert set(ANSWER_SCHEMA_KEYS).issubset(payload.keys())


# Stable documented JSON schema surface.  If this set changes, the README
# table and the MCP tool response schema must be updated in lockstep.
ANSWER_SCHEMA_KEYS: frozenset[str] = frozenset(
    {
        "query",
        "answer",
        "chunks",
        "metadata",
        "model",
        "latency_s",
        "error",
        "judge_score",
        "judge_reasoning",
        "judge_error",
        "token_f1",
        "exact_match",
        "answer_in_context",
        "failure_mode",
    }
)


# ---------------------------------------------------------------------------
# retriever mcp
# ---------------------------------------------------------------------------


def test_retriever_mcp_help_lists_serve() -> None:
    """`retriever mcp --help` must list the `serve` subcommand."""
    result = RUNNER.invoke(app, ["mcp", "--help"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "serve" in result.stdout


def test_retriever_mcp_serve_help_shows_required_flags() -> None:
    """`retriever mcp serve --help` must list the required flags."""
    result = RUNNER.invoke(app, ["mcp", "serve", "--help"])
    assert result.exit_code == 0
    for required in ("--lancedb-uri", "--model"):
        assert required in result.stdout


def test_answer_tool_schema_is_stable() -> None:
    """The MCP ``answer`` tool schema must remain stable.

    Agents pin against this shape in their ``mcp.json`` configuration, so
    changes here constitute a breaking API change and must be called out
    in release notes.
    """
    assert ANSWER_TOOL_NAME == "answer"
    schema = ANSWER_TOOL_INPUT_SCHEMA
    assert schema["type"] == "object"
    assert schema["required"] == ["question"]
    props = schema["properties"]
    assert set(props) == {"question", "top_k", "reference"}
    assert props["question"]["type"] == "string"
    assert props["top_k"]["type"] == "integer"
    assert props["reference"]["type"] == "string"


def test_retriever_mcp_serve_without_mcp_extras_exits_with_friendly_error() -> None:
    """When the ``mcp`` SDK is not importable, exit code 1 with a hint.

    We simulate the missing-extras path by patching the lazy importer.
    """
    from nemo_retriever import mcp_server as mcp_server_mod

    def _raise_import_error() -> tuple[Any, Any, Any]:
        raise typer.Exit(code=1)

    with patch.object(mcp_server_mod, "_require_mcp", side_effect=_raise_import_error):
        result = RUNNER.invoke(
            app,
            ["mcp", "serve", "--lancedb-uri", "/tmp/x", "--model", "mock/llm"],
        )
    assert result.exit_code == 1
