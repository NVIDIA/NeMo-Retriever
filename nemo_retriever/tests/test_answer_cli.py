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
import pytest
import typer
from typer.testing import CliRunner

from nemo_retriever.adapters.cli.main import app
from nemo_retriever.answer_cli import (
    EXIT_BELOW_JUDGE_THRESHOLD,
    EXIT_EMPTY_RETRIEVAL,
    EXIT_GENERATION_FAILED,
    EXIT_JUDGE_FAILED,
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


def test_retriever_answer_requires_model(tmp_path: Any) -> None:
    """Missing ``--model`` exits 4 (usage error), not 2.

    2 is reserved for "answered but below judge threshold" in the
    documented exit-code contract, so conflating the two would break
    agents relying on the schema.  ``--lancedb-uri`` is supplied so the
    Click-level "missing required option" check does not fire first.
    """
    result = RUNNER.invoke(app, ["answer", "hello", "--lancedb-uri", str(tmp_path)])
    assert result.exit_code == EXIT_USAGE
    assert "--model is required" in result.stderr


def test_retriever_answer_judge_without_reference_exits_usage(tmp_path: Any) -> None:
    """`--judge-model` without `--reference` is a usage error."""
    result = RUNNER.invoke(
        app,
        ["answer", "hi", "--lancedb-uri", str(tmp_path), "--model", "x", "--judge-model", "y"],
    )
    assert result.exit_code == EXIT_USAGE
    assert "--judge-model requires --reference" in result.stderr


def test_retriever_answer_min_judge_score_without_judge_exits_usage(tmp_path: Any) -> None:
    """`--min-judge-score` needs a judge."""
    result = RUNNER.invoke(
        app,
        ["answer", "hi", "--lancedb-uri", str(tmp_path), "--model", "x", "--min-judge-score", "4"],
    )
    assert result.exit_code == EXIT_USAGE
    assert "--min-judge-score requires --judge-model" in result.stderr


def test_retriever_answer_missing_lancedb_uri_exits_usage() -> None:
    """``--lancedb-uri`` is a required Typer option.

    Missing required options surface as a Click usage error (exit 2 with
    ``Missing option`` in stderr), not our own ``EXIT_USAGE`` (4), because
    Typer rejects the invocation before ``answer_command`` runs.  The
    separation between Click's argument-parsing errors and our internal
    usage errors is intentional so agents can distinguish them.

    We intentionally do not grep for ``--lancedb-uri`` in the rendered
    error panel: Click+Rich word-wrap the panel based on the test
    process's detected terminal width, which can split the option name
    across a border.  The structural invariant (Click flags the option
    as required) is asserted at the param level by
    :func:`test_retriever_mcp_serve_requires_lancedb_uri_and_model`
    and, for ``answer``, by :func:`test_retriever_answer_requires_lancedb_uri`.
    """
    result = RUNNER.invoke(app, ["answer", "hi", "--model", "x"])
    assert result.exit_code == 2
    assert "Missing option" in result.stderr


def test_retriever_answer_requires_lancedb_uri() -> None:
    """``--lancedb-uri`` is declared required on ``retriever answer``.

    Asserts directly against the Click command object rather than the
    rendered help panel so the contract does not depend on terminal
    width, ANSI colour, or Rich's panel wrapping.
    """
    answer_cmd = typer.main.get_command(app).commands["answer"]
    by_opt = {opt: p for p in answer_cmd.params for opt in p.opts}
    assert "--lancedb-uri" in by_opt, "retriever answer must expose --lancedb-uri"
    assert by_opt["--lancedb-uri"].required, "--lancedb-uri must be marked required"


def _resolve_command(*path: str) -> Any:
    """Walk a Click command group tree to a leaf command.

    ``path`` is the chain of subcommand names from the root app to the
    leaf, mirroring the way a user types it (e.g. ``("mcp", "serve")``
    or ``("eval", "batch")``).  Returns the underlying Click command
    object so callers can introspect its ``params`` list directly.
    """
    cmd = typer.main.get_command(app)
    for name in path:
        cmd = cmd.commands[name]
    return cmd


# Each tuple is (command-path, flag-name).  Parametrising over the cross
# product of "command -> flag" sites locks the convention in once across
# the four CLI surfaces that accept these bearer tokens, so a future
# refactor that drops ``envvar=`` on any of them fails CI loudly.
_NVIDIA_API_KEY_ENVVAR_SITES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("answer",), "--reranker-api-key"),
    (("answer",), "--judge-api-key"),
    (("mcp", "serve"), "--reranker-api-key"),
    (("retrieve",), "--reranker-api-key"),
    (("eval", "batch"), "--reranker-api-key"),
    (("eval", "batch"), "--judge-api-key"),
)


@pytest.mark.parametrize(("command_path", "flag"), _NVIDIA_API_KEY_ENVVAR_SITES)
def test_cli_flag_reads_nvidia_api_key_envvar(command_path: tuple[str, ...], flag: str) -> None:
    """Bearer-token flags fall back to ``$NVIDIA_API_KEY`` when not passed.

    The ``--api-key`` and ``--embedding-api-key`` flags already declare
    ``envvar="NVIDIA_API_KEY"``; ``--reranker-api-key`` and
    ``--judge-api-key`` should follow the same convention so a single
    exported token covers the whole live-RAG stack.  Asserts against the
    Click command object rather than the rendered help panel for the
    same reason as :func:`test_retriever_answer_requires_lancedb_uri`.
    """
    cmd = _resolve_command(*command_path)
    by_opt = {opt: p for p in cmd.params for opt in p.opts}
    assert flag in by_opt, f"retriever {' '.join(command_path)} must expose {flag}"
    envvar = by_opt[flag].envvar
    envvars = [envvar] if isinstance(envvar, str) else list(envvar or ())
    assert "NVIDIA_API_KEY" in envvars, (
        f"retriever {' '.join(command_path)} {flag} must declare envvar='NVIDIA_API_KEY' "
        f"to match the convention used by --api-key / --embedding-api-key"
    )


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
    """Generation failure with non-empty retrieval -> exit 5.

    Exit 5 is reserved for "retrieval returned real chunks but the LLM
    failed on them" (network timeout, model refusal, malformed response).
    ``chunks`` is left at the default non-empty value so this test
    isolates the ``gen_error`` branch; the empty-retrieval case is
    covered by :func:`test_retriever_answer_empty_retrieval_wins_over_gen_error`.
    """
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


def test_retriever_answer_empty_retrieval_wins_over_gen_error(tmp_path: Any) -> None:
    """Empty retrieval must surface as exit 3 even when ``gen_error`` is set.

    The documented contract assigns exit 3 to "retrieval returned zero
    chunks" and exit 5 to "generation failed".  Empty retrieval is
    typically the root cause of the LLM failure (invoked with empty
    context), so agents/CI branching on exit codes to diagnose failures
    need the root cause reported first.  Regression guard for the
    original ordering bug where ``gen_error`` was checked before
    ``chunks``, mislabeling knowledge-base outages as generation bugs.
    """
    result = _make_answer_df(chunks=[], metadata=[], gen_error="context too short")
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


def test_retriever_answer_judge_error_exits_six(tmp_path: Any) -> None:
    """Judge call failure (`judge_error` populated, no score) -> exit 6.

    Distinguished from exit 2 (below threshold) and exit 5 (generation
    failure) so agents can retry the judge tier without regenerating
    the answer.  Also takes precedence over exit 2 even when
    ``--min-judge-score`` is set, because a missing judge score is not
    the same as a low one.
    """
    result = _make_answer_df(judge_score=None, judge_error="judge timeout")
    exit_code, _out, err = _invoke_answer_with_patched_generation(
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
        "--quiet",
    )
    assert exit_code == EXIT_JUDGE_FAILED
    assert "judge call failed" in err


def test_retriever_answer_judge_error_wins_over_below_threshold(tmp_path: Any) -> None:
    """`judge_error` takes precedence over `--min-judge-score` (exit 6, not 2).

    Regression guard: before the EXIT_JUDGE_FAILED branch, a judge that
    errored out and returned no score would fall through to the
    ``judge_score < min_judge_score`` check and exit 2, conflating
    infrastructure failure with quality failure.
    """
    result = _make_answer_df(judge_score=None, judge_error="judge transport error")
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
    assert exit_code == EXIT_JUDGE_FAILED


def test_retriever_answer_summary_includes_judge_reasoning(tmp_path: Any) -> None:
    """When the judge tier ran, the human summary renders ``judge_reasoning``.

    Guards against the common regression where a refactor of
    ``_emit_human_summary`` drops the reasoning line silently -- it's
    already in the JSON, so a CLI user watching the terminal could go a
    long time before noticing the TTY summary is stale.
    """
    result = _make_answer_df(
        judge_score=5,
        judge_reasoning="Exact address match with the reference.",
    )
    _exit, out, _err = _invoke_answer_with_patched_generation(
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
    )
    assert "Judge:" in out
    assert "Exact address match with the reference." in out


def test_retriever_answer_summary_omits_judge_line_without_judge(tmp_path: Any) -> None:
    """No judge tier -> no ``Judge:`` line (avoids confusing log parsers)."""
    result = _make_answer_df()
    _exit, out, _err = _invoke_answer_with_patched_generation(
        result,
        "hello",
        "--model",
        "mock/llm",
        "--lancedb-uri",
        str(tmp_path),
    )
    assert "Judge:" not in out


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
    """``retriever mcp serve --help`` must exit cleanly.

    The structural check that ``--lancedb-uri`` and ``--model`` are
    present and marked required lives in
    :func:`test_retriever_mcp_serve_requires_lancedb_uri_and_model`,
    which introspects the Click command object and is therefore
    immune to Rich's terminal-width-dependent help panel rendering.
    """
    result = RUNNER.invoke(app, ["mcp", "serve", "--help"])
    assert result.exit_code == 0


def test_retriever_mcp_serve_requires_lancedb_uri_and_model() -> None:
    """``mcp serve`` must declare ``--lancedb-uri`` and ``--model`` required.

    Reads the contract directly from the Click command's ``params``
    list.  This is the representation Click uses at invocation time
    to produce ``Missing option`` errors, so it is the authoritative
    source for whether an option is required.
    """
    serve_cmd = typer.main.get_command(app).commands["mcp"].commands["serve"]
    by_opt = {opt: p for p in serve_cmd.params for opt in p.opts}
    for flag in ("--lancedb-uri", "--model"):
        assert flag in by_opt, f"retriever mcp serve must expose {flag}"
        assert by_opt[flag].required, f"{flag} must be marked required"


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
