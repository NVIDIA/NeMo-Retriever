# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``retriever answer`` Typer subcommand.

Single-query live RAG entrypoint: retrieve from LanceDB, generate with an
LLM, and optionally score against a reference answer.  The command
delegates to :func:`nemo_retriever.generation.answer` (and, when a
``--reference`` is supplied, :func:`nemo_retriever.generation.eval`) and
projects the resulting single DataFrame row back onto the historical
``AnswerResult``-shaped JSON payload.

This command is strictly a *query-time* entrypoint.  It does not ingest
documents; the caller is expected to have already populated the target
LanceDB table via ``retriever pipeline run`` (see
:mod:`nemo_retriever.pipeline`).  Keeping ingestion and retrieval behind
separate CLI verbs mirrors the package boundary between
:mod:`nemo_retriever.pipeline` and :mod:`nemo_retriever.generation` and
lets each surface evolve independently.

Design notes
------------
Heavy imports (``litellm``, ``lancedb``, :mod:`nemo_retriever.retriever`)
are deferred to inside the command body so that ``retriever --help`` and
``retriever answer --help`` stay responsive for environments without the
``[llm]`` extra installed.

The command supports a machine-readable JSON mode (``--json -`` or
``--json <path>``) intended for agent / shell / CI consumption; the JSON
schema is stable and mirrors the historical
:class:`nemo_retriever.llm.types.AnswerResult` (with ``gen_error``
renamed to ``error`` for backward compatibility with prior tooling).

Exit-code contract (schema'd for agents and CI)::

    0 -- success; answer generated and (if ``--min-judge-score`` was set)
         the judge score met the threshold.
    2 -- answered but the judge score was below ``--min-judge-score``.
    3 -- retrieval returned zero chunks; no answer generated.
    4 -- usage error (e.g. ``--judge-model`` without ``--reference``).
    5 -- generation failed (``AnswerResult.error`` populated).
    6 -- judge call failed (``judge_error`` populated; the answer itself
         may still be usable).  Distinguished from exit 5 so agents can
         retry the judge without regenerating the answer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import typer

EXIT_OK = 0
EXIT_BELOW_JUDGE_THRESHOLD = 2
EXIT_EMPTY_RETRIEVAL = 3
EXIT_USAGE = 4
EXIT_GENERATION_FAILED = 5
EXIT_JUDGE_FAILED = 6


# Columns in the generation DataFrame that we project onto the
# historical AnswerResult JSON schema, in the historical key order.
# ``(out_key, src_col)`` pairs.  ``gen_error`` in the DataFrame maps to
# the historical ``error`` field.
ANSWER_DICT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("query", "query"),
    ("answer", "answer"),
    ("chunks", "chunks"),
    ("metadata", "metadata"),
    ("model", "model"),
    ("latency_s", "latency_s"),
    ("error", "gen_error"),
    ("judge_score", "judge_score"),
    ("judge_reasoning", "judge_reasoning"),
    ("judge_error", "judge_error"),
    ("token_f1", "token_f1"),
    ("exact_match", "exact_match"),
    ("answer_in_context", "answer_in_context"),
    ("failure_mode", "failure_mode"),
)


def row_to_answer_dict(row: Any) -> dict[str, Any]:
    """Project a single DataFrame row onto the AnswerResult-shaped dict.

    Columns that were never produced (no reference -> no score columns;
    no judge -> no judge columns) default to ``None`` so the JSON schema
    stays stable regardless of which scoring tiers were exercised.

    Numpy scalars (``np.int64``, ``np.float64``, ``np.bool_``) that
    pandas returns for ``pd.Series.get`` are unboxed to native Python
    types so ``json.dumps`` produces ``5`` / ``0.8`` / ``false`` rather
    than their string fallbacks via ``default=str``.
    """
    import numpy as np
    import pandas as pd

    def _unbox(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (list, dict)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if pd.isna(value):
            return None
        return value

    def _get(column: str) -> Any:
        value = row.get(column, None) if hasattr(row, "get") else None
        return _unbox(value)

    return {out_key: _get(src_col) for out_key, src_col in ANSWER_DICT_COLUMNS}


def _emit_json(result_dict: dict[str, Any], json_out: Optional[Path]) -> None:
    """Emit the answer payload as JSON to stdout or a file.

    ``json_out`` semantics:
        * ``None`` -- no JSON emission (human mode only).
        * ``Path("-")`` -- write JSON to stdout.
        * any other ``Path`` -- write JSON to that file.
    """
    payload = json.dumps(result_dict, default=str, indent=2, ensure_ascii=False)

    if json_out is None:
        return
    if str(json_out) == "-":
        typer.echo(payload)
        return
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(payload, encoding="utf-8")


def _emit_human_summary(result_dict: dict[str, Any], *, to_stderr: bool) -> None:
    """Emit a short human-readable summary to stdout or stderr."""
    stream = sys.stderr if to_stderr else sys.stdout

    def _w(line: str) -> None:
        stream.write(line + "\n")

    _w("=" * 60)
    _w(f"Query:   {result_dict['query']}")
    _w(f"Model:   {result_dict['model']}")
    latency = result_dict.get("latency_s")
    _w(f"Latency: {latency:.2f}s" if isinstance(latency, (int, float)) else "Latency: N/A")
    chunks = result_dict.get("chunks") or []
    _w(f"Chunks:  {len(chunks)}")
    if result_dict.get("error"):
        _w(f"Error:   {result_dict['error']}")
    _w("-" * 60)
    _w(f"Answer:\n{result_dict['answer']}")
    score_bits: list[str] = []
    if result_dict.get("judge_score") is not None:
        score_bits.append(f"judge_score={result_dict['judge_score']}/5")
    if result_dict.get("token_f1") is not None:
        score_bits.append(f"token_f1={result_dict['token_f1']:.3f}")
    if result_dict.get("exact_match") is not None:
        score_bits.append(f"exact_match={result_dict['exact_match']}")
    if result_dict.get("answer_in_context") is not None:
        score_bits.append(f"answer_in_context={result_dict['answer_in_context']}")
    if result_dict.get("failure_mode"):
        score_bits.append(f"failure_mode={result_dict['failure_mode']}")
    if score_bits:
        _w("-" * 60)
        _w("Scoring: " + "  ".join(score_bits))
    # Render the judge's one-sentence rationale on its own line when the
    # judge tier ran.  It's too long to pack into the space-joined
    # score_bits summary, so keep it structurally distinct.  Hard-cap at
    # 500 chars to protect the terminal from a verbose judge; full text
    # remains available in the JSON payload under ``judge_reasoning``.
    if result_dict.get("judge_score") is not None:
        reasoning = result_dict.get("judge_reasoning")
        if reasoning:
            trimmed = reasoning if len(reasoning) <= 500 else f"{reasoning[:500]}... (truncated; see JSON)"
            _w(f"Judge:   {trimmed}")
    _w("=" * 60)


def answer_command(
    question: str = typer.Argument(
        ...,
        help="Natural-language question to answer.",
        show_default=False,
    ),
    # ── Retrieval storage ─────────────────────────────────────────────────
    lancedb_uri: Path = typer.Option(
        ...,
        "--lancedb-uri",
        help="Path to the LanceDB directory populated by `retriever pipeline run`. "
        "Required; this command does not ingest.",
    ),
    lancedb_table: str = typer.Option(
        "nv-ingest",
        "--lancedb-table",
        help="LanceDB table name (default matches `retriever pipeline run`).",
    ),
    # ── Retrieval tuning ─────────────────────────────────────────────────
    top_k: int = typer.Option(5, "--top-k", help="Number of chunks to retrieve."),
    hybrid: bool = typer.Option(False, "--hybrid", help="Enable hybrid (vector + BM25) search."),
    nprobes: int = typer.Option(0, "--nprobes", help="LanceDB nprobes for IVF search."),
    refine_factor: int = typer.Option(10, "--refine-factor", help="LanceDB refine factor."),
    # ── Embedding ────────────────────────────────────────────────────────
    embedder: Optional[str] = typer.Option(
        None,
        "--embedder",
        help="Embedding model name. Defaults to the same VL embedder used by the ingestion pipeline.",
    ),
    embedding_endpoint: Optional[str] = typer.Option(
        None,
        "--embedding-endpoint",
        help="HTTP endpoint for a remote embedding NIM. If unset, local HF inference is used.",
    ),
    embedding_api_key: Optional[str] = typer.Option(
        None,
        "--embedding-api-key",
        envvar="NVIDIA_API_KEY",
        help="Bearer token for the embedding endpoint.",
    ),
    # ── Reranking ────────────────────────────────────────────────────────
    reranker: Optional[str] = typer.Option(
        None,
        "--reranker",
        help="Reranker model name (e.g. 'nvidia/llama-nemotron-rerank-1b-v2'). " "Omit to disable reranking.",
    ),
    reranker_endpoint: Optional[str] = typer.Option(
        None,
        "--reranker-endpoint",
        help="Base URL of a remote rerank endpoint (NIM/vLLM). If unset, local HF inference is used.",
    ),
    reranker_api_key: Optional[str] = typer.Option(
        None,
        "--reranker-api-key",
        envvar="NVIDIA_API_KEY",
        help="Bearer token for the remote rerank endpoint.",
    ),
    reranker_modality: str = typer.Option(
        "text",
        "--reranker-modality",
        help="Reranking modality; 'text' or 'text_image'.",
    ),
    # ── Generation ───────────────────────────────────────────────────────
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="LLM model identifier in litellm notation, e.g. 'nvidia_nim/meta/llama-3.3-70b-instruct'.",
    ),
    api_base: Optional[str] = typer.Option(None, "--api-base", help="API base URL for the LLM."),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="NVIDIA_API_KEY",
        help="Bearer token for the LLM.",
    ),
    temperature: float = typer.Option(0.0, "--temperature", help="Sampling temperature."),
    top_p: Optional[float] = typer.Option(None, "--top-p", help="Nucleus sampling parameter."),
    max_tokens: int = typer.Option(4096, "--max-tokens", help="Max tokens to generate."),
    # ── Judging / reference ──────────────────────────────────────────────
    reference: Optional[str] = typer.Option(
        None,
        "--reference",
        help="Ground-truth answer for token-F1, exact-match, and judge scoring.",
    ),
    judge_model: Optional[str] = typer.Option(
        None,
        "--judge-model",
        help="Judge model identifier in litellm notation. Requires --reference.",
    ),
    judge_api_base: Optional[str] = typer.Option(
        None,
        "--judge-api-base",
        help="API base URL for the judge LLM (defaults to --api-base).",
    ),
    judge_api_key: Optional[str] = typer.Option(
        None,
        "--judge-api-key",
        envvar="NVIDIA_API_KEY",
        help="Bearer token for the judge LLM (defaults to --api-key).",
    ),
    min_judge_score: Optional[int] = typer.Option(
        None,
        "--min-judge-score",
        help="Exit 2 if judge_score < this value. Requires --judge-model and --reference.",
    ),
    # ── Output ───────────────────────────────────────────────────────────
    json_out: Optional[Path] = typer.Option(
        None,
        "--json",
        help="Path for JSON output, or '-' for stdout. Human summary goes to the other stream.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        help="Suppress the human-readable summary (JSON output, if requested, still emits).",
    ),
) -> None:
    """Answer a single question using live RAG.

    The command performs ``retrieve -> generate`` via
    :func:`nemo_retriever.generation.answer` and, when a ``--reference``
    is supplied (optionally with ``--judge-model``), routes through
    :func:`nemo_retriever.generation.eval` to add token-F1 / exact-match /
    answer-in-context / LLM-as-judge scoring.  Pipes cleanly into agents
    and CI via schema'd ``--json`` output and the exit-code contract
    documented on this module.

    ``--lancedb-uri`` is required.  Populate the target LanceDB with
    ``retriever pipeline run`` before invoking this command.
    """
    if model is None:
        typer.echo("Error: --model is required.", err=True)
        raise typer.Exit(code=EXIT_USAGE)

    if judge_model is not None and reference is None:
        typer.echo("Error: --judge-model requires --reference.", err=True)
        raise typer.Exit(code=EXIT_USAGE)

    if min_judge_score is not None and judge_model is None:
        typer.echo("Error: --min-judge-score requires --judge-model.", err=True)
        raise typer.Exit(code=EXIT_USAGE)

    resolved_lancedb_uri = str(Path(lancedb_uri).expanduser().resolve())

    from nemo_retriever.generation import answer as answer_fn
    from nemo_retriever.generation import eval as eval_fn
    from nemo_retriever.llm.clients import LiteLLMClient, LLMJudge
    from nemo_retriever.model import VL_EMBED_MODEL
    from nemo_retriever.retriever import Retriever

    resolved_embedder = embedder or VL_EMBED_MODEL

    retriever = Retriever(
        lancedb_uri=resolved_lancedb_uri,
        lancedb_table=lancedb_table,
        embedder=resolved_embedder,
        embedding_http_endpoint=embedding_endpoint,
        embedding_api_key=embedding_api_key or "",
        top_k=top_k,
        hybrid=hybrid,
        nprobes=nprobes,
        refine_factor=refine_factor,
        reranker=bool(reranker),
        reranker_model_name=reranker,
        reranker_endpoint=reranker_endpoint,
        reranker_api_key=reranker_api_key or "",
        rerank_modality=reranker_modality,
    )

    llm = LiteLLMClient.from_kwargs(
        model=model,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    judge = None
    if judge_model is not None:
        judge = LLMJudge.from_kwargs(
            model=judge_model,
            api_base=judge_api_base or api_base,
            api_key=judge_api_key or api_key,
        )

    if reference is not None:
        df = eval_fn(
            retriever,
            [question],
            llm=llm,
            reference=reference,
            judge=judge,
            top_k=top_k,
        )
    else:
        df = answer_fn(
            retriever,
            [question],
            llm=llm,
            top_k=top_k,
        )

    row = df.iloc[0]
    result_dict = row_to_answer_dict(row)

    # ``--json -`` routes JSON to stdout and the human summary to stderr.
    json_to_stdout = json_out is not None and str(json_out) == "-"
    _emit_json(result_dict, json_out)
    if not quiet:
        _emit_human_summary(result_dict, to_stderr=json_to_stdout)

    gen_error = result_dict.get("error")
    chunks = result_dict.get("chunks") or []
    judge_score = result_dict.get("judge_score")
    judge_error = result_dict.get("judge_error")

    # Exit 3 (empty retrieval) precedes exit 5 (generation failed) so callers
    # are pointed at the root cause rather than the downstream symptom.
    # Exit 6 (judge failed) is distinguished from exit 5 so agents can retry
    # the judge without regenerating the answer; it also takes precedence
    # over exit 2 because a missing judge score is not the same as a low one.
    if not chunks:
        raise typer.Exit(code=EXIT_EMPTY_RETRIEVAL)
    if gen_error is not None:
        raise typer.Exit(code=EXIT_GENERATION_FAILED)
    if judge_error is not None:
        typer.echo(f"Error: judge call failed: {judge_error}", err=True)
        raise typer.Exit(code=EXIT_JUDGE_FAILED)
    if min_judge_score is not None and (judge_score is None or judge_score < min_judge_score):
        raise typer.Exit(code=EXIT_BELOW_JUDGE_THRESHOLD)

    raise typer.Exit(code=EXIT_OK)
