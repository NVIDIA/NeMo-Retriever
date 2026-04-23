# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``retriever answer`` Typer subcommand.

Single-query live RAG entrypoint: retrieve from LanceDB, generate with an
LLM, and optionally score against a reference answer.  The command now
delegates to :func:`nemo_retriever.generation.answer` (and, when a
``--reference`` is supplied, :func:`nemo_retriever.generation.eval`) and
projects the resulting single DataFrame row back onto the historical
``AnswerResult``-shaped JSON payload.

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
"""

from __future__ import annotations

import atexit
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import typer

logger = logging.getLogger(__name__)


EXIT_OK = 0
EXIT_BELOW_JUDGE_THRESHOLD = 2
EXIT_EMPTY_RETRIEVAL = 3
EXIT_USAGE = 4
EXIT_GENERATION_FAILED = 5


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
    _w(f"Latency: {result_dict['latency_s']:.2f}s")
    _w(f"Chunks:  {len(result_dict['chunks'])}")
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
    _w("=" * 60)


def _ingest_files(
    files: list[Path],
    *,
    lancedb_uri: str,
    lancedb_table: str,
    embedder: str,
    embedding_endpoint: Optional[str],
    embedding_api_key: Optional[str],
    hybrid: bool,
) -> None:
    """One-shot ingestion into an existing or newly created LanceDB URI.

    Runs the inprocess GraphIngestor chain (extract -> split -> embed) and
    writes the result DataFrame to LanceDB via the same
    :func:`~nemo_retriever.vector_store.lancedb_store.handle_lancedb` helper
    that the main :mod:`~nemo_retriever.examples.graph_pipeline` uses.
    """
    # Deferred imports so that ``retriever answer --help`` does not pull in
    # pyarrow / ray / lancedb at argparse time.
    from nemo_retriever import create_ingestor
    from nemo_retriever.params import EmbedParams, ExtractParams, TextChunkParams
    from nemo_retriever.vector_store.lancedb_store import handle_lancedb

    file_strs = [str(p) for p in files]
    logger.info("Ingesting %d file(s) into %s::%s", len(file_strs), lancedb_uri, lancedb_table)

    embed_kwargs: dict[str, Any] = {"model_name": embedder}
    if embedding_endpoint:
        embed_kwargs["embed_invoke_url"] = embedding_endpoint
    if embedding_api_key:
        embed_kwargs["api_key"] = embedding_api_key

    ingestor = (
        create_ingestor(run_mode="inprocess")
        .files(file_strs)
        .extract(ExtractParams())
        .split(TextChunkParams())
        .embed(EmbedParams(**embed_kwargs))
    )
    result_df = ingestor.ingest()
    handle_lancedb(result_df, lancedb_uri, lancedb_table, mode="overwrite", hybrid=hybrid)
    logger.info("Ingestion complete.")


def answer_command(
    question: str = typer.Argument(
        ...,
        help="Natural-language question to answer.",
        show_default=False,
    ),
    # ── Retrieval storage ─────────────────────────────────────────────────
    lancedb_uri: Optional[Path] = typer.Option(
        None,
        "--lancedb-uri",
        help="Path to the LanceDB directory. Required unless --ingest is used, "
        "in which case a temporary directory is created and cleaned up on exit.",
    ),
    lancedb_table: str = typer.Option(
        "nv-ingest",
        "--lancedb-table",
        help="LanceDB table name.",
    ),
    # ── Optional one-shot ingestion ──────────────────────────────────────
    ingest: list[Path] = typer.Option(
        [],
        "--ingest",
        help="File(s) to ingest into LanceDB before answering. " "Multiple --ingest flags are allowed.",
    ),
    keep_ingest: bool = typer.Option(
        False,
        "--keep-ingest",
        help="Keep the temporary LanceDB directory created by --ingest " "instead of deleting it on exit.",
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

    ingest_files = [p for p in (ingest or []) if p is not None]
    resolved_lancedb_uri: Optional[str] = None

    if ingest_files and lancedb_uri is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="retriever-answer-"))
        resolved_lancedb_uri = str(tmp_dir)
        if not keep_ingest:
            atexit.register(shutil.rmtree, str(tmp_dir), True)
        logger.info("Created temp LanceDB dir: %s", tmp_dir)
    elif lancedb_uri is not None:
        resolved_lancedb_uri = str(Path(lancedb_uri).expanduser().resolve())

    if resolved_lancedb_uri is None:
        typer.echo(
            "Error: --lancedb-uri is required (or use --ingest to create a temp DB on the fly).",
            err=True,
        )
        raise typer.Exit(code=EXIT_USAGE)

    from nemo_retriever.generation import answer as answer_fn
    from nemo_retriever.generation import eval as eval_fn
    from nemo_retriever.llm.clients import LiteLLMClient, LLMJudge
    from nemo_retriever.model import VL_EMBED_MODEL
    from nemo_retriever.retriever import Retriever

    resolved_embedder = embedder or VL_EMBED_MODEL

    if ingest_files:
        _ingest_files(
            ingest_files,
            lancedb_uri=resolved_lancedb_uri,
            lancedb_table=lancedb_table,
            embedder=resolved_embedder,
            embedding_endpoint=embedding_endpoint,
            embedding_api_key=embedding_api_key,
            hybrid=hybrid,
        )

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

    # --json - routes JSON to stdout and the summary to stderr so the two
    # streams can be piped independently (e.g. agents capture stdout, humans
    # watch stderr).  Any other --json target keeps the summary on stdout.
    json_to_stdout = json_out is not None and str(json_out) == "-"
    _emit_json(result_dict, json_out)
    if not quiet:
        _emit_human_summary(result_dict, to_stderr=json_to_stdout)

    gen_error = result_dict.get("error")
    chunks = result_dict.get("chunks") or []
    judge_score = result_dict.get("judge_score")

    if gen_error is not None:
        raise typer.Exit(code=EXIT_GENERATION_FAILED)
    if not chunks:
        raise typer.Exit(code=EXIT_EMPTY_RETRIEVAL)
    if min_judge_score is not None and (judge_score is None or judge_score < min_judge_score):
        raise typer.Exit(code=EXIT_BELOW_JUDGE_THRESHOLD)

    raise typer.Exit(code=EXIT_OK)
