# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typer subcommands for batch ``retrieve`` and ``eval batch``.

``retrieve_command`` is wired to ``retriever retrieve``: a read-only
batch retrieval tool that loads one or more questions from a JSONL
file (or a single ``--query`` argument), runs
:func:`~nemo_retriever.generation.retrieve`, and emits the resulting
DataFrame as JSONL.

``eval_batch_command`` is wired to ``retriever eval batch``: the full
evaluation chain that calls :func:`~nemo_retriever.generation.eval` and
emits the per-row metric columns.  It is intentionally a peer of the
legacy ``retriever eval run`` YAML-config entrypoint, not a replacement.

The single-question flow is ``retriever answer`` which lives in
:mod:`nemo_retriever.answer_cli` and is retained for backwards
compatibility.

Heavy imports (``pandas``, ``litellm``, ``lancedb``,
:mod:`nemo_retriever.retriever`) are deferred until the command body
runs so that ``retriever --help`` and ``retriever <subcommand> --help``
stay responsive in installations without the ``[llm]`` extra.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import typer

logger = logging.getLogger(__name__)


EXIT_OK = 0
EXIT_USAGE = 4


def _load_queries(
    queries_file: Optional[Path],
    query: Optional[str],
) -> tuple[list[str], list[Optional[str]]]:
    """Parse ``--queries`` / ``--query`` into aligned ``(queries, references)``.

    ``queries_file`` is a JSONL file with one object per line.  Each
    object must have a ``query`` field; ``reference_answer`` is optional
    for retrieve and required for eval.  ``query`` supplies a single-query
    fast path without a file.  Exactly one of the two must be given.
    """
    if bool(queries_file) == bool(query):
        typer.echo(
            "Error: exactly one of --queries (JSONL file) or --query (single string) is required.",
            err=True,
        )
        raise typer.Exit(code=EXIT_USAGE)

    if query is not None:
        return [query], [None]

    assert queries_file is not None
    questions: list[str] = []
    references: list[Optional[str]] = []
    with queries_file.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                typer.echo(
                    f"Error: --queries line {line_no} is not valid JSON: {exc}",
                    err=True,
                )
                raise typer.Exit(code=EXIT_USAGE) from exc
            if not isinstance(record, dict) or "query" not in record:
                typer.echo(
                    f"Error: --queries line {line_no} must be a JSON object with a 'query' field.",
                    err=True,
                )
                raise typer.Exit(code=EXIT_USAGE)
            questions.append(str(record["query"]))
            references.append(record.get("reference_answer"))

    if not questions:
        typer.echo("Error: --queries file contained no rows.", err=True)
        raise typer.Exit(code=EXIT_USAGE)

    return questions, references


def _emit_jsonl(df: Any, output: Optional[Path]) -> None:
    """Stream one JSON object per DataFrame row to *output* or stdout.

    ``output=None`` or ``output=Path('-')`` writes to stdout; any other
    path is opened for writing.
    """
    to_stdout = output is None or str(output) == "-"
    stream = sys.stdout if to_stdout else output.open("w", encoding="utf-8")
    try:
        for _, row in df.iterrows():
            stream.write(json.dumps(row.to_dict(), default=str, ensure_ascii=False) + "\n")
        if hasattr(stream, "flush"):
            stream.flush()
    finally:
        if not to_stdout:
            stream.close()


def _build_retriever(
    *,
    lancedb_uri: Path,
    lancedb_table: str,
    embedder: Optional[str],
    embedding_endpoint: Optional[str],
    embedding_api_key: Optional[str],
    top_k: int,
    hybrid: bool,
    reranker: Optional[str],
    reranker_endpoint: Optional[str],
    reranker_api_key: Optional[str],
) -> Any:
    """Construct a :class:`Retriever` from the common CLI flags.

    Factored out so that every batch CLI in this module shares identical
    retriever wiring, preventing flag-drift across subcommands.
    """
    from nemo_retriever.model import VL_EMBED_MODEL
    from nemo_retriever.retriever import Retriever

    resolved_embedder = embedder or VL_EMBED_MODEL

    return Retriever(
        lancedb_uri=str(Path(lancedb_uri).expanduser().resolve()),
        lancedb_table=lancedb_table,
        embedder=resolved_embedder,
        embedding_http_endpoint=embedding_endpoint,
        embedding_api_key=embedding_api_key or "",
        top_k=top_k,
        hybrid=hybrid,
        reranker=bool(reranker),
        reranker_model_name=reranker,
        reranker_endpoint=reranker_endpoint,
        reranker_api_key=reranker_api_key or "",
    )


def retrieve_command(
    query: Optional[str] = typer.Option(
        None,
        "--query",
        "-q",
        help="Single question string.  Mutually exclusive with --queries.",
    ),
    queries_file: Optional[Path] = typer.Option(
        None,
        "--queries",
        help="JSONL file with one object per line containing a 'query' field.",
    ),
    lancedb_uri: Path = typer.Option(
        ...,
        "--lancedb-uri",
        help="Path to the LanceDB directory.",
    ),
    lancedb_table: str = typer.Option(
        "nv-ingest",
        "--lancedb-table",
        help="LanceDB table name.",
    ),
    top_k: int = typer.Option(5, "--top-k", help="Number of chunks to retrieve."),
    hybrid: bool = typer.Option(False, "--hybrid", help="Enable hybrid (vector + BM25) search."),
    embedder: Optional[str] = typer.Option(None, "--embedder", help="Embedding model name."),
    embedding_endpoint: Optional[str] = typer.Option(
        None,
        "--embedding-endpoint",
        help="HTTP endpoint for a remote embedding NIM.",
    ),
    embedding_api_key: Optional[str] = typer.Option(
        None,
        "--embedding-api-key",
        envvar="NVIDIA_API_KEY",
        help="Bearer token for the embedding endpoint.",
    ),
    reranker: Optional[str] = typer.Option(None, "--reranker", help="Reranker model name."),
    reranker_endpoint: Optional[str] = typer.Option(
        None,
        "--reranker-endpoint",
        help="Base URL of a remote rerank endpoint.",
    ),
    reranker_api_key: Optional[str] = typer.Option(
        None,
        "--reranker-api-key",
        envvar="NVIDIA_API_KEY",
        help="Bearer token for the remote rerank endpoint.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to write JSONL output.  '-' (or omitted) writes to stdout.",
    ),
) -> None:
    """Batch retrieval over LanceDB -- emit ``[query, chunks, metadata]`` rows."""
    questions, _references = _load_queries(queries_file, query)

    from nemo_retriever.generation.retrieve import retrieve as retrieve_fn

    retriever = _build_retriever(
        lancedb_uri=lancedb_uri,
        lancedb_table=lancedb_table,
        embedder=embedder,
        embedding_endpoint=embedding_endpoint,
        embedding_api_key=embedding_api_key,
        top_k=top_k,
        hybrid=hybrid,
        reranker=reranker,
        reranker_endpoint=reranker_endpoint,
        reranker_api_key=reranker_api_key,
    )

    df = retrieve_fn(retriever, questions, top_k=top_k)
    _emit_jsonl(df, output)
    raise typer.Exit(code=EXIT_OK)


def eval_batch_command(
    queries_file: Optional[Path] = typer.Option(
        None,
        "--queries",
        help="JSONL file with 'query' and 'reference_answer' per line.",
    ),
    query: Optional[str] = typer.Option(
        None,
        "--query",
        "-q",
        help="Single question string (requires --reference for scoring).",
    ),
    reference: Optional[str] = typer.Option(
        None,
        "--reference",
        help="Ground-truth answer used when --query is supplied.",
    ),
    lancedb_uri: Path = typer.Option(..., "--lancedb-uri", help="Path to LanceDB directory."),
    lancedb_table: str = typer.Option("nv-ingest", "--lancedb-table", help="LanceDB table name."),
    top_k: int = typer.Option(5, "--top-k", help="Number of chunks to retrieve."),
    hybrid: bool = typer.Option(False, "--hybrid", help="Enable hybrid search."),
    embedder: Optional[str] = typer.Option(None, "--embedder", help="Embedding model name."),
    embedding_endpoint: Optional[str] = typer.Option(None, "--embedding-endpoint", help="Embedding NIM URL."),
    embedding_api_key: Optional[str] = typer.Option(
        None,
        "--embedding-api-key",
        envvar="NVIDIA_API_KEY",
        help="Bearer token for the embedding endpoint.",
    ),
    reranker: Optional[str] = typer.Option(None, "--reranker", help="Reranker model name."),
    reranker_endpoint: Optional[str] = typer.Option(None, "--reranker-endpoint", help="Rerank endpoint URL."),
    reranker_api_key: Optional[str] = typer.Option(
        None, "--reranker-api-key", envvar="NVIDIA_API_KEY", help="Rerank API key."
    ),
    model: str = typer.Option(..., "--model", help="LLM model identifier in litellm notation."),
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
    judge_model: Optional[str] = typer.Option(
        None,
        "--judge-model",
        help="Judge model identifier; enables Tier-3 LLM-as-judge scoring.",
    ),
    judge_api_base: Optional[str] = typer.Option(
        None, "--judge-api-base", help="Judge API base (defaults to --api-base)."
    ),
    judge_api_key: Optional[str] = typer.Option(
        None, "--judge-api-key", envvar="NVIDIA_API_KEY", help="Judge API key (defaults to --api-key)."
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to write JSONL output.  '-' (or omitted) writes to stdout.",
    ),
) -> None:
    """Batch evaluation: retrieve -> answer -> score -> judge for every row."""
    questions, references = _load_queries(queries_file, query)

    if query is not None:
        if reference is None:
            typer.echo("Error: --reference is required when using --query.", err=True)
            raise typer.Exit(code=EXIT_USAGE)
        references = [reference]

    missing = [i for i, r in enumerate(references) if r is None]
    if missing:
        typer.echo(
            f"Error: {len(missing)} row(s) in --queries are missing a 'reference_answer' field.",
            err=True,
        )
        raise typer.Exit(code=EXIT_USAGE)

    from nemo_retriever.generation.eval import eval as eval_fn
    from nemo_retriever.llm.clients import LiteLLMClient, LLMJudge

    retriever = _build_retriever(
        lancedb_uri=lancedb_uri,
        lancedb_table=lancedb_table,
        embedder=embedder,
        embedding_endpoint=embedding_endpoint,
        embedding_api_key=embedding_api_key,
        top_k=top_k,
        hybrid=hybrid,
        reranker=reranker,
        reranker_endpoint=reranker_endpoint,
        reranker_api_key=reranker_api_key,
    )

    llm = LiteLLMClient.from_kwargs(
        model=model,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    judge_client: Optional[Any] = None
    if judge_model is not None:
        judge_client = LLMJudge.from_kwargs(
            model=judge_model,
            api_base=judge_api_base or api_base,
            api_key=judge_api_key or api_key,
        )

    df = eval_fn(
        retriever,
        questions,
        llm=llm,
        reference=[r for r in references if r is not None],
        judge=judge_client,
        top_k=top_k,
    )
    _emit_jsonl(df, output)
    raise typer.Exit(code=EXIT_OK)
