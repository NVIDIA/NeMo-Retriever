# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``retriever mcp serve`` -- stdio MCP server over ``generation.answer``.

This module exposes a single ``answer`` MCP tool over stdio so that an
agent runtime (Cursor, Claude Desktop, Cline, etc.) can call into the
:mod:`nemo_retriever.generation` library without spawning a new process
per query.

Why a separate module
---------------------
The ``mcp`` Python SDK is an optional dependency declared under the
``[mcp]`` extras in ``pyproject.toml``.  All ``mcp.*`` imports are
performed inside the command body so that:

    pip install nemo-retriever              # no [mcp] extra
    retriever --help                        # still works
    retriever mcp --help                    # still works
    retriever mcp serve ...                 # raises a friendly error
                                            # if ``mcp`` is not installed

The runtime error points the user at ``pip install nemo-retriever[mcp]``
rather than a raw :class:`ModuleNotFoundError`.

Protocol
--------
The server exposes exactly one tool, ``answer``, whose input schema is::

    {
      "type": "object",
      "properties": {
        "question": {"type": "string"},
        "top_k": {"type": "integer", "default": <server default>},
        "reference": {"type": "string"}
      },
      "required": ["question"]
    }

The tool returns a single ``TextContent`` block whose ``text`` field is a
JSON object mirroring the historical
:class:`~nemo_retriever.llm.types.AnswerResult` schema (see
:func:`nemo_retriever.answer_cli.row_to_answer_dict`).  Agents are
expected to ``json.loads`` that text to recover structured fields.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import typer

logger = logging.getLogger(__name__)


ANSWER_TOOL_NAME = "answer"
ANSWER_TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question": {
            "type": "string",
            "description": "Natural-language question to answer.",
        },
        "top_k": {
            "type": "integer",
            "description": "Number of chunks to retrieve (defaults to server's --top-k).",
        },
        "reference": {
            "type": "string",
            "description": "Optional ground-truth reference for scoring.",
        },
    },
    "required": ["question"],
}


def _require_mcp() -> tuple[Any, Any, Any]:
    """Lazy-import the ``mcp`` SDK with a friendly error on miss.

    Returns a 3-tuple of the modules needed by the serve path:
    ``(mcp.server, mcp.server.stdio, mcp.types)``.
    """
    try:
        import mcp.server  # noqa: I001
        import mcp.server.stdio
        import mcp.types
    except ImportError as exc:
        typer.echo(
            "Error: the `mcp` Python SDK is required for `retriever mcp serve`.\n"
            "Install it with:\n"
            "    pip install nemo-retriever[mcp]",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    return mcp.server, mcp.server.stdio, mcp.types


def serve_command(
    lancedb_uri: Path = typer.Option(
        ...,
        "--lancedb-uri",
        help="Path to the LanceDB directory to serve.",
    ),
    lancedb_table: str = typer.Option(
        "nv-ingest",
        "--lancedb-table",
        help="LanceDB table name.",
    ),
    top_k: int = typer.Option(5, "--top-k", help="Default number of chunks to retrieve per call."),
    hybrid: bool = typer.Option(False, "--hybrid", help="Enable hybrid (vector + BM25) search."),
    embedder: Optional[str] = typer.Option(
        None,
        "--embedder",
        help="Embedding model name (defaults to the VL embedder used by ingestion).",
    ),
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
    reranker: Optional[str] = typer.Option(
        None,
        "--reranker",
        help="Reranker model name. Omit to disable reranking.",
    ),
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
    model: str = typer.Option(
        ...,
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
    max_tokens: int = typer.Option(4096, "--max-tokens", help="Max tokens per generation."),
) -> None:
    """Serve :func:`nemo_retriever.generation.answer` as an MCP tool over stdio.

    Intended to be registered in ``mcp.json`` by an agent runtime::

        {
          "mcpServers": {
            "nemo-retriever": {
              "command": "retriever",
              "args": ["mcp", "serve",
                       "--lancedb-uri", "/path/to/db",
                       "--model", "nvidia_nim/meta/llama-3.3-70b-instruct"]
            }
          }
        }

    The server exposes a single ``answer`` tool; its schema is identical
    to :data:`ANSWER_TOOL_INPUT_SCHEMA`.  Each tool call maps 1:1 to
    :func:`nemo_retriever.generation.answer` (or
    :func:`nemo_retriever.generation.eval` when the call includes a
    ``reference``) and returns an ``AnswerResult``-shaped JSON dict in a
    single text content block.
    """
    import asyncio

    from nemo_retriever.answer_cli import row_to_answer_dict
    from nemo_retriever.generation import answer as answer_fn
    from nemo_retriever.generation import eval as eval_fn
    from nemo_retriever.llm.clients import LiteLLMClient
    from nemo_retriever.model import VL_EMBED_MODEL
    from nemo_retriever.retriever import Retriever

    mcp_server, mcp_stdio, mcp_types = _require_mcp()

    # Construct the shared Retriever + LiteLLMClient pair once at startup
    # so the LanceDB connection pool, NIM auth-token resolution, and
    # litellm retry bookkeeping are reused across every tool call.
    retriever = Retriever(
        lancedb_uri=str(Path(lancedb_uri).expanduser().resolve()),
        lancedb_table=lancedb_table,
        embedder=embedder or VL_EMBED_MODEL,
        embedding_http_endpoint=embedding_endpoint,
        embedding_api_key=embedding_api_key or "",
        top_k=top_k,
        hybrid=hybrid,
        reranker=bool(reranker),
        reranker_model_name=reranker,
        reranker_endpoint=reranker_endpoint,
        reranker_api_key=reranker_api_key or "",
    )

    llm = LiteLLMClient.from_kwargs(
        model=model,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    server = mcp_server.Server("nemo-retriever")

    @server.list_tools()
    async def _list_tools() -> list:
        return [
            mcp_types.Tool(
                name=ANSWER_TOOL_NAME,
                description=(f"Answer a question using LanceDB at {lancedb_uri} " f"and LLM {model}."),
                inputSchema=ANSWER_TOOL_INPUT_SCHEMA,
            )
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list:
        if name != ANSWER_TOOL_NAME:
            raise ValueError(f"unknown tool: {name}")

        question = arguments.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("`question` must be a non-empty string")

        call_top_k = int(arguments.get("top_k", top_k))
        reference = arguments.get("reference")
        if reference is not None and not isinstance(reference, str):
            raise ValueError("`reference`, if supplied, must be a string")

        def _invoke() -> dict[str, Any]:
            if reference is not None:
                df = eval_fn(
                    retriever,
                    [question],
                    llm=llm,
                    reference=reference,
                    top_k=call_top_k,
                )
            else:
                df = answer_fn(
                    retriever,
                    [question],
                    llm=llm,
                    top_k=call_top_k,
                )
            return row_to_answer_dict(df.iloc[0])

        # Offload the synchronous generation call to a thread so we don't
        # block the event loop that's driving stdio I/O.
        loop = asyncio.get_running_loop()
        result_dict = await loop.run_in_executor(None, _invoke)
        payload = json.dumps(result_dict, default=str, ensure_ascii=False)
        return [mcp_types.TextContent(type="text", text=payload)]

    async def _run() -> None:
        async with mcp_stdio.stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(_run())
