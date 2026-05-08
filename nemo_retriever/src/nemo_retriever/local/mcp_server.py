# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP tools for agent-friendly local document search."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from nemo_retriever.local.document_search import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX,
    DEFAULT_MAX_DOCS,
    DEFAULT_TOP_K,
    _error_payload,
    _search_index_for_output,
    _status_payload,
    ask_documents,
)

mcp = FastMCP("nemo-retriever-local-search")


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".nemo-retriever-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _ensure_writable_hf_modules_cache() -> None:
    """Avoid root-owned Hugging Face dynamic module caches in agent harnesses."""

    configured = os.environ.get("HF_MODULES_CACHE")
    if configured and _is_writable_dir(Path(configured).expanduser()):
        return

    base = Path(
        os.environ.get(
            "NEMO_RETRIEVER_MCP_CACHE_DIR",
            str(Path.cwd() / ".nemo-retriever" / "mcp-cache"),
        )
    ).expanduser()
    modules_cache = base / "huggingface" / "modules"
    modules_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = str(modules_cache)


def _patterns(value: list[str] | None) -> list[str]:
    return [item for item in value or [] if item]


def _default_index_for_input_path(input_path: str) -> Path:
    resolved = Path(input_path).expanduser().resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
    return Path(".nemo-retriever") / f"local-index-{digest}"


def _resolve_index_arg(index: str | None, input_path: str | None = None) -> Path:
    if index:
        return Path(index)
    if input_path is None:
        return DEFAULT_INDEX
    return _default_index_for_input_path(input_path)


def _tool_error(command: str, exc: Exception) -> dict[str, Any]:
    return _error_payload(command, exc)


def local_document_ask_payload(
    input_path: str,
    query: str,
    index: str | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    max_docs: int = DEFAULT_MAX_DOCS,
    max_pages: int | None = None,
    inference: str = "local",
    embed_invoke_url: str | None = None,
    api_key: str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = DEFAULT_TOP_K,
    show_context: bool = True,
) -> dict[str, Any]:
    """Return an ask payload, or a structured error payload."""

    try:
        _ensure_writable_hf_modules_cache()
        return ask_documents(
            input_path=Path(input_path),
            query=query,
            index=_resolve_index_arg(index, input_path),
            include=_patterns(include),
            exclude=_patterns(exclude),
            max_docs=max_docs,
            max_pages=max_pages,
            inference=inference,
            embed_invoke_url=embed_invoke_url,
            api_key=api_key,
            embedding_model=embedding_model,
            top_k=top_k,
            output="json",
            show_context=show_context,
        )
    except Exception as exc:
        return _tool_error("ask", exc)


def local_document_search_payload(
    query: str,
    index: str = str(DEFAULT_INDEX),
    top_k: int = DEFAULT_TOP_K,
    show_context: bool = True,
    inference: str | None = None,
    embed_invoke_url: str | None = None,
    api_key: str | None = None,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    """Return a search payload for an existing index, or a structured error payload."""

    try:
        _ensure_writable_hf_modules_cache()
        return _search_index_for_output(
            output="json",
            query=query,
            index=Path(index),
            top_k=top_k,
            show_context=show_context,
            inference=inference,
            embed_invoke_url=embed_invoke_url,
            api_key=api_key,
            embedding_model=embedding_model,
        )
    except Exception as exc:
        return _tool_error("search", exc)


def local_document_status_payload(index: str = str(DEFAULT_INDEX)) -> dict[str, Any]:
    """Return local index health and staleness, or a structured error payload."""

    try:
        return _status_payload(Path(index))
    except Exception as exc:
        return _tool_error("status", exc)


@mcp.tool(
    name="local_document_ask",
    description=(
        "Index a local file or directory if needed, then retrieve evidence for a question. "
        "Use for small, single-user local corpora."
    ),
)
def local_document_ask(
    input_path: str,
    query: str,
    index: str | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    max_docs: int = DEFAULT_MAX_DOCS,
    max_pages: int | None = None,
    inference: str = "local",
    embed_invoke_url: str | None = None,
    api_key: str | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = DEFAULT_TOP_K,
    show_context: bool = True,
) -> dict[str, Any]:
    return local_document_ask_payload(
        input_path=input_path,
        query=query,
        index=index,
        include=include,
        exclude=exclude,
        max_docs=max_docs,
        max_pages=max_pages,
        inference=inference,
        embed_invoke_url=embed_invoke_url,
        api_key=api_key,
        embedding_model=embedding_model,
        top_k=top_k,
        show_context=show_context,
    )


@mcp.tool(
    name="local_document_search",
    description="Search an existing NeMo Retriever local document index and return cited evidence chunks.",
)
def local_document_search(
    query: str,
    index: str = str(DEFAULT_INDEX),
    top_k: int = DEFAULT_TOP_K,
    show_context: bool = True,
    inference: str | None = None,
    embed_invoke_url: str | None = None,
    api_key: str | None = None,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    return local_document_search_payload(
        query=query,
        index=index,
        top_k=top_k,
        show_context=show_context,
        inference=inference,
        embed_invoke_url=embed_invoke_url,
        api_key=api_key,
        embedding_model=embedding_model,
    )


@mcp.tool(
    name="local_document_status",
    description="Inspect a NeMo Retriever local document index for health, staleness, and chunk counts.",
)
def local_document_status(index: str = str(DEFAULT_INDEX)) -> dict[str, Any]:
    return local_document_status_payload(index=index)


def main() -> None:
    _ensure_writable_hf_modules_cache()
    mcp.run(transport="stdio", show_banner=False, log_level="ERROR")


if __name__ == "__main__":
    main()
