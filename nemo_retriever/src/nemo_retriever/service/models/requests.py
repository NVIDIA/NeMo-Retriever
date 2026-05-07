# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Metadata JSON sent alongside the uploaded file."""

    job_id: str | None = None
    filename: str | None = None
    content_type: str | None = None
    page_number: int | None = None
    total_pages: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """JSON body for ``POST /v1/query``.

    ``query`` accepts a single string or a list of strings.  When a list
    is provided the response contains one :class:`QueryResultSet` per
    input query, in the same order.
    """

    query: str | list[str] = Field(
        ...,
        description=(
            "One or more natural-language text queries to run against the "
            "vector store.  Pass a single string for a one-shot query or a "
            "list of strings for batch retrieval."
        ),
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="Maximum number of results per query. Falls back to the server default when omitted.",
    )
    lancedb_uri: str | None = Field(
        default=None,
        description="Override the LanceDB URI. Falls back to the server default when omitted.",
    )
    lancedb_table: str | None = Field(
        default=None,
        description="Override the LanceDB table name. Falls back to the server default when omitted.",
    )
    hybrid: bool = Field(
        default=False,
        description="Enable hybrid vector + BM25 search with RRF reranking.",
    )


class RerankRequest(BaseModel):
    """JSON body for ``POST /v1/rerank``."""

    query: str = Field(
        ...,
        min_length=1,
        description="The query string to score each passage against.",
    )
    passages: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description=(
            "Passages to rerank.  Each dict must contain a ``text`` key.  "
            "Any additional keys (source, page_number, metadata, …) are "
            "preserved in the response so callers can round-trip full hit "
            "dicts from ``/v1/query``."
        ),
    )
    model_name: str | None = Field(
        default=None,
        description="Override the reranker model name. Falls back to the server default when omitted.",
    )
    top_n: int | None = Field(
        default=None,
        ge=1,
        le=1000,
        description="Return only the top-N results after reranking. Falls back to the server default when omitted.",
    )
