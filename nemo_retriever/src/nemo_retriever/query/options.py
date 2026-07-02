# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence


QueryRetrievalMode = Literal["auto", "dense", "hybrid", "sparse"]


@dataclass(frozen=True)
class QueryRetrievalOptions:
    top_k: int = 10
    candidate_k: int | None = None
    page_dedup: bool = False
    content_types: str | Sequence[str] | None = None
    # ``auto`` lets LanceDB table capability detection choose dense, hybrid, or
    # sparse retrieval. Explicit modes are expert overrides.
    retrieval_mode: QueryRetrievalMode = "auto"


@dataclass(frozen=True)
class QueryEmbedOptions:
    embed_invoke_url: str | None = None
    embed_model_name: str | None = None
    embed_model_name_prefix: str | None = None


@dataclass(frozen=True)
class QueryRerankOptions:
    enabled: bool = False
    reranker_invoke_url: str | None = None
    reranker_model_name: str | None = None
    reranker_backend: str | None = None
    reranker_api_key: str | None = None


@dataclass(frozen=True)
class QueryStorageOptions:
    lancedb_uri: str = "lancedb"
    table_name: str = "nemo-retriever"


@dataclass(frozen=True)
class QueryServiceOptions:
    service_url: str = "http://localhost:7670"
    service_api_token: str | None = None


@dataclass(frozen=True)
class QueryAgenticOptions:
    """Options for the agentic (ReAct) retrieval strategy."""

    enabled: bool = False
    llm_model: str | None = None
    invoke_url: str | None = None
    reasoning_effort: str | None = "high"
    backend_top_k: int = 20
    react_max_steps: int = 50
    text_truncation: int = 0
    temperature: float = 0.0


@dataclass(frozen=True)
class QueryRequest:
    query: str
    retrieval: QueryRetrievalOptions = field(default_factory=QueryRetrievalOptions)
    embed: QueryEmbedOptions = field(default_factory=QueryEmbedOptions)
    rerank: QueryRerankOptions = field(default_factory=QueryRerankOptions)
    storage: QueryStorageOptions = field(default_factory=QueryStorageOptions)
    agentic: QueryAgenticOptions = field(default_factory=QueryAgenticOptions)


@dataclass(frozen=True)
class ServiceQueryRequest:
    query: str
    retrieval: QueryRetrievalOptions = field(default_factory=QueryRetrievalOptions)
    service: QueryServiceOptions = field(default_factory=QueryServiceOptions)
