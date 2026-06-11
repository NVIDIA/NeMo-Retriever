# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence


QueryRunModeValue = Literal["inprocess", "service"]


@dataclass(frozen=True)
class QueryRuntimeOptions:
    run_mode: QueryRunModeValue = "inprocess"


@dataclass(frozen=True)
class QueryRetrievalOptions:
    top_k: int = 10
    candidate_k: int | None = None
    page_dedup: bool = False
    content_types: str | Sequence[str] | None = None


@dataclass(frozen=True)
class QueryEmbedOptions:
    embed_invoke_url: str | None = None
    embed_model_name: str | None = None


@dataclass(frozen=True)
class QueryRerankOptions:
    enabled: bool = False
    reranker_invoke_url: str | None = None
    reranker_model_name: str | None = None
    reranker_backend: str | None = None


@dataclass(frozen=True)
class QueryStorageOptions:
    lancedb_uri: str = "lancedb"
    table_name: str = "nemo-retriever"


@dataclass(frozen=True)
class QueryServiceOptions:
    service_url: str = "http://localhost:7670"
    service_api_token: str | None = None


@dataclass(frozen=True)
class QueryRequest:
    query: str
    runtime: QueryRuntimeOptions = field(default_factory=QueryRuntimeOptions)
    retrieval: QueryRetrievalOptions = field(default_factory=QueryRetrievalOptions)
    embed: QueryEmbedOptions = field(default_factory=QueryEmbedOptions)
    rerank: QueryRerankOptions = field(default_factory=QueryRerankOptions)
    storage: QueryStorageOptions = field(default_factory=QueryStorageOptions)
    service: QueryServiceOptions = field(default_factory=QueryServiceOptions)
