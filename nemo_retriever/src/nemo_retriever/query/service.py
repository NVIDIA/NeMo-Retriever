# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from nemo_retriever.query.options import QueryRequest
from nemo_retriever.retriever import _shape_query_hits
from nemo_retriever.service.client import RetrieverServiceClient
from nemo_retriever.vdb.records import RetrievalHit


def _service_retrieval_top_k(request: QueryRequest) -> int:
    effective_top_k = int(request.retrieval.top_k)
    candidate_top_k = (
        int(request.retrieval.candidate_k) if request.retrieval.candidate_k is not None else effective_top_k
    )
    if candidate_top_k < effective_top_k:
        raise ValueError(f"candidate_k ({candidate_top_k}) must be greater than or equal to top_k ({effective_top_k}).")
    return candidate_top_k


def query_documents(request: QueryRequest) -> list[RetrievalHit]:
    """Run root query through the retriever service and preserve local result shaping."""
    retrieval_top_k = _service_retrieval_top_k(request)
    client = RetrieverServiceClient(
        base_url=request.service.service_url,
        api_token=request.service.service_api_token,
    )
    raw_result_sets = client.query(request.query, top_k=retrieval_top_k)
    raw_hits = raw_result_sets[0] if raw_result_sets else []
    return _shape_query_hits(
        raw_hits,
        top_k=request.retrieval.top_k,
        page_dedup=request.retrieval.page_dedup,
        content_types=request.retrieval.content_types,
    )
