# SPDX-FileCopyrightText: Copyright (c) 2024-26, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from nemo_retriever.query.options import QueryRequest
from nemo_retriever.query.workflow import agentic_query_documents as run_agentic_query_documents
from nemo_retriever.query.workflow import query_documents as run_query_documents
from nemo_retriever.common.vdb.records import RetrievalHit


def query_documents(request: QueryRequest) -> list[RetrievalHit]:
    """Run the typed root query workflow."""
    return run_query_documents(request)


def agentic_query_documents(request: QueryRequest) -> list[dict[str, Any]]:
    """Run the typed root agentic (ReAct) query workflow."""
    return run_agentic_query_documents(request)
