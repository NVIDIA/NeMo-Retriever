# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Retriever adapters for nv-ingest-client VDB implementations."""

from nemo_retriever.vdb.operators import IngestVdbOperator, RetrieveVdbOperator
from nemo_retriever.vdb.records import normalize_retrieval_results, to_client_vdb_records
from nemo_retriever.vdb.vector_search import search_vdb_with_vectors, supports_vector_search_vdb

__all__ = [
    "IngestVdbOperator",
    "RetrieveVdbOperator",
    "normalize_retrieval_results",
    "search_vdb_with_vectors",
    "supports_vector_search_vdb",
    "to_client_vdb_records",
]
