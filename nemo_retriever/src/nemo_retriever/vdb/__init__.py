# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo Retriever VDB implementations and graph operators."""

from nemo_retriever.vdb.adt_vdb import VDB
from nemo_retriever.vdb.factory import get_vdb_op_cls
from nemo_retriever.vdb.operators import IngestVdbOperator, RetrieveVdbOperator
from nemo_retriever.vdb.records import normalize_retrieval_results, to_client_vdb_records
from nemo_retriever.vdb.sidecar_metadata import (
    apply_sidecar_metadata_to_client_batches,
    filter_hits_by_content_metadata,
    load_meta_dataframe,
    normalize_sidecar_cell_value,
    parse_hit_content_metadata,
    split_sidecar_from_vdb_kwargs,
)

__all__ = [
    "VDB",
    "get_vdb_op_cls",
    "IngestVdbOperator",
    "RetrieveVdbOperator",
    "normalize_retrieval_results",
    "to_client_vdb_records",
    "apply_sidecar_metadata_to_client_batches",
    "filter_hits_by_content_metadata",
    "load_meta_dataframe",
    "normalize_sidecar_cell_value",
    "parse_hit_content_metadata",
    "split_sidecar_from_vdb_kwargs",
]
