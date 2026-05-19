# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tabular-data-specific VDB subclasses.

These extend the generic :mod:`nemo_retriever.vdb` operators with the schema
and per-row projections needed by the tabular-data ingest / retrieval
pipeline (e.g. promoting ``label`` and ``database_name`` to first-class
columns so they can be filtered with plain SQL).
"""

from nemo_retriever.tabular_data.vdb.tabular_lancedb import TabularLanceDB

__all__ = ["TabularLanceDB"]
