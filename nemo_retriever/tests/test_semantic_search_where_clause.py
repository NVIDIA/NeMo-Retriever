# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the plain-SQL ``where``-clause builder used by tabular semantic search.

The builder targets the **top-level** ``label`` / ``database_name`` columns
exposed by both the Postgres tabular schema and ``TabularLanceDB`` — no
per-backend JSON tricks, just standard SQL column equality.
"""

from __future__ import annotations

from nemo_retriever.tabular_data.retrieval.data_access.semantic_search import (
    _build_metadata_where_clause,
)


def test_no_filters_returns_none() -> None:
    assert _build_metadata_where_clause() is None
    assert _build_metadata_where_clause(labels=[], database_name=None) is None
    assert _build_metadata_where_clause(labels=None, database_name="") is None


def test_single_label_emits_equality() -> None:
    assert _build_metadata_where_clause(labels=["Column"]) == "label = 'Column'"


def test_multiple_labels_emit_in_list() -> None:
    out = _build_metadata_where_clause(labels=["Column", "Table"])
    assert out == "label IN ('Column', 'Table')"


def test_label_and_database_name_combined() -> None:
    out = _build_metadata_where_clause(labels=["Column"], database_name="dor_prod")
    assert out == "label = 'Column' AND database_name = 'dor_prod'"


def test_database_name_only() -> None:
    out = _build_metadata_where_clause(database_name="dor_prod")
    assert out == "database_name = 'dor_prod'"


def test_identifier_with_underscore_and_hyphen() -> None:
    out = _build_metadata_where_clause(labels=["Column_1", "Table-2"], database_name="dor-prod_eu1")
    assert out == "label IN ('Column_1', 'Table-2') AND database_name = 'dor-prod_eu1'"
