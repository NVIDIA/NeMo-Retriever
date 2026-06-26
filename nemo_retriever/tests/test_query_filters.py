# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from nemo_retriever.query.filters import build_query_where_clause, has_query_filters, query_filter_payload
from nemo_retriever.query.options import QueryFilterOptions


def test_structured_source_id_and_page_filters_build_lancedb_predicate() -> None:
    assert build_query_where_clause(QueryFilterOptions(source_id="docs/a.pdf", page_number=3)) == (
        "(source LIKE '%\"source_id\":\"docs/a.pdf\"%' ESCAPE '\\' "
        "OR source LIKE '%\"source_id\": \"docs/a.pdf\"%' ESCAPE '\\' "
        "OR source = 'docs/a.pdf') "
        "AND (metadata LIKE '%\"page_number\":3,%' "
        "OR metadata LIKE '%\"page_number\":3}%' "
        "OR metadata LIKE '%\"page_number\": 3,%' "
        "OR metadata LIKE '%\"page_number\": 3}%' "
        'OR metadata LIKE \'%"page_number":"3"%\' '
        'OR metadata LIKE \'%"page_number": "3"%\')'
    )


def test_source_filter_matches_source_identifier_name_or_raw_source() -> None:
    assert build_query_where_clause(QueryFilterOptions(source="a.pdf")) == (
        "(source LIKE '%\"source_id\":\"a.pdf\"%' ESCAPE '\\' "
        "OR source LIKE '%\"source_id\": \"a.pdf\"%' ESCAPE '\\' "
        "OR source LIKE '%\"source_name\":\"a.pdf\"%' ESCAPE '\\' "
        "OR source LIKE '%\"source_name\": \"a.pdf\"%' ESCAPE '\\' "
        "OR source = 'a.pdf')"
    )


def test_source_filters_escape_lancedb_like_wildcards() -> None:
    where = build_query_where_clause(QueryFilterOptions(source_id="my_report%2026.pdf"))

    assert where is not None
    assert "my\\_report\\%2026.pdf" in where
    assert "ESCAPE '\\'" in where
    assert "source = 'my_report%2026.pdf'" in where


def test_advanced_where_is_pass_through_and_combines_with_structured_filters() -> None:
    assert build_query_where_clause(QueryFilterOptions(where="text = 'alpha'")) == "text = 'alpha'"
    assert build_query_where_clause(QueryFilterOptions(page_number=2, where="text = 'alpha'")) == (
        "(metadata LIKE '%\"page_number\":2,%' "
        "OR metadata LIKE '%\"page_number\":2}%' "
        "OR metadata LIKE '%\"page_number\": 2,%' "
        "OR metadata LIKE '%\"page_number\": 2}%' "
        'OR metadata LIKE \'%"page_number":"2"%\' '
        'OR metadata LIKE \'%"page_number": "2"%\') '
        "AND (text = 'alpha')"
    )


def test_has_query_filters_reports_non_empty_options() -> None:
    assert not has_query_filters(QueryFilterOptions())
    assert has_query_filters(QueryFilterOptions(source_id="docs/a.pdf"))
    assert has_query_filters(QueryFilterOptions(where="text = 'alpha'"))


def test_negative_page_filter_is_rejected() -> None:
    with pytest.raises(ValueError, match="page_number must be greater than or equal to 0"):
        build_query_where_clause(QueryFilterOptions(page_number=-1))

    with pytest.raises(ValueError, match="page_number must be greater than or equal to 0"):
        query_filter_payload(QueryFilterOptions(page_number=-1))
