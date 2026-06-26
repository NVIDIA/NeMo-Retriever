# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate root query filter options into LanceDB's native ``where`` predicate.

LanceDB owns predicate execution. This module only maps stable root query
options such as source and page filters onto the table columns used by both
older local indexes and the VectorDB service, where ``metadata`` and ``source``
are JSON strings at rest.
"""

from __future__ import annotations

import json
from typing import Any

from nemo_retriever.query.options import QueryFilterOptions


def _clean_str(value: str | None) -> str | None:
    cleaned = (value or "").strip()
    return cleaned or None


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _like_clause(column: str, pattern: str, *, escape: bool = False) -> str:
    clause = f"{column} LIKE {_sql_string(pattern)}"
    if escape:
        clause += " ESCAPE '\\'"
    return clause


def _like_pattern_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _json_member_patterns(key: str, value: str) -> list[str]:
    encoded_value = _like_pattern_literal(json.dumps(value, ensure_ascii=False))
    return [
        f'%"{key}":{encoded_value}%',
        f'%"{key}": {encoded_value}%',
    ]


def _source_json_member_clause(key: str, value: str) -> str:
    return " OR ".join(_like_clause("source", pattern, escape=True) for pattern in _json_member_patterns(key, value))


def _source_id_clause(source_id: str) -> str:
    clauses = [
        _source_json_member_clause("source_id", source_id),
        f"source = {_sql_string(source_id)}",
    ]
    return "(" + " OR ".join(clauses) + ")"


def _source_clause(source: str) -> str:
    clauses = [
        _source_json_member_clause("source_id", source),
        _source_json_member_clause("source_name", source),
        f"source = {_sql_string(source)}",
    ]
    return "(" + " OR ".join(clauses) + ")"


def _page_number_clause(page_number: int) -> str:
    page = int(page_number)
    if page < 0:
        raise ValueError("page_number must be greater than or equal to 0.")
    patterns = [
        f'%"page_number":{page},%',
        f'%"page_number":{page}}}%',
        f'%"page_number": {page},%',
        f'%"page_number": {page}}}%',
        f'%"page_number":"{page}"%',
        f'%"page_number": "{page}"%',
    ]
    return "(" + " OR ".join(_like_clause("metadata", pattern) for pattern in patterns) + ")"


def build_query_where_clause(options: QueryFilterOptions) -> str | None:
    """Build the LanceDB predicate for root query filters."""
    clauses: list[str] = []

    source_id = _clean_str(options.source_id)
    if source_id is not None:
        clauses.append(_source_id_clause(source_id))

    source = _clean_str(options.source)
    if source is not None:
        clauses.append(_source_clause(source))

    if options.page_number is not None:
        clauses.append(_page_number_clause(options.page_number))

    where = _clean_str(options.where)
    if where is not None:
        clauses.append(where if not clauses else f"({where})")

    if not clauses:
        return None
    return " AND ".join(clauses)


def has_query_filters(options: QueryFilterOptions) -> bool:
    """Return whether any query filter option is set."""
    return build_query_where_clause(options) is not None


def query_filter_payload(options: QueryFilterOptions) -> dict[str, Any]:
    """Return the public service query filter payload for non-empty fields."""
    payload: dict[str, Any] = {}

    source_id = _clean_str(options.source_id)
    if source_id is not None:
        payload["source_id"] = source_id

    source = _clean_str(options.source)
    if source is not None:
        payload["source"] = source

    if options.page_number is not None:
        page_number = int(options.page_number)
        if page_number < 0:
            raise ValueError("page_number must be greater than or equal to 0.")
        payload["page_number"] = page_number

    where = _clean_str(options.where)
    if where is not None:
        payload["where"] = where

    return payload
