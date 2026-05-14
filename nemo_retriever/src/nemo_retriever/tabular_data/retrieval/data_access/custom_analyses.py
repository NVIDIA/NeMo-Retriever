# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CustomAnalysis fetch + selection helpers.

Read-only operations on the ``CustomAnalysis`` / ``Sql`` subgraph plus the
small selection / rendering helpers used by the text-to-SQL agents to
filter classified analyses and turn them into markdown.
"""

from __future__ import annotations

import logging

from nemo_retriever.tabular_data.ingestion.model.reserved_words import Edges, Labels
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

logger = logging.getLogger(__name__)


def fetch_custom_analyses() -> list[dict[str, str]]:
    """Fetch all CustomAnalysis nodes from Neo4j and return as domain rules.

    Each analysis becomes ``{"name": <name>, "description": <sql>}``.
    """
    query = (
        f"MATCH (n:{Labels.CUSTOM_ANALYSIS})-[:{Edges.HAS_SQL}]->(sql:{Labels.SQL}) "
        "RETURN n.name AS name, n.description AS description, sql.sql_full_query AS sql_code"
    )
    try:
        results = get_neo4j_conn().query_read(query=query, parameters={})
    except Exception as e:
        logger.warning("Failed to fetch custom analyses from Neo4j: %s", e)
        return []

    rules: list[dict[str, str]] = []
    for row in results or []:
        name = row.get("name", "")
        description = row.get("description", "")
        sql_code = row.get("sql_code", "")
        if not name:
            continue
        parts = []
        if description:
            parts.append(description)
        if sql_code:
            parts.append(f"SQL: {sql_code}")
        if parts:
            rules.append({"name": name, "description": "\n".join(parts)})
    logger.info("Fetched %d custom analyses from Neo4j as domain rules", len(rules))
    return rules


def get_custom_analyses_ids(items):
    """Filter custom analyses by classification flag and return their IDs."""
    if not items:
        return []

    def _get(obj, key, default=None):
        """Safe getter for both Pydantic-style objects and plain dicts."""
        if hasattr(obj, key):
            return getattr(obj, key, default)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    classified_ids_and_labels = []
    for item in items:
        is_relevant = bool(_get(item, "classification", False))
        if not is_relevant:
            continue
        item_id = _get(item, "id")
        item_label = _get(item, "label")
        if item_id and item_label:
            classified_ids_and_labels.append({"id": item_id, "label": item_label})

    return classified_ids_and_labels


def build_custom_analyses_section(items, candidates):
    """Build a markdown section listing custom analyses that were used."""
    if not items:
        return ""

    # Normalize to attribute access via getattr (fallback to dict.get)
    def _get(obj, key, default=None):
        return getattr(obj, key, obj.get(key, default) if isinstance(obj, dict) else default)

    # Map candidate id -> candidate object
    by_id = {_get(c, "id"): c for c in candidates if _get(c, "id")}

    matched_lines = []
    for item in items:
        cid = _get(item, "id")
        candidate = by_id.get(cid)
        if not candidate:
            continue

        name = _get(candidate, "name", "<unknown name>")
        relevant = _get(item, "classification", False)
        if relevant:
            matched_lines.append(f"- [[[{name}/{cid}]]]")

    if not matched_lines:
        return ""

    return "\n\n**Semantic items used**:\n" + "\n".join(matched_lines)


def get_relevant_queries(candidates):
    """Collect SQL snippets from custom-analysis candidates (deduped, in order)."""
    snippet_queries = []
    for candidate in candidates:
        if candidate.get("label", "") == Labels.CUSTOM_ANALYSIS:
            analysis_sql = candidate.get("sql", [])
            if not analysis_sql:
                continue
            s_query = analysis_sql[0].get("sql_code", "")
            if s_query and s_query not in snippet_queries:
                snippet_queries.append(s_query)
    return snippet_queries
