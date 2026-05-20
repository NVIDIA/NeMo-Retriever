# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Foreign-key / join discovery against Neo4j.

* :func:`_get_relevant_fks` expands outward up to 3 levels through ``fk`` /
  ``join`` edges to gather all FK relationships among the connected tables.
* :func:`_apply_foreign_key_hints` decorates relevant-table dicts in place.
* :func:`get_relevant_fks_from_candidates_tables` and
  :func:`get_relevant_tables_with_fks` are the two public combinators used
  by the deep_agent and text-to-SQL pipelines respectively.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from nemo_retriever.tabular_data.neo4j import get_neo4j_conn
from nemo_retriever.tabular_data.retrieval.data_access.relevant_tables import (
    get_relevant_tables,
    get_relevant_tables_from_candidates,
)

if TYPE_CHECKING:
    from nemo_retriever.retriever import Retriever

logger = logging.getLogger(__name__)


def _get_relevant_fks(tables_ids):
    # Build a connected graph by expanding from target tables through FK relationships
    query = """
    // Start with target tables and expand outward to find connected tables
    WITH $tables_ids as current_ids

    // Level 1: Find tables connected via FK
    OPTIONAL MATCH (t0:table WHERE t0.id IN current_ids)
          -[:schema]->(:column)-[:fk]-(:column)<-[:schema]-(t1:table)
    WITH current_ids, collect(DISTINCT t1.id) as new_ids_1
    WITH current_ids + new_ids_1 as level_1_ids

    // Level 2
    OPTIONAL MATCH (t1:table WHERE t1.id IN level_1_ids)
          -[:schema]->(:column)-[:fk]-(:column)<-[:schema]-(t2:table)
    WITH level_1_ids, collect(DISTINCT t2.id) as new_ids_2
    WITH level_1_ids + new_ids_2 as level_2_ids

    // Level 3
    OPTIONAL MATCH (t2:table WHERE t2.id IN level_2_ids)
          -[:schema]->(:column)-[:fk]-(:column)<-[:schema]-(t3:table)
    WITH level_2_ids, collect(DISTINCT t3.id) as new_ids_3
    WITH level_2_ids + new_ids_3 as all_table_ids

    // Get all FK relationships between these tables
    MATCH (t1:table)-[:schema]->(col1:column)-[:fk]-(col2:column)<-[:schema]-(t2:table)
    WHERE t1.id IN all_table_ids AND t2.id IN all_table_ids
      AND t1.id < t2.id  // Avoid duplicates by keeping only one direction

    RETURN collect(DISTINCT {
        table1: t1.schema_name + '.' + t1.name,
        column1: col1.name,
        column1_datatype: coalesce(col1.data_type, 'None'),
        table2: t2.schema_name + '.' + t2.name,
        column2: col2.name,
        column2_datatype: coalesce(col2.data_type, 'None')
    }) as list_of_foreign_keys
    """
    results = get_neo4j_conn().query_read(query, {"tables_ids": tables_ids})
    if len(results) > 0:
        result_fks = results[0]["list_of_foreign_keys"]
    else:
        result_fks = []

    # Build a connected graph by expanding from target tables through FK relationships
    query = """
    // Start with target tables and expand outward to find connected tables

    // Level 1: Find tables connected via FK
    OPTIONAL MATCH (t0:table WHERE t0.id IN $tables_ids)-[:join]-(t1:table)
    WITH collect(DISTINCT t1.id) as new_ids_1
    WITH $tables_ids + new_ids_1 as level_1_ids

    // Level 2
    OPTIONAL MATCH (t1:table WHERE t1.id IN level_1_ids)-[:join]-(t2:table)
    WITH level_1_ids, collect(DISTINCT t2.id) as new_ids_2
    WITH level_1_ids + new_ids_2 as level_2_ids

    // Level 3
    OPTIONAL MATCH (t2:table WHERE t2.id IN level_2_ids)-[:join]-(t3:table)
    WITH level_2_ids, collect(DISTINCT t3.id) as new_ids_3
    WITH level_2_ids + new_ids_3 as all_table_ids

    // Get all join relationships between these tables and parse the join property
    MATCH (t1:table)-[rel:join]-(t2:table)
    WHERE t1.id IN all_table_ids AND t2.id IN all_table_ids
      AND t1.id < t2.id  // Avoid duplicates by keeping only one direction
      AND rel.join IS NOT NULL

    // Parse the join property: split by operators and extract left/right sides
    WITH t1, t2, rel,
         trim(apoc.text.split(rel.join, '<=|>=|=|<|>')[0]) as left_side,
         trim(apoc.text.split(rel.join, '<=|>=|=|<|>')[1]) as right_side

    // Parse left side: SCHEMA.TABLE.COLUMN (handle potential whitespace)
    WITH t1, t2, rel, left_side, right_side,
         trim(split(left_side, '.')[0]) as left_schema,
         trim(split(left_side, '.')[1]) as left_table,
         trim(split(left_side, '.')[2]) as left_column,
         trim(split(right_side, '.')[0]) as right_schema,
         trim(split(right_side, '.')[1]) as right_table,
         trim(split(right_side, '.')[2]) as right_column
    WHERE left_schema IS NOT NULL AND left_table IS NOT NULL AND left_column IS NOT NULL
      AND right_schema IS NOT NULL AND right_table IS NOT NULL AND right_column IS NOT NULL

    // Match the actual column nodes for left side
    OPTIONAL MATCH (s1:schema{name: left_schema})-[:schema]->
        (tbl1:table{name: left_table})-[:schema]->(col1:column{name: left_column})

    // Match the actual column nodes for right side
    OPTIONAL MATCH (s2:schema{name: right_schema})-[:schema]->
        (tbl2:table{name: right_table})-[:schema]->(col2:column{name: right_column})

    // Return the structured format
    RETURN collect(DISTINCT {
        table1: t1.schema_name + '.' + t1.name,
        column1: coalesce(col1.name, left_column),
        column1_datatype: coalesce(col1.data_type, 'None'),
        table2: t2.schema_name + '.' + t2.name,
        column2: coalesce(col2.name, right_column),
        column2_datatype: coalesce(col2.data_type, 'None')
    }) as list_of_foreign_keys
    """
    results = get_neo4j_conn().query_read(query, {"tables_ids": tables_ids})
    if len(results) > 0:
        result_joins = results[0]["list_of_foreign_keys"]
    else:
        result_joins = []
    results = result_fks + result_joins

    # Convert to JSON strings, use set to remove duplicates, then convert back
    unique_strings = set(json.dumps(d, sort_keys=True) for d in results)
    unique_results = [json.loads(s) for s in unique_strings]

    key_order = [
        "table1",
        "column1",
        "column1_datatype",
        "table2",
        "column2",
        "column2_datatype",
    ]
    sorted_results = [{key: d[key] for key in key_order} for d in unique_results]
    return sorted_results


def _apply_foreign_key_hints(tables: list[dict], relevant_fks: list) -> None:
    """Set ``foreign_key`` on tables when name matches FK side (same as ``get_relevant_tables``)."""
    for table in tables:
        for fk in relevant_fks:
            if table["name"] == fk["table1"]:
                table["foreign_key"] = f"'{table['name']}.{fk['column1']}' = '{fk['table2']}.{fk['column2']}'"


def get_relevant_fks_from_candidates_tables(
    candidates: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Extract tables and foreign keys from flat candidate dicts.

    Wraps :func:`~nemo_retriever.tabular_data.retrieval.data_access.relevant_tables.get_relevant_tables_from_candidates`
    and additionally fetches FK / join relationships for the resulting tables
    via :func:`_get_relevant_fks`, then applies FK hints in-place.

    Returns:
        ``(relevant_tables, relevant_fks)``.
    """
    relevant_tables = get_relevant_tables_from_candidates(candidates)
    if not relevant_tables:
        return [], []

    try:
        relevant_fks = _get_relevant_fks([x["id"] for x in relevant_tables])
    except Exception:
        logger.exception("_get_relevant_fks failed for candidate tables")
        relevant_fks = []

    _apply_foreign_key_hints(relevant_tables, relevant_fks)
    return relevant_tables, relevant_fks


def get_relevant_tables_with_fks(
    retriever: "Retriever",
    initial_question,
    k=15,
    database_name: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Like :func:`get_relevant_tables` but also returns FK relationships, with FK hints applied in-place."""
    relevant_tables_list = get_relevant_tables(retriever, initial_question, k=k, database_name=database_name)

    relevant_fks: list = []
    if relevant_tables_list:
        try:
            relevant_fks = _get_relevant_fks([x["id"] for x in relevant_tables_list])
        except Exception:
            logger.exception("_get_relevant_fks failed in get_relevant_tables_with_fks")
            relevant_fks = []
    _apply_foreign_key_hints(relevant_tables_list, relevant_fks)

    return relevant_tables_list, relevant_fks
