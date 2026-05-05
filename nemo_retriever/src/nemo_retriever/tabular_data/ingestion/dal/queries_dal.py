# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from nemo_retriever.tabular_data.ingestion.utils import chunks
from nemo_retriever.tabular_data.ingestion.dal.utils_dal import prepare_edge, add_edges
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn
from nemo_retriever.tabular_data.ingestion.model.neo4j_node import Neo4jNode


def add_query(edges):
    """Add the nodes and edges of the parsed query to the graph."""
    edges_data = [prepare_edge(edge) for edge in edges]
    for chunk in chunks(edges_data, 10):
        add_edges(chunk)


def get_sql_by_full_query(sql_full_query: str):
    query = f"MATCH (n:{Labels.SQL} {{sql_full_query: $sql_full_query}}) RETURN n.id AS id"
    result = get_neo4j_conn().query_read(
        query=query,
        parameters={"sql_full_query": sql_full_query},
    )
    if result:
        return result[0]["id"]
    return None


def get_sql_counters(sql_node):
    total_counter = sql_node.get_properties()["total_counter"]
    count_per_month = {}
    for prop_key, prop_val in sql_node.get_properties().items():
        if prop_key.startswith("count_"):
            count_per_month.update({prop_key: prop_val})
    return total_counter, count_per_month


def update_counters_and_timestamps_for_query_and_affected_data(
    identical_sql_id: str,
    sql_node: Neo4jNode,
    update_data_last_query_timestamp: bool = True,
):
    latest_timestamp = sql_node.get_properties()["last_query_timestamp"]
    total_counter, count_per_month = get_sql_counters(sql_node)
    # Counter values are bound as parameters (not f-string interpolated) to
    # avoid building Cypher from arbitrary node-property values. Property
    # names cannot be parameterised in Cypher, so the month key stays
    # interpolated; get_sql_counters already constrains it to "count_*".
    set_counts_str = ""
    count_params: dict[str, int] = {}
    for idx, (month, count) in enumerate(count_per_month.items()):
        if not isinstance(count, int):
            raise TypeError(f"Expected int counter for {month}, got {type(count).__name__}")
        param_key = f"count_{idx}"
        set_counts_str += f"SET s.{month} = coalesce(s.{month}, 0) + ${param_key}\n"
        count_params[param_key] = count
    # if the sql already exists in the graph, then update the "last_query_timestamp" property
    # of the sql_node and the table and column nodes that appear as part of the sql.
    cypher_query = f"""MATCH (s:{Labels.SQL} {{id: $id}})
                                SET s.last_query_timestamp = $latest_timestamp
                                SET s.total_counter = s.total_counter + $total_counter
                                {set_counts_str}
                            """.strip()
    get_neo4j_conn().query_write(
        query=cypher_query,
        parameters={
            "latest_timestamp": latest_timestamp,
            "id": identical_sql_id,
            "total_counter": total_counter,
            **count_params,
        },
    )

    if update_data_last_query_timestamp:
        cypher_query = f"""
                MATCH (sql:{Labels.SQL} {{id:$id}})
                WITH sql
                CALL apoc.path.subgraphNodes(sql, {{
                    relationshipFilter: "SQL>",
                    labelFilter: "/{Labels.COLUMN}|/{Labels.TABLE}"}})
                    YIELD node
                    WHERE coalesce(node.deleted, false) = false
                    SET node.last_query_timestamp = $latest_timestamp
                """
        get_neo4j_conn().query_write(
            cypher_query,
            parameters={
                "latest_timestamp": latest_timestamp,
                "id": identical_sql_id,
            },
        )


def load_sqls_to_tables() -> pd.DataFrame:
    """Load all Sql nodes with their connected Table and Column IDs from the graph."""
    query = f"""
        MATCH (s:{Labels.SQL})
        WITH s
        CALL apoc.path.subgraphNodes(s, {{
            relationshipFilter: "SQL>",
            labelFilter: "/{Labels.TABLE}|/{Labels.COLUMN}",
            minLevel: 0}})
        YIELD node
        WHERE coalesce(node.deleted, false) = false
        WITH s,
             [n IN collect(DISTINCT node) WHERE n:{Labels.TABLE} | n.id] AS tbls,
             [n IN collect(DISTINCT node) WHERE n:{Labels.COLUMN} | n.id] AS cols
        RETURN collect({{
            sql_id: s.id,
            tbls: tbls,
            cols: cols,
            nodes_count: s.nodes_count,
            sql_full_query: s.sql_full_query
        }}) AS sqls_tbls
    """
    result = get_neo4j_conn().query_read(query=query)
    if not result or not result[0].get("sqls_tbls"):
        return pd.DataFrame(columns=["sql_id", "tbls", "cols", "nodes_count", "sql_full_query"])
    return pd.DataFrame(result[0]["sqls_tbls"])


def get_candidate_sql_ids(
    tbl_ids: list[str],
    col_ids: list[str],
    nodes_count: int,
    sqls_tbls_df: pd.DataFrame,
) -> pd.DataFrame:
    """Pre-filter graph SQLs by table set, column set (leaves), and AST node count.

    Mirrors the old ``get_sqls_connected_to_tables`` heuristic: same tables,
    same leaf columns, same structural size.  Only candidates passing all
    three gates need the expensive sqlglot structural comparison.
    """
    if sqls_tbls_df.empty:
        return sqls_tbls_df.iloc[0:0]

    tbl_set = set(tbl_ids)
    col_set = set(col_ids)
    mask = (
        (sqls_tbls_df["nodes_count"] == nodes_count)
        & sqls_tbls_df["tbls"].apply(lambda t: set(t) == tbl_set)
        & sqls_tbls_df["cols"].apply(lambda c: set(c) == col_set)
    )
    return sqls_tbls_df.loc[mask]
