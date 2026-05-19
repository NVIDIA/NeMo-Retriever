# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neo4j-backed schema / node lookups.

Builds the :class:`Schema` objects consumed by the SQL parser plus generic
``_get_node_properties_by_id`` / ``get_item_by_id`` helpers.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from nemo_retriever.tabular_data.ingestion.model.neo4j_node import Neo4jNode
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Edges, Labels
from nemo_retriever.tabular_data.ingestion.model.schema import Schema
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

logger = logging.getLogger(__name__)

_ALLOWED_NODE_LABELS = frozenset(Labels.LIST_OF_ALL)


def _get_schemas_from_graph_by_ids(
    relevant_schemas_ids: list | None = None,
) -> list[dict[str, str]]:
    schema_ids = relevant_schemas_ids or []
    query = f"""
    MATCH (db:{Labels.DB})-[:{Edges.CONTAINS}]->(schema:{Labels.SCHEMA})
          -[:{Edges.CONTAINS}]->(table:{Labels.TABLE})
          -[:{Edges.CONTAINS}]->(column:{Labels.COLUMN})
    WHERE size($relevant_schemas_ids) = 0
       OR schema.id IN $relevant_schemas_ids
    RETURN collect({{
        column_name:  column.name,
        column_id:    column.id,
        table_name:   table.name,
        table_id:     table.id,
        db_name:      db.name,
        table_schema: schema.name,
        data_type:    column.data_type
    }}) AS data
    """
    result = get_neo4j_conn().query_read(query, {"relevant_schemas_ids": schema_ids})
    if len(result) > 0:
        return result[0]["data"]
    return []


def get_all_schemas_ids():
    query = f"""MATCH(s:{Labels.SCHEMA}) RETURN s.id as schema_id"""
    result = pd.DataFrame(
        get_neo4j_conn().query_read(
            query=query,
            parameters=None,
        )
    )
    return result["schema_id"].tolist()


def get_schemas_by_ids(relevant_schemas_ids: list = None):
    before_get_all = time.time()
    data_array = _get_schemas_from_graph_by_ids(relevant_schemas_ids)
    logger.info(f"time took to get all data from graph: {time.time() - before_get_all}")
    data_df = pd.DataFrame(data_array)
    dbs = list(data_df["db_name"].unique())

    schemas = data_df[["db_name", "table_schema"]]
    schemas = schemas.drop_duplicates().to_dict(orient="records")

    all_schemas = {}
    schema_dfs = {}
    dbs_nodes = {}
    for db_name in dbs:
        db_node = Neo4jNode(name=db_name, label=Labels.DB, props={"name": db_name})
        dbs_nodes[db_name] = db_node

    tables_df = data_df[["db_name", "table_schema", "table_name", "table_id"]].drop_duplicates(
        subset=["db_name", "table_schema", "table_name"]
    )
    tables_df = tables_df.rename(columns={"table_id": "id"})

    unique_schemas = data_df.table_schema.unique()
    for table_schema in unique_schemas:
        schema_tables_df = tables_df.loc[tables_df["table_schema"] == table_schema]
        schema_dfs[table_schema] = {"tables": schema_tables_df.to_dict(orient="records")}

    for table_schema in unique_schemas:
        columns_df = data_df.loc[data_df["table_schema"] == table_schema].rename(columns={"column_id": "id"})
        schema_dfs[table_schema]["columns"] = columns_df.to_dict(orient="records")

    before_modify_all = time.time()
    for schema in schemas:
        table_schema: str = schema.get("table_schema")
        if not table_schema:
            continue

        schema_db_name: str = schema["db_name"]
        schema_db_node = dbs_nodes[schema_db_name]
        tables_df = pd.DataFrame(schema_dfs[table_schema]["tables"])
        columns_df = pd.DataFrame(schema_dfs[table_schema]["columns"])

        all_schemas[table_schema.lower()] = Schema(
            schema_db_node,
            tables_df,
            columns_df,
            table_schema,
            is_creation_mode=False,
        )
    logger.info(f"total time it took to create all schemas nodes: {time.time() - before_modify_all}")
    logger.info(f"total time for get_schemas_by_ids(): {time.time() - before_get_all}")
    return all_schemas


def _get_node_properties_by_id(id, label: str | list[str]):
    labels_list = label if isinstance(label, list) else [label]
    for lbl in labels_list:
        if lbl not in _ALLOWED_NODE_LABELS:
            logger.warning("Rejecting unknown label %r in _get_node_properties_by_id", lbl)
            return None
    label_filter = "|".join(labels_list)
    query = f"""
        MATCH(n:{label_filter}{{id:$id}})
        RETURN apoc.map.setKey(properties(n),"label", labels(n)[0]) as props
    """

    props = get_neo4j_conn().query_read_only(query, parameters={"id": id})
    if len(props) == 0:
        return None
    else:
        return props[0]["props"]


def get_item_by_id(item_id, label):
    result = _get_node_properties_by_id(item_id, label)
    if result:
        return result
    else:
        logger.error(f"The required item with id : {item_id} is not found in graph. ERROR.")
        return None
