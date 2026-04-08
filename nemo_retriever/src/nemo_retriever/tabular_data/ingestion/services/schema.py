# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from nemo_retriever.tabular_data.ingestion.utils import chunks
from nemo_retriever.tabular_data.ingestion.dal.schemas_dal import (
    add_schemas_edge,
    merge_schema_edges,
    merge_schema_nodes,
)
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels
from nemo_retriever.tabular_data.ingestion.model.schema import Schema
import logging

logger = logging.getLogger(__name__)


def add_table(table_edges):
    merge_schema_edges(table_edges, Labels.TABLE, Labels.COLUMN)


def add_schema(
    schema: Schema,
    latest_timestamp: datetime,
    num_workers: int,
):
    """
    Add all the nodes and edges of the given schema.
    If the schema exists then:
    new nodes - will be added with the given latest_timestamp.
    remaining nodes - the latest_timestamp will be updated.
    missing nodes - the property delete=True will be added to these nodes.
    """

    try:
        # add db->schema edge
        db_schema_edge = schema.get_db_schema_edge()
        add_schemas_edge(db_schema_edge, latest_timestamp)

        # add all table and column nodes
        table_column_nodes_chunks = list(
            chunks(
                [
                    {
                        "label": [x["props"]["label"]],
                        "match_props": x["match_props"],
                        "props": x["props"],
                    }
                    for x in schema.get_table_nodes()
                ],
                500,
            )
        ) + list(
            chunks(
                [
                    {
                        "label": [x["props"]["label"]],
                        "match_props": x["match_props"],
                        "props": x["props"],
                    }
                    for x in schema.get_column_nodes()
                ],
                500,
            )
        )
        for table_column_nodes in table_column_nodes_chunks:
            merge_schema_nodes(table_column_nodes, latest_timestamp)

        # add schema->table edges
        edges_chunks = list(
            chunks(
                schema.get_schema_to_tables_edges(),
                500,
            )
        )
        for edges in edges_chunks:
            merge_schema_edges(edges, Labels.SCHEMA, Labels.TABLE)

        # for each table, add table->column edges
        edges_per_table = schema.get_edges_per_table()
        with ThreadPoolExecutor(num_workers) as executor:
            executor.map(add_table, edges_per_table)

    except Exception as err:
        logger.error(f"Failed adding schema: {schema.get_schema_name()}")
        logger.exception(err)
        raise
