# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels
from nemo_retriever.tabular_data.ingestion.model.neo4j_node import Neo4jNode
from nemo_retriever.tabular_data.ingestion.model.schema import Schema
import logging

logger = logging.getLogger(__name__)


def parse_df(tables_df, columns_df, db_node=None):
    """
    Every schema manager assumes a single database in the input file
    :param filename: a csv file with the following columns:
    database,schema,table_name,column_name,ordinal_position,data_type
    Assumption: the file contains schemas of a single database
    :return:
    """
    db_name = tables_df.iloc[0]["database"]
    if not db_node:
        db_node = Neo4jNode(
            name=db_name,
            label=Labels.DB,
            props={"name": db_name, "pulled": datetime.now()},
            match_props={"name": db_name},
        )

    unique_schema_names = tables_df.schema.unique()
    schemas = {}

    for schema_name in unique_schema_names:
        schema_tables_df = tables_df.loc[tables_df["schema"] == schema_name]
        schema_columns_df = columns_df.loc[columns_df["schema"] == schema_name]
        logger.info(f"Started parsing schema {schema_name}.")
        schema = Schema(db_node, schema_tables_df, schema_columns_df)
        schema.create_schema_node(schema_name)
        schemas.update({schema.get_schema_name().lower(): schema})
        logger.info(f"Finished parsing schema {schema.get_schema_name()}.")

    return schemas, db_node
