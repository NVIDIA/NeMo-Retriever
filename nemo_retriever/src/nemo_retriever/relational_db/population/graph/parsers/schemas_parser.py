from datetime import datetime

from shared.graph.model.reserved_words import Labels
from shared.graph.model.node import Node
from shared.graph.model.schema import Schema, TEMP_SCHEMA_NAME
import logging

logger = logging.getLogger("schemas_parser.py")


def parse_df(
    tables_df, columns_df, account_id, db_node=None, temp_schema_creation_flag=True
):
    """
    Every schema manager assumes a single database in the input file
    :param filename: a csv file with the following columns:
    database,schema,table_name,column_name,ordinal_position,data_type
    Assumption: the file contains schemas of a single database
    :return:
    """
    db_name = tables_df.iloc[0]["database"]
    if not db_node:
        db_node = Node(
            name=db_name,
            label=Labels.DB,
            props={"name": db_name, "pulled": datetime.now()},
            match_props={"name": db_name, "account_id": account_id},
        )

    unique_schema_names = tables_df.schema.unique()
    schemas = {}

    # create schema for temporary tables
    if temp_schema_creation_flag:
        schema = Schema(account_id, db_node)
        schema.create_schema_node(TEMP_SCHEMA_NAME, account_id, is_temp=True)
        schemas.update({TEMP_SCHEMA_NAME: schema})

    for schema_name in unique_schema_names:
        schema_tables_df = tables_df.loc[tables_df["schema"] == schema_name]
        schema_columns_df = columns_df.loc[columns_df["schema"] == schema_name]
        schema_tables_df["is_temp"] = False
        schema_columns_df["is_temp"] = False
        logger.info(f"Started parsing schema {schema_name}.")
        schema = Schema(account_id, db_node, schema_tables_df, schema_columns_df)
        schema.create_schema_node(schema_name, account_id)
        schemas.update({schema.get_schema_name().lower(): schema})
        logger.info(f"Finished parsing schema {schema.get_schema_name()}.")

    return schemas, db_node
