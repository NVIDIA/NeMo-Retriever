from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from shared.graph.model.node import Node
from shared.graph.utils import chunks
from shared.graph.dal.schemas_dal import (
    load_schema_from_graph,
    get_db_ids_and_names,
    get_schemas_ids_and_names,
    get_slim_account_schemas,
    add_schemas_edge,
    merge_schema_edges,
    merge_schema_nodes,
)
from shared.graph.model.reserved_words import Labels
from shared.graph.model.schema import TEMP_SCHEMA_NAME, Schema
import pandas as pd
import time
import logging

logger = logging.getLogger("schemas_service")


def by_label(k):
    return k["label"]


def by_schema_name(k):
    return k["schema_name"]


def get_schemas_slim(account_id: str, relevant_schemas_ids: list = None):
    ## This function is for sql validations in the app - (for example metrics or analyses sql validations)
    ## in these cases we get a slim version of the data so the validation is faster
    before_get_all = time.time()
    data_array = get_slim_account_schemas(account_id, relevant_schemas_ids)
    logger.info(f"time took to get all data from graph: {time.time() - before_get_all}")
    data_df = pd.DataFrame(data_array)
    dbs = list(data_df["db_name"].unique())

    schemas = data_df[["schema_name", "db_name"]]
    schemas = schemas.drop_duplicates().to_dict(orient="records")

    all_schemas = {}
    schema_dfs = {}
    dbs_nodes = {}
    for db_name in dbs:
        db_node = Node(name=db_name, label=Labels.DB, props={"name": db_name})
        dbs_nodes[db_name] = db_node

    tables_df = data_df[["schema_name", "db_name", "table_name"]]
    tables_df = tables_df.drop_duplicates()
    # TODO: add db_name to the schema's identifying key in the all_schemas map
    unique_schemas = data_df.schema_name.unique()
    for schema_name in unique_schemas:
        schema_tables_df = tables_df.loc[tables_df["schema_name"] == schema_name]
        schema_dfs[schema_name] = {"tables": schema_tables_df.to_dict(orient="records")}

    for schema_name in unique_schemas:
        columns_df = data_df.loc[data_df["schema_name"] == schema_name]
        schema_dfs[schema_name]["columns"] = columns_df.to_dict(orient="records")

    def create_schema_node(schema):
        schema_name: str = schema["schema_name"]
        if schema_name != "":
            schema_db_name: str = schema["db_name"]
            schema_db_node = dbs_nodes[schema_db_name]
            tables_df = pd.DataFrame(schema_dfs[schema_name]["tables"])
            tables_df = tables_df.rename(columns={"schema_name": "schema"})
            tables_df["name"] = tables_df["table_name"]
            tables_df["is_temp"] = False
            columns_df = pd.DataFrame(schema_dfs[schema_name]["columns"])
            columns_df["column_name"] = columns_df["name"]
            columns_df = columns_df.rename(columns={"schema_name": "schema"})
            columns_df["is_temp"] = False
            all_schemas[schema_name.lower()] = Schema(
                account_id,
                schema_db_node,
                tables_df,
                columns_df,
                schema_name,
                is_creation_mode=False,
            )

    before_modify_all = time.time()
    for schema in schemas:
        create_schema_node(schema)
    logger.info(
        f"total time it took to create all schemas nodes: {time.time() - before_modify_all}"
    )
    logger.info(f"total time for get_schemas_slim(): {time.time() - before_get_all}")
    return all_schemas


def get_account_schemas(account_id, connection_id=None, include_deleted=False):
    all_schemas = {}
    dbs = get_db_ids_and_names(account_id, connection_id)
    for db in dbs:
        db_id = db["id"]
        db_name = db["name"]
        db_node = Node(
            name=db_name, label=Labels.DB, props={"name": db_name}, existing_id=db_id
        )
        schemas = get_schemas_ids_and_names(
            account_id, db_id, include_deleted=include_deleted
        )
        for s in schemas:
            all_schemas.update(
                {
                    s["schema_name"].lower(): load_schema_from_graph(
                        db_name,
                        s["schema_name"],
                        account_id,
                        db_node,
                        is_temp=(s["schema_name"] == TEMP_SCHEMA_NAME),
                        include_deleted=include_deleted,
                    )
                }
            )
    return all_schemas


def add_table(table_edges, account_id):
    merge_schema_edges(table_edges, Labels.TABLE, Labels.COLUMN, account_id)


def add_schema(
    schema: Schema,
    account_id: str,
    latest_timestamp: datetime,
    num_workers: int,
    added_or_modified_tables_dict: dict,
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
        add_schemas_edge(db_schema_edge, account_id, latest_timestamp)

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
        if len(table_column_nodes_chunks) > 0:
            added_or_modified_tables_dict["tables"] = set(
                [x["props"]["name"] for x in table_column_nodes_chunks[0]]
            )  # take all table names
        else:
            added_or_modified_tables_dict["tables"] = set()
        for table_column_nodes in table_column_nodes_chunks:
            merge_schema_nodes(table_column_nodes, account_id, latest_timestamp)

        # add schema->table edges
        edges_chunks = list(
            chunks(
                schema.get_schema_to_tables_edges(),
                500,
            )
        )
        for edges in edges_chunks:
            merge_schema_edges(edges, Labels.SCHEMA, Labels.TABLE, account_id)

        # for each table, add table->column edges
        edges_per_table = schema.get_edges_per_table()
        with ThreadPoolExecutor(num_workers) as executor:
            executor.map(
                lambda table_edges: add_table(table_edges, account_id), edges_per_table
            )

    except Exception as err:
        logger.info(f"Failed adding schema: {schema.get_schema_name()}")
        logger.exception(err)
        added_or_modified_tables_dict["tables"] = set()
        return added_or_modified_tables_dict

    return added_or_modified_tables_dict
