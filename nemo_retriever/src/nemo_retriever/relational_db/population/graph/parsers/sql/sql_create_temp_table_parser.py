from shared.graph.model.reserved_words import Props, Labels, label_to_type
from shared.graph.model.node import Node
from shared.graph.model.schema import TEMP_SCHEMA_NAME

from shared.graph.model.query import MissingDataError
from shared.graph.model.query import Query
from typing import Callable


def build_query_obj(
    query_obj: Query,
    parsed_query: list[dict[str, str]],
    keep_string_vals: bool,
    schemas: dict[str, any],
    add_temp_schema_to_graph: Callable[[str, Node], None],
    is_full_parse: bool = False,
):
    if len(parsed_query[0]["CreateTable"]["name"]) == 2:
        schema_name = parsed_query[0]["CreateTable"]["name"][0]["value"]
        table_name = parsed_query[0]["CreateTable"]["name"][1]["value"]
    elif len(parsed_query[0]["CreateTable"]["name"]) == 1:
        schema_name = ""
        table_name = parsed_query[0]["CreateTable"]["name"][0]["value"]
    else:
        schema_name = parsed_query[0]["CreateTable"]["name"][1]["value"]
        table_name = parsed_query[0]["CreateTable"]["name"][2]["value"]

    if len(query_obj.default_schema) == 0 and len(schema_name) > 0:
        query_obj.default_schema = schema_name

    is_temporary_table = parsed_query[0]["CreateTable"]["temporary"]
    is_transient_table = parsed_query[0]["CreateTable"]["transient"]
    is_temporary_table = is_temporary_table or is_transient_table
    is_clone = parsed_query[0]["CreateTable"]["clone"]
    if is_temporary_table:
        if len(schema_name) == 0:
            schema_name = (
                query_obj.default_schema
                if query_obj.default_schema is not None
                else TEMP_SCHEMA_NAME
            )
        schema_name = schema_name.lower()
        if schema_name not in schemas:
            raise MissingDataError(f"Schema {schema_name} is not present in the data.")
        account_id = schemas[schema_name].schema_node.match_props["account_id"]
        if not schemas[schema_name].table_exists(table_name):
            schemas[schema_name].create_table_node(
                table_name,
                account_id,
                table_type=label_to_type(Labels.TEMP_TABLE),
                is_temp=True,
            )
            table_node = schemas[schema_name].get_table_node(table_name)
            add_temp_schema_to_graph(
                schema_name, schemas[schema_name].schema_node, table_node
            )
    elif is_clone is not None:
        raise Exception("Clone is not supported.")
    else:
        raise Exception(
            "This is supposed to be a create temp table query, but neither temporary nor transient is True."
        )
    table_node = query_obj.get_table_node_from_schema(table_name, schema_name)
    if table_node is None:
        raise MissingDataError(
            f"Table {table_name} does not exist in the schema {schema_name}."
        )

    source_table_exists = False
    if parsed_query[0]["CreateTable"]["like"] is not None:
        like_table = parsed_query[0]["CreateTable"]["like"]
        source_table_name = like_table[1]["value"]
        source_table_exists = True

    columns = parsed_query[0]["CreateTable"]["columns"]
    data_types = {}
    if len(columns) == 0 and source_table_exists:
        columns = schemas[schema_name].get_table_columns_by_table_name(
            source_table_name
        )
    elif len(columns) > 0:

        def prepare_data_types(temp_column: dict, data_types: dict):
            data_type = (
                next(iter(temp_column["data_type"]))
                if "data_type" in temp_column
                else None
            )
            data_types[temp_column["name"]["value"]] = data_type
            return temp_column["name"]["value"]

        columns = [prepare_data_types(column, data_types) for column in columns]

    for temp_column_name in columns:
        if not schemas[schema_name].is_column_in_table(table_node, temp_column_name):
            schemas[schema_name].create_column_node(
                temp_column_name,
                account_id,
                table_name,
                data_type=data_types.get(temp_column_name),
                is_temp=True,
            )
            target_col = schemas[schema_name].get_column_node(
                temp_column_name, table_name
            )
            schemas[schema_name].add_column_to_table(table_node, target_col, None)
            add_temp_schema_to_graph(schema_name, table_node, target_col)
        else:
            target_col = schemas[schema_name].get_column_node(
                temp_column_name, table_name
            )
        if source_table_exists:
            source_col = schemas[schema_name].get_column_node(
                temp_column_name, table_name
            )
            edge_params = {Props.SOURCE_SQL_ID: [str(query_obj.get_id())]}
            query_obj.get_source_to_target_edges().append(
                (source_col, target_col, edge_params)
            )

    if len(query_obj.get_edges()) == 0:
        # This query creates the temp table without connections to the data,
        # so there are no edges to add to the graph.
        # Do not proceed with adding query to the graph.
        return False
    return True
