from shared.graph.parsers.sql import sql_select_parser as sql_select
from shared.graph.model.reserved_words import Props, Labels, label_to_type
from shared.graph.model.node import Node
from shared.graph.model.schema import TEMP_SCHEMA_NAME

# from shared.graph.dal.schemas_dal import add_schemas_edge
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

    if (query_obj.default_schema is None or len(query_obj.default_schema) == 0) and len(
        schema_name
    ) > 0:
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
    table_node = query_obj.get_table_node_from_schema(table_name, schema_name)
    if table_node is None:
        raise MissingDataError(
            f"Table {table_name} does not exist in the schema {schema_name}."
        )

    if "query" in parsed_query[0]["CreateTable"]:
        source_sql_query = parsed_query[0]["CreateTable"]["query"]
        if (
            not source_sql_query and len(parsed_query) > 1
        ):  # for ms dialect queries like create->insert
            source_sql_query = parsed_query[1]["Insert"]["source"]
        sql_select.build_query_obj(
            query_obj=query_obj,
            parsed_query=source_sql_query,
            keep_string_vals=keep_string_vals,
            is_full_parse=is_full_parse,
        )
        selected_nodes = query_obj.get_projection_nodes()
    else:
        return False

    for source_col in selected_nodes:
        if is_temporary_table:
            account_id = schemas[schema_name].schema_node.match_props["account_id"]
            if not schemas[schema_name].is_column_in_table(
                table_node, source_col.get_name()
            ):
                schemas[schema_name].create_column_node(
                    source_col.get_name(),
                    account_id,
                    table_name,
                    data_type=source_col.props["data_type"]
                    if "data_type" in source_col.props
                    else None,
                    is_temp=True,
                )
                target_col = schemas[schema_name].get_column_node(
                    source_col.get_name(), table_name
                )
                schemas[schema_name].add_column_to_table(table_node, target_col, None)
                add_temp_schema_to_graph(schema_name, table_node, target_col)
            else:
                target_col = schemas[schema_name].get_column_node(
                    source_col.get_name(), table_name
                )
        else:
            target_col = query_obj.get_column_node(source_col.get_name(), table_node)
            if target_col.get_match_props()["table_name"] == "":
                raise MissingDataError(
                    f"Column {source_col.get_name()} does not exist in the target table {schema_name}.{table_name}."
                )
        edge_params = {Props.SOURCE_SQL_ID: [str(query_obj.get_id())]}
        query_obj.get_source_to_target_edges().append(
            (source_col, target_col, edge_params)
        )
    return True
