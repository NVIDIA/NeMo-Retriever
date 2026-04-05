from nemo_retriever.tabular_data.ingestion.graph.parsers.sql import sql_select_parser as sql_select
from nemo_retriever.tabular_data.ingestion.graph.model.reserved_words import Props
from nemo_retriever.tabular_data.ingestion.graph.model.query import Query


def build_query_obj(
    query_obj: Query,
    parsed_query: list[dict],
    keep_string_vals: bool,
    is_full_parse: bool,
):
    if len(parsed_query[0]["Insert"]["table_name"]) == 2:
        schema_name = parsed_query[0]["Insert"]["table_name"][0]["value"]
        table_name = parsed_query[0]["Insert"]["table_name"][1]["value"]
    elif len(parsed_query[0]["Insert"]["table_name"]) == 1:
        schema_name = ""
        table_name = parsed_query[0]["Insert"]["table_name"][0]["value"]
    else:
        schema_name = parsed_query[0]["Insert"]["table_name"][1]["value"]
        table_name = parsed_query[0]["Insert"]["table_name"][2]["value"]

    if (
        query_obj.default_schema is None
        or len(query_obj.default_schema) == 0
        and len(schema_name) > 0
    ):
        query_obj.default_schema = schema_name

    table_node = query_obj.get_table_node_from_schema(table_name, schema_name)
    columns = parsed_query[0]["Insert"]["columns"]
    column_nodes = [
        query_obj.get_column_node(col_name["value"], table_node) for col_name in columns
    ]

    source_sql_query = parsed_query[0]["Insert"]["source"]
    if next(iter(source_sql_query["body"])) == "Values":
        raise Exception("Parsing Insert Into queries with values is not supported.")
    else:
        sql_select.build_query_obj(
            query_obj=query_obj,
            parsed_query=source_sql_query,
            keep_string_vals=keep_string_vals,
            is_full_parse=is_full_parse,
        )
        selected_nodes = query_obj.get_projection_nodes()
    if len(column_nodes) == 0:
        for i, s_col in enumerate(selected_nodes):
            try:
                target_col = query_obj.get_column_node(s_col.get_name(), table_node)
            except Exception:
                schema = query_obj.schemas[schema_name.lower()]
                target_col_name = schema.tables_columns_pos[(table_name.lower(), i + 1)]
                target_col = query_obj.get_column_node(target_col_name, table_node)
            if target_col.get_match_props()["table_name"] == "":
                raise Exception(
                    f"Column {s_col.get_name()} does not exist in the target table {schema_name}.{table_name}."
                )
            column_nodes.append(target_col)
    elif len(column_nodes) != len(selected_nodes):
        raise Exception(
            "The number of target columns is not equal to the number of source columns."
        )
    for target_col, source_col in zip(column_nodes, selected_nodes):
        edge_params = {Props.SOURCE_SQL_ID: [str(query_obj.get_id())]}
        query_obj.get_source_to_target_edges().append(
            (source_col, target_col, edge_params)
        )
