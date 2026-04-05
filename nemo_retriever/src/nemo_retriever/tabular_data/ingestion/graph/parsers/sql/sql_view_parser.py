from nemo_retriever.tabular_data.ingestion.graph.parsers.sql import sql_select_parser

from nemo_retriever.tabular_data.ingestion.graph.model.reserved_words import Views, Props, Labels
from nemo_retriever.tabular_data.ingestion.graph.model.query import Query


def get_projection_nodes(query_obj):
    projs = query_obj.get_projection_nodes()
    if len(projs) > 0:
        return projs
    if len(query_obj.subselects) > 0:
        return list(query_obj.subselects.values())[0].get_projection_nodes()
    return []


def build_query_obj(
    schema_name: str,
    table_name: str,
    with_no_binding: bool,
    query_obj: Query,
    parsed_query: list[dict[str, str]],
    keep_string_vals: bool,
    is_full_parse: bool = False,
):
    type = Views.NON_BINDING_VIEW if with_no_binding else Views.VIEW

    sql_type = next(iter(parsed_query[0]))
    if sql_type.lower() == "query":
        source_sql_query = parsed_query[0]["Query"]

    elif sql_type.lower() == "CreateView".lower():
        if parsed_query[0]["CreateView"]["materialized"] in [True, "true"]:
            type = Views.MATERIALIZED_VIEW

        tbl_idx = 0  # CompoundIdentifier does not include schema_name
        if (
            len(parsed_query[0]["CreateView"]["name"]) == 2
        ):  # CompoundIdentifier does not include db_name
            tbl_idx = 1
            schema_name = parsed_query[0]["CreateView"]["name"][0]["value"]
        elif (
            len(parsed_query[0]["CreateView"]["name"]) == 3
        ):  # CompoundIdentifier includes db_name and schema_name
            tbl_idx = 2
            schema_name = parsed_query[0]["CreateView"]["name"][1]["value"]

        table_name = parsed_query[0]["CreateView"]["name"][tbl_idx]["value"]
        source_sql_query = parsed_query[0]["CreateView"]["query"]
    else:
        raise Exception(f"Unknown view sql type: {sql_type}.")

    table_node = query_obj.get_table_node_from_schema(table_name, schema_name)
    sql_select_parser.build_query_obj(
        query_obj=query_obj,
        parsed_query=source_sql_query,
        keep_string_vals=keep_string_vals,
        is_full_parse=is_full_parse,
    )

    selected_nodes = get_projection_nodes(query_obj)
    for source_col in selected_nodes:
        if source_col.name == "Eq" and source_col.label == Labels.OPERATOR:
            source_col = query_obj.get_left_side_of_assignment_operation(source_col)
            if source_col is None:
                continue
        target_col = query_obj.get_column_node(source_col.get_name(), table_node)
        if target_col.get_match_props()["table_name"] == "":
            raise Exception(
                f"Column {source_col.get_name()} does not exist in the target table {schema_name}.{table_name}."
            )
        edge_params = {Props.SOURCE_SQL_ID: [str(query_obj.get_id())]}
        query_obj.get_source_to_target_edges().append(
            (source_col, target_col, edge_params)
        )

    # TODO: remove this after verifying that the types are set correctly
    table_node.add_property("table_type", type)
    table_node.add_property("type", type)
    return True
