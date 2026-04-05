from nemo_retriever.tabular_data.ingestion.graph.parsers.sql import sql_select_parser as sql_select
from nemo_retriever.tabular_data.ingestion.graph.parsers.sql.utils import get_name_and_parent_name
from nemo_retriever.tabular_data.ingestion.graph.model.reserved_words import Props, Parser, SQL
from nemo_retriever.tabular_data.ingestion.graph.model.query import Query


def build_query_obj(
    query_obj: Query,
    parsed_query: list[dict[str, str]],
    keep_string_vals: bool,
    is_full_parse: bool = False,
):
    schema_name, table_name = get_name_and_parent_name(
        parsed_query[0]["Merge"]["table"]["Table"]["name"]
    )

    if (
        query_obj.default_schema is None
        or len(query_obj.default_schema) == 0
        and len(schema_name) > 0
    ):
        query_obj.default_schema = schema_name

    table_node = query_obj.get_table_node_from_schema(table_name, schema_name)

    source = parsed_query[0]["Merge"]["source"]
    if "Derived" in source:
        source_subquery = source["Derived"]["subquery"]
        sql_select.build_query_obj(
            query_obj=query_obj,
            parsed_query=source_subquery,
            keep_string_vals=keep_string_vals,
            is_full_parse=is_full_parse,
        )
        selected_nodes = query_obj.get_projection_nodes()
    elif "Table" in source:
        source_schema_name, source_table_name = get_name_and_parent_name(
            source["Table"]["name"]
        )
        selected_nodes = None

        # add source table to the queries tables list, it will be used for searching for equal query in the graph
        source_table_node = query_obj.get_table_node_from_schema(
            source_table_name, source_schema_name
        )
        query_obj.add_table_to_query(source_table_node, source_table_name)

        from_edge_params = {Props.SQL_ID: str(query_obj.get_id())}
        query_obj.add_edge(
            (query_obj.sql_node, source_table_node, from_edge_params), SQL.FROM
        )
    else:
        raise Exception("Not supported source type.")

    if len(parsed_query[0]["Merge"]["clauses"]) == 2:
        assignments = parsed_query[0]["Merge"]["clauses"][1]["action"]["Insert"]
    else:
        assignments = parsed_query[0]["Merge"]["clauses"][0]["action"]["Insert"]
    columns = assignments["columns"]
    values = assignments["kind"]["Values"]["rows"][0]
    for target_col_name, source_col_name in zip(columns, values):
        # get the column node from the schema
        target_col_name = target_col_name["value"]
        target_node = query_obj.get_column_node(target_col_name, table_node)
        # get the source node
        if Parser.COMPOUND_IDENTIFIER in source_col_name:
            source_col_name = source_col_name[Parser.COMPOUND_IDENTIFIER][1]["value"]
        elif Parser.IDENTIFIER in source_col_name:
            source_col_name = source_col_name[Parser.IDENTIFIER]["value"]
            if source_col_name == "CURRENT_USER":
                continue
        elif Parser.FUNCTION in source_col_name or Parser.VALUE in source_col_name:
            continue
        if selected_nodes:
            source_node = list(
                filter(
                    lambda selected_nodes: selected_nodes.name.lower()
                    == source_col_name.lower(),
                    selected_nodes,
                )
            )
        else:
            if len(source_schema_name) > 0:
                source_node = query_obj.schemas[
                    source_schema_name.lower()
                ].get_column_node(source_col_name, source_table_name)
            else:
                source_table_node = query_obj.get_table_node_from_schema(
                    source_table_name, source_schema_name
                )
                source_node = query_obj.get_column_node(
                    source_col_name, source_table_node
                )
            source_node = [source_node]

        if len(source_node) == 1:
            if "Table" in source:
                # the source is a table, hence no select query has been parsed, therefore it is required to
                # add the sql node explicitly.
                select_edge_params = {Props.SQL_ID: str(query_obj.get_id())}
                query_obj.add_edge(
                    (query_obj.sql_node, source_node[0], select_edge_params), SQL.SELECT
                )

            source_of_edge_params = {Props.SOURCE_SQL_ID: [str(query_obj.get_id())]}
            query_obj.get_source_to_target_edges().append(
                (source_node[0], target_node, source_of_edge_params)
            )
