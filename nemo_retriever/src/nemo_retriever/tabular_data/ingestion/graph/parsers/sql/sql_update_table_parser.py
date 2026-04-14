from nemo_retriever.tabular_data.ingestion.graph.parsers.sql.sql_select_parser import (
    handle_expression,
    build_query_obj as build_select_query,
    add_alias,
)
from nemo_retriever.tabular_data.ingestion.graph.parsers.sql.utils import get_name_and_parent_name
from nemo_retriever.tabular_data.ingestion.graph.model.reserved_words import Props, SQL
from nemo_retriever.tabular_data.ingestion.graph.model.query import Query


def build_query_obj(
    query_obj: Query,
    parsed_query: list[dict[str, str]],
    keep_string_vals: bool,
    is_full_parse: bool = False,
):
    schema_name, table_name = get_name_and_parent_name(
        parsed_query[0]["Update"]["table"]["relation"]["Table"]["name"]
    )
    if (
        query_obj.default_schema is None
        or len(query_obj.default_schema) == 0
        and len(schema_name) > 0
    ):
        query_obj.default_schema = schema_name

    table_node = query_obj.get_table_node_from_schema(table_name, schema_name)

    # retrieve from_table if it exists
    from_table_node = None
    alias_to_table_map = {}
    if (
        "from" in parsed_query[0]["Update"]
        and parsed_query[0]["Update"]["from"] is not None
    ):
        from_dict = parsed_query[0]["Update"]["from"]
        if "Table" in from_dict["relation"]:
            from_schema, from_table = get_name_and_parent_name(
                from_dict["relation"]["Table"]["name"]
            )
            from_table_node = query_obj.get_table_node_from_schema(
                from_table, from_schema
            )
            if "name" in from_dict["relation"]["Table"]["alias"]:
                alias_name = from_dict["relation"]["Table"]["alias"]["name"]["value"]
                alias_to_table_map[alias_name] = from_table_node

            from_edge_params = {Props.SQL_ID: str(query_obj.get_id())}
            query_obj.add_edge(
                (query_obj.sql_node, from_table_node, from_edge_params), SQL.FROM
            )

    assignments = parsed_query[0]["Update"]["assignments"]
    for assignment in assignments:
        if "target" in assignment:
            if len(assignment["target"]["ColumnName"]) == 1:
                column_name = assignment["target"]["ColumnName"]
                if isinstance(column_name, list):
                    column_name = column_name[0]["value"]
            elif len(assignment["target"]["ColumnName"]) == 2:
                # ["ColumnName"][0]["value"] holds the target's table alias name
                column_name = assignment["target"]["ColumnName"][1]["value"]
            else:
                raise Exception("Not supported sqloxide parsing of update query.")
        else:
            column_name = (
                assignment["id"][0]["value"]
                if len(assignment["id"]) == 1
                else assignment["id"][1]["value"]
            )
        column_node = query_obj.get_column_node(column_name, table_node)

        if "Subquery" in assignment["value"]:
            source_sql_query = assignment["value"]["Subquery"]
            build_select_query(
                query_obj=query_obj,
                parsed_query=source_sql_query,
                keep_string_vals=keep_string_vals,
                is_full_parse=is_full_parse,
            )
            selected_nodes = query_obj.get_projection_nodes()
            if len(selected_nodes) > 1:
                raise Exception(f"Too many values assigned to {column_name}.")
            edge_params = {Props.SOURCE_SQL_ID: [str(query_obj.get_id())]}
            query_obj.get_source_to_target_edges().append(
                (selected_nodes[0], column_node, edge_params)
            )
        elif "CompoundIdentifier" in assignment["value"]:
            source_table_name, source_column_name = get_name_and_parent_name(
                assignment["value"]["CompoundIdentifier"]
            )
            if source_table_name in alias_to_table_map:
                source_table_node = alias_to_table_map[source_table_name]
            else:
                source_table_node = from_table_node
            source_column_node = query_obj.get_column_node(
                source_column_name, source_table_node
            )

            # the source is not a Subquery, hence no select query has been parsed, therefore it is required to
            # add the sql node explicitly.
            select_edge_params = {Props.SQL_ID: str(query_obj.get_id())}
            query_obj.add_edge(
                (query_obj.sql_node, source_column_node, select_edge_params), SQL.SELECT
            )

            source_of_edge_params = {Props.SOURCE_SQL_ID: [str(query_obj.get_id())]}
            query_obj.get_source_to_target_edges().append(
                (source_column_node, column_node, source_of_edge_params)
            )
            # add source table to the queries tables list, it will be used for searching for equal query in the graph
            query_obj.add_table_to_query(source_table_node, source_table_name)
        elif "Function" in assignment["value"]:
            ## For use cases like set current_date = current_date()
            if is_full_parse:
                root_node = query_obj.get_sql_node()
            else:
                ## when inserting to the graph we create a fake alias for the function expression
                # because function nodes are not inserted to the graph
                root_node = add_alias(
                    query_obj=query_obj,
                    sql_id=str(query_obj.get_id()),
                    section_type=SQL.SELECT,
                    parent=query_obj.get_sql_node(),
                    alias_name=assignment["value"],
                    is_full_parse=False,
                )
            expression_root_node, _, _, _ = handle_expression(
                query_obj,
                "select",
                assignment["value"],
                root_node,
                str(query_obj.get_id()),
            )
            source_of_edge_params = {Props.SOURCE_SQL_ID: [str(query_obj.get_id())]}
            query_obj.get_source_to_target_edges().append(
                (expression_root_node, column_node, source_of_edge_params)
            )
        else:
            raise Exception("Not supported sqloxide parsing of update query.")
    return True
