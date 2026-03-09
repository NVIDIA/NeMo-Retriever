from pandas import DataFrame
from typing import Literal

from shared.graph.dal.sql_snippet_dal import (
    get_upstream_source,
    replace_definition_sql,
    save_custom_snippet_in_graph,
    save_analysis_in_graph,
    delete_snippet,
)
from shared.graph.utils import remove_redundant_parentheses

from shared.graph.dal.tables_dal import get_join_columns
from shared.graph.parsers.sql.queries_parser import parse_single
from shared.graph.services.queries_comparison.queries_comparison import (
    find_identical_queries,
)
from shared.graph.dal.utils_dal import (
    prepare_edge,
    add_edges,
)
from shared.graph.dal.usages.semantic.terms_attributes import (
    update_single_attr_usage,
    update_single_term_usage,
    update_certified_for_zones,
    update_document_reference_certified_for_zones,
)
from shared.graph.model.reserved_words import Labels, SQLType
from shared.graph.model.query import Query, NotSelectSqlTypeError, NoFKError


import logging

logger = logging.getLogger("sql_snippet.py")


def get_is_simple_snippet(parsed_query: Query):
    one_column_selected = len(parsed_query.get_reached_columns_ids()) == 1
    no_where_clause = len(parsed_query.where_edges) == 0
    no_group_by_clause = len(parsed_query.group_by_edges) == 0
    if one_column_selected and no_where_clause and no_group_by_clause:
        return True
    return False


def get_definition_sql_snippet(
    account_id: str, parsed_query: Query, is_simple_snippet: bool
):
    query_clauses = {
        "select": parsed_query.sql_node.props["select_str"],
        "from": parsed_query.sql_node.props["from_str"],
        "where": parsed_query.sql_node.props["where_str"],
        "select_tc": parsed_query.sql_node.props["select_str"],
    }
    if is_simple_snippet:
        is_compound_identifier = len(query_clauses["select"].split(".")) > 1
        column_name = (
            query_clauses["select"]
            if not is_compound_identifier
            else query_clauses["select"].split(".")[-1]
        )
        query_clauses["select"] = column_name
    columns_sources = get_upstream_source(
        account_id, parsed_query.get_reached_columns_ids()
    )
    source_clauses = create_definition_sql_snippet(
        account_id,
        columns_sources,
        query_clauses["select"],
        query_clauses["where"],
    )
    return source_clauses


def parse_custom_snippet(
    account_id: str,
    sql_snippet: str,
    schemas: dict,
    dialects: list[str],
    sqls_tbls_df: DataFrame = None,
    is_population: bool = False,
) -> tuple[
    Query,
    Literal["single_column", "synthetic_sql", "existing_sqls"],
    list[dict],
    dict,
    dict,
]:
    parsed_query: Query = parse_single(
        q=sql_snippet,
        schemas=schemas,
        dialects=dialects,
        keep_string_vals=False,
        allow_only_select=True,
        sql_type=SQLType.SEMANTIC,
        is_full_parse=False,
    )
    is_simple_snippet = get_is_simple_snippet(parsed_query)

    snippet_sql_clauses = {
        "select": parsed_query.sql_node.props["select_str"],
        "from": parsed_query.sql_node.props["from_str"],
        "where": parsed_query.sql_node.props["where_str"],
        "select_tc": parsed_query.sql_node.props["select_str"],
    }

    if is_simple_snippet:
        mode = "single_column"
        selected_column_id = parsed_query.get_reached_columns_ids()[0]
        nodes_to_connect_to = [
            {
                "node_id": str(selected_column_id),
                "node_label": Labels.COLUMN,
            }
        ]
    elif not is_population:
        # happens only when creating the custom attribute
        mode = "synthetic_sql"
        edges_data = [
            prepare_edge(edge, account_id) for edge in parsed_query.get_edges()
        ]
        add_edges(account_id, edges_data)
        nodes_to_connect_to = [
            {
                "node_id": str(parsed_query.id),
                "node_label": Labels.SQL,
            }
        ]
    elif is_population:
        identical_ids_from_graph, _, subset_ids_from_graph, _ = find_identical_queries(
            account_id=account_id,
            main_sql=sql_snippet,
            get_parsed_query=lambda sql_query: parse_single(
                q=sql_query,
                schemas=schemas,
                dialects=dialects,
                is_full_parse=True,
                allow_only_select=True,
                sql_type=SQLType.SEMANTIC,
                default_schema=parsed_query.default_schema,
            ),
            sqls_tbls_from_graph_df=sqls_tbls_df,
            is_subgraph=True,
            remove_aliases=True,
        )
        containing_ids_from_graph = identical_ids_from_graph + subset_ids_from_graph

        if containing_ids_from_graph:
            mode = "existing_sqls"
            nodes_to_connect_to = [
                {
                    "node_id": str(id),
                    "node_label": Labels.SQL,
                }
                for id in containing_ids_from_graph
            ]
        else:
            # No matching SQLs found in population mode, fall back to synthetic SQL
            mode = "synthetic_sql"
            edges_data = [
                prepare_edge(edge, account_id) for edge in parsed_query.get_edges()
            ]
            add_edges(account_id, edges_data)
            nodes_to_connect_to = [
                {
                    "node_id": str(parsed_query.id),
                    "node_label": Labels.SQL,
                }
            ]
    definition_sql_snippet = get_definition_sql_snippet(
        account_id, parsed_query, is_simple_snippet
    )
    return (
        parsed_query,
        mode,
        nodes_to_connect_to,
        definition_sql_snippet,
        snippet_sql_clauses,
    )


def save_custom_snippet(
    account_id: str,
    sql_snippet: str,
    schemas: dict,
    dialects: list[str],
    attribute_id: str,
    sqls_tbls_df: DataFrame = None,
    snippet_id: str = None,
    is_population: bool = False,
    user_id: str = None,
):
    try:
        (
            parsed_snippet,
            mode,
            nodes_to_connect_to,
            definition_sql_snippet,
            snippet_sql_clauses,
        ) = parse_custom_snippet(
            account_id=account_id,
            sql_snippet=sql_snippet,
            schemas=schemas,
            dialects=dialects,
            sqls_tbls_df=sqls_tbls_df,
            is_population=is_population,
        )
        snippet_id = save_custom_snippet_in_graph(
            account_id=account_id,
            attr_id=attribute_id,
            snippet_sql=sql_snippet,
            mode=mode,
            nodes_to_connect=nodes_to_connect_to,
            snippet_id=snippet_id,
            definition_sql_clauses=definition_sql_snippet,
            sql_clauses=snippet_sql_clauses,
            user_id=user_id,
            snippet_data_type=parsed_snippet.sql_node.get_properties().get("data_type"),
            reached_columns=parsed_snippet.get_reached_columns_ids(),
        )
        return snippet_id
    except Exception as ex:
        if snippet_id:
            logger.error(
                f"Something went wrong parsing snippet {snippet_id}. error: {str(ex)}"
            )
        else:
            raise Exception("Something went wrong. Please try again.")


def edit_snippet(
    account_id: str,
    schemas: dict,
    snippet_id: str,
    snippet_code: str,
    certified_for_zones: list,
    dialects: list,
    attribute_id: str,
    user_id: str,
):
    new_snippet_id = snippet_id  # Default to original snippet_id
    if snippet_code is not None:
        attr_id, term_id = delete_snippet(account_id, attribute_id, snippet_id)

        new_snippet_id = save_custom_snippet(
            account_id=account_id,
            sql_snippet=remove_redundant_parentheses(snippet_code),
            schemas=schemas,
            dialects=dialects,
            attribute_id=attribute_id,
            snippet_id=None,
        )
        update_single_attr_usage(account_id, attr_id)
        update_single_term_usage(account_id, term_id)
    if certified_for_zones is not None:
        update_certified_for_zones(
            account_id, attribute_id, new_snippet_id, certified_for_zones
        )
    return new_snippet_id


def edit_document_reference(
    account_id: str,
    document_reference_id: str,
    certified_for_zones: list,
    attribute_id: str,
    user_id: str,
):
    new_document_reference_id = (
        document_reference_id  # Default to original document_reference_id
    )
    if certified_for_zones is not None:
        update_document_reference_certified_for_zones(
            account_id, attribute_id, new_document_reference_id, certified_for_zones
        )
    return new_document_reference_id


def parse_analysis(
    account_id: str,
    sql: str,
    schemas: dict,
    dialects: list[str],
    sqls_tbls_df: DataFrame = None,
    analysis_id: str = None,
    name: str = None,
    description: str = None,
    owner_id: str = None,
    recommended: bool = False,
    user_id: str = None,
):
    try:
        parsed_query: Query = parse_single(
            q=sql,
            schemas=schemas,
            dialects=dialects,
            keep_string_vals=False,
            sql_type=SQLType.SEMANTIC,
            is_full_parse=False,
        )
        if not analysis_id:
            edges_data = [
                prepare_edge(edge, account_id) for edge in parsed_query.get_edges()
            ]
            add_edges(account_id, edges_data)
            sqls_ids = [str(parsed_query.id)]
        else:
            identical_sqls_ids, _, subset_sqls_ids, _ = find_identical_queries(
                account_id=account_id,
                main_sql=sql,
                get_parsed_query=lambda sql_query: parse_single(
                    q=sql_query,
                    schemas=schemas,
                    dialects=dialects,
                    default_schema=parsed_query.default_schema,
                    is_full_parse=True,
                ),
                sqls_tbls_from_graph_df=sqls_tbls_df,
                is_subgraph=True,
                remove_aliases=True,
            )
            sqls_ids = identical_sqls_ids + subset_sqls_ids
            if not sqls_ids:
                return
        analysis_id = save_analysis_in_graph(
            account_id=account_id,
            analysis_id=analysis_id,
            sqls_ids=sqls_ids,
            sql=sql,
            name=name,
            description=description,
            owner_id=owner_id,
            recommended=recommended,
            user_id=user_id,
            reached_columns=parsed_query.reached_columns_ids,
        )
        return analysis_id
    except Exception as ex:
        if not analysis_id:
            raise Exception("Something went wrong. Please try again.")
        else:
            logger.error(
                f"Something went wrong looking for identical sqls to analysis {analysis_id}. error: {str(ex)}"
            )


def update_definition_sql(
    account_id: str,
    attr_id: str,
    snippet_id: str,
    reached_columns: list,
    snippet_select: list,
):
    # Calculate definition sql of the snippet.
    # For each column find the upstream source (if exists).
    columns_sources = get_upstream_source(account_id, reached_columns)
    source_clauses = create_definition_sql_snippet(
        account_id, columns_sources, snippet_select, ""
    )

    replace_definition_sql(account_id, attr_id, snippet_id, source_clauses)


def create_definition_sql_snippet(
    account_id, columns_sources, select_clause, where_clause
):
    source_clauses = dict()
    source_clauses["source_select"] = select_clause
    source_clauses["source_where"] = where_clause
    source_clauses["from_source_ids"] = []
    source_clauses["from_source"] = []
    from_sources = []
    for column in columns_sources:
        # construct the source expression for every column
        upstream_path = column["upstream_path"]
        for source_of_rel in upstream_path:
            # search first for the full column name (schema+table+column) and then for the column name only
            source_clauses["source_select"] = search_and_replace_column(
                source_of_rel["column"],
                source_of_rel["src"],
                source_clauses["source_select"],
            )
            source_clauses["source_where"] = search_and_replace_column(
                source_of_rel["column"],
                source_of_rel["src"],
                source_clauses["source_where"],
            )

        from_sources.extend(column["from_sources"])
        source_clauses["from_source_ids"].extend(
            [c["id"] for c in column["from_sources"]]
        )
        source_clauses["from_source"].extend(
            [c["name"] for c in column["from_sources"]]
        )
    # construct the "from" clause
    source_from_str = None
    if len(set(source_clauses["from_source"])) > 1:
        sorted_sources = sorted(from_sources, key=lambda x: x["name"])
        tables_ids = [t["id"] for t in sorted_sources]
        source_from_str = get_all_joins(tables_ids, account_id)
    elif len(set(source_clauses["from_source"])) == 1:
        source_from_str = from_sources[0]["name"]
    else:
        source_from_str = None

    if source_from_str:
        source_clauses["definition_sql"] = (
            f"""SELECT {source_clauses["source_select"]} FROM {source_from_str}"""
        )
        if len(source_clauses["source_where"]) > 0:
            source_clauses["definition_sql"] += (
                f" WHERE {source_clauses['source_where']}"
            )
    else:
        source_clauses["definition_sql"] = None
    return source_clauses


def replace_insensitive(string: str, source: str, replace: str):
    start_index = string.lower().find(source.lower())
    if start_index == -1:
        raise Exception(f"Column {replace} not found in {source}")
    end_index = start_index + len(source)
    return string[:start_index] + replace + string[end_index:]


def search_and_replace_column(full_column_name, source_of_column, target_string):
    try:
        new_str = replace_insensitive(target_string, full_column_name, source_of_column)
        return new_str
    except Exception:
        pass
    try:
        new_str = replace_insensitive(
            target_string, full_column_name.split(".")[-1], source_of_column
        )
        return new_str
    except Exception:
        pass
    return target_string


def get_all_joins(tables: list[str], account_id: str) -> str:
    seed_table_id = tables[0]
    other_tables_ids = [t for t in tables[1:]]
    branches = get_join_columns(account_id, seed_table_id, other_tables_ids)
    ## sorting the branch so we get consistent sqls
    sorted_branches = sorted(
        branches, key=lambda branch: branch[0]["start_table"] + branch[0]["end_table"]
    )
    tables = [sorted_branches[0][0]["start_table"]]
    tables_names = [sorted_branches[0][0]["start_table"]]
    ## a branch looks like this:
    ## [
    ##     {"start_table": "table1", "end_table": "table2", "join_columns": "table1.column1 = table2.column2"},
    ##     {"start_table": "table2", "end_table": "table3", "join_columns": "table2.column2 = table3.column3"},
    ## ]
    ## we want to get the following join:
    ## table1 JOIN table2 ON table1.column1 = table2.column2
    ## JOIN table3 ON table2.column2 = table3.column3s
    for tables_branch in sorted_branches:
        for table in tables_branch:
            ## if the end table is already in the list, we'll join the start table
            if table["end_table"] in tables_names:
                tables.append(f"JOIN {table['start_table']} ON {table['join_columns']}")
                tables_names.append(table["start_table"])
            ## if the start table is already in the list, we'll join the end table
            elif table["start_table"] in tables_names:
                tables.append(f"JOIN {table['end_table']} ON {table['join_columns']}")
                tables_names.append(table["end_table"])
    from_str = "\n".join(tables)
    return from_str


def is_valid_sql(
    schemas,
    sql: str,
    dialects: list,
    check_single_projection: bool = True,
    allow_only_select: bool = False,
    fks=None,
    query_obj: Query = None,
):
    try:
        query = (
            query_obj
            if query_obj
            else parse_single(
                q=sql,
                schemas=schemas,
                dialects=dialects,
                sql_type=SQLType.SEMANTIC,
                allow_only_select=allow_only_select,
                fks=fks,
            )
        )
        if check_single_projection:
            if len(query.get_projection_nodes()) > 1:
                return {"error": "A snippet cannot select more than one item"}
            elif len(query.get_projection_nodes()) == 0:
                return {"error": "This snippet selects no data"}
        return {"success": True}

    except NoFKError as error:
        return {"error": str(error), "another_try": 1}
    except NotSelectSqlTypeError as error:
        return {"error": str(error), "another_try": 0}
    except Exception as error:
        return {"error": str(error), "another_try": 1}
