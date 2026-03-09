from networkx import DiGraph
from pandas import DataFrame
from typing import Callable, Union

from shared.graph.dal.queries_dal import get_sql_by_id
from shared.graph.model.query import Query
from shared.graph.model.reserved_words import SQLType
from shared.graph.services.queries_comparison.compare_queries import (
    compare_query_to_list_of_queries,
    get_slim_graph_from_edges,
)


def get_sqls_connected_to_tables(
    sql_type: SQLType,
    tbl_ids: list[str],
    nodes_count: int,
    sqls_tbls_df: DataFrame,
    parsed_query_leaves: list[tuple[str, str]],
    is_subgraph: bool = False,
) -> list[str]:
    if sqls_tbls_df.empty:
        return []

    # s.id as sql_id, collect(distinct node.is) as tbls, s.nodes_count as nodes_count
    if not is_subgraph:
        potential_sqls_list = list(
            sqls_tbls_df.loc[
                (sqls_tbls_df.sql_type == sql_type)
                & (sqls_tbls_df.nodes_count == nodes_count)
                & sqls_tbls_df.apply(lambda x: set(x.tbls) == set(tbl_ids), axis=1)
                & sqls_tbls_df.apply(
                    lambda x: x.leaves is None
                    or set(x.leaves) == set(parsed_query_leaves),
                    axis=1,
                )
            ].sql_id
        )
    else:  # is subgraph
        potential_sqls_list = list(
            sqls_tbls_df.loc[
                (sqls_tbls_df.nodes_count >= nodes_count)
                & sqls_tbls_df.apply(
                    lambda x: set(x.tbls).issubset(set(tbl_ids)), axis=1
                )
                & sqls_tbls_df.apply(
                    lambda x: x.leaves is None
                    or set(x.leaves).issubset(set(parsed_query_leaves)),
                    axis=1,
                )
            ].sql_id
        )
    return potential_sqls_list


def get_leafs_from_graph(query_slim_graph: DiGraph):
    return [
        (
            query_slim_graph.nodes[node_id]["label"],
            query_slim_graph.nodes[node_id]["name"],
        )
        for node_id in query_slim_graph.nodes()
        if query_slim_graph.out_degree(node_id) == 0
    ]


def find_identical_queries(
    account_id: str,
    get_parsed_query: Callable[[str], Query],
    main_sql: str,
    sqls_tbls_from_graph_df: Union[DataFrame, None] = None,
    is_subgraph: bool = False,
    remove_aliases: bool = False,
    in_memory_queries: Union[dict[str, Query], None] = None,
    potential_sqls_ids: Union[list[str], None] = None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    if sqls_tbls_from_graph_df is None and potential_sqls_ids is None:
        raise ValueError(
            "sqls_tbls_from_graph_df or potential_sqls_ids must be provided"
        )
    parsed_query_obj = get_parsed_query(main_sql)
    main_graph = get_slim_graph_from_edges(
        parsed_query_obj.get_edges(),
        remove_aliases=remove_aliases,
    )
    parsed_query_leaves = get_leafs_from_graph(main_graph)

    if potential_sqls_ids is None:
        potential_sqls_ids = get_sqls_connected_to_tables(
            parsed_query_obj.sql_node.props["sql_type"],
            parsed_query_obj.get_tables_ids(),
            parsed_query_obj.get_nodes_counter(),
            sqls_tbls_from_graph_df,
            parsed_query_leaves,
            is_subgraph,
        )

    def save_leafs_in_df(query_id: str, query_slim_graph: DiGraph):
        if sqls_tbls_from_graph_df is not None:
            saved_query_leaves = sqls_tbls_from_graph_df.loc[
                sqls_tbls_from_graph_df.sql_id == query_id, "leaves"
            ]
            if list(saved_query_leaves)[0] is None:
                query_leafs = get_leafs_from_graph(query_slim_graph)
                mask = sqls_tbls_from_graph_df.sql_id == query_id
                row_idx = sqls_tbls_from_graph_df.index[mask][0]
                sqls_tbls_from_graph_df.at[row_idx, "leaves"] = query_leafs

    identical_sqls_ids_from_graph = []
    subset_ids_from_graph = []
    if potential_sqls_ids:
        ## search in graph
        identical_sqls_ids_from_graph, subset_ids_from_graph = (
            compare_query_to_list_of_queries(
                main_graph=main_graph,
                queries_ids=potential_sqls_ids,
                get_sql_string_by_id=lambda id: get_sql_by_id(account_id, id),
                get_parsed_query=get_parsed_query,
                save_leafs_in_df=save_leafs_in_df,
                remove_aliases=remove_aliases,
                is_subgraph=is_subgraph,
            )
        )
    identical_sqls_ids_in_memory = []
    subset_ids_from_memory = []
    # we get in_memory_queries dictionary during population
    if len(identical_sqls_ids_from_graph) == 0 and in_memory_queries:
        identical_sqls_ids_in_memory, subset_ids_from_memory = (
            compare_query_to_list_of_queries(
                main_graph=main_graph,
                queries_ids=in_memory_queries.keys(),
                get_parsed_query=get_parsed_query,
                get_sql_string_by_id=lambda id: in_memory_queries[id].string_query,
                is_subgraph=is_subgraph,
                remove_aliases=remove_aliases,
            )
        )
    # when we're checking full equality we expect to have only one identical sql id (either from graph or in memory)
    return (
        identical_sqls_ids_from_graph,
        identical_sqls_ids_in_memory,
        subset_ids_from_graph,
        subset_ids_from_memory,
    )
