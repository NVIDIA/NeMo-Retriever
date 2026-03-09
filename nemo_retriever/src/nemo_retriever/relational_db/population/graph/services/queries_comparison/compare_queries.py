import logging

from networkx import Graph, DiGraph
from networkx.algorithms.isomorphism import DiGraphMatcher
from typing import Callable

from shared.graph.model.query import Query
from shared.graph.model.reserved_words import Labels
from shared.graph.model.node import Node

logger = logging.getLogger("compare_queries.py")


def compare_query_to_list_of_queries(
    main_graph: Graph,
    queries_ids: list[str],
    get_sql_string_by_id: Callable,
    get_parsed_query: Callable[[str], Query],
    save_leafs_in_df: Callable = None,
    is_subgraph: bool = False,
    remove_aliases: bool = False,
):
    ids_of_matching_sqls = []
    ids_of_subset_sqls = []
    for sql_id in queries_ids:
        compared_sql = get_sql_string_by_id(sql_id)
        try:
            compared_query = get_parsed_query(compared_sql)
            compared_graph = get_slim_graph_from_edges(
                compared_query.get_edges(),
                remove_aliases=remove_aliases,
            )
            if save_leafs_in_df is not None:
                save_leafs_in_df(sql_id, compared_graph)
            is_equal = compare_queries(
                main_graph, compared_graph, is_subgraph=is_subgraph
            )
        except Exception as err:
            logger.exception(err)
            logger.info("Failed comparing.")
            is_equal = False

        if is_equal:
            ## heuristic: if a query is a subgraph of another query and the nodes count is equal- then it's identical
            if is_subgraph:
                if (
                    main_graph.number_of_nodes() != compared_graph.number_of_nodes()
                    or main_graph.number_of_edges() != compared_graph.number_of_edges()
                ):
                    ids_of_subset_sqls.append(sql_id)
                else:
                    ids_of_matching_sqls.append(sql_id)
            else:
                ids_of_matching_sqls.append(sql_id)
                return ids_of_matching_sqls, ids_of_subset_sqls
    return ids_of_matching_sqls, ids_of_subset_sqls


def edge_compare(
    edge1: dict[str, str | None],
    edge2: dict[str, str | None],
):
    if (
        "child_idx" in edge1
        and edge1["child_idx"] is not None
        and "child_idx" in edge2
        and edge2["child_idx"] is not None
    ):
        return edge1["child_idx"] == edge2["child_idx"]
    return True


def node_compare(node1: dict[str, str], node2: dict[str, str]):
    if node1["label"] != node2["label"]:
        return False
    if node1["label"] == Labels.SQL:
        return True
    if node1["label"] in [
        Labels.TABLE,
        Labels.COLUMN,
        Labels.TEMP_TABLE,
        Labels.TEMP_COLUMN,
    ]:
        return node1["id"] == node2["id"]
    return node1["name"] == node2["name"]


def get_set_of_edges(graph: Graph):
    edges = set()
    for edge in graph.edges.data():
        try:
            edge_label = edge[2].get("child_idx")
            edges.add(
                graph.nodes[edge[0]]["node_label"].lower()
                + "_"
                + graph.nodes[edge[1]]["node_label"].lower()
                + "_"
                + (f"{edge_label}" if edge_label else "")
            )
        except Exception as e:
            print(f"Error getting edge label for edge {edge}: {e}")
    return edges


def compare_sets_of_edges_by_order(
    main_set: set, compared_set: set, is_full_comparison=False
):
    if is_full_comparison and len(main_set) != len(compared_set):
        return False

    # main_hashes_list = []
    # compared_hashes_list = []
    # for edge in main_set:
    #     main_hashes_list.append(hash(edge))
    # for edge in compared_set:
    #     compared_hashes_list.append(hash(edge))
    # if sum(main_hashes_list) == sum(compared_hashes_list):
    #     main_hashes_list.sort()
    #     compared_hashes_list.sort()
    #     if main_hashes_list == compared_hashes_list:
    #         return True

    for edge in main_set:
        if edge not in compared_set:
            return False
    return True


def compare_queries(
    main_graph: Graph, compared_graph: Graph, is_subgraph=False
) -> bool:
    if is_subgraph:
        ## subgraph isomorphism
        return DiGraphMatcher(
            G1=compared_graph,  # Check whether a subgraph of G1 is isomorphic to G2
            G2=main_graph,
            node_match=node_compare,
            edge_match=edge_compare,
        ).subgraph_is_isomorphic()
    # full comparison
    main_set = get_set_of_edges(main_graph)
    compared_set = get_set_of_edges(compared_graph)
    return compare_sets_of_edges_by_order(
        main_set, compared_set, is_full_comparison=True
    )


aliases_labels = [Labels.ALIAS, Labels.SET_OP_COLUMN]


def get_node_label(node):
    if node.label == Labels.SQL:
        return node.label
    node_label = f"{node.label}_{node.name}"
    if node.label in [Labels.TABLE, Labels.TEMP_TABLE]:
        identity_props = node.match_props
        return f"{identity_props['db_name']}.{identity_props['schema_name']}.{identity_props['name']}"
    if node.label in [Labels.COLUMN, Labels.TEMP_COLUMN]:
        identity_props = node.match_props
        return f"{identity_props['db_name']}.{identity_props['schema_name']}.{identity_props['table_name']}.{identity_props['name']}"
    return node_label


def get_slim_graph_from_edges(
    edges: list[tuple[Node, Node, dict, str]], remove_aliases=False
):
    g = DiGraph()
    aliases_ids = set()
    for e in edges:
        node_e0 = e[0].get_sql_node() if isinstance(e[0], Query) else e[0]
        node_e1 = e[1].get_sql_node() if isinstance(e[1], Query) else e[1]
        if node_e0.label in aliases_labels:
            aliases_ids.add(node_e0.id)
        if node_e1.label in aliases_labels:
            aliases_ids.add(node_e1.id)
        g.add_nodes_from(
            [
                (
                    node_e0.id,
                    {
                        "name": node_e0.name,
                        "label": node_e0.label,
                        "id": node_e0.id,
                        "node_label": get_node_label(node_e0),
                    },
                ),
                (
                    node_e1.id,
                    {
                        "name": node_e1.name,
                        "label": node_e1.label,
                        "id": node_e1.id,
                        "node_label": get_node_label(node_e1),
                    },
                ),
            ]
        )
        child_idx = None if not e[2] or "child_idx" not in e[2] else e[2]["child_idx"]
        g.add_edges_from([(node_e0.id, node_e1.id)], child_idx=child_idx)
    if remove_aliases:
        g = remove_aliases_from_graph(g, aliases_ids)
    return g


def remove_aliases_from_graph(g: DiGraph, aliases_ids: set[str]):
    removed_aliases_ids = []
    for alias_id in aliases_ids:
        for predecessor in g.predecessors(alias_id):
            for successor in g.successors(alias_id):
                g.add_edge(predecessor, successor)
        try:
            g.remove_node(alias_id)
            removed_aliases_ids.append(alias_id)
        except Exception:
            pass
    return g
