import copy
import time

import pandas as pd
import logging
import networkx as nx

# from scipy import spatial

from infra.Neo4jConnection import get_neo4j_conn

# from sklearn.metrics.pairwise import cosine_similarity
from networkx import isomorphism
from tqdm import tqdm

logger = logging.getLogger("embedding_structure_similarity.py")
conn = get_neo4j_conn()


def calculate_embedding_similarity(account_id, query_metric_obj):
    logger.info("Encoding names...")
    encode_query = """MATCH (item{account_id:$account_id})
                      WITH item
                      ORDER BY item.name
                      WITH collect(distinct item.name) AS items
                      MATCH (item{account_id:$account_id})
                      SET item.embedding = gds.alpha.ml.oneHotEncoding(items, [item.name])"""
    conn.query_write(query=encode_query, parameters={"account_id": account_id})

    # create graph in memory
    # run FastRP on all nodes
    logger.info("Building graph projection...")
    if_exists_query = (
        f"RETURN gds.graph.exists('names-graph-{account_id}') AS graphExists"
    )
    exists = conn.query_read_only(
        query=if_exists_query, parameters={"account_id": account_id}
    )
    if exists[0]["graphExists"]:
        drop_graph_query = (
            f"""call gds.graph.drop('names-graph-{account_id}') yield graphName"""
        )
        conn.query_write(query=drop_graph_query, parameters={"account_id": account_id})

    create_graph_query = f""" 
                            CALL gds.graph.project.cypher(
                                  'names-graph-{account_id}',
                                  'MATCH (n{{account_id:"{account_id}"}})
                                   RETURN
                                    id(n) AS id,
                                    n.embedding AS embedding',
                                  'MATCH (n)-[r]->(m{{account_id:"{account_id}"}}) RETURN id(n) AS source, id(m) AS target, type(r) AS type'
                                )
                            YIELD
                                  graphName, nodeCount AS nodes, relationshipCount AS rels
                            RETURN graphName, nodes, rels
                          """
    conn.query_write(query=create_graph_query, parameters={"account_id": account_id})

    logger.info("Encoding queries...")
    start = time.time()
    run_node_embedding_query = f"""
        MATCH (n:sql{{account_id:$account_id, is_sub_select:false}})
        WITH collect(ID(n)) as allsqls, n
        CALL gds.fastRP.stream('names-graph-{account_id}',
          {{
              embeddingDimension: 100,
              randomSeed: 42,
              featureProperties: ['embedding'],
              propertyRatio: 1.0,
              iterationWeights:  [0.5, 0.6, 0.7, 0.8, 0.9,1,1,1,1,1,1,1,1,1]
          }}
          )
        YIELD nodeId, embedding
        WHERE nodeId in allsqls
        RETURN nodeId, embedding, n.id as sql_id, n.sql_full_query
        """
    embs = pd.DataFrame(
        conn.query_read_only(
            query=run_node_embedding_query, parameters={"account_id": account_id}
        )
    )
    end = time.time()
    logger.info(f"Embedding time: {end - start} seconds")
    # metric_sql_embedding = embs.loc[
    #     embs["sql_id"] == str(query_metric_obj.id)
    # ].reset_index(drop=True)["embedding"][0]
    # embs["cosine"] = embs["embedding"].apply(
    #     lambda x: 1 - spatial.distance.cosine([x], [metric_sql_embedding])
    # )
    embs = embs.sort_values(by=["cosine"], ascending=False)[1:]
    top_similar = embs.loc[embs["cosine"] >= 0.5]
    return top_similar


def filter_nodes(graph_f, start_node, parent, new_graph, new_graph_no_cols, operators):
    neibrs = [
        x[0]
        for x in sorted(
            graph_f[start_node].items(), key=lambda edge: edge[1]["child_idx"]
        )
    ]
    if start_node == "start":
        for neibr in neibrs:
            new_graph.add_node(start_node, name="start")
            new_graph_no_cols.add_node(start_node, name="start")
            new_graph, new_graph_no_cols, operators = filter_nodes(
                graph_f, neibr, start_node, new_graph, new_graph_no_cols, operators
            )

    elif graph_f.nodes.get(start_node)["label"] in [
        "function",
        "operator",
        "table",
        "column",
        "constant",
        "set_op_column",
    ]:
        # new_graph.add_edge(graph_f.nodes.get(parent), graph_f.nodes.get(start_node))
        if graph_f.nodes.get(start_node)["label"] not in [
            "table",
            "column",
            "constant",
            "set_op_column",
        ]:
            operators.setdefault(graph_f.nodes.get(start_node)["name"], 0)
            operators[graph_f.nodes.get(start_node)["name"]] += 1
            if parent is not None:
                new_graph_no_cols.add_node(
                    start_node, name=graph_f.nodes.get(start_node)["name"]
                )
                new_graph_no_cols.add_edge(parent, start_node)
            else:
                new_graph_no_cols.add_node(
                    start_node, name=graph_f.nodes.get(start_node)["name"]
                )
        if parent is not None:
            new_graph.add_node(start_node, name=graph_f.nodes.get(start_node)["name"])
            new_graph.add_edge(parent, start_node)
        else:
            new_graph.add_node(start_node, name=graph_f.nodes.get(start_node)["name"])

        for neibr in neibrs:
            new_graph, new_graph_no_cols, operators = filter_nodes(
                graph_f, neibr, start_node, new_graph, new_graph_no_cols, operators
            )

    elif graph_f.nodes.get(start_node)["label"] in ["sql", "command", "alias"]:
        if graph_f.nodes.get(start_node)["name"].lower() in ["where", "from"]:
            return new_graph, new_graph_no_cols, operators
        for neibr in neibrs:
            new_graph, new_graph_no_cols, operators = filter_nodes(
                graph_f, neibr, parent, new_graph, new_graph_no_cols, operators
            )

    else:
        logger.error(f"Not supported label {graph_f.nodes.get(start_node)['label']}")
        raise

    return new_graph, new_graph_no_cols, operators


def get_filtered_graph(q_id, account_id):
    query = """
                        MATCH (f{account_id:$account_id, id:$id})
                        CALL apoc.path.subgraphAll(f,{
                            relationshipFilter: ">SQL",
                            labelFilter: "",
                            minLevel: 0})
                            YIELD nodes ,relationships
                    RETURN  nodes,   [p IN relationships | [coalesce(properties(p).child_idx,0),p]] AS edges
                    """
    func_subgraph = conn.query_read_only(
        query=query, parameters={"account_id": account_id, "id": q_id}
    )[0]
    func_nx_graph = nx.DiGraph(name=q_id)
    func_nx_graph.add_nodes_from([(y["id"], y) for y in func_subgraph["nodes"]])
    func_nx_graph.add_edges_from(
        [
            (y[1][0]["id"], y[1][2]["id"], {"child_idx": y[0]})
            for y in func_subgraph["edges"]
            if y[1][1] == "SQL"
        ]
    )
    func_nx_graph.add_edge("start", list(func_nx_graph.nodes)[0], child_idx=0)
    new_graph, new_graph_no_cols, operators = filter_nodes(
        graph_f=func_nx_graph,
        start_node="start",
        parent=None,
        new_graph=nx.DiGraph(name=f"{q_id}_filtered"),
        new_graph_no_cols=nx.DiGraph(name=f"{q_id}_filtered_nc"),
        operators={},
    )
    return pd.Series([func_nx_graph, new_graph, new_graph_no_cols, operators])


def build_new_sql(account_id, x, no_cols=False):
    if no_cols:
        mapping = x.no_cols_isomorphism.mapping
    else:
        mapping = x.full_isomorphism.mapping
    root = ""
    for mapped_nodes_key, mapped_nodes_val in mapping.items():
        if (
            mapped_nodes_key != "start" and root == ""
        ):  # next node after start - function root
            root = mapped_nodes_key
            break
    # Find path via "select", hopefully only one path, (e.g. order_by could add one more path)
    query = """
                MATCH (fn{account_id:$account_id, id:$root_id})
                MATCH (q:sql{account_id:$account_id, id:$sql_id})-[:SQL]-(sql:command{name:"Select"})
                        CALL apoc.path.expandConfig(sql,{
                            relationshipFilter: ">SQL",
                            labelFilter: "",
                            terminatorNodes:[fn],
                            minLevel: 0})
                            YIELD path
                    RETURN    q+nodes(path) as nodes_path
            """

    select_path = conn.query_read_only(
        query=query,
        parameters={"account_id": account_id, "sql_id": x.sql_id, "root_id": root},
    )
    if select_path is None or len(select_path[0]) == 0:
        logger.error("The path between sql and function root node not found. Bug.")
        return None
    if len(select_path) > 1:  # is this the way to check??
        logger.error("Multiple paths from sql to function root. TODO: implement.")
        return None
    select_path = select_path[0]
    if select_path["nodes_path"][0]["label"] != "sql":
        logger.error("First node in path is not sql. Bug.")
        return None
    where_str = select_path["nodes_path"][0]["where_str"]
    from_str = select_path["nodes_path"][0]["from_str"]
    if (
        select_path["nodes_path"][-2]["label"] == "alias"
    ):  # If alias points on  root function node take expr_str of alias
        select_str = f"SELECT {select_path['nodes_path'][-2]['expr_str']}"
    else:
        select_str = f"SELECT {select_path['nodes_path'][-1]['expr_str']}"

    if len(where_str) > 0:
        full_sql = f"{select_str} FROM {from_str} WHERE {where_str}"
    else:
        full_sql = f"{select_str} FROM {from_str}"

    formula_siblings = list(x.graph.neighbors(list(x.graph.predecessors(root))[0]))
    for f_sibling in formula_siblings:
        if f_sibling != root:
            desc = nx.descendants(x.graph, f_sibling)
            desc.add(f_sibling)
            x.graph.remove_nodes_from(desc)

    return full_sql


def check_additional_filters(x, sql_metric_full_query_graph, sql_metric_graph):
    def split_graph(
        full_graph,
    ):  # split into from and where subgraphs' formula  graph is already built, compare graphs separately.
        subgraphs = []
        for node_name in ["where", "from"]:
            nodes = [
                x
                for x, y in full_graph.nodes(data=True)
                if x != "start" and y["name"].lower() == node_name
            ]
            to_add = None
            for node in nodes:
                if (
                    to_add is None and full_graph.out_degree(node) > 0
                ):  # find first not empty from / where. There are some froms that has no out degree - not good
                    desc = nx.descendants(full_graph, node)
                    if (
                        node_name == "from"
                    ):  # do not add where node, bad for finding mapping in subgraph isomorphism
                        desc.add(node)
                    sub_g = copy.deepcopy(full_graph.subgraph(desc))
                    to_add = sub_g
                    subgraphs.append(to_add)
            if to_add is None:
                subgraphs.append(to_add)

        return subgraphs[0], subgraphs[1]

    sql_formula_graph = x.graph
    sql_full_graph = x.full_query_graph

    sql_metric_where, sql_metric_from = split_graph(sql_metric_full_query_graph)
    sql_graph_where, sql_graph_from = split_graph(sql_full_graph)

    if (
        nx.graph_edit_distance(
            sql_formula_graph,
            sql_metric_graph,
            node_match=lambda a, b: a["name"] == b["name"],
            roots=("start", "start"),
            timeout=5,
        )
        == 0
    ):
        if sql_graph_from is None or sql_metric_from is None:
            logger.error("Non empty from not found. Stange.")
            raise
        sql_from_node = [
            x for x, y in sql_graph_from.nodes(data=True) if y["name"] == "From"
        ][0]
        metric_from_node = [
            x for x, y in sql_metric_from.nodes(data=True) if y["name"] == "From"
        ][0]
        if (
            nx.graph_edit_distance(
                sql_graph_from,
                sql_metric_from,
                node_match=lambda a, b: a["name"] == b["name"],
                roots=(sql_from_node, metric_from_node),
                timeout=5,
            )
            == 0
        ):  # for now consider exactly same froms
            if (
                sql_graph_where is None and sql_metric_where is None
            ):  # isomorphic by default
                return True
            if (
                sql_graph_where is None and sql_metric_where is not None
            ):  # missing filter, possible misuse
                return False
            if (
                sql_graph_where is not None and sql_metric_where is None
            ):  # additional filter, not interesting
                return True
            where_iso = isomorphism.DiGraphMatcher(
                sql_graph_where,
                sql_metric_where,
                node_match=lambda a, b: a["name"] == b["name"],
            )
            if where_iso.subgraph_is_isomorphic():  # Returns True if a subgraph of sql_graph_where is isomorphic to sql_metric_where.
                return True  # additional filter, further remove the row in df
    return False


def retrieve_formula_cols(account_id, mapping):
    root_query = ""
    root_metric = ""
    for mapped_nodes_key, mapped_nodes_val in mapping.items():
        if (
            mapped_nodes_key != "start" and root_query == "" and root_metric == ""
        ):  # next node after start - function root
            root_query = mapped_nodes_key
            root_metric = mapped_nodes_val
            break

    formula_cols_query_metric = []
    for root in [root_query, root_metric]:
        query = """
                        MATCH (fn{account_id:$account_id, id:$root_id})
                            CALL apoc.path.subgraphNodes(fn,{
                                relationshipFilter: ">SQL",
                                labelFilter: "/column",
                                minLevel: 0})
                                YIELD node
                                WHERE coalesce(node.deleted, false) = false 
                        RETURN    collect(node.name) as col_names
                """

        col_names = conn.query_read_only(
            query=query, parameters={"account_id": account_id, "root_id": root}
        )
        if col_names is None or len(col_names[0]) == 0:
            logger.error("No columns found in function. Probably bug.")
            raise

        formula_cols_query_metric.append(col_names[0]["col_names"])

    if len(formula_cols_query_metric) != 2:
        logger.error("Formula columns in query or metrica not found, Bug.")
        raise

    return pd.Series(formula_cols_query_metric)


def get_similar_structures(account_id, query_metric_obj, most_similar):
    most_similar[["full_query_graph", "graph", "graph_no_cols", "operators"]] = (
        most_similar["sql_id"].progress_apply(
            lambda x: get_filtered_graph(x, account_id)
        )
    )
    (
        sql_metric_full_query_graph,
        sql_metric_graph,
        sql_metric_graph_no_cols,
        operators,
    ) = get_filtered_graph(str(query_metric_obj.id), account_id)
    most_similar["same_keys"] = most_similar["operators"].apply(
        lambda x: set(operators.keys()) == set(x.keys())
    )
    most_similar = most_similar.loc[most_similar["same_keys"] is True].copy()
    most_similar["same_values"] = most_similar["operators"].apply(
        lambda x: operators == x
    )
    most_similar["edit_distance"] = most_similar["graph"].progress_apply(
        lambda x: nx.graph_edit_distance(
            sql_metric_graph,
            x,
            node_match=lambda a, b: a["name"] == b["name"],
            roots=("start", "start"),
            timeout=5,
        )
    )
    most_similar["full_isomorphism"] = most_similar["graph"].progress_apply(
        lambda x: isomorphism.DiGraphMatcher(
            x, sql_metric_graph, node_match=lambda a, b: a["name"] == b["name"]
        )
    )
    most_similar["map"] = most_similar["full_isomorphism"].apply(
        lambda x: x.subgraph_is_isomorphic()
    )
    most_similar["no_cols_isomorphism"] = most_similar["graph_no_cols"].progress_apply(
        lambda x: isomorphism.DiGraphMatcher(
            x, sql_metric_graph_no_cols, node_match=lambda a, b: a["name"] == b["name"]
        )
    )
    most_similar["no_col_map"] = most_similar["no_cols_isomorphism"].apply(
        lambda x: x.subgraph_is_isomorphic()
    )
    if len(most_similar) > 0:
        most_similar[["query_formula_cols", "metric_formula_cols"]] = most_similar[
            "no_cols_isomorphism"
        ].apply(lambda x: retrieve_formula_cols(account_id, x.mapping))
        most_similar["cols_intersect_len"] = most_similar.apply(
            lambda x: len(
                set(x.query_formula_cols).intersection(set(x.metric_formula_cols))
            ),
            axis=1,
        )
        most_similar = most_similar.loc[most_similar.cols_intersect_len > 0]

    same_values_df = (
        most_similar.loc[most_similar["same_values"] is True]
        .reset_index(drop=True)
        .sort_values(by=["edit_distance"])
    )
    same_values_df["new_sql"] = same_values_df["n.sql_full_query"]
    sub_iso = most_similar.loc[
        (most_similar["same_values"] is False) & (most_similar["map"] is True)
    ].reset_index(drop=True)

    if len(sub_iso) > 0:
        sub_iso["new_sql"] = sub_iso.apply(
            lambda x: build_new_sql(account_id, x), axis=1
        )

    sub_iso_no_cols = most_similar.loc[
        (most_similar["same_values"] is False)
        & (most_similar["map"] is False)
        & (most_similar["no_col_map"] is True)
    ].reset_index(drop=True)
    if len(sub_iso_no_cols) > 0:
        sub_iso_no_cols["new_sql"] = sub_iso_no_cols.apply(
            lambda x: build_new_sql(account_id, x, no_cols=True), axis=1
        )

    ordered_most_similar = same_values_df
    if len(sub_iso) > 0:
        ordered_most_similar = pd.concat(
            [ordered_most_similar, sub_iso], ignore_index=True
        )

    if len(sub_iso_no_cols) > 0:
        ordered_most_similar = pd.concat(
            [ordered_most_similar, sub_iso_no_cols], ignore_index=True
        )

    logger.info("metric's sql:")
    logger.info(query_metric_obj.string_query)
    if len(ordered_most_similar) > 0:
        ordered_most_similar["additional_filter"] = ordered_most_similar.apply(
            lambda x: check_additional_filters(
                x, sql_metric_full_query_graph, sql_metric_graph
            ),
            axis=1,
        )
        ordered_most_similar = ordered_most_similar.loc[
            ordered_most_similar["additional_filter"] is False
        ].copy()
        ordered_most_similar["temp"] = ordered_most_similar["new_sql"].str.lower()
        ordered_most_similar = ordered_most_similar.drop_duplicates("temp")

    return ordered_most_similar


def calculate_embedding_structure_similarity(account_id, query_metric_obj):
    tqdm.pandas(desc="Sqls similarity")
    most_similar = calculate_embedding_similarity(account_id, query_metric_obj)
    most_similar_structure = get_similar_structures(
        account_id, query_metric_obj, most_similar
    )
    return most_similar_structure
