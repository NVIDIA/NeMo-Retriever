import logging
from shared.graph.utils import chunks
from shared.graph.dal.utils_dal import add_edges, prepare_edge
from shared.graph.model.node import Node

from infra.Neo4jConnection import get_neo4j_conn
from shared.graph.model.query import get_sql_counters


logger = logging.getLogger("queries_dal.py")
conn = get_neo4j_conn()


def get_all_sql_nodes(account_id):
    query = "MATCH(n:SQL_node{account_id:$account_id,is_sub_select:False}) return collect (n.id) as all_sqls"
    result = conn.query_read_only(query=query, parameters={"account_id": account_id})
    return result[0]["all_sqls"]


def get_sql_by_id(account_id: str, sql_id: str):
    query = "MATCH(n:sql{account_id:$account_id,id:$sql_id}) return n.sql_full_query as sql_full_query"
    result = conn.query_read_only(
        query=query, parameters={"account_id": account_id, "sql_id": sql_id}
    )
    return result[0]["sql_full_query"]


def get_sql_by_full_query(account_id: str, sql_full_query: str):
    query = "MATCH(n:sql{account_id:$account_id,sql_full_query:$sql_full_query}) return n.id as id"
    result = conn.query_read_only(
        query=query,
        parameters={"account_id": account_id, "sql_full_query": sql_full_query},
    )
    if result:
        return result[0]["id"]
    return None


def update_counters_and_timestamps_for_query_and_affected_data(
    account_id: str,
    identical_sql_id: str,
    sql_node: Node,
    update_data_last_query_timestamp: bool = True,
):
    # If sql is already in graph and sql is not temporary metric sql - add +1 to sql's counter
    # sql_node = edges[0][0]
    latest_timestamp = sql_node.get_properties()["last_query_timestamp"]
    total_counter, cnt_per_month = get_sql_counters(sql_node)
    set_cnts_str = ""
    for month, cnt in cnt_per_month.items():
        set_cnts_str = f"{set_cnts_str}SET s.{month} = coalesce(s.{month}, 0) + {cnt}\n"
    # if the sql already exists in the graph, then update the "last_query_timestamp" property
    # of the sql_node and the table and column nodes that appear as part of the sql.
    cypher_query = f"""MATCH (s:sql {{id: $id, account_id: $account_id}}) 
                                SET s.last_query_timestamp = $latest_timestamp 
                                SET s.total_counter = s.total_counter + $total_counter
                                {set_cnts_str}
                            """.strip()
    conn.query_write(
        query=cypher_query,
        parameters={
            "latest_timestamp": latest_timestamp,
            "account_id": account_id,
            "id": identical_sql_id,
            "total_counter": total_counter,
        },
    )

    if update_data_last_query_timestamp:
        cypher_query = """
                MATCH (v: sql{account_id:$account_id, id:$id})
                WITH v
                CALL apoc.path.subgraphNodes(v, {
                    relationshipFilter: "SQL>", 
                    labelFilter: "/column|/table"})
                    YIELD node
                    WHERE coalesce(node.deleted, false) = false 
                    SET node.last_query_timestamp = $latest_timestamp
                """
        conn.query_write(
            cypher_query,
            parameters={
                "latest_timestamp": latest_timestamp,
                "account_id": account_id,
                "id": identical_sql_id,
            },
        )


def add_query(edges, account_id):
    """
    Add the nodes and edges of the parsed query to the graph
    :param edges: edges ready for insertion
    :return:
    """
    edges_data = [prepare_edge(edge, account_id) for edge in edges]
    edges_data_chunks = list(chunks(edges_data, 10))
    for i, chunk in enumerate(edges_data_chunks):
        add_edges(account_id, chunk)


def get_sql_type(account_id, query_id):
    query = """
        MATCH (s:sql {id:$query_id, account_id:$account_id})
        RETURN s.sql_type as sql_type 
    """
    result = conn.query_read_only(
        query, parameters={"query_id": query_id, "account_id": account_id}
    )
    return result[0]["sql_type"]


def get_all_snippets_without_description_suggestion(account_id):
    query = """MATCH(a:attribute{account_id: $account_id})-[r:attr_of]->(n) 
                where r.description_suggestion is null OR r.description_suggestion = ''
                RETURN r.sql_snippet_id as id, r.sql_snippet as query_text
    """
    result = conn.query_read_only(
        query,
        parameters={
            "account_id": account_id,
        },
    )
    return result
