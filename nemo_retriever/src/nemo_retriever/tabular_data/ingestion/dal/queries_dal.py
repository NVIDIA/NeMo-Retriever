import pandas as pd
from nemo_retriever.tabular_data.ingestion.utils import chunks
from nemo_retriever.tabular_data.ingestion.dal.utils_dal import prepare_edge, add_edges
from nemo_retriever.tabular_data.ingestion.model.reserved_words import SQLType
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn

conn = get_neo4j_conn()

def get_sql_by_id(sql_id: str):
    query = "MATCH(n:sql{id:$sql_id}) return n.sql_full_query as sql_full_query"
    result = conn.query_read_only(
        query=query, parameters={"sql_id": sql_id}
    )
    return result[0]["sql_full_query"]


def load_sqls_to_tables(is_view: bool = False):
    query = """
            MATCH (s:sql {is_sub_select:false} WHERE ($is_view=TRUE AND s.sql_type="view") OR ($is_view=FALSE AND s.sql_type<>"view"))
            WITH s
            CALL apoc.path.subgraphNodes(s, {
            relationshipFilter: "SQL>",
            labelFilter: "/table|-column",
            minLevel: 0}) 
            YIELD node
            WITH s.id as sql_id, s.sql_type as sql_type, collect(distinct node.id) as tbls, s.nodes_count as nodes_count
            RETURN collect({sql_id:sql_id, sql_type:sql_type, tbls:tbls, nodes_count:nodes_count, leaves: NULL}) as sqls_tbls
            """
    sqls_tbls_df = pd.DataFrame(
        conn.query_read_only(
            query=query,
            parameters={
                "is_view": is_view,
                "sql_type": SQLType.VIEW,
            },
        )[0]["sqls_tbls"]
    )
    return sqls_tbls_df


def add_query(edges):
    """Add the nodes and edges of the parsed query to the graph."""
    edges_data = [prepare_edge(edge) for edge in edges]
    for chunk in chunks(edges_data, 10):
        add_edges(chunk)

def get_sql_by_full_query(sql_full_query: str):
    query = "MATCH(n:sql{sql_full_query:$sql_full_query}) return n.id as id"
    result = conn.query_read_only(
        query=query,
        parameters={"sql_full_query": sql_full_query},
    )
    if result:
        return result[0]["id"]
    return None
