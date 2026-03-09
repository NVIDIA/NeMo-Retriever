from datetime import datetime, timedelta
import logging
import pandas as pd

from infra.Neo4jConnection import get_neo4j_conn
from shared.graph.model.reserved_words import SQLType

logger = logging.getLogger("utils_dal.py")
conn = get_neo4j_conn()


def get_tables_queried_today(account_id: str, tables_ids: list[str]):
    today = datetime.now()
    yesterday = today - timedelta(days=1)

    query = """
    MATCH(table:table{account_id:$account_id})
    WHERE table.id in $tables_ids AND table.last_query_timestamp is not null
    WITH table, DateTime(table.last_query_timestamp) as queried_date
    WHERE queried_date >= DateTime($yesterday)
    RETURN collect(table.id) as tables
    """
    tables_queried_today = conn.query_read_only(
        query=query,
        parameters={
            "yesterday": yesterday,
            "tables_ids": tables_ids,
            "account_id": account_id,
        },
    )
    if len(tables_queried_today) > 0:
        return tables_queried_today[0]["tables"]
    return []


def get_join_columns(
    account_id, root_table_id: str, tables_ids: list[str]
) -> list[list[str]]:
    query = """MATCH (t:table{account_id:$account_id,id:$root_table_id})
               MATCH(other_table:table{account_id:$account_id} WHERE other_table.id in $tables_ids)
               MATCH P = SHORTEST 1 (t)-[:join]-+(other_table)
               WITH relationships(P) as rels
               WITH [rel in rels |
               CASE WHEN apoc.coll.sort([startNode(rel).name, endNode(rel).name])[0] = startNode(rel).name 
               THEN {start_table: startNode(rel).schema_name + "." + startNode(rel).name, 
               end_table :endNode(rel).schema_name + "." + endNode(rel).name, 
               join_columns:rel.join}
               ELSE {start_table: endNode(rel).schema_name + "." + endNode(rel).name, 
               end_table :startNode(rel).schema_name + "." + startNode(rel).name, 
               join_columns:rel.join}
               END] as rels
               return collect(rels) as branches
            """
    result = conn.query_read_only(
        query=query,
        parameters={
            "root_table_id": root_table_id,
            "tables_ids": tables_ids,
            "account_id": account_id,
        },
    )
    return result[0]["branches"]


def load_sqls_to_tables(account_id: str, is_view: bool = False):
    query = """
            MATCH (s:sql {account_id:$account_id, is_sub_select:false} WHERE ($is_view=TRUE AND s.sql_type="view") OR ($is_view=FALSE AND s.sql_type<>"view"))
            WITH s
            CALL apoc.path.subgraphNodes(s, {
            relationshipFilter: "SQL>",
            labelFilter: "/table|/temp_table|-column|-temp_column",
            minLevel: 0}) 
            YIELD node
            WHERE coalesce(node.deleted, false) = false
            WITH s.id as sql_id, s.sql_type as sql_type, collect(distinct node.id) as tbls, s.nodes_count as nodes_count
            RETURN collect({sql_id:sql_id, sql_type:sql_type, tbls:tbls, nodes_count:nodes_count, leaves: NULL}) as sqls_tbls
            """
    sqls_tbls_df = pd.DataFrame(
        conn.query_read_only(
            query=query,
            parameters={
                "account_id": account_id,
                "is_view": is_view,
                "sql_type": SQLType.VIEW,
            },
        )[0]["sqls_tbls"]
    )
    return sqls_tbls_df
