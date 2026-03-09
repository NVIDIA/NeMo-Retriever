from shared.graph.dal.usage_dal import get_count_str_by_month
from infra.Neo4jConnection import get_neo4j_conn
from shared.graph.model.reserved_words import SQLType
import logging

logger = logging.getLogger("usages/queries_usage.py")
conn = get_neo4j_conn()


def calculate_queries_usage(account_id: str):
    logger.info("Updating queries usage")
    cnt_str = get_count_str_by_month("sql_node")

    query = f"""
    call apoc.periodic.iterate(
        " MATCH(sql_node:sql{{account_id:$account_id, sql_type:$sql_type, is_sub_select:False}}) RETURN sql_node", 
        " SET sql_node.usage={cnt_str} ",
        {{batchSize:1000, params: {{account_id:$account_id, sql_type:$sql_type}}}}) 
    YIELD batches, total 
    RETURN batches, total
    """
    result = conn.query_write(
        query, parameters={"account_id": account_id, "sql_type": SQLType.QUERY}
    )
    logger.info("Completed updating queries usage. total:")
    logger.info(result)
