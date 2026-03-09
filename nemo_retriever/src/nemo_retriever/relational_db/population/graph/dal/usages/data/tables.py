from infra.Neo4jConnection import get_neo4j_conn
from shared.graph.model.reserved_words import SQLType
import logging

logger = logging.getLogger("usages/tables_usage.py")

conn = get_neo4j_conn()


##TODO: add last timestamp calculation
def calculate_tables_queries_and_usage(account_id: str):
    logger.info("Updating tables usage")
    set_table_usage_and_queries = """
    CALL (table){
        CALL apoc.path.subgraphNodes(table, { relationshipFilter: '<SQL', labelFilter: '>sql|-table', minLevel:0 }) 
        YIELD node as sql_node 
        WHERE coalesce(sql_node.is_sub_select,False)=False AND (sql_node.sql_type=$sql_type) 
        RETURN count(sql_node) as num_of_queries, sum(sql_node.usage) as usage 
    }
    WITH table, num_of_queries, usage 
    SET table.usage=usage, table.num_of_queries=num_of_queries
    """
    query = f"""
    call apoc.periodic.iterate(
        "MATCH(table:table{{account_id:$account_id}}) RETURN table", 
        "{set_table_usage_and_queries}",
        {{batchSize:300, params: {{account_id:$account_id, sql_type:$sql_type}}}}) 
    YIELD batches, total 
    RETURN batches, total
    """
    result = conn.query_write(
        query, parameters={"account_id": account_id, "sql_type": SQLType.QUERY}
    )
    logger.info("Completed updating tables usage. total:")
    logger.info(result)
