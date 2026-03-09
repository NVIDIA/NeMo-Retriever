from infra.Neo4jConnection import get_neo4j_conn
from shared.graph.model.reserved_words import SQLType
import logging

logger = logging.getLogger("usages/analsyis.py")

conn = get_neo4j_conn()

set_analysis_usage = """
CALL apoc.path.subgraphNodes(sql_node, { relationshipFilter: '<SQL', labelFilter: '>sql', filterStartNode: true })
YIELD node as main_sql
WHERE main_sql.is_sub_select = false AND main_sql.sql_type=$sql_type
WITH distinct analysis, COUNT(main_sql) as num_of_queries, SUM(coalesce(main_sql.usage,0)) as usage
SET analysis.num_of_queries=num_of_queries, analysis.usage=usage
"""


def calculate_analyses_usage(account_id: str):
    logger.info("Updating analyses usage")
    query = f"""
    CALL apoc.periodic.iterate(
        "MATCH(analysis:analysis{{account_id:$account_id}} WHERE NOT coalesce(analysis.recommended, FALSE))-[:analysis_of]->(sql_node:sql{{account_id:$account_id}}) RETURN analysis, sql_node", 
        "{set_analysis_usage}",
        {{batchSize:1000, params: {{account_id:$account_id, sql_type: $sql_type}} }}) 
    YIELD batches, total 
    RETURN batches, total
    """
    result = conn.query_write(
        query, parameters={"account_id": account_id, "sql_type": SQLType.QUERY}
    )
    logger.info("Completed updating analyses usage , total: ")
    logger.info(result)


def update_single_analysis_usage(account_id: str, analysis_id: str):
    query = f"""
    MATCH(analysis:analysis{{account_id:$account_id, id:$analysis_id}})-[:analysis_of]->(sql_node:sql{{account_id:$account_id}})
    {set_analysis_usage}
    """
    conn.query_write(
        query,
        parameters={
            "analysis_id": analysis_id,
            "account_id": account_id,
            "sql_type": SQLType.QUERY,
        },
    )
