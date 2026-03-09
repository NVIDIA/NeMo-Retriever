from infra.Neo4jConnection import get_neo4j_conn
from shared.graph.model.reserved_words import SQLType, Parser
import logging

logger = logging.getLogger("usages/columns_usage.py")

conn = get_neo4j_conn()


def get_column_queries_cypher(table_alias: str, column_alias: str):
    wildcards = f"""OPTIONAL MATCH({table_alias})<-[r:SQL]-(w:constant WHERE w.name in $wildcards)
                    MATCH(sql_node:sql{{is_sub_select:false, id:r.sql_id, account_id:$account_id, sql_type:$sql_type}})"""
    direct = f"""CALL apoc.path.subgraphNodes({column_alias}, {{ relationshipFilter: $rel_filter, labelFilter: $label_filter, minLevel:0 }})
                YIELD node as sql_node WHERE (not sql_node.is_sub_select) AND (sql_node.sql_type = $sql_type)"""
    return f"""
        CALL ({table_alias}){{ {wildcards} RETURN collect(sql_node) as wildcard_queries }}
        CALL ({column_alias}){{ {direct} RETURN collect(sql_node) as direct_queries }}
        """


def get_column_usage_parameters(account_id: str):
    return {
        "account_id": account_id,
        "sql_type": SQLType.QUERY,
        "rel_filter": "<SQL",
        "label_filter": ">sql|-table",
        "wildcards": [Parser.WILDCARD, Parser.Q_WILDCARD],
    }


batch_params = """account_id:$account_id, sql_type:$sql_type, rel_filter: $rel_filter, label_filter:$label_filter, wildcards:$wildcards"""


##TODO: add last timestamp calculation


def calculate_column_queries_and_usage(account_id: str):
    logger.info("Updating columns usage")
    get_columns = """
    MATCH(column:column{account_id:$account_id})<-[:schema]-(table:table{account_id:$account_id})
    RETURN column, table
    """
    set_column_usage = f""" 
    {get_column_queries_cypher("table", "column")}
    WITH column, apoc.coll.toSet(wildcard_queries+direct_queries) as queries
    CALL (queries){{
        UNWIND queries as sql_node
        RETURN count(sql_node) as num_of_queries, sum(sql_node.usage) as usage 
    }}
    SET column.usage=usage, column.num_of_queries=num_of_queries
    """
    query = f"""
    CALL apoc.periodic.iterate(
        "{get_columns}", 
        "{set_column_usage}",
        {{batchSize:300, params: {{{batch_params}}} }}) 
    YIELD batches, total 
    RETURN batches, total
            """
    result = conn.query_write(query, parameters=get_column_usage_parameters(account_id))
    logger.info("Completed updating columns usage, total:")
    logger.info(result)
