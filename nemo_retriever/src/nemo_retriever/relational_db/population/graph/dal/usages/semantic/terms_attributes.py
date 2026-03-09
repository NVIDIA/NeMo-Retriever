from shared.graph.dal.usages.data.columns import (
    get_column_queries_cypher,
    get_column_usage_parameters,
    batch_params,
)
from infra.Neo4jConnection import get_neo4j_conn
import logging

logger = logging.getLogger("usages/terms_attributes.py")

conn = get_neo4j_conn()

get_sql_or_alias_roots = """
            CALL apoc.path.subgraphNodes(item, { relationshipFilter: $rel_filter, filterStartNode: True, minLevel:0 })
            YIELD node WHERE node.sql_type=$sql_type and node.is_sub_select=false
            return collect(node) as queries """

set_attribute_usage = f""" 
    MATCH(attribute)-[:attr_of]->(item:column|alias|sql|error_node{{account_id:$account_id}})
    CALL apoc.case([
        item:column, '
        MATCH(item)<-[:schema]-(table:table{{account_id:$account_id}})
        {get_column_queries_cypher("table", "item")}
        RETURN apoc.coll.toSet(wildcard_queries+direct_queries) as queries',
        item:sql or item:alias, '{get_sql_or_alias_roots}'
        ],
        'return [] as queries', //we don't check the queries of error nodes (deleted data)
        {{item:item, {batch_params}}})
    YIELD value as result
    WITH attribute, apoc.coll.toSet(apoc.coll.flatten(collect(result.queries))) as queries
    CALL (attribute, queries){{
        UNWIND queries as sql_node
        RETURN count(sql_node) as num_of_queries, sum(sql_node.usage) as usage 
    }}
    CALL (attribute){{
        MATCH (attribute)-[:attr_of|reaching]->(column:column)
        RETURN count(column) as num_of_columns
    }}
    SET attribute.usage=usage, attribute.num_of_queries=num_of_queries, attribute.num_of_columns=num_of_columns
    """


def update_single_attr_usage(account_id: str, attr_id: str):
    query = f"""
    MATCH(attribute:attribute{{account_id:$account_id,id:$attr_id}})
    {set_attribute_usage}
    RETURN attribute
    """
    parameters = get_column_usage_parameters(account_id)
    parameters["attr_id"] = attr_id
    result = conn.query_write(query, parameters)
    return result[0]["attribute"]["usage"]


def update_term_attributes_usage(account_id: str, term_id: str):
    query = f"""
    MATCH(term:term{{account_id:$account_id,id:$term_id}})-[:term_of]->(attribute:attribute{{account_id:$account_id}})
    {set_attribute_usage}
    RETURN attribute
    """
    parameters = get_column_usage_parameters(account_id)
    parameters["term_id"] = term_id
    result = conn.query_write(query, parameters)
    return result[0]["attribute"]["usage"]


def calculate_attributes_usage(account_id: str):
    logger.info("Updating attributes usage")
    query = f"""
    CALL apoc.periodic.iterate(
        " MATCH(attribute:attribute{{account_id:$account_id}}) RETURN attribute", 
        "{set_attribute_usage}",
        {{batchSize:1000, params: {{{batch_params}}} }}) 
    YIELD batches, total 
    RETURN batches, total
    """
    result = conn.query_write(query, parameters=get_column_usage_parameters(account_id))
    logger.info("Completed updating attributes usage , total: ")
    logger.info(result)


set_term_usage = f""" 
    MATCH(term)-[:term_of]->(attribute:attribute{{account_id:$account_id}})-[:attr_of]->(item:column|alias|sql|error_node{{account_id:$account_id}})
    CALL apoc.case([
        item:column, '
        MATCH(item)<-[:schema]-(table:table{{account_id:$account_id}})
        {get_column_queries_cypher("table", "item")}
        RETURN apoc.coll.toSet(wildcard_queries+direct_queries) as queries',
        item:sql or item:alias, '{get_sql_or_alias_roots}'
        ],
        'return [] as queries', //we don't check the queries of error nodes (deleted data)
        {{item:item, {batch_params}}})
    YIELD value as result
    WITH term, apoc.coll.toSet(apoc.coll.flatten(collect(result.queries))) as queries
    CALL (queries){{
        UNWIND queries as sql_node
        RETURN count(sql_node) as num_of_queries, sum(sql_node.usage) as usage 
    }}
    CALL (term){{
        MATCH(term)-[:term_of]->(attribute:attribute)-[:attr_of|reaching]->(column:column)<-[:schema]-(table:table)
        RETURN count(distinct table) as num_of_tables, count(distinct column) as num_of_columns
    }}
    SET term.usage=usage, term.num_of_queries=num_of_queries, term.num_of_columns=num_of_columns, term.num_of_tables=num_of_tables
    """


def update_single_term_usage(account_id: str, term_id: str):
    query = f"""
    MATCH(term:term{{account_id:$account_id,id:$term_id}})
    {set_term_usage}
    RETURN term
    """
    parameters = get_column_usage_parameters(account_id)
    parameters["term_id"] = term_id
    result = conn.query_write(query, parameters)
    return result[0]["term"]["usage"]


def update_certified_for_zones(
    account_id: str, attribute_id: str, snippet_id: str, certified_for_zones: list
):
    query = """
    Match(attr: attribute {account_id: $account_id, id: $id})-[snippet:attr_of {sql_snippet_id: $snippet_id}]->()
    SET snippet.certified_for_zones=$certified_for_zones
    """
    parameters = {
        "account_id": account_id,
        "id": attribute_id,
        "snippet_id": snippet_id,
        "certified_for_zones": certified_for_zones,
    }
    conn.query_write(query, parameters)


def update_document_reference_certified_for_zones(
    account_id: str,
    attribute_id: str,
    document_reference_id: str,
    certified_for_zones: list,
):
    query = """
    Match(attr: attribute {account_id: $account_id, id: $id})-[document_reference:attr_of_file]->(doc:document {account_id: $account_id, id: $document_reference_id})
    SET document_reference.certified_for_zones=$certified_for_zones
    """
    parameters = {
        "account_id": account_id,
        "id": attribute_id,
        "document_reference_id": document_reference_id,
        "certified_for_zones": certified_for_zones,
    }
    conn.query_write(query, parameters)


def calculate_terms_usage(account_id: str):
    logger.info("Updating terms usage")
    query = f"""
    CALL apoc.periodic.iterate(
        "MATCH(term:term{{account_id:$account_id}}) RETURN term", 
        "{set_term_usage}",
        {{batchSize:300, params: {{ {batch_params} }}}}) 
    YIELD batches, total 
    RETURN batches, total
    """
    result = conn.query_write(query, parameters=get_column_usage_parameters(account_id))
    logger.info("Completed updating terms usage")
    logger.info(result)
