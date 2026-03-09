import logging
from infra.Neo4jConnection import get_neo4j_conn
from shared.graph.utils import chunks
import re

conn = get_neo4j_conn()
logger = logging.getLogger("bi_dal.py")


def is_cte_a_table_or_custom_query(cte_name: str):
    return cte_name.startswith("cte_table_") or cte_name.startswith("cte_custom_")


def get_field_relevant_columns(
    account_id: str, field_id: str
) -> dict[str, dict[str, str | list[str]]]:
    query = """
    MATCH(field:field{id:$field_id,account_id:$account_id})
    CALL apoc.path.subgraphNodes(field, { relationshipFilter: "CTE>|cte_depends_on>", labelFilter:'>cte' })
    YIELD node as root_cte
    WITH field, root_cte WHERE root_cte.name STARTS WITH 'cte_table_'
    MATCH(root_cte)<-[:CTE]-(:field)-[:depends_on]->(column:column{account_id:$account_id})<-[:schema]-(table:table{account_id:$account_id})
    WHERE EXISTS((column)<-[:depends_on*]-(field))
    WITH distinct root_cte, table, collect(distinct column.name) as columns_names
    WITH collect(distinct {cte_name:root_cte.name, schema_name: table.schema_name, table_name: table.name, columns_names: columns_names}) as cte_and_columns
    RETURN apoc.map.groupBy(cte_and_columns,'cte_name') as cte_and_columns
    """
    result = conn.query_read_only(
        query=query, parameters={"account_id": account_id, "field_id": field_id}
    )
    if not result:
        return {}
    return result[0]["cte_and_columns"]


def get_ctes_by_field_id(account_id: str, field_id: str):
    query = """
    MATCH(field:field{id:$field_id,account_id:$account_id})-[:CTE]->(cte:cte{account_id:$account_id})
    OPTIONAL MATCH(cte)-[:cte_depends_on]->(source_cte:cte{account_id:$account_id})
    WHERE EXISTS((source_cte)<-[:CTE]-(:field)<-[:depends_on*]-(field))
    WITH cte, source_cte order by source_cte.sql
    RETURN collect(properties(source_cte)) as sources, properties(cte) as cte 
    """
    result = conn.query_read_only(
        query=query,
        parameters={"account_id": account_id, "field_id": field_id},
    )
    if not result:
        return {}
    return result[0]


def get_next_level_of_ctes(account_id: str, ctes_ids: list[str], field_id: str):
    query = """
    MATCH(root_field:field{account_id:$account_id,id:$field_id})
    MATCH(cte:cte{account_id:$account_id} WHERE cte.id IN $ctes_ids)
    OPTIONAL MATCH(cte)-[:cte_depends_on]->(source_cte:cte{account_id:$account_id})
    WHERE EXISTS((source_cte)<-[:CTE]-(:field)<-[:depends_on*]-(root_field))
    WITH source_cte order by source_cte.sql
    RETURN collect(properties(source_cte)) as sources
    """
    result = conn.query_read_only(
        query=query,
        parameters={
            "account_id": account_id,
            "ctes_ids": ctes_ids,
            "field_id": field_id,
        },
    )
    if not result:
        return []
    return result[0]["sources"]


def get_root_cte_columns(account_id: str, field_id: str):
    query = """
    MATCH(:field{account_id:$account_id,id:$field_id})-[:depends_on]->(:column{account_id:$account_id})<-[:schema]-(table:table{account_id:$account_id})
    MATCH(table)-[:schema]->(column:column{account_id:$account_id})
    WHERE EXISTS((column)<-[:depends_on]-(:field))
    WITH column ORDER BY column.name
    RETURN collect(column.name) AS columns
    """
    result = conn.query_read_only(
        query=query,
        parameters={
            "account_id": account_id,
            "field_id": field_id,
        },
    )
    return result[0]["columns"]


def get_root_cte_custom_sql(account_id: str, field_id: str):
    query = """
    MATCH(:field{account_id:$account_id,id:$field_id})<-[:bi_custom_sql_field]-(sql:sql{account_id:$account_id})
    RETURN sql.sql_full_query as sql, sql.name as name
    """
    result = conn.query_read_only(
        query=query,
        parameters={
            "account_id": account_id,
            "field_id": field_id,
        },
    )
    return result[0]["sql"], result[0]["name"]


def create_cte(
    account_id: str,
    cte_sql: str,
    cte_name: str,
    cte_alias: str,
    field_id: str,
    created_time: str,
    is_constant: bool = False,
):
    ## connect field to cte
    query = """
    MATCH(field:field{account_id:$account_id,id:$field_id})
    MERGE(cte:cte{account_id:$account_id,name:$cte_name,sql:$cte_sql,type:'cte'})
    ON CREATE SET cte.id=randomUUID(), cte.is_constant=$is_constant, cte.created_time=$created_time
    MERGE(field)-[cte_link:CTE{alias:$cte_alias}]->(cte)
    RETURN cte.id as cte_id, cte.name as cte_name
    """
    result = conn.query_write(
        query=query,
        parameters={
            "account_id": account_id,
            "cte_sql": cte_sql,
            "cte_name": cte_name,
            "cte_alias": cte_alias,
            "field_id": field_id,
            "created_time": created_time,
            "is_constant": is_constant,
        },
    )
    cte_name = result[0]["cte_name"]
    ## connect cte to ancestor ctes- if it is not a root cte
    if is_cte_a_table_or_custom_query(cte_name):
        return
    cte_id = result[0]["cte_id"]
    query = """
    MATCH(field:field{account_id:$account_id, id:$field_id})-[:CTE]->(cte:cte{account_id:$account_id, id:$cte_id})
    MATCH(field)-[:depends_on]->(source_field:field{account_id:$account_id})-[:CTE]->(source_cte:cte{account_id:$account_id})
    WHERE source_cte<>cte
    MERGE(cte)-[:cte_depends_on]->(source_cte)
    RETURN collect(source_cte.id) as connected_ctes_ids
    """
    result = conn.query_write(
        query=query,
        parameters={"account_id": account_id, "field_id": field_id, "cte_id": cte_id},
    )
    return


def delete_ctes(account_id):
    logger.info("Delete CTEs")
    query = """ 
    CALL apoc.periodic.iterate('MATCH(cte:cte{account_id:$account_id}) RETURN cte','DETACH DELETE cte',
    {batchSize:1000,params:{account_id:$account_id}}) 
    yield batches, total 
    return batches, total
    """
    result = conn.query_write(
        query=query,
        parameters={"account_id": account_id},
    )
    logger.info(f"{result[0]['total']} CTEs deleted")


## some fields have constants as formulas like numeric values or strings
def get_constant_fields(account_id):
    query = """ 
    MATCH (field:field {account_id:$account_id})
    WHERE NOT EXISTS((field)-[:CTE]->(:cte)) 
      AND NOT EXISTS((field)-[:depends_on]->(:column|field{account_id:$account_id}))
    AND field.formula IS NOT NULL
    WITH field, apoc.text.regexGroups(field.formula,'(^[0-9]+)|(^\\"[a-zA-Z]+\\")') AS output
    WITH field,output WHERE size(output)>0
    RETURN collect({field:field{ .id, .name, .formula, type: 'constant_field'}, ancestors:[]}) as constant_fields
    """
    result = conn.query_read_only(query=query, parameters={"account_id": account_id})
    if not result:
        return []
    return result[0]["constant_fields"]


def get_fields_ids_chunks(account_id: str):
    query = """
    MATCH (field:field {account_id:$account_id})-[:depends_on]->(ancestor:column|field{account_id:$account_id})
    WHERE NOT EXISTS((field)-[:CTE]->(:cte))
    RETURN collect(distinct field.id) as fields_ids
    """
    result = conn.query_read_only(query=query, parameters={"account_id": account_id})
    if not result:
        return []
    fields_chunks = chunks(result[0]["fields_ids"], 5000)
    return fields_chunks


federated_pattern = r"\[([a-zA-Z_]+)([\.]*)([a-z0-9 \(\)]*)\]\."


def handle_federated(field_formula: str, ancestors: list[dict[str, str]]):
    formula_copy = field_formula
    formula_copy = re.sub(federated_pattern, "", formula_copy)
    if formula_copy == field_formula:
        return formula_copy
    for ancestor in ancestors:
        if (
            not ancestor["fullyQualifiedName"]
            or ancestor["fullyQualifiedName"] == ancestor["name"]
            or ancestor["fullyQualifiedName"] not in formula_copy
        ):
            continue
        formula_copy = formula_copy.replace(
            ancestor["fullyQualifiedName"], f"[{ancestor['name']}]"
        )
    return formula_copy


def get_fields_without_cte(account_id, processed_ids):
    logger.info("Collecting fields without CTEs")
    fields_chunks = get_fields_ids_chunks(account_id)
    fields = []
    for i, fields_ids in enumerate(fields_chunks):
        logger.info(f"chunk {i}")
        query = """ 
            UNWIND $fields_ids as field_id
            MATCH (field:field {account_id:$account_id,id:field_id} )-[:depends_on]->(ancestor:column|field{account_id:$account_id})
            WITH field, collect(ancestor) as ancestors
            WITH field, ancestors WHERE ALL(x IN ancestors WHERE x:column or (x:field and EXISTS((x)-[:CTE]->(:cte))))
            UNWIND ancestors as ancestor
            CALL (ancestor){
                WITH ancestor, [(ancestor)-[:depends_on]->(grandparent) | grandparent.name] as grandparents
                RETURN CASE WHEN ancestor.formula IS NOT NULL
                        AND size(grandparents)=1 
                        AND apoc.text.join(['[',grandparents[0],']'],'')=ancestor.formula
                        THEN True ELSE False END AS is_referencing_ancestor
            }
            CALL apoc.case([
                ancestor:field and ancestor.formula IS NOT NULL AND NOT is_referencing_ancestor, // calculated field
                'MATCH(ancestor)-[cte_alias:CTE]->(cte:cte{account_id:$account_id})
                RETURN ancestor{type:"calculated_field", .fullyQualifiedName, .id, .name, .formula, cte:properties(cte), cte_alias: cte_alias.alias } as ancestor',
                
                ancestor:field AND ancestor.custom_sql IS NOT NULL 
                            AND EXISTS((:sql)-[:bi_custom_sql_field]->(field)) 
                            AND NOT is_referencing_ancestor, // custom BI SQL field
                'MATCH(ancestor)-[cte_alias:CTE]->(cte:cte{account_id:$account_id})
                RETURN ancestor{type:"custom_sql_field", .fullyQualifiedName, .id ,.name, .custom_sql, cte:properties(cte), cte_alias: cte_alias.alias } as ancestor',

                ancestor:field, // referencing field
                'MATCH(ancestor)-[cte_alias:CTE]->(cte:cte{account_id:$account_id})
                RETURN ancestor{type:"referencing_field",.fullyQualifiedName,  .id ,.name, cte:properties(cte), cte_alias: cte_alias.alias } as ancestor'
            ], 
            'RETURN ancestor{.type, .id, .name, .table_name, .schema_name } as ancestor', // column
            {ancestor:ancestor, account_id:$account_id, is_referencing_ancestor:is_referencing_ancestor})
            YIELD value 
            WITH value.ancestor as ancestor, field
            WITH collect(ancestor) as ancestors, field
            WITH field, ancestors, CASE WHEN field.formula IS NOT NULL
                        AND size(ancestors)=1 
                        AND apoc.text.join(['[',ancestors[0].name,']'],'') =field.formula
                        THEN True ELSE False
                        END AS is_referencing_field
            WITH field, ancestors,
                    CASE WHEN field.formula IS NOT NULL AND NOT is_referencing_field THEN 'calculated_field' 
                    WHEN field.custom_sql IS NOT NULL AND EXISTS((field)<-[:bi_custom_sql_field]-()) THEN 'custom_sql_field' 
                    WHEN size(ancestors)>1 AND field.formula IS NULL THEN 'error_field'
                    ELSE 'referencing_field' END AS field_type
            RETURN collect({field:field{.id,.name,.formula,.custom_sql, .fullyQualifiedName, type: field_type}, ancestors:ancestors}) as fields_and_ancestors
        """
        result = conn.query_read_only(
            query=query, parameters={"account_id": account_id, "fields_ids": fields_ids}
        )
        if result:
            fields.extend(result[0]["fields_and_ancestors"])

    for f in fields:
        if "formula" in f["field"] and f["field"]["formula"] is not None:
            f["field"]["formula"] = handle_federated(
                f["field"]["formula"], f["ancestors"]
            )
    if not fields:
        return []
    if not processed_ids:
        return fields
    processed_set = set(processed_ids)
    fields_to_connect_to_ctes = list(
        filter(lambda field: field["field"]["id"] not in processed_set, fields)
    )
    processed_ids = []
    return fields_to_connect_to_ctes
