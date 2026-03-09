import logging
import re

import pandas as pd
from infra.Neo4jConnection import get_neo4j_conn
from shared.graph.services.sql_snippet import (
    save_custom_snippet,
    update_definition_sql,
)
from shared.graph.model.reserved_words import Labels, label_to_type

logger = logging.getLogger("attribute_dal.py")
conn = get_neo4j_conn()


def _ensure_unique_attr_name(account_id: str, term_id: str, base_name: str) -> str:
    """
    Append a running number suffix when an attribute with the same name already exists
    for the given term/account.
    """
    query = """
        MATCH (:term {account_id: $account_id, id: $term_id})-[:term_of]->(attr:attribute {account_id: $account_id})
        WHERE attr.name STARTS WITH $base_name
        RETURN attr.name AS name
    """
    rows = conn.query_read_only(
        query=query,
        parameters={
            "account_id": account_id,
            "term_id": term_id,
            "base_name": base_name,
        },
    )
    existing = {row["name"] for row in rows}
    if base_name not in existing:
        return base_name

    pattern = re.compile(rf"^{re.escape(base_name)} \((\d+)\)$")
    used_numbers = {
        int(match.group(1))
        for name in existing
        if (match := pattern.match(name)) is not None
    }
    next_number = 1
    while next_number in used_numbers:
        next_number += 1
    return f"{base_name} ({next_number})"


def attr_name_exists_insensitive(account_id: str, attr_name: str, term_id: str) -> bool:
    """
    Check if an attribute with the given name (case-insensitive) already exists
    for the specified term and account.
    """
    attr_name = attr_name.strip()
    query = """
        match (bt:term {id: $term_id, account_id: $account_id})-[:term_of]->(attr:attribute)
        where toLower(attr.name) = toLower($attr_name)
        return distinct true
    """
    result = conn.query_read_only(
        query=query,
        parameters={
            "account_id": account_id,
            "term_id": term_id,
            "attr_name": attr_name,
        },
    )
    return True if len(result) > 0 else False


def get_generated_attr_snippets(account_id):
    query = """MATCH (attr:attribute{account_id:$account_id, status:"generated"})-[snippet:attr_of]->(:column|alias|sql|error_node{account_id:$account_id})
               MATCH (attr)-[:attr_of|reaching {sql_snippet_id: snippet.sql_snippet_id}]->(column:column{account_id:$account_id})
               RETURN distinct attr.id as attr_id, snippet.sql_snippet_id as snippet_id, coalesce(snippet.snippet_select_tc, snippet.snippet_select) as select,
               collect(distinct column.id) as reached_columns
            """
    result = pd.DataFrame(
        conn.query_read_only(query=query, parameters={"account_id": account_id})
    )
    return result.to_dict(orient="records")


def get_created_attributes(account_id):
    query = """MATCH(attr:attribute{account_id:$account_id})-[r:attr_of]->(:sql)
               RETURN distinct attr.id as attr_id, attr.status as status, attr.name as attr_name, 
               r.sql_snippet as snippet, r.sql_snippet_id as snippet_id
            """

    return conn.query_read_only(query=query, parameters={"account_id": account_id})


def populate_created_attributes(account_id, schemas, dialect, keep_string_values):
    logger.info("Update attributes.")
    logger.info("Update manually created attributes (parse sql snippets).")
    # get from graph_dal all created attributes and their snippets
    created_attrs = get_created_attributes(account_id)
    # for each attribute
    for attr in created_attrs:
        save_custom_snippet(
            account_id=account_id,
            sql_snippet=attr["snippet"],
            schemas=schemas,
            dialects=[dialect],
            attribute_id=attr["attr_id"],
            snippet_id=attr["snippet_id"],
            is_population=True,
        )


def update_generated_attributes(account_id):
    logger.info(
        'Update the definition sql and the "source" properties of automatically generated attribute snippets.'
    )
    generated_snippets = get_generated_attr_snippets(account_id)
    for snippet in generated_snippets:
        update_definition_sql(
            account_id,
            snippet["attr_id"],
            snippet["snippet_id"],
            snippet["reached_columns"],
            snippet["select"],
        )


def get_term_attr_zones(term_or_attribute_alias: str):
    return f"""
        CALL ({term_or_attribute_alias}){{
            CALL apoc.path.subgraphNodes({term_or_attribute_alias}, {{
                relationshipFilter: "term_of>|attr_of>|reaching>|<schema|<zone_of",
                labelFilter: "/zone",   
                minLevel: 0}})
            YIELD node as zone
            RETURN collect(zone{{.id,.name,.color}}) as zones, collect(zone.id) as zone_ids
        }}
        """


def get_analysis_zones(analysis_alias: str):
    return f"""
        CALL ({analysis_alias}){{
            CALL apoc.path.subgraphNodes({analysis_alias}, {{
                relationshipFilter: "analysis_of>|SQL>|<schema|<zone_of",
                labelFilter: "/zone",
                minLevel: 0}})
            YIELD node as z
            RETURN collect(distinct {{id:z.id, name:z.name, color:z.color}}) as zones, collect(distinct z.id) as zone_ids
        }}
        """


def save_attr(
    account_id: str,
    attr_name: str,
    attr_description: str,
    term_id: str,
    user_id: str,
    source_file_name: str | None = None,
    source_file_id: str | None = None,
    use_as_reference: bool = False,
    value: str | None = None,
):
    if source_file_name is not None:
        attr_name = _ensure_unique_attr_name(account_id, term_id, attr_name)
    set_clauses = [
        "attr.description = $attr_description",
        'attr.status = "created"',
        "attr.id = randomUUID()",
        "attr.type = $type",
        "attr.created_date = datetime.realtime()",
    ]

    set_clause = ", ".join(set_clauses)

    # Build query with optional edge to document node
    query = f""" MATCH (n:term{{account_id: $account_id, id: $term_id}})
                MERGE (n)-[:term_of]->(attr:attribute{{account_id: $account_id, name:$attr_name}})
                ON CREATE 
                SET {set_clause}
                WITH attr"""

    if source_file_id:
        query += """
                MATCH (doc:document {account_id: $account_id, id: $source_file_id})
                MERGE (attr)-[attr_of_file:attr_of_file]->(doc)"""
        if attr_description:
            query += """
                SET attr_of_file.description = $attr_description
                """
        if source_file_name is not None:
            query += """
                SET attr_of_file.source_file_name = $source_file_name
                """
        if source_file_id is not None:
            query += """
                SET attr_of_file.source_file_id = $source_file_id
                """
        if use_as_reference is not None:
            query += """
                SET attr_of_file.use_as_reference = $use_as_reference
                """
    query += """
                RETURN attr.id as id
            """
    res = conn.query_write(
        query=query,
        parameters={
            "attr_name": attr_name,
            "attr_description": attr_description,
            "account_id": account_id,
            "term_id": term_id,
            "label": Labels.ATTR,
            "type": label_to_type(Labels.ATTR),
            "source_file_name": source_file_name,
            "source_file_id": source_file_id,
            "use_as_reference": use_as_reference,
            "value": value,
        },
    )
    return res[0]


def connect_attribute_to_file(
    account_id: str,
    attr_id: str,
    description: str,
    source_file_name: str | None = None,
    source_file_id: str | None = None,
    use_as_reference: bool = False,
) -> None:
    """
    Connect an existing attribute to a document via attr_of_file edge.
    Does not modify any attribute/doc properties.
    """
    query = """
        MATCH (attr:attribute {account_id: $account_id, id: $attr_id})
        MATCH (doc:document {account_id: $account_id, id: $source_file_id})
        MERGE (attr)-[attr_of_file:attr_of_file]->(doc)
        SET attr_of_file.description = $description
        SET attr_of_file.source_file_name = $source_file_name
        SET attr_of_file.source_file_id = $source_file_id
        SET attr_of_file.use_as_reference = $use_as_reference
        RETURN attr.id as attr_id, doc.id as doc_id
    """
    conn.query_write(
        query=query,
        parameters={
            "account_id": account_id,
            "attr_id": attr_id,
            "description": description,
            "source_file_name": source_file_name,
            "source_file_id": source_file_id,
            "use_as_reference": use_as_reference,
        },
    )
