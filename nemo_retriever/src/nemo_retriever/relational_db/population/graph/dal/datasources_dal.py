from shared.graph.model.reserved_words import data_relationships
from infra.Neo4jConnection import get_neo4j_conn

conn = get_neo4j_conn()


def get_parent_id_of_column_or_field(account_id, id):
    upstream_relationships = "|".join(data_relationships)

    query = f"""
        match(f:field | column {{id: $id, account_id: $account_id}})<-[:{upstream_relationships}]-(parent{{account_id: $account_id}})
        RETURN parent.id as id
        """
    result = conn.query_read_only(
        query=query,
        parameters={"account_id": account_id, "id": id},
    )
    if result[0]:
        return result[0]["id"]
    return None
