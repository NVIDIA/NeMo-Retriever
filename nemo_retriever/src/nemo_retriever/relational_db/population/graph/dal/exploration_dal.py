from infra.Neo4jConnection import get_neo4j_conn

conn = get_neo4j_conn()


def get_term_relationships(account_id: str, by_id: str = None):
    id_filter = ""
    if by_id is not None:
        id_filter = f"""{{id:'{by_id}'}} """
    terms_rels = f"""
        MATCH (target:term{id_filter})
        CALL apoc.path.subgraphNodes(target, {{
            relationshipFilter: "bt_join",
            labelFilter: ">term",
            minLevel: 0}})
        YIELD node as source
        WHERE source.id <> target.id
        RETURN collect({{source_id:source.id, target_id:target.id, rel_type:"bt_join", target_label: "term"}}) as relationships
        """
    return conn.query_read_only(query=terms_rels, parameters={"account_id": account_id})


def get_metric_relationships(account_id: str, by_id: str = None, term_id: str = None):
    id_filter = ""
    if by_id is not None:
        id_filter = f""", id:'{by_id}' """
    metrics_rels = f"""
        MATCH (target:metric{{account_id: $account_id{id_filter}}} WHERE coalesce(target.recommended, false) = false)
        MATCH (target)-[:metric_formula*]->(:attribute)<-[:term_of]-(term:term)
        CALL apoc.path.subgraphNodes(term, {{
            relationshipFilter: "bt_join",
            labelFilter: ">term",
            filterStartNode: true,
            minLevel: 0}})
        YIELD node as source
        {f"WHERE source.id = '{term_id}'" if term_id else ""}
        WITH distinct source, target
        RETURN collect(distinct {{source_id:source.id, target_id:target.id, rel_type:"metric_formula", target_label: "metric"}}) as relationships
        """
    return conn.query_read_only(
        query=metrics_rels, parameters={"account_id": account_id}
    )


def get_analysis_relationships(account_id: str, by_id: str = None, term_id: str = None):
    id_filter = ""
    if by_id is not None:
        id_filter = f""", id:'{by_id}' """
    analyses_rels = f"""
        MATCH (target:analysis{{account_id: $account_id{id_filter}}} WHERE coalesce(target.recommended, false) = false)
        MATCH (target)-[:analysis_of]->(query:sql{{account_id: $account_id}})-[:SQL]->(column:column{{account_id: $account_id}})
        OPTIONAL MATCH(column)<-[:attr_of|reaching]-(attribute:attribute{{account_id: $account_id}})<-[:term_of]-(term:term{{account_id: $account_id}})
        CALL apoc.path.subgraphNodes(term, {{
            relationshipFilter: "bt_join",
            labelFilter: ">term",
            filterStartNode: true,
            minLevel: 0}})
        YIELD node as source
        {f"WHERE source.id = '{term_id}'" if term_id else ""}
        WITH distinct source, target
        RETURN collect(distinct{{source_id:source.id, target_id:target.id, rel_type:"analysis_of", target_label: "analysis"}}) as relationships
    """
    return conn.query_read_only(
        query=analyses_rels, parameters={"account_id": account_id}
    )
