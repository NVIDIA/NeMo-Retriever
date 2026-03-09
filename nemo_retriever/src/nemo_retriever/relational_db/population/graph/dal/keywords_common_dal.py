from infra.Neo4jConnection import get_neo4j_conn
import logging

from shared.graph.model.reserved_words import (
    fields_relationships,
    Labels,
    DatasourcesRelationships,
)

conn = get_neo4j_conn()
logger = logging.getLogger("keywords_common_dal.py")


def get_node_fields(account_id, id, label, fields):
    fields_str = ", ".join([f"n.{f} as {f}" for f in fields])
    query = f"""
            MATCH (n:{label}{{account_id:$account_id, id:$id}})
            RETURN {fields_str}
            """
    result = conn.query_read_only(
        query, parameters={"id": id, "account_id": account_id}
    )
    if len(result) > 0:
        return result[0]
    else:
        return None


def get_num_of_dashboards_and_visuals(node, relationship):
    bi_relationships = "|<".join(fields_relationships + [DatasourcesRelationships.BI])
    bi_labels = "|>".join(Labels.LIST_OF_VISUALS + Labels.LIST_OF_DASHBOARDS)
    query = f""" 
            CALL ({node}){{
                CALL apoc.path.subgraphNodes({node}, {{
                    relationshipFilter: "{relationship}|reaching>",
                    labelFilter: "/column",
                    filterStartNode: true,
                    minLevel: 0
                }}) 
                YIELD node as c
                CALL apoc.path.subgraphNodes(c, {{
                    relationshipFilter: "<depends_on|<{bi_relationships}",
                    labelFilter: ">{bi_labels}",
                    minLevel: 0
                }})
                YIELD node
                WITH distinct node
                WITH apoc.map.groupByMulti(collect(node),'type') as types_map
                RETURN coalesce(size(types_map['sheet']),0) as num_of_visuals, 
                coalesce(size(types_map['dashboard']),0) + coalesce(size(types_map['report']),0) 
                as num_of_dashboards
            }}
            """
    return query
