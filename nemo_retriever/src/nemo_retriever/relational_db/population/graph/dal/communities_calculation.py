import logging
from typing import Union
import pandas as pd
from infra.Neo4jConnection import get_neo4j_conn

logger = logging.getLogger("communities_calculation.py")


conn = get_neo4j_conn()


def calculate_data_communities(account_id):
    try:
        reset_data_communities(account_id)
        create_outlier_community(account_id)
        index = 0
        table_id = get_table_without_community(account_id)
        prev_table_id = None
        while table_id and table_id != prev_table_id:
            prev_table_id = table_id
            community_tables_ids = find_community_of_table(account_id, table_id)
            community_name = f"data-{index}"
            create_community(account_id, community_tables_ids, community_name)
            index = index + 1
            table_id = get_table_without_community(account_id)
        logger.info(f"{index} clusters created in data layer")
    except Exception as e:
        logger.info(f"data layer clustering went wrong, error: {e}")


def reset_data_communities(account_id: str):
    query = """
                MATCH (community:community{account_id:$account_id} where community.type="data_community")
                DETACH DELETE community
            """
    pd.DataFrame(conn.query_write(query, parameters={"account_id": account_id}))


def create_community(account_id: str, ids_in_community: list, community_name: str):
    query = """
                CREATE(community:community{account_id:$account_id, 
                                            id: randomUUID(), 
                                            name: $community_name, 
                                            created_date: datetime.realtime(),
                                            size: $ids_count, 
                                            type: "data_community" })
                WITH community
                MATCH (table:table where table.id in $ids)
                WITH table, community
                MERGE (community)-[:community_of]->(table)
                RETURN community, table
                """
    community = pd.DataFrame(
        conn.query_write(
            query,
            parameters={
                "account_id": account_id,
                "ids": ids_in_community,
                "ids_count": len(ids_in_community),
                "community_name": community_name,
            },
        )
    )
    return community


def create_outlier_community(account_id: str):
    find_outlier = """
        MATCH (table:table{account_id:$account_id})
        WHERE NOT EXISTS
        {
            MATCH p = (table)-[:join]-(other:table{account_id:$account_id})
            WHERE other<>table
        }
        RETURN collect(table.id) as tables
        """
    outlier_result = conn.query_read_only(
        find_outlier, parameters={"account_id": account_id}
    )

    if len(outlier_result[0]["tables"]) > 0:
        create_community(account_id, outlier_result[0]["tables"], "data-outlier")


def get_table_without_community(account_id: str) -> Union[str, None]:
    query = """
    MATCH (table:table{account_id:$account_id})
    WHERE NOT EXISTS ((table)<-[:community_of]-(:community{account_id:$account_id}))
    WITH table, (apoc.node.degree.in(table,"join") + apoc.node.degree.out(table,"join")) as degree
    WITH table ORDER BY degree DESC
    LIMIT 1
    RETURN table.id as table_id
    """
    result = conn.query_read_only(query, parameters={"account_id": account_id})
    if result:
        return result[0]["table_id"]
    return None


def find_community_of_table(account_id: str, table_id: str) -> list[str]:
    query = """
    MATCH (table:table{account_id:$account_id, id:$table_id})
    CALL apoc.path.subgraphNodes(table, {
        relationshipFilter: "join",
        labelFilter: ">table"
    })
    YIELD node as other
    WITH table, collect(distinct other.id) as other_tables_ids
    RETURN other_tables_ids+[$table_id] as community_tables_ids
    """
    result = conn.query_read_only(
        query, parameters={"account_id": account_id, "table_id": table_id}
    )
    return result[0]["community_tables_ids"]
