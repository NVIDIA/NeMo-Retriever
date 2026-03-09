import logging
from infra.Neo4jConnection import get_neo4j_conn
import numpy as np
from datetime import date, timedelta
from cachetools import cached, TTLCache
from shared.graph.model.reserved_words import SQLType, Labels, UsageDesc


conn = get_neo4j_conn()
logger = logging.getLogger("usage_dal.py")


QUERIES_USAGE_PERCENTILE = "queries_usage_percentile"
TABLES_USAGE_PERCENTILE = "tables_usage_percentile"
COLUMNS_USAGE_PERCENTILE = "columns_usage_percentile"

usage_entities = [
    Labels.BT,
    Labels.METRIC,
    Labels.ATTR,
    Labels.TABLE,
    Labels.COLUMN,
    Labels.ANALYSIS,
    Labels.SQL,
]


def get_usage_percentile_vars():
    usage_percentile_vars = []
    for usage_entity in usage_entities:
        usage_percentile_vars.append(f"{usage_entity}_percentile_25")
        usage_percentile_vars.append(f"{usage_entity}_percentile_75")
    return usage_percentile_vars


def get_stored_usage_percentiles(account_id: str, percentiles_type_name: str):
    query = f"""
                MATCH (n:db{{account_id:$account_id}})
                RETURN n.{f"{percentiles_type_name}_25"} as usage_percentile_25, n.{f"{percentiles_type_name}_75"} as usage_percentile_75
                """
    results = conn.query_read_only(
        query=query,
        parameters={
            "account_id": account_id,
        },
    )
    return results


def store_usage_percentiles(
    account_id: str,
    percentiles_type_name: str,
    usage_percentile_25: int,
    usage_percentile_75: int,
):
    query = """
            MATCH (n:db{account_id:$account_id})
            WITH n
            CALL apoc.create.setProperties(n, [$percentiles_type_name_25, $percentiles_type_name_75], [$usage_percentile_25, $usage_percentile_75]) 
            YIELD node
            RETURN n
            """
    conn.query_write(
        query=query,
        parameters={
            "account_id": account_id,
            "percentiles_type_name_25": f"{percentiles_type_name}_25",
            "percentiles_type_name_75": f"{percentiles_type_name}_75",
            "usage_percentile_25": usage_percentile_25,
            "usage_percentile_75": usage_percentile_75,
        },
    )


def get_usage_bin(usage_percentile_25, usage_percentile_75, usage_number):
    if usage_number == 0:
        return UsageDesc.UNUSED
    elif 0 < usage_number <= usage_percentile_25:
        return UsageDesc.LOW
    elif usage_percentile_25 < usage_number <= usage_percentile_75:
        return UsageDesc.MEDIUM
    else:
        return UsageDesc.HIGH


def get_usage_status_cypher(
    num_of_usage_alias: str,
    usage_alias: str,
    usage_25_param="$usage_percentile_25",
    usage_75_param="$usage_percentile_75",
):
    return f""" CASE 
    WHEN {num_of_usage_alias} > {usage_75_param} THEN "{UsageDesc.HIGH}"
    WHEN {num_of_usage_alias} > {usage_25_param} THEN "{UsageDesc.MEDIUM}"
    WHEN {num_of_usage_alias} > 0 THEN "{UsageDesc.LOW}"
    ELSE "{UsageDesc.UNUSED}" END AS {usage_alias}
    """


def get_usage_by_entity(entity, account_id):
    num_of_usage = entity["num_of_usage"]
    label = entity["label"]
    if num_of_usage == 0:
        return "unused"
    if label in [Labels.BT, Labels.METRIC, Labels.ATTR]:
        percentile_25, percentile_75 = get_node_usage_percentiles(account_id, label)
    elif label == Labels.TABLE:
        percentile_25, percentile_75 = get_tables_usage_percentiles(account_id)
    elif label == Labels.COLUMN:
        percentile_25, percentile_75 = get_columns_usage_percentiles(account_id)
    elif label == Labels.ANALYSIS or label == Labels.SQL:
        percentile_25, percentile_75 = get_usage_percentiles(account_id)
    else:
        raise Exception(f"usage not available for entities of label '{label}'")

    return get_usage_bin(percentile_25, percentile_75, num_of_usage)


@cached(cache=TTLCache(maxsize=4096, ttl=3600))
def get_node_usage_percentiles(account_id, node_type):
    query = f""" match(node:{node_type}{{account_id:$account_id}}) where coalesce(node.usage,0)>0 return collect(coalesce(node.usage, 0)) as usages """
    result_bt = conn.query_read_only(query=query, parameters={"account_id": account_id})
    usage = result_bt[0]["usages"]
    if len(usage) == 0:
        return 0, 0
    usage_percentile_25 = np.percentile(usage, 25)
    usage_percentile_75 = np.percentile(usage, 75)
    return usage_percentile_25, usage_percentile_75


def get_usage_filtering_str(val_str, usages, usage_percentile_25, usage_percentile_75):
    us = []
    for usage in usages:
        ubc_str = _get_usage_bin_condition(
            val_str, usage, usage_percentile_25, usage_percentile_75
        )
        us.append(f"({ubc_str})")
    return " or ".join(us)


def _get_usage_bin_condition(val_str, usage, usage_percentile_25, usage_percentile_75):
    if not usage:
        return f"{val_str}>=0"
    elif usage == UsageDesc.UNUSED:
        return f"{val_str}=0"
    elif usage == UsageDesc.LOW:
        return f"0<{val_str}<={usage_percentile_25}"
    elif usage == UsageDesc.MEDIUM:
        return f"{usage_percentile_25}<{val_str}<={usage_percentile_75}"
    elif usage == UsageDesc.HIGH:
        return f"{val_str}>{usage_percentile_75}"
    else:
        logger.error(f"Unknown usage: {usage}")
        return f"{val_str}>=0"


def get_count_str_by_month(alias: str):
    current = date.today().replace(day=1)
    count_3_month = []

    for i in range(0, 3):
        count_3_month.append(f"coalesce({alias}.cnt_{current.month}_{current.year}, 0)")
        prev = current - timedelta(days=1)
        current = prev.replace(day=1)

    count_str = "+".join(count_3_month)
    return count_str


def init_queries_usage_percentiles(account_id: str):
    count_string = get_count_str_by_month("n")
    query_all = f"""match(n:sql{{account_id: $account_id, is_sub_select:FALSE}})
                     return collect({count_string}) as usages """
    usages_result = conn.query_read_only(
        query=query_all, parameters={"account_id": account_id}
    )
    usages = usages_result[0]["usages"]
    if len(usages) == 0:
        return 0, 0
    usage_percentile_25 = np.percentile(usages, 25)
    usage_percentile_75 = np.percentile(usages, 75)
    store_usage_percentiles(
        account_id, QUERIES_USAGE_PERCENTILE, usage_percentile_25, usage_percentile_75
    )
    return usage_percentile_25, usage_percentile_75


@cached(cache=TTLCache(maxsize=4096, ttl=3600))
def get_usage_percentiles(account_id: str):
    stored_percentiles = get_stored_usage_percentiles(
        account_id, QUERIES_USAGE_PERCENTILE
    )
    if len(stored_percentiles) == 0 or (
        stored_percentiles[0]["usage_percentile_25"] is None
    ):
        usage_percentile_25, usage_percentile_75 = init_queries_usage_percentiles(
            account_id
        )
    else:
        usage_percentile_25 = stored_percentiles[0]["usage_percentile_25"]
        usage_percentile_75 = stored_percentiles[0]["usage_percentile_75"]

    return usage_percentile_25, usage_percentile_75


def get_queries_usage_percentiles(account_id, node_str="node"):
    count_str = get_count_str_by_month(node_str)
    usage_percentile_25, usage_percentile_75 = get_usage_percentiles(account_id)
    return usage_percentile_25, usage_percentile_75, count_str


def init_tables_usage_percentiles(account_id: str):
    cnt_str = get_count_str_by_month("query")
    query = f"""
                match(t:table{{account_id:$account_id}})
                CALL apoc.path.subgraphNodes(t, {{
                    relationshipFilter: "<SQL",
                    labelFilter: ">sql|-table",     
                    minLevel: 0}})
                YIELD node as query
                WHERE (not query.is_sub_select) and (query.sql_type = $sql_type)
                WITH t, reduce(sum = 0, u IN apoc.coll.toSet(collect([query.id, {cnt_str}])) | sum + u[1]) as num_of_usage
                RETURN collect(case when num_of_usage>0 then num_of_usage end) as usages
                """
    usages_result = conn.query_read_only(
        query=query,
        parameters={
            "account_id": account_id,
            "sql_type": SQLType.QUERY,
        },
    )
    usages = usages_result[0]["usages"]
    usage_percentile_25 = np.percentile(usages, 25) if len(usages) > 0 else 0
    usage_percentile_75 = np.percentile(usages, 75) if len(usages) > 0 else 0
    store_usage_percentiles(
        account_id, TABLES_USAGE_PERCENTILE, usage_percentile_25, usage_percentile_75
    )
    return usage_percentile_25, usage_percentile_75


@cached(cache=TTLCache(maxsize=4096, ttl=3600))
def get_tables_usage_percentiles(account_id):
    stored_percentiles = get_stored_usage_percentiles(
        account_id, TABLES_USAGE_PERCENTILE
    )
    if len(stored_percentiles) == 0 or (
        stored_percentiles[0]["usage_percentile_25"] is None
    ):
        usage_percentile_25, usage_percentile_75 = init_tables_usage_percentiles(
            account_id
        )
    else:
        usage_percentile_25 = stored_percentiles[0]["usage_percentile_25"]
        usage_percentile_75 = stored_percentiles[0]["usage_percentile_75"]

    return usage_percentile_25, usage_percentile_75


queries_for_columns_params = {
    "wildcard_names": ["Wildcard", "QualifiedWildcard"],
    "sql_subgraph_rel": "<SQL",
    "sql_subgraph_labels": ">sql|-table",
    "sql_type": SQLType.QUERY,
}
queries_for_columns_params_keys = ", ".join(
    [f"{key}:${key}" for key in queries_for_columns_params.keys()]
)


def get_queries_for_column(table_node, column_node, cnt_str):
    query = f"""
            //get wildcard queries on tables (select * from table T)
                CALL ({table_node}, {column_node}){{
                    CALL ({table_node}){{
                        OPTIONAL MATCH({table_node})<-[r:SQL]-(w:constant)
                        WHERE w.name in $wildcard_names
                        OPTIONAL MATCH(sql_node:sql{{is_sub_select:false, id:r.sql_id, account_id:$account_id, sql_type:$sql_type}})
                        RETURN collect(distinct sql_node) as wildcard_queries, 
                        collect(distinct sql_node.id) as wildcard_queries_ids,
                        SUM({cnt_str}) AS wildcard_usage
                    }}
                    //get direct queries on column
                    CALL ({column_node}){{
                        CALL apoc.path.subgraphNodes({column_node}, {{
                            relationshipFilter: $sql_subgraph_rel,
                            labelFilter: $sql_subgraph_labels,     
                            minLevel: 0}})
                        YIELD node as sql_node
                        //Do not include wildcards because we already queried them
                        WHERE (not sql_node.is_sub_select) 
                        AND (sql_node.sql_type = $sql_type)
                        RETURN collect(distinct sql_node) as direct_queries, 
                        collect(distinct sql_node.id) as direct_queries_ids, 
                        SUM({cnt_str}) AS direct_usage                    
                    }}
                    RETURN apoc.coll.toSet(direct_queries+wildcard_queries) as queries, 
                    apoc.coll.toSet(direct_queries_ids+wildcard_queries_ids) as queries_ids,
                    direct_usage + wildcard_usage as usage
                }}
            """
    return query


def calculate_queries_usage(
    usage_calc_string: str, query_alias: str, all_queries_array_name: str
):
    return f"""reduce(sum = 0, {query_alias} IN {all_queries_array_name} | sum + {usage_calc_string})"""


def init_columns_usage_percentiles(account_id: str):
    cnt_str = get_count_str_by_month("sql_node")
    query = """
            match(c:column{account_id:$account_id})<-[:schema]-(t:table)
            """
    query += get_queries_for_column("t", "c", cnt_str)
    query += """WITH t, c, """
    query += """ usage as num_of_usage
                RETURN collect(case when num_of_usage>0 then num_of_usage end) as usages
            """
    usages_result = conn.query_read_only(
        query=query,
        parameters={
            "account_id": account_id,
            "sql_type": SQLType.QUERY,
        }
        | queries_for_columns_params,
    )
    usages = usages_result[0]["usages"]
    usage_percentile_25 = np.percentile(usages, 25) if len(usages) > 0 else 0
    usage_percentile_75 = np.percentile(usages, 75) if len(usages) > 0 else 0
    store_usage_percentiles(
        account_id, COLUMNS_USAGE_PERCENTILE, usage_percentile_25, usage_percentile_75
    )
    return usage_percentile_25, usage_percentile_75


@cached(cache=TTLCache(maxsize=4096, ttl=3600))
def get_columns_usage_percentiles(account_id):
    stored_percentiles = get_stored_usage_percentiles(
        account_id, COLUMNS_USAGE_PERCENTILE
    )
    if len(stored_percentiles) == 0 or (
        stored_percentiles[0]["usage_percentile_25"] is None
    ):
        usage_percentile_25, usage_percentile_75 = init_columns_usage_percentiles(
            account_id
        )
    else:
        usage_percentile_25 = stored_percentiles[0]["usage_percentile_25"]
        usage_percentile_75 = stored_percentiles[0]["usage_percentile_75"]

    return usage_percentile_25, usage_percentile_75
