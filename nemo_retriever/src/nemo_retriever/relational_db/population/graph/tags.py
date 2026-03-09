from shared.graph.dal.datasources_dal import get_parent_id_of_column_or_field
from shared.graph.model.reserved_words import Labels
from shared.graph.lineage.service import get_lineage
from shared.graph.search import (
    SearchObjectOptions,
    get_discovery_base,
    get_query_params,
)
from shared.graph.dal.usage_dal import get_usage_percentile_vars
from shared.graph.model.reserved_words import data_tree_labels
from itertools import groupby
from infra.Neo4jConnection import get_neo4j_conn

conn = get_neo4j_conn()

propagation_query_str = """
    CALL (r, item){
        OPTIONAL MATCH (item)-[:schema]->(col_propagated:column {account_id: $account_id})
        FOREACH (cp IN CASE WHEN col_propagated IS NOT NULL THEN [col_propagated] ELSE [] END |
            MERGE (r)-[:applies_to {propagated_from: item.id}]->(cp)
        )
        RETURN col_propagated
    }
"""


rule_query_str = """with collect(distinct n) as items
            MATCH (r:rule {account_id:$account_id, id: $rule_id})
            UNWIND items as item
            return r, item
            """


def get_param_string(filters):
    params_string = ""
    if "owner_ids" in filters:
        params_string += ", owners: $owners"
    if "tags" in filters:
        params_string += ", tags: $tags"
    if "data_types" in filters:
        params_string += ", data_types: $data_types"
    if "data" in filters:
        params_string += ", data_entities_ids: $data_entities_ids"
    if "usage" in filters:
        for percentile in get_usage_percentile_vars():
            params_string += f""", {percentile}: ${percentile}"""
    return params_string


def tag_rule_items(account, rule, items_query, filters, search_term):
    propagation_query = ""
    if rule["propagated"]:
        propagation_query = propagation_query_str
    rule_query = rule_query_str

    parameters = get_query_params(account, filters, search_term)
    parameters["rule_id"] = rule["id"]
    params_string = get_param_string(filters)

    query = f""" call 
                apoc.periodic.iterate('{items_query + rule_query}',
                "{propagation_query} MERGE (r)-[:applies_to]->(item)",
                    {{
                        batchSize:1000, 
                        params: {{account_id: $account_id, rule_id: $rule_id, sql_type: $sql_type, search_term: $search_term {params_string}}}
                    }}
                )
                yield batches, total return batches, total
            """

    res = conn.query_write(
        query=query,
        parameters=parameters,
    )

    update_query = """
        MATCH (r:rule{id: $rule_id, account_id: $account_id})
        SET r.affected_items_count = $affected_items_count
    """
    parameters["affected_items_count"] = res[0]["total"]

    conn.query_write(
        query=update_query,
        parameters=parameters,
    )

    return res


def tag_downstream_entities(account, downstream_entities_ids, rule_id, parent_id):
    downstream_entity_labels = "|".join(data_tree_labels)

    query = f"""
            UNWIND $entity_ids as entity_id
            MATCH  (item:{downstream_entity_labels}{{account_id:$account_id, id: entity_id}})
            MATCH  (rule:rule{{account_id:$account_id, id: $rule_id}})
            MERGE (rule)-[:applies_to{{downstream_of:$parent_id}}]->(item)
            return rule, collect(entity_id) as downstream_tagged_items
        """

    res = conn.query_write(
        query=query,
        parameters={
            "account_id": account,
            "entity_ids": downstream_entities_ids,
            "rule_id": rule_id,
            "parent_id": parent_id,
        },
    )
    return res


def flatten_lineage(lineage_tree: dict) -> list[str]:
    if (
        not lineage_tree
        or "downstream" not in lineage_tree
        or not len(lineage_tree["downstream"])
    ):
        return []
    ids = set()

    def get_downstream_ids(node: dict):
        ids.add(node["id"])
        [
            get_downstream_ids(n)
            for n in node["downstream"]
            if "downstream" in n and len(n["downstream"])
        ]

    get_downstream_ids(lineage_tree)
    return list(ids)


def tag_downstream(account, rule_id, rule_info):
    filters = rule_info["filters"]
    search_term = rule_info["search_term"]
    filters["objects"] = [
        SearchObjectOptions.ONLY_COLUMN,
        SearchObjectOptions.ONLY_TABLE,
    ]
    items_query, _ = get_discovery_base(search_term, rule_info["match"], filters)

    items_query += "return collect(distinct apoc.map.setKey(properties(n), 'label', labels(n)[0])) as items"

    parameters = get_query_params(account, filters, search_term)
    discovery_result = conn.query_read_only(
        query=items_query,
        parameters=parameters,
    )

    def key_func(k):
        return k["label"]

    for label, results in groupby(discovery_result[0]["items"], key_func):
        if label == Labels.TABLE:
            for tbl in list(results):
                table_downstream_items = []
                downstream_item_for_table = flatten_lineage(
                    get_lineage(account, tbl["id"], [Labels.TABLE], None, None, True)
                )
                table_downstream_items += downstream_item_for_table
                tag_downstream_entities(
                    account, table_downstream_items, rule_id, tbl["id"]
                )
        if label == Labels.COLUMN:
            for col in list(results):
                column_downstream_items = []
                parent_table_id = get_parent_id_of_column_or_field(account, col["id"])
                downstream_item_for_column = flatten_lineage(
                    get_lineage(
                        account,
                        parent_table_id,
                        [Labels.TABLE],
                        [col["id"]],
                        None,
                        True,
                    )
                )
                column_downstream_items += downstream_item_for_column
                tag_downstream_entities(
                    account, column_downstream_items, rule_id, col["id"]
                )
