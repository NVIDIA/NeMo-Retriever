import logging
import pandas as pd
from datetime import datetime
import json

from infra.Neo4jConnection import get_neo4j_conn
from infra.PostgresConnection import get_postgres_conn
from shared.graph.model.reserved_words import (
    Labels,
    label_to_type,
    SQLFunctions,
    DatasourcesRelationships as Rels,
)
from shared.graph.model.snippet import Snippet

from itertools import product
import re
import uuid

conn = get_neo4j_conn()
pg = get_postgres_conn()
logger = logging.getLogger("metrics_dal.py")


class SnippetError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def get_metric_snippets_combinations(
    account_id: str,
    metric_formula: str,
    metric_id: str,
    metric_name: str,
    population: bool = False,
) -> pd.DataFrame:
    terms_attrs_names = get_terms_attributes_metrics_tuples_from_formula(metric_formula)
    if terms_attrs_names:
        terms_attributes = validate_term_and_attribute_relation(
            account_id, terms_attrs_names
        )

    attributes_ids = [term_attr["attr"]["id"] for term_attr in terms_attributes]
    snippets_map: dict[str, Snippet] = {}
    for term_attr in terms_attributes:
        attr = term_attr["attr"]
        term = term_attr["term"]
        snippets_of_attribute = get_snippets_of_attr(
            account_id,
            attr["id"],
            attr["name"],
            term["name"],
            metric_id,
            metric_name,
            population,
        )
        snippets_map = snippets_map | snippets_of_attribute

    snippets_df = pd.DataFrame(columns=attributes_ids)
    if len(terms_attrs_names) == 1:
        only_attr_id = terms_attributes[0]["attr"]["id"]
        snippets_df[only_attr_id] = snippets_map.values()
    else:
        snippets_ids = list(snippets_map.keys())
        attributes_snippets_combinations = get_snippets_combinations(
            account_id, attributes_ids, terms_attrs_names, metric_id, snippets_ids
        )
        snippets_arrays: list[dict[str, Snippet]] = []
        [
            create_combinations_array(combination, snippets_arrays, snippets_map)
            for combination in attributes_snippets_combinations
        ]
        snippets_df = pd.DataFrame(snippets_arrays, columns=attributes_ids)

    return snippets_df


def create_combinations_array(
    combination: dict[str, list[str]],  ## { attributeId: [snippet1,snippet2,snippet3] }
    snippets_arrays: list[dict[str, Snippet]],
    snippets_map: dict[str, Snippet],
):
    ## create a cartesian product of the attributes and their snippets (=all possible combinations)
    comibations_matrix = list(product(combination.values()))
    snippet_to_attr: dict[str, Snippet] = {}
    for snippet in comibations_matrix:
        snippet_id = snippet[0][0]
        snippet_obj = snippets_map[snippet_id]
        attr = snippet_obj.attribute_id
        snippet_to_attr[attr] = snippet_obj
    snippets_arrays.append(snippet_to_attr)


def get_terms_attributes_metrics_tuples_from_formula(
    formula: str,
) -> list[tuple[str, str]]:
    terms_attributes_names_tuples = []
    term_attr_pattern = r"\[([a-zA-Z0-9 ]+\.[a-zA-Z0-9%\-_ ]+)\]"
    try:
        matches = re.findall(term_attr_pattern, formula)
        if matches:
            for match in matches:
                try:
                    parts = str(match).split(".")
                    term = parts[0]
                    attr = parts[1]
                    terms_attributes_names_tuples.append([term, attr])
                except Exception:
                    raise SnippetError(
                        "Terms and Attributes must be in the following format: [Term name.Attribute name]"
                    )
        else:
            raise SnippetError("formula must contain at least one Attribute")
    except Exception:
        raise SnippetError("formula must contain at least one Attribute")
    return terms_attributes_names_tuples


def add_to_ambiguous_table(account_id: str, metric_id: str, metric_name: str):
    snapshot_month = datetime.now().month
    snapshot_year = datetime.now().year

    delete_query = """
        DELETE FROM public.ambiguous_entities_snapshots
        WHERE account_id = %(account_id)s AND entity_id = %(metric_id)s AND entity_type = %(entity_type)s AND snapshot_month = %(snapshot_month)s AND snapshot_year = %(snapshot_year)s;
    """

    insert_query = """
        INSERT INTO public.ambiguous_entities_snapshots
        (account_id, entity_id, entity_name, entity_type, snapshot_month, snapshot_year)
        VALUES %;
    """

    insert_template = "(%(account_id)s, %(metric_id)s, %(metric_name)s, %(entity_type)s, %(snapshot_month)s, %(snapshot_year)s)"
    try:
        pg.delete_rows(
            delete_query,
            {
                "account_id": account_id,
                "metric_id": metric_id,
                "snapshot_month": snapshot_month,
                "snapshot_year": snapshot_year,
                "entity_type": Labels.METRIC,
            },
        )

        pg.execute_values(
            insert_query,
            {
                "account_id": account_id,
                "metric_id": metric_id,
                "metric_name": metric_name,
                "entity_type": Labels.METRIC,
                "snapshot_month": snapshot_month,
                "snapshot_year": snapshot_year,
            },
            insert_template,
        )
    except Exception as e:
        logger.error(f"Failed to update ambiguous_entities_snapshots: {e}")


## this function receives an array of tuples of names of terms and attributes (from a formula),
def validate_term_and_attribute_relation(
    account_id: str, terms_attrs_names: list[tuple[str, str]]
) -> dict[str, dict[str, str]]:
    terms_attributes: list[dict[str, str]] = []
    for term_attr in terms_attrs_names:
        query = """
        OPTIONAL MATCH (term:term{account_id:$account_id, name: $term_name})-[:term_of]->(attr:attribute{account_id:$account_id, name: $attribute_name})
        RETURN attr{.id,.name} as attr, term{.id,.name} as term
        """
        result: list[dict] = conn.query_read_only(
            query,
            parameters={
                "account_id": account_id,
                "term_name": term_attr[0],
                "attribute_name": term_attr[1],
            },
        )[0]
        terms_attributes.append(result)

    term_atts_not_found = []
    for index, term_attr in enumerate(terms_attributes):
        if term_attr["attr"] is None:
            term_attr_name = ".".join(terms_attrs_names[index])
            term_atts_not_found.append(term_attr_name)
    if len(term_atts_not_found) > 0:
        invalid_combinations = ",".join(term_atts_not_found)
        raise SnippetError(
            f"The following Terms and Attributes are misspelled or not connected: {invalid_combinations}"
        )
    return terms_attributes


## this function receives an array of tuples of names of terms and attributes (from a formula),
def validate_metrics_names_and_field_id(
    account_id: str, field_id: str, metrics_names: list[str]
) -> dict[str, dict[str, str]]:
    query = """
    UNWIND $metrics_names as metric_name
    MATCH(metric:metric{account_id:$account_id, id: $metric_name})
    MATCH(metric)-[:metric_field]->(metric_field:field{account_id:$account_id})
    MATCH(metric_field)-[:CTE]->(metric_field_cte:cte{account_id:$account_id})
    MATCH(field:field{account_id:$account_id, id: $field_id})-[:CTE]->(cte:cte{account_id:$account_id})
    MATCH(cte)-[:cte_depends_on]->(metric_field_cte)
    MATCH(metric)-[:metric_sql]->(attr:attribute{account_id:$account_id})<-[:term_of]-(term:term{account_id:$account_id})
    RETURN collect(distinct { attr: attr{.id,.name}, term: term{.id,.name} }) as terms_attributes
    """
    result: list[dict] = conn.query_read_only(
        query,
        parameters={
            "account_id": account_id,
            "field_id": field_id,
            "metrics_names": metrics_names,
        },
    )
    return result[0]["terms_attributes"]


def get_snippets_of_attr(
    account_id: str,
    attribute_id: str,
    attribute_name: str,
    term_name: str,
    metric_id: str,
    metric_name: str,
    population: bool,
) -> dict[str, Snippet]:
    query = """
    MATCH(attribute:attribute{account_id:$account_id, id:$attribute_id})-[snippet:attr_of]->(item:column|alias|sql)
    CALL (attribute, snippet){
        MATCH(attribute)-[:attr_of|reaching{sql_snippet_id:snippet.sql_snippet_id}]->(column:column)<-[:schema]-(table:table)
        WHERE coalesce(column.deleted,False)=False
        CALL (table){
            CALL apoc.path.subgraphNodes(table, { relationshipFilter: '<schema|<zone_of', labelFilter: '/zone' })
            YIELD node 
            WITH collect(node) as related_zones
            WITH CASE WHEN size(related_zones)=0 THEN [{name: 'no zones', id: 'no zones'}] ELSE related_zones END as zones
            UNWIND zones as zone
            RETURN zone{ .name, .id} as zone_name_id
        }
        RETURN zone_name_id, collect(distinct table{ .id, .name}) as tables_of_snippet_names_ids
    }
    CALL(attribute,snippet){
        UNWIND snippet.from_source_ids as source_id
        MATCH(table:table{id:source_id,account_id:$account_id})
        RETURN collect(distinct table{ .id, .name}) as source_tables_of_snippet_name_ids
    }
    WITH collect(distinct {zone_name: zone_name_id.name, zone_id: zone_name_id.id, snippet_id:snippet.sql_snippet_id, snippet_props:properties(snippet),
    data_item: {data_type: snippet.data_type, label: labels(item)[0], id: item.id}, tables_of_snippet_names_ids: tables_of_snippet_names_ids, 
                source_tables_of_snippet_name_ids: source_tables_of_snippet_name_ids }) as zone_to_snippets
    WITH apoc.map.groupByMulti(zone_to_snippets, 'zone_id') as zones_to_data_map
    RETURN zones_to_data_map
    """
    zones_to_data_map: dict[str, list[dict[str, str]]] = conn.query_read_only(
        query,
        parameters={"account_id": account_id, "attribute_id": attribute_id},
    )[0]["zones_to_data_map"]

    snippets_map: dict[str, Snippet] = {}
    for zone, snippets_per_zone in zones_to_data_map.items():
        for snippet in snippets_per_zone:
            if len(snippets_per_zone) == 1:
                data_item = snippet["data_item"]
                snippet_id = snippet["snippet_id"]
                snippets_map[snippet_id] = Snippet(
                    id=snippet_id,
                    account_id=account_id,
                    attribute_id=attribute_id,
                    attribute_name=attribute_name,
                    term_name=term_name,
                    data_item_id=data_item["id"],
                    data_item_label=data_item["label"],
                    data_item_data_type=data_item["data_type"],
                    props=snippet["snippet_props"],
                    tables_names_ids=snippet["tables_of_snippet_names_ids"],
                    source_table_name_ids=snippet["source_tables_of_snippet_name_ids"],
                )

    if len(snippets_map.keys()) == 0:
        if population:
            add_to_ambiguous_table(account_id, metric_id, metric_name)
        raise SnippetError(
            f"The Attribute '[{term_name}.{attribute_name}]' doesn't have a unique definition in any Zone."
        )
    return snippets_map


def get_snippets_of_bi_fields(
    account_id: str, fields_ids: list[str], snippet_id: str
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    query = """
    MATCH(field:field{account_id:$account_id} WHERE field.id in $fields_ids)-[:depends_on]->(ancestor_field:field{account_id:$account_id})
    CALL apoc.when(
        EXISTS((ancestor_field)<-[:reaching_field]-(:attribute)),
        'MATCH(ancestor_field)<-[attr_snippet:reaching_field]-(attribute:attribute{account_id:$account_id})
        RETURN attribute, attr_snippet' ,
        'MATCH(ancestor_field)<-[:metric_field]-(:metric)-[metric_snippet:metric_sql]->(attribute:attribute{account_id:$account_id})
        MATCH(attribute)-[attr_snippet:attr_of]->(:sql)
        WHERE attr_snippet.sql_snippet_id in metric_snippet.snippets_ids
        RETURN attribute, attr_snippet',
        {ancestor_field:ancestor_field, account_id:$account_id}
        )
    YIELD value
    WITH distinct value.attribute as attribute, value.attr_snippet as attr_snippet
    RETURN collect(distinct attr_snippet.sql_snippet_id) as snippets_ids, collect(distinct attribute.id) as attributes_ids
    """
    field_snippets: dict[str, list[dict[str, str]]] = conn.query_read_only(
        query,
        parameters={
            "account_id": account_id,
            "fields_ids": fields_ids,
            "snippet_id": snippet_id,
        },
    )[0]

    snippets_ids = field_snippets["snippets_ids"]
    attributes_ids = field_snippets["attributes_ids"]

    return snippets_ids, attributes_ids


def group_tables_by_communities(account_id: str, tables_ids: list[str]):
    query = """
    MATCH(table:table{account_id:$account_id} WHERE table.id in $tables_ids)<-[:community_of]-(community:community{account_id:$account_id})
    WHERE NOT community.name IN ['data-outlier','semantic-outlier']
    with community.id as community_id, collect(distinct table.id) as tables_ids
    return collect(tables_ids) as tables_communities
    """
    tables_communities = conn.query_read_only(
        query,
        parameters={"account_id": account_id, "tables_ids": tables_ids},
    )[0]["tables_communities"]

    query = """
    MATCH(table:table{account_id:$account_id} WHERE table.id in $tables_ids)<-[:community_of]-(community:community{account_id:$account_id})
    WHERE community.name IN ['data-outlier','semantic-outlier']
    return collect([table.id]) as tables_outliers
    """
    tables_outliers = conn.query_read_only(
        query,
        parameters={"account_id": account_id, "tables_ids": tables_ids},
    )[0]["tables_outliers"]

    tables_communities.extend(tables_outliers)
    return tables_communities


# a single combination is a grouping of snippets that could form a valid sql
# (they are all connected to joined tables/the same table)
# the structure is a matrix of dictionaries of attributes ids to their snippets ids
# a row in the matrix is a single valid combination
# for example:
# snippets 1,2,3,4 are connected to table1 and table2 -> so we can compose sqls "from table1 join table2"
#    [ { attrId1: [snippetId1, snippetId2], attrId2: [snippetId3, snippetId4] } ]
# snippets 6,7,8 are connected to table3 -> so we can compose sqls "from table3"
#    [ { attrId1: [snippetId6, snippetId7], attrId2: [snippetId8] } ] // from table3
def get_snippets_combinations(
    account_id: str,
    attributes_ids: list[str],
    term_attr_names: list[tuple[str, str]],
    metric_id: str,
    valid_snippets_ids: list[str],
) -> list[dict[str, list[str]]]:
    query = """
    MATCH(attr:attribute{account_id:$account_id} WHERE attr.id in $attributes_ids)
    MATCH(attr)-[snippet:attr_of|reaching WHERE snippet.sql_snippet_id in $snippets_ids]->(column:column)<-[:schema]-(table:table)
    WHERE coalesce(column.deleted,False)=False // don't allow deleted data
    CALL (attr){
        OPTIONAL MATCH p=(attr)-[r:reaching_field WHERE r.sql_snippet_id in $snippets_ids]->(:field)
        RETURN p IS NOT NULL AS has_fields
    }
    WITH attr, snippet, table WHERE not has_fields 
    WITH collect(distinct {attr_id:attr.id,snippet_id:snippet.sql_snippet_id, table_id:table.id}) as combinations
    RETURN combinations
    """
    combinations = pd.DataFrame(
        conn.query_read_only(
            query,
            parameters={
                "account_id": account_id,
                "attributes_ids": attributes_ids,
                "snippets_ids": valid_snippets_ids,
            },
        )[0]["combinations"],
        columns=["attr_id", "snippet_id", "table_id"],
    )
    tables_ids = list(set(combinations["table_id"]))
    tables_communities: list[list[str]] = []
    if len(tables_ids) > 1:
        tables_communities = group_tables_by_communities(account_id, tables_ids)
    else:
        tables_communities = [[tables_ids[0]]]

    # a valid combination is a combination from which we can compose an sql.
    # the condiitons for composing a valid sql:
    # 1. if there is more than 1 table, the tables must be joined (they are in the same community)
    # 2. every attribute in the metric should have a snippet to the table/s
    valid_snippets_combinations: list[dict[str, list[str]]] = []
    all_combinations: list[dict[str, list[str]]] = [
        find_community_combinations(community, combinations)
        for community in tables_communities
    ]

    def is_combination_connected_to_all_atts(combination: pd.DataFrame):
        return all(att_id in list(combination.keys()) for att_id in attributes_ids)

    valid_snippets_combinations = [
        combination
        for combination in all_combinations
        if is_combination_connected_to_all_atts(combination)
    ]

    if len(valid_snippets_combinations) == 0:
        terms_attrs_names_strings = [
            f"{term_attr[0]}.{term_attr[1]}" for term_attr in term_attr_names
        ]
        raise SnippetError(
            f"No joins could be found between the unique snippets of {', '.join(terms_attrs_names_strings)}."
        )
    return valid_snippets_combinations


def find_community_combinations(community: list[str], combinations: pd.DataFrame):
    atts_snippets_map: dict[str, list[str]] = {}
    [
        find_attributes_of_table(table_id, combinations, atts_snippets_map)
        for table_id in community
    ]
    return atts_snippets_map


def find_attributes_of_table(
    table_id: str,
    combinations: pd.DataFrame,
    atts_snippets_map: dict[str, list[str]],
):
    table_combinations = combinations[combinations["table_id"] == table_id]
    attributes_of_table = table_combinations["attr_id"]
    attributes_of_table.apply(
        lambda att_id: map_attributes_to_snippets(
            att_id, table_combinations, atts_snippets_map
        )
    )
    return atts_snippets_map


def map_attributes_to_snippets(
    attr_id: str, combinations: pd.DataFrame, atts_snippets_map: dict[str, list[str]]
):
    snippets_of_att = combinations[combinations["attr_id"] == attr_id]["snippet_id"]
    atts_snippets_map.setdefault(attr_id, []).extend(list(snippets_of_att))
    return atts_snippets_map


def get_child_metric_id(metric_name: str, account_id: str, field_id: str) -> str:
    query = """
    MATCH(field:field{id:$field_id, account_id: $account_id})-[:CTE]->(field_cte:cte{account_id: $account_id})
    MATCH(field_cte)-[:cte_depends_on]->(ancestor_cte:cte{account_id: $account_id})
    MATCH(ancestor_cte)<-[:CTE]-(ancestor_field:field)<-[:metric_field]-(n:metric{account_id: $account_id, name: $metric_name})
    WHERE EXISTS((field)-[:depends_on*]->(ancestor_field))
    RETURN n.id as metric_id """
    metric_result = conn.query_read_only(
        query=query,
        parameters={
            "account_id": account_id,
            "metric_name": metric_name,
            "field_id": field_id,
        },
    )
    if not len(metric_result):
        raise SnippetError(
            f"Couldn't find a valid BI Metric {metric_name} to connect to field {field_id}"
        )
    return metric_result[0]["metric_id"]


def get_child_metrics(account_id: str, field_id: str) -> str:
    query = """
    MATCH(field:field{id:$field_id, account_id: $account_id})-[:CTE]->(field_cte:cte{account_id: $account_id})
    MATCH(field_cte)-[:cte_depends_on]->(ancestor_cte:cte{account_id: $account_id})
    MATCH(ancestor_cte)<-[:CTE]-(ancestor_field:field)<-[:metric_field]-(metric:metric{account_id: $account_id})
    RETURN collect(distinct metric{.id,.name,.formula}) as metrics """
    metric_result = conn.query_read_only(
        query=query,
        parameters={
            "account_id": account_id,
            "field_id": field_id,
        },
    )
    if not metric_result:
        return []
    return metric_result[0]["metrics"]


def add_metric_to_graph(account_id, metric_obj):
    ident_props = {
        "name": str(metric_obj.name),
        "account_id": str(metric_obj.account_id),
        "formula": str(metric_obj.formula),
    }
    description = (
        metric_obj.props["description"] if "description" in metric_obj.props else None
    )
    metric_obj.props["created_date"] = datetime.now()
    metric_obj.props["type"] = label_to_type(Labels.METRIC)
    query = """
    call apoc.merge.node.eager([$metric_label], $ident_props, $on_create, $on_match) 
    yield node 
    return node.id as id, exists((node)-[:metric_formula]->()) as metric_already_exists"""
    result = conn.query_write(
        query=query,
        parameters={
            "account_id": account_id,
            "metric_label": metric_obj.metric_node.label,
            "ident_props": ident_props,
            "on_create": metric_obj.props,
            "on_match": {"description": description},
        },
    )
    metric_id = result[0]["id"]
    metric_already_exists = result[0]["metric_already_exists"]

    if metric_obj.formula and not metric_already_exists:
        edges = metric_obj.get_edges().copy()
        edges_data = [_prepare_edge(edge, account_id) for edge in edges]
        _add_edges(account_id, edges_data)
    return metric_id


def _prepare_edge(edge, account_id):
    ident_prop_key = "name" if edge[0].get_label() == Labels.METRIC else "id"
    ident_prop_val = (
        edge[0].props["name"]
        if edge[0].get_label() == Labels.METRIC
        else str(edge[0].get_id())
    )
    if len(edge) == 3 and edge[2] is not None:
        edge_props = edge[2]
    else:
        edge_props = {}
    edge = {
        "v1_label": edge[0].get_label(),
        "v1_props": {ident_prop_key: ident_prop_val, "account_id": account_id},
        "v1_on_create": edge[0].get_properties(),
        "v2_label": edge[1].get_label(),
        "v2_props": {"id": str(edge[1].get_id()), "account_id": account_id},
        "v2_on_create": edge[1].get_properties(),
        "edge_props": edge_props,
    }
    return edge


def _add_edges(account_id, edges_data):
    """
    If the nodes do not exist in the Neo4j graph, the function adds them.
    Add to the Neo4j graph the given edge.
    :param edge: edge is a tuple of the form (from_node, to_node, edge_properties)
    :return:
    """
    query = """
            UNWIND $edges_data as data
            call apoc.merge.node.eager([data.v1_label], data.v1_props, data.v1_on_create, {})
            yield node as v1 
            call apoc.merge.node.eager([data.v2_label], data.v2_props, data.v2_on_create, {})
            yield node as v2
            call apoc.merge.relationship.eager(v1, "metric_formula", {}, {}, v2)
            YIELD rel  
            SET rel = data.edge_props          
            RETURN DISTINCT 'true'       
            """
    conn.query_write(
        query=query, parameters={"edges_data": edges_data, "account_id": account_id}
    )


def delete_metric_subgraph(account_id: str, id: str):
    query = """ MATCH (n:metric {id:$id, account_id:$account_id})
                CALL apoc.path.subgraphNodes(n, {
                    relationshipFilter: "metric_formula>",
                    labelFilter: "-metric|-attribute",     
                    minLevel: 0})
                YIELD node
                where not node:metric and not node:attribute
                DETACH DELETE node
            """
    conn.query_write(query=query, parameters={"id": id, "account_id": account_id})

    query = """ MATCH (n:metric {id:$id, account_id:$account_id})-[r:metric_sql]->() DELETE r """
    conn.query_write(query=query, parameters={"id": id, "account_id": account_id})


def get_metric_sqls(account_id: str, id: str) -> list[dict[str, str]]:
    query = """ MATCH (n:metric {id:$id, account_id:$account_id})-[r:metric_sql]->() 
                RETURN collect(distinct r{.sql, .definition_sql}) as sqls """
    return conn.query_read_only(
        query=query, parameters={"id": id, "account_id": account_id}
    )[0]["sqls"]


def detach_metric_queries(account_id: str, id: str, sqls_to_keep: list[str]):
    query = """ 
    MATCH (n:metric {id:$id, account_id:$account_id})-[r:identical|subset]->()
    WHERE not coalesce(r.sql,'') in $sqls
    DELETE r """
    conn.query_write(
        query=query,
        parameters={"id": id, "account_id": account_id, "sqls": sqls_to_keep},
    )


def set_deleted_property_to_true(account_id: str, id: str):
    query = """ MATCH (n:metric {id:$id, account_id:$account_id})
                SET n.invalid = true
                SET n.invalid_time = datetime.realtime()
                SET n.certified_formula = false
            """
    conn.query_write(query=query, parameters={"id": id, "account_id": account_id})


def delete_metric(account_id: str, ids: list):
    query = """ MATCH (n:metric{account_id:$account_id}) WHERE n.id in $ids
                CALL apoc.path.subgraphNodes(n, {
                    relationshipFilter: "metric_formula>|metric_sql>",
                    labelFilter: "-metric|-attribute",     
                    minLevel: 0})
                YIELD node
                where not node:metric and not node:attribute
                DETACH DELETE node
                DETACH DELETE n
                RETURN distinct True as delete_succuss
            """
    result = conn.query_write(
        query=query, parameters={"ids": ids, "account_id": account_id}
    )

    if len(result) == 0:
        conn.query_write(
            """ MATCH (n:metric{account_id:$account_id}) WHERE n.id in $ids DETACH DELETE n""",
            parameters={"ids": ids, "account_id": account_id},
        )


def delete_all_metrics(account_id: str, recommended=True):
    query = """ MATCH (n:metric {account_id:$account_id, recommended: $recommended}) RETURN collect(n.id) as ids"""
    result = conn.query_write(
        query=query, parameters={"account_id": account_id, "recommended": recommended}
    )
    if len(result) > 0 and len(result[0]["ids"]) > 0:
        delete_metric(account_id, result)


def recommended_to_real(account_id, id, name, description, user_id, owner_id):
    query = """MATCH (m:metric{id:$id, account_id:$account_id})
                SET m.name = $name 
                SET m.clean_name = $name
                SET m.description = $description
                SET m.recommended=False
                SET m.created_date = datetime.realtime()
                SET m.created_by = $user_id
              """
    if owner_id is not None:
        query += """set m.owner_id=$owner_id"""
    conn.query_write(
        query=query,
        parameters={
            "id": id,
            "account_id": account_id,
            "name": name,
            "description": description,
            "user_id": user_id,
            "owner_id": owner_id,
        },
    )


def update_metric_node(
    account_id: str,
    id: str,
    user_id: str,
    name: dict,
    description: dict,
    owner_id: dict,
    formula: dict,
    owner_notes: str,
    tags: list,
):
    update_query = "MATCH (n:metric {id:$metric_id, account_id:$account_id}) "
    if name:
        if "value" in name:
            update_query += """
            SET n.name=$name
            SET n.certified_name=False
            REMOVE n.description_suggestion
            """
        else:
            update_query += """
            SET n.certified_name=$certified_name 
            SET n.last_certified=datetime.realtime()
            SET n.last_certified_by=$user_id 
            """
    if description:
        if "value" in description:
            if description["value"]:
                update_query += """SET n.description=$description """
            else:
                update_query += """REMOVE n.description """
            update_query += """SET n.certified_description=False """
        else:
            update_query += """
            SET n.certified_description=$certified_description
            SET n.last_certified=datetime.realtime()
            SET n.last_certified_by=$user_id
            """
    if formula:
        if "value" in formula:
            if formula["value"]:
                update_query += """SET n.formula=$formula """
            else:
                update_query += """REMOVE n.formula """
            update_query += """
            SET n.certified_formula=False
            REMOVE n.description_suggestion
            """
        else:
            update_query += """
            SET n.certified_formula=$certified_formula
            SET n.last_certified=datetime.realtime() 
            SET n.last_certified_by=$user_id 
            """
    if tags is not None:
        update_query += """
        WITH n
        CALL (n){
            OPTIONAL MATCH(n)<-[r:tag_of]-(t:tag{account_id:$account_id})
            where not t.id in $tags
            DELETE r
        }
        CALL (n){
            CALL apoc.do.when(size($tags)>0,
            'UNWIND $tags as t_id
            MATCH(t:tag{id: t_id})
            MERGE(n)<-[tr:tag_of]-(t)
            ON CREATE SET tr.tagged_by=$user_id, tr.tagged_date=datetime.realtime()',
            '',
            {n:n,tags:$tags,user_id:$user_id}
            )
            YIELD value
        }
        """
    if owner_id is not None:
        if owner_id != "":
            update_query += """SET n.owner_id=$owner_id """
        else:
            update_query += """REMOVE n.owner_id """
    elif owner_notes is not None:
        if owner_notes != "":
            update_query += """SET n.owner_notes=$owner_notes """
        else:
            update_query += """REMOVE n.owner_notes """

    update_query += """
    SET n.last_modified = datetime.realtime()
    SET n.last_modified_by = $user_id

    RETURN n.id as id
    """
    name_value = name["value"] if name and "value" in name else None
    certified_name = name["certified"] if name and "certified" in name else None
    description_value = (
        description["value"] if description and "value" in description else None
    )
    certified_description = (
        description["certified"] if description and "certified" in description else None
    )
    formula_value = formula["value"] if formula and "value" in formula else None
    certified_formula = (
        formula["certified"] if formula and "certified" in formula else None
    )

    result = conn.query_write(
        query=update_query,
        parameters={
            "account_id": account_id,
            "metric_id": id,
            "user_id": user_id,
            "tags": tags,
            "name": name_value,
            "certified_name": certified_name,
            "description": description_value,
            "certified_description": certified_description,
            "formula": formula_value,
            "certified_formula": certified_formula,
            "owner_notes": owner_notes,
            "owner_id": owner_id,
        },
    )

    return result[0]


def get_metrics_with_props(account_id: str, recommended: bool):
    query = """MATCH (n:metric {account_id:$account_id})
                WHERE coalesce(n.recommended,False)=$recommended
                CALL (n){
                    MATCH(n)-[metric_sql:metric_sql]->(:attribute)
                    RETURN collect(distinct metric_sql.sql) as sqls
               }
               return n.id as id, n.name as name, coalesce(n.invalid, false) as invalid, n.description as description, 
               n.formula as formula, n.sql as sql, n.owner_id as owner_id, sqls,
               coalesce(n.certified_name, false) as certified_name, 
               coalesce(n.certified_description, false) as certified_description, 
               coalesce(n.certified_formula, false) as certified_formula,
               n.last_modified as last_modified_date, n.last_modified_by as last_modified_by,
               n.last_certified as last_certified_date, n.last_certified_by as last_certified_by,
               n.created_date as created_date, n.created_by as created_by,
               n.owner_notes as owner_notes, coalesce(n.recommended, false) as recommended """
    return conn.query_read_only(
        query=query, parameters={"account_id": account_id, "recommended": recommended}
    )


def get_metric_node_dict(account_id: str, id: str):
    query_metric = """MATCH (n:metric {id:$id, account_id:$account_id}) 
                           return n.id as id, n.name as name, n.description as description, 
                           n.formula as formula, n.sql as sql, n.owner_id as owner_id, 
                           coalesce(n.certified_name, false) as certified_name, 
                           coalesce(n.certified_description, false) as certified_description, 
                           coalesce(n.certified_formula, false) as certified_formula, 
                           n.owner_notes as owner_notes,
                           n.last_modified as last_modified_date, n.last_modified_by as last_modified_by,
                           n.last_certified as last_certified_date, n.last_certified_by as last_certified_by,
                           n.created_date as created_date, n.created_by as created_by """
    metric_dict = conn.query_read_only(
        query=query_metric, parameters={"id": id, "account_id": account_id}
    )[0]
    return metric_dict


def connect_metric_sql_edges(
    account_id: str,
    metric_id: str,
    sql_edge_props: dict[str, str | Snippet],
    attributes_ids: list[str],
) -> str:
    sql = sql_edge_props["sql"]
    definition_sql = (
        None
        if "definition_sql" not in sql_edge_props
        else sql_edge_props["definition_sql"]
    )
    snippets_ids: list[Snippet] = sql_edge_props["snippets"]
    sql_snippet_id = str(uuid.uuid4())
    query = """
                MATCH (m:metric{id:$metric_id, account_id: $account_id}) 
                MATCH (attr:attribute{account_id: $account_id})
                WHERE attr.id in $attributes_ids
                MERGE(m)-[metric_sql:metric_sql{sql:$sql}]->(attr)
                ON CREATE SET metric_sql.sql_snippet_id=$sql_snippet_id, metric_sql.definition_sql=$definition_sql, 
                    metric_sql.snippets_ids=$snippets_ids
                RETURN True
            """

    conn.query_write(
        query=query,
        parameters={
            "account_id": account_id,
            "metric_id": metric_id,
            "sql": sql,
            "definition_sql": definition_sql,
            "attributes_ids": attributes_ids,
            "snippets_ids": snippets_ids,
            "sql_snippet_id": sql_snippet_id,
        },
    )
    return sql_snippet_id


def connect_metric_to_bi_fields(
    account_id: str, metric_id: str, fields_ids: list[str]
) -> str:
    connect_to_metric = """
                MATCH (m:metric{id:$metric_id, account_id: $account_id}) 
                MATCH (field:field{account_id: $account_id} WHERE field.id in $fields_ids)
                MERGE(m)-[:metric_field]->(field)
                RETURN True
            """

    conn.query_write(
        query=connect_to_metric,
        parameters={
            "account_id": account_id,
            "metric_id": metric_id,
            "fields_ids": fields_ids,
        },
    )


def get_name_by_id(category, id, account_id):
    if category == Labels.TABLE:
        query = 'match (n: table {account_id: $account_id, id: $id})<-[:schema]-(s:schema{account_id: $account_id}) return s.name + "." + n.name as name'
    else:
        query = f"match (n: {category} {{account_id: $account_id, id: $id}}) return n.name as name"
    q_result = conn.query_read_only(
        query=query, parameters={"account_id": account_id, "id": id}
    )
    return q_result[0]["name"]


def add_edges_to_queries(
    metric_id: str,
    nodes_to_connect: list[str],
    edge_label: str,
    account_id: str,
    sql: str,
):
    query = f"""
            unwind $nodes_to_connect as node_id
            match (s:sql{{id:node_id, account_id: $account_id}}) 
            match(k:metric{{id:$metric_id}})
            merge (k)-[r:{edge_label}]->(s)
            SET r.sql=$sql
            return s,k"""
    conn.query_write(
        query=query,
        parameters={
            "nodes_to_connect": nodes_to_connect,
            "account_id": account_id,
            "metric_id": metric_id,
            "sql": sql,
        },
    )


def add_usage_to_metric(account_id, metric_id) -> int:
    query = """
    MATCH(metric:metric{account_id:$account_id, id:$metric_id})
    OPTIONAL MATCH(metric)-[:subset|identical]->(sql:sql)
    WITH metric, coalesce(sql.usage,0) as query_usage, sql
    WITH metric, SUM(query_usage) as metric_usage, COUNT(distinct sql) as num_of_queries
    SET metric.usage=metric_usage, metric.num_of_queries=num_of_queries
    RETURN metric.usage as usage
    """
    return conn.query_write(
        query=query, parameters={"metric_id": metric_id, "account_id": account_id}
    )[0]["usage"]


def find_metric_filters(metric_id, account_id):
    query = """MATCH (m:metric{id:$metric_id, account_id:$account_id}) 
                call (m){
                    CALL apoc.path.subgraphNodes(m, {
                                relationshipFilter: "metric_formula>",
                                labelFilter: "/attribute",
                                minLevel: 0})
                    YIELD node as attr
                    MATCH (bt:term)-[:term_of]->(attr)
                    return attr.id as filter1, attr
                }
                
                WITH filter1, attr
                call (attr){
                    CALL apoc.path.subgraphNodes(attr, {
                            relationshipFilter: "attr_of>|SQL>",
                            labelFilter: "/column",
                            minLevel: 0})
                    YIELD node as col
                    WHERE coalesce(col.deleted, false) = false 
                    MATCH (col)<-[:attr_of]-(at:attribute)<-[:term_of]-(bt:term)
                    return at.id as filter2
                }
                
                return apoc.coll.toSet(collect(filter1)+collect(filter2)) as filter
            
            """

    all_filters = pd.DataFrame(
        conn.query_read_only(
            query=query, parameters={"metric_id": metric_id, "account_id": account_id}
        )
    )
    if "filter" in all_filters and all_filters is not None and len(all_filters) > 0:
        return all_filters["filter"][0]
    return []


def add_common_dimensions_to_metric(metric_id, nodes_for_dimensions, account_id):
    query = """UNWIND $nodes_for_dimensions as node_id
           MATCH (s:sql{id:node_id,account_id:$account_id})-[:SQL]->(:command{name:"Select"})-[:SQL]->(gb:command{name:"Group_by"})
           -[:SQL]->(col:column)<-[:attr_of|reaching]-(at:attribute)-[:term_of]-(bt:term)
           WHERE coalesce(col.deleted, false) = false 
           return bt.name as term_name, bt.id as term_id, at.name as attr_name, at.id as attr_id"""
    all_dimensions = pd.DataFrame(
        conn.query_read_only(
            query=query,
            parameters={
                "nodes_for_dimensions": nodes_for_dimensions,
                "account_id": account_id,
            },
        )
    )
    if (
        "term_name" in all_dimensions
        and all_dimensions is not None
        and len(all_dimensions) > 0
    ):
        query = "match (k:metric{id:$metric_id,account_id:$account_id}) set k.common_dimensions = $all_dimensions"
        conn.query_write(
            query=query,
            parameters={
                "metric_id": metric_id,
                "account_id": account_id,
                "all_dimensions": json.dumps(
                    all_dimensions.drop_duplicates().to_dict("records")
                ),
            },
        )


def add_common_filters_to_metric(metric_id, nodes_for_filters, account_id):
    query_where = """
                MATCH (where_node:command{name:"Where", account_id:$account_id})
                WITH collect(where_node) AS endNodes

                UNWIND $nodes_for_filters as node_id
                MATCH (s:sql{id:node_id,account_id:$account_id})
                CALL apoc.path.subgraphNodes(s, {
                                relationshipFilter: "SQL>",
                                labelFilter: ">command",
                                minLevel: 0,
                                endNodes: endNodes})
                YIELD node as wh
    
                CALL apoc.path.subgraphNodes(wh, {
                            relationshipFilter: "SQL>|<attr_of|<reaching",
                            labelFilter: "/attribute",
                            minLevel: 0})
                YIELD node as k
                
                MATCH (bt:term)-[:term_of]->(k)
                return bt.name as term_name, bt.id as term_id, k.name as attr_name, k.id as attr_id
               """
    all_filters_where = pd.DataFrame(
        conn.query_read_only(
            query=query_where,
            parameters={
                "nodes_for_filters": nodes_for_filters,
                "account_id": account_id,
            },
        )
    )

    query_join = """
                MATCH (join_node:command{name:"Joins", account_id:$account_id})
                WITH collect(join_node) AS endNodes

                UNWIND $nodes_for_filters as node_id
                MATCH (s:sql{id:node_id,account_id:$account_id})
                CALL apoc.path.subgraphNodes(s, {
                                relationshipFilter: "SQL>",
                                labelFilter: ">command",
                                minLevel: 0,
                                endNodes: endNodes})
                YIELD node as join
                
                CALL apoc.path.subgraphNodes(join, {
                            relationshipFilter: "SQL>",
                            labelFilter: ">operator",
                            minLevel: 0})
                YIELD node as op
                
                MATCH (op)-[:SQL]->(:constant)
                CALL apoc.path.subgraphNodes(op, {
                            relationshipFilter: "SQL>|<attr_of",
                            labelFilter: "/attribute",
                            minLevel: 0})
                
                YIELD node as k
                
                MATCH (bt:term)-[:term_of]->(k)
                return bt.name as term_name, bt.id as term_id, k.name as attr_name, k.id as attr_id
                
                """
    all_filters_join = pd.DataFrame(
        conn.query_read_only(
            query=query_join,
            parameters={
                "nodes_for_filters": nodes_for_filters,
                "account_id": account_id,
            },
        )
    )

    all_filters = pd.concat([all_filters_where, all_filters_join])
    metric_filters = find_metric_filters(metric_id, account_id)

    if "term_name" in all_filters and len(all_filters) > 0:
        # query_filters = list(set(all_filters['filter']))
        relevant_filters = get_relevant_filters(
            metric_filters, all_filters
        )  # list(set(query_filters) - set(metric_filters))
        if len(relevant_filters) > 0:
            query = "match (k:metric{id:$metric_id,account_id:$account_id}) set k.common_filters = $relevant_filters"
            conn.query_write(
                query=query,
                parameters={
                    "metric_id": metric_id,
                    "account_id": account_id,
                    "relevant_filters": json.dumps(
                        relevant_filters.drop_duplicates().to_dict("records")
                    ),
                },
            )


def get_relevant_filters(metric_filter, all_filters):
    return all_filters[~all_filters["attr_id"].isin(metric_filter)]


def name_constructor(node):
    if node.label == Labels.COLUMN:
        return f"{node.match_props['schema_name']}.{node.match_props['table_name']}.{node.match_props['name']}"
    if node.label == Labels.TABLE:
        return f"{node.match_props['schema_name']}.{node.match_props['name']}"
    return node.props["name"]


def filter_attrs_with_same_name(row):
    def select_min_length_row(group):
        # choose row with the minimal number of tables,shortest 'from' and 'select', hopefully only one table will remain in list with appropriate column and table names
        # If there are several rows with dufferent columns and one table - leave them, further it will be filtered in reccomendations check
        min_length = (
            group["from_id"].apply(len)
            + group["from"].str.split().apply(len)
            + group["select"].str.split().apply(len)
        ).min()
        return group[
            (
                group["from_id"].apply(len)
                + group["from"].str.split().apply(len)
                + group["select"].str.split().apply(len)
            )
            == min_length
        ]

    # If we get several attributes with the same name and id -> this is an attribute on aliases with different 'from' and 'select',
    # In addition to aliases, there should be attribute on column with single 'select' and 'from' that was found for kpi recommendation.
    # Find the row with single from/select and drop others
    df = pd.DataFrame(row)
    result = df.groupby("attr").apply(select_min_length_row).reset_index(drop=True)
    return result.to_dict(orient="records")


def retrieve_metrics_subgraphs(account_id, metric_id):
    query = """   
            MATCH (m:metric{account_id:$account_id, id:$metric_id})
            CALL apoc.path.subgraphAll(m,{ relationshipFilter: "-metric_sql|metric_formula> "})
            YIELD nodes ,relationships
            RETURN m.name as name, m.id as id, nodes,
            [p IN relationships | [coalesce(properties(p).child_idx,coalesce(properties(p).case_part,0)),p]] AS edges,
            [n IN nodes | apoc.map.setKey(properties(n), "label", labels(n)[0])] as nodes_props
            """
    return pd.DataFrame(
        conn.query_read_only(
            query=query, parameters={"account_id": account_id, "metric_id": metric_id}
        )
    )


def get_values_list(node_id, account_id):
    query = """ MATCH (n:sql{account_id: $account_id, id: $node_id})
                CALL apoc.path.subgraphNodes(n,{
                labelFilter: ">constant",
                relationshipFilter: "SQL>",
                filterStartNode: true,
                minLevel: 0})
                YIELD node
                with n,node 
                MATCH (c)-[]->(node) 
                return n.id as id, collect(node.name) as values, collect(c.name) as operators

    """
    result = conn.query_read_only(
        query, parameters={"account_id": account_id, "node_id": node_id}
    )
    if len(result) > 0:
        result = result[0]
        result["ops_values"] = [
            (x, y) for x, y in zip(result["operators"], result["values"])
        ]
        return result["ops_values"]
    return []


def get_operators(account_id):
    query = """
            MATCH (n:sql{account_id: $account_id})
            CALL apoc.path.subgraphNodes(n,{
                labelFilter: ">operator|>function",
                relationshipFilter: "SQL>",
                filterStartNode: true,
                minLevel: 0})
            YIELD node
            where not toLower(node.name) in ["case"]
            with collect (distinct toLower(node.name)) as distinct_operators, collect(toLower(node.name)) as operators, n           
            RETURN n.id as node_id, n.sql_id as sql_id, operators as operators_list
            """
    roots = pd.DataFrame(
        conn.query_read_only(query=query, parameters={"account_id": account_id})
    )
    roots["operators_list"] = roots["operators_list"].apply(
        lambda x: [
            (
                y
                if y
                not in set(SQLFunctions.all_funcs.keys()).difference(
                    set(SQLFunctions.agg_funcs)
                )
                else "to_delete"
            )
            for y in x
        ]
    )
    return roots


def get_queries_for_metric_usage(
    account_id, metric_id: str, sql: str, tables_ids: list[str]
) -> list[str]:
    ## we are looking for the queries to compare to the metric's sql for usage calculations
    ## as an optimization we take only the queries that use the same tables
    ## and we filter out queries that were already compared
    query = """
                MATCH(metric:metric{account_id:$account_id,id:$metric_id})
                CALL (metric){
                    OPTIONAL MATCH(metic)-[r:identical|subset]->(metric_query:sql)
                    WHERE r.sql=$sql
                    RETURN collect(metric_query) as metric_existing_queries
                }
                MATCH(table:table{account_id: $account_id})
                WHERE coalesce(table.deleted, false) = false
                AND table.id in $tables_ids
                MATCH(sql:sql{account_id: $account_id,is_sub_select:False})-[:SQL*1..2]->(table)
                WHERE not sql in metric_existing_queries
                WITH sql, collect(table.id) as sql_tables_ids
                WHERE all(table_id in $tables_ids WHERE table_id in sql_tables_ids)
                RETURN collect(sql.id) as sqls_ids
                """

    sqls_ids = conn.query_read_only(
        query=query,
        parameters={
            "account_id": account_id,
            "tables_ids": tables_ids,
            "metric_id": metric_id,
            "sql": sql,
        },
    )[0]["sqls_ids"]
    return sqls_ids


def delete_query_by_id(account_id, query_id):
    query = """
                MATCH(s: sql{id:$query_id, account_id:$account_id})
                CALL
                apoc.path.subgraphNodes(s, {
                relationshipFilter: "SQL>|source_of>",
                minLevel: 0})
                YIELD
                node
                WHERE NOT (node:table or node:column)
                DETACH DELETE node
                DETACH DELETE s
            """
    conn.query_write(
        query=query, parameters={"account_id": account_id, "query_id": query_id}
    )
    logger.info(f"Removed sql id: {query_id}")


def get_metric_id_by_formula(account_id, formula):
    query = """MATCH(m:metric{account_id:$account_id,formula:$metric_formula}) 
               RETURN m.id as metric_id"""
    return conn.query_read_only(
        query=query, parameters={"account_id": account_id, "metric_formula": formula}
    )


def connect_metric_to_possibly_misused(account_id, most_similar, metric_id):
    query = """MATCH(n{account_id:$account_id}) WHERE id(n) in $most_similar 
               MATCH (m:metric{account_id:$account_id, id:$metric_id})
               MERGE (m)-[:possibly_misused]->(n)
            """
    return conn.query_write(
        query=query,
        parameters={
            "account_id": account_id,
            "most_similar": most_similar,
            "metric_id": metric_id,
        },
    )


def get_new_possibly_misused(account_id, most_similar, metric_id):
    query = """MATCH(m:metric{account_id:$account_id, id:$metric_id})-[:possibly_misused]->(n)
               RETURN collect(id(n)) as ids
            """
    result = conn.query_read_only(
        query=query,
        parameters={
            "account_id": account_id,
            "most_similar": most_similar,
            "metric_id": metric_id,
        },
    )
    if len(result) > 0:
        checked_ids = result[0]["ids"]
        most_similar = list(set(most_similar).difference(checked_ids))
    return most_similar


def get_reached_columns(account_id, id):
    query = """
                MATCH (metric:metric{account_id:$account_id,id:$id})-[metric_snippet:metric_sql]->(attribute:attribute{account_id:$account_id}) 
                MATCH (attribute)-[attr_snippet:attr_of|reaching WHERE attr_snippet.sql_snippet_id in metric_snippet.snippets_ids]->(column:column{account_id:$account_id})<-[:schema]-(table:table{account_id:$account_id})
                RETURN collect(distinct {id:column.id, name:column.name, table_id: table.id, table_name:column.table_name, schema_name: column.schema_name}) as reached_columns
            """
    result = conn.query_read_only(
        query, parameters={"id": id, "account_id": account_id}
    )
    if len(result) > 0:
        return result[0]["reached_columns"]
    else:
        return []


def get_metric_fields_by_snippets(account_id, id):
    query = f""" 
    MATCH (metric:metric{{account_id:$account_id,id:$id}})-[metric_field:metric_field]->(field:field{{account_id:$account_id}})
    MATCH(field)<-[:{Rels.EMBEDDED_FIELD}|{Rels.PUBLISHED_FIELD}]-(parent:{Labels.PUBLISHED_DS}|{Labels.EMBEDDED_DS}{{account_id:$account_id}})
    CALL (parent) {{
        CALL apoc.when(
            parent:{Labels.EMBEDDED_DS},
            "MATCH (parent)<-[:bi_rel]-(workbook:workbook)<-[:bi_rel*]-(project:project{{account_id:$account_id}})<-[:bi_rel]-(site:site{{account_id:$account_id}})
            WITH parent, site, workbook, collect(distinct project) as projects
            WITH distinct parent, workbook, site, projects limit 1
            RETURN [site]+projects+[workbook]+[parent] as path_to_site, site{{.name,.id}} as site",
            "MATCH (parent)<-[:bi_rel*]-(project:project{{account_id:$account_id}})<-[:bi_rel]-(site:site{{account_id:$account_id}})
            WITH parent, site, collect(distinct project) as projects
            WITH distinct parent, site, projects limit 1
            RETURN [site]+projects+[parent] as path_to_site, site{{.name,.id}} as site",
            {{parent:parent,account_id:$account_id}}
        )
        YIELD value
        WITH value.path_to_site AS path_to_site, value.site AS site
        WITH collect(distinct path_to_site)[0] as path_to_site, site
        RETURN apoc.text.join([p in path_to_site | p.name]," \u2022 ") as breadcrumbs, site
    }}
    RETURN collect(distinct field{{ .id,.name,.formula, datasource_name: parent.name, datasource_id: parent.id, datasource_type: parent.type, breadcrumbs,
    site:site.name }}) as fields, collect(distinct site) as sites
    """
    result = conn.query_read_only(
        query, parameters={"id": id, "account_id": account_id}
    )
    if len(result) > 0:
        return result[0]["fields"], result[0]["sites"]
    else:
        return [], []


def get_metrics_related_zones_cypher(metric_alias: str = "n"):
    return f""" CALL ({metric_alias}){{
                MATCH({metric_alias})-[metric_snippet:metric_sql]->(attribute:attribute)
                MATCH(attribute)-[attr_snippet:attr_of|reaching WHERE attr_snippet.sql_snippet_id in metric_snippet.snippets_ids]->(column:column)
                CALL apoc.path.subgraphNodes(column, {{
                    relationshipFilter: "<schema|<zone_of",
                    labelFilter: "/zone",   
                    minLevel: 0}})
                YIELD node as zone
                RETURN collect(distinct zone{{.id,.name,.color}}) as zones, collect(distinct zone.id) as zone_ids
            }}
            """
