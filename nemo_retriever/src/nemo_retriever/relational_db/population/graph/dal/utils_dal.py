import logging

from shared.graph.model.query import Query, NestedPropertyError
from shared.graph.model.node import Node
from shared.graph.model.reserved_words import (
    Props,
    Labels,
    label_to_type,
    entities_without_owners,
    data_relationships,
    BiConnectors,
)
from infra.Neo4jConnection import get_neo4j_conn
from notifications.certification_dal import get_certification
from notifications.queries_dal import get_num_of_queries_by_label
from notifications.types import EntitiesFamilies
from notifications.utils import get_entity_family

logger = logging.getLogger("utils_dal.py")
conn = get_neo4j_conn()


def entity_exists_in_graph_insensitive(account_id: str, name: str, label: Labels):
    name = name.strip()
    query = f"""MATCH(n:{label}{{account_id:$account_id}})
            WHERE toLower(n.name) = toLower($name)
            RETURN n 
            """
    result = conn.query_read_only(
        query=query, parameters={"name": name, "account_id": account_id}
    )
    return True if len(result) > 0 else False


def is_flat_dict(properties: dict):
    for key, value in properties.items():
        if isinstance(value, list):
            if value:
                if any(
                    isinstance(item, list) or isinstance(item, dict) for item in value
                ):
                    raise NestedPropertyError(
                        f"Invalid property name: {key}\nThe property value: {value}"
                    )
        if isinstance(value, dict):
            raise NestedPropertyError(
                f"Invalid property name: {key}\nThe property value: {value}"
            )


def check_properties_compatibility_with_neo4j(
    node_from: Node, node_to: Node, edge_props: dict
):
    is_flat_dict(node_from.get_properties())
    if node_from.get_override_existing_props():
        is_flat_dict(node_from.get_override_existing_props())
    is_flat_dict(node_to.get_properties())
    if node_to.get_override_existing_props():
        is_flat_dict(node_to.get_override_existing_props())
    is_flat_dict(edge_props)


def prepare_edge(edge, account_id, provider=None):
    node_from = edge[0].get_sql_node() if isinstance(edge[0], Query) else edge[0]
    node_to = edge[1].get_sql_node() if isinstance(edge[1], Query) else edge[1]

    e_label = _get_edge_label(edge)
    v1_label, v1_identity_props, v1_on_create_props, v1_on_match_props = prepare_node(
        node_from, account_id, provider
    )
    v2_label, v2_identity_props, v2_on_create_props, v2_on_match_props = prepare_node(
        node_to, account_id, provider
    )
    edge_props = edge[2].copy()
    edge_props.update({"account_id": account_id})

    check_properties_compatibility_with_neo4j(node_from, node_to, edge[2])

    if e_label == Props.JOIN:
        edge_identity_props = {Props.JOIN: edge_props[Props.JOIN]}
    elif "child_idx" in edge_props and edge_props["child_idx"] is not None:
        edge_identity_props = {"child_idx": edge_props["child_idx"]}
    else:
        edge_identity_props = {}

    return {
        "v1_label": v1_label,
        "v1_identity_props": v1_identity_props,
        "v1_on_create_props": v1_on_create_props,
        "v1_on_match_props": v1_on_match_props,
        "v2_label": v2_label,
        "v2_identity_props": v2_identity_props,
        "v2_on_create_props": v2_on_create_props,
        "v2_on_match_props": v2_on_match_props,
        "edge_props": edge_props,
        "edge_label": e_label,
        "edge_identity_props": edge_identity_props,
    }


def _get_edge_label(edge):
    if Props.JOIN in edge[2]:
        return "join"
    elif Props.SOURCE_SQL_ID in edge[2]:
        return "source_of"
    elif Props.UNION in edge[2]:
        return "union"
    elif Props.SQL_ID in edge[2]:
        return "SQL"
    else:
        return next(iter(edge[2]))


def prepare_node(node: Node, account_id, provider=None):
    label = node.get_label()
    props = node.get_properties()

    if "operators" in props:
        props.pop("operators")

    if provider == BiConnectors.QUICKSIGHT and label not in [
        Labels.COLUMN,
        Labels.TABLE,
    ]:
        if label == Labels.QS:
            identity_props = {"name": props["name"], "account_id": account_id}
        else:
            identity_props = {"qs_id": str(props["qs_id"]), "account_id": account_id}
    elif (provider == BiConnectors.SISENSE and label == Labels.SISENSE) or (
        provider == BiConnectors.LOOKER and label == Labels.LOOKER
    ):
        identity_props = {"name": props["name"], "account_id": account_id}
    else:
        identity_props = node.get_match_props() | {"account_id": account_id}
    on_create_props = props.copy()
    on_create_props.update({"account_id": account_id})
    if "type" not in on_create_props:
        on_create_props.update({"type": label_to_type(on_create_props["label"])})
    is_bi_insertion = provider in BiConnectors.ALL
    if is_bi_insertion:
        override_props = on_create_props.copy()
        override_props.pop("id")
        if "created" in override_props and (
            label in [Labels.PROJECT, Labels.QS, Labels.SISENSE, Labels.LOOKER]
        ):
            override_props.pop("created")
        if node.get_override_existing_props():
            override_props = node.get_override_existing_props()
    elif node.get_override_existing_props():
        override_props = node.get_override_existing_props()
    else:
        override_props = {}
    return [label], identity_props, on_create_props, override_props


def add_edges(account_id, edges_data):
    """
    If the nodes do not exist in the Neo4j graph, the function adds them.
    Add to the Neo4j graph the given edge.
    :param edge: edge is a tuple of the form (from_node, to_node, edge_properties)
    :return:
    """
    query = """
            unwind $edges_data as data
            call apoc.merge.node.eager(data.v1_label, data.v1_identity_props, data.v1_on_create_props, data.v1_on_match_props)
            yield node as v1
            call apoc.merge.node.eager(data.v2_label, data.v2_identity_props, data.v2_on_create_props, data.v2_on_match_props)
            yield node as v2
            with v1, v2, data, data.edge_identity_props as e_identity_props
            call apoc.merge.relationship.eager(v1, data.edge_label, e_identity_props, {}, v2)
            YIELD rel
            with rel, case when not rel.source_sql_id is null and rel.join_sql_id is null then {source_sql_id: apoc.coll.toSet(rel.source_sql_id + data.edge_props.source_sql_id)} 
            when rel.source_sql_id is null and not rel.join_sql_id is null then {join_sql_id: apoc.coll.toSet(rel.join_sql_id + data.edge_props.join_sql_id), join: rel.join}
            else data.edge_props end as props 
            SET rel = props
            RETURN DISTINCT 'true'    
            """
    conn.query_write(
        query=query, parameters={"account_id": account_id, "edges_data": edges_data}
    )


def get_node_properties_by_id(account_id, id, label: str | list[str]):
    if isinstance(label, list):
        label_filter = "|".join(label)
    else:
        label_filter = label
    query = f"""
        MATCH(n:{label_filter}{{account_id:$account_id, id:$id}})
        RETURN apoc.map.setKey(properties(n),"label", labels(n)[0]) as props
    """

    props = conn.query_read_only(query, parameters={"id": id, "account_id": account_id})
    if len(props) == 0:
        return None
    else:
        remove_embeddings_keys(props[0]["props"])
        return props[0]["props"]


def get_node_properties_and_tags_by_id(account_id, id, label: str | list[str]):
    if isinstance(label, list):
        label_filter = "|".join(label)
    else:
        label_filter = label
    query = (
        """MATCH(n:"""
        + label_filter
        + """{account_id:$account_id, id:$id})
        """
        + get_tags_query_by_node_alias("n")
        + """
        WITH apoc.map.setKey(properties(n),"label", labels(n)[0]) as props, value.tags_ids as tags
        RETURN distinct apoc.map.setKey(props, "tags", tags) as props
        """
    )

    props = conn.query_read_only(query, parameters={"id": id, "account_id": account_id})
    if len(props) == 0:
        return None
    else:
        return props[0]["props"]


def get_tags_query_by_node_alias(node_name: str):
    return f"""
            CALL ({node_name}){{
                optional match ({node_name})<-[:tag_of]-(direct_tag:tag)
                optional match ({node_name})<-[:applies_to]-(rule:rule)-[:rule_of]->(rule_tag:tag)
                optional match ({node_name})<-[:applies_to]-(rule:rule)-[:rule_of]->(dual_tag:tag)-[tag_of:tag_of]->({node_name})

                with direct_tag, dual_tag, tag_of,
                CASE WHEN rule_tag IS NOT NULL 
                THEN apoc.map.merge(rule_tag, {{rule_id: rule_tag.id}}) 
                ELSE NULL END as rule_tag

                with direct_tag, rule_tag, tag_of,
                CASE WHEN dual_tag IS NOT NULL
                THEN apoc.map.merge(dual_tag, {{manually_tagged_by: tag_of.tagged_by, rule_id: rule_tag.id, dual: true}}) 
                ELSE NULL END as dual_tag

                with
                collect(distinct rule_tag) as rule_tags,
                collect(distinct direct_tag) as direct_tags,
                collect(distinct dual_tag) as dual_tags,
                collect(distinct rule_tag.id) as rule_tags_ids,
                collect(distinct direct_tag.id) as direct_tags_ids,
                collect(distinct dual_tag.id) as dual_tags_ids

                with
                dual_tags, rule_tags_ids, direct_tags_ids, dual_tags_ids,
                [rt in rule_tags WHERE NOT rt.id IN dual_tags_ids] as filtered_rule_tags,
                [dt in direct_tags WHERE NOT dt.id IN dual_tags_ids] as filtered_direct_tags
                
                with apoc.coll.toSet(dual_tags + filtered_rule_tags + filtered_direct_tags) as tags,
                apoc.coll.toSet(dual_tags_ids + rule_tags_ids + direct_tags_ids) as tags_ids
                with [tag IN tags WHERE tag IS NOT NULL] as filtered_tags, [tag IN tags_ids WHERE tag IS NOT NULL] as filtered_tags_ids

                return {{tags: filtered_tags, tags_ids: filtered_tags_ids}} as value
            }}
        """


def get_node_parent_owner_by_id(account_id, node_id, label: str = None):
    concatenated_relationships = "|".join(data_relationships)
    query = (
        """MATCH(n:"""
        + label
        + """{account_id:$account_id, id:$id})
           CALL apoc.case([
            n:attribute,
            'match(n)<-[:term_of]-(bt:term) 
            return bt.owner_id as owner_id',

            n:column,
            'match(n)<-[:schema]-(t:table)
            return t.owner_id as owner_id',

            n:field,
            'MATCH (n)<-[r:"""
        + concatenated_relationships
        + """]-(parent)
            
            with collect(parent.owner_id) as owner_ids
            return case when size(owner_ids) > 0 then owner_ids[0] else NULL end as owner_id'
            ],
            '', {n:n})

            YIELD value
            return value.owner_id as owner_id
        """
    )

    result = conn.query_read_only(
        query, parameters={"id": node_id, "account_id": account_id}
    )
    return result[0]["owner_id"]


def get_entity_before_update(account_id: str, node_id: str, label: Labels):
    entity_before_update: dict = get_node_properties_and_tags_by_id(
        account_id, node_id, label
    )

    if label in entities_without_owners:
        owner_id = get_node_parent_owner_by_id(account_id, node_id, label)
        if owner_id:
            entity_before_update["owner_id"] = owner_id

    is_semantic_entity = get_entity_family(label) == EntitiesFamilies.SEMANTIC_OBJECTS
    is_data_entity = get_entity_family(label) == EntitiesFamilies.DATA_OBJECTS
    if is_semantic_entity is True:
        entity_before_update["certified"] = get_certification(
            label, account_id, node_id
        )
    if is_data_entity:
        entity_before_update["num_of_queries"] = get_num_of_queries_by_label(
            account_id, node_id, label
        )["num_of_queries"]
    remove_embeddings_keys(entity_before_update)
    return entity_before_update


def remove_embeddings_keys(entity: dict):
    embedding_keys = [key for key in entity.keys() if "embedding" in key]
    if len(embedding_keys) > 0:

        def remove_embedding_key(key: str):
            del entity[key]

        [remove_embedding_key(key) for key in embedding_keys]


def delete_bulk_of_nodes(account_id, ids, labels):
    for label in labels:
        query = f"""match(n:{label}{{account_id:$account_id}})
                    where n.id in $ids 
                    detach delete n
                """
        conn.query_write(query, parameters={"ids": ids, "account_id": account_id})


def detach_bulk_of_nodes(account_id, ids):
    query = """ unwind $ids as id 
                match(n:field{account_id:$account_id, qs_id:id})-[r:depends_on]->() 
                delete r
            """
    conn.query_write(query, parameters={"ids": ids, "account_id": account_id})


def get_node_id_by_name_and_label(account_id: str, name: str, label: Labels):
    query = f"""MATCH (n:{label}{{account_id:$account_id, name:$name}})
               RETURN n.id as id"""
    result = conn.query_read_only(
        query=query,
        parameters={"account_id": account_id, "name": name},
    )
    if len(result) > 0:
        return result[0]["id"]
    return None


def get_analyses_queries(account_id: str) -> list[str]:
    query = """
        MATCH (s:sql{account_id: $account_id})<-[:analysis_of]-(analysis:analysis) 
        WHERE coalesce(analysis.recommended, false) = false 
        RETURN collect(distinct s.id) as sqls_ids
        """
    all_queries = conn.query_read_only(
        query=query, parameters={"account_id": account_id}
    )[0]["sqls_ids"]
    return all_queries


def get_queries_for_table(table_node):
    query = f"""
            CALL ({table_node}){{
                //get direct queries on table
                CALL ({table_node}){{
                    CALL apoc.path.subgraphNodes({table_node}, {{
                        relationshipFilter: "<SQL",
                        labelFilter: ">sql|-table",
                        minLevel: 0}})
                    YIELD node as sql_node
                    WHERE (not sql_node.is_sub_select) AND (sql_node.sql_type=$sql_type)
                    RETURN collect(distinct sql_node) as queries, MAX(sql_node.last_query_timestamp) as last_query_timestamp
                }}
                RETURN queries,last_query_timestamp
            }}
            """
    return query


def remove_account_from_graph(account_id):
    parameters = {"account_id": account_id}
    delete_query = f"""call apoc.periodic.iterate("MATCH (n{{account_id:'{account_id}'}}) return n", "DETACH DELETE n",
                                        {{batchSize:1000}}) yield batches, total return batches, total"""
    conn.query_write(query=delete_query, parameters=parameters)
