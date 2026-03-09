import logging

from infra.Neo4jConnection import get_neo4j_conn
from shared.graph.dal.utils_dal import (
    get_tags_query_by_node_alias,
)
from shared.graph.model.reserved_words import (
    Labels,
    DatasourcesRelationships,
    type_to_label,
    BiConnectors,
    data_relationships,
    label_to_type,
    bi_roots,
    get_number_of_sheets_by_connector,
)

logger = logging.getLogger("lineage_dal")
conn = get_neo4j_conn()


children_relationship = {
    Labels.TABLE: DatasourcesRelationships.SCHEMA,
    Labels.TEMP_TABLE: DatasourcesRelationships.SCHEMA,
    Labels.SHEET: DatasourcesRelationships.SHEET_FIELD,
    Labels.PUBLISHED_DS: DatasourcesRelationships.PUBLISHED_FIELD,
    Labels.EMBEDDED_DS: DatasourcesRelationships.EMBEDDED_FIELD,
    Labels.SISENSE_TABLE: DatasourcesRelationships.FIELD,
    Labels.SISENSE_WIDGET: DatasourcesRelationships.WIDGET_FIELD,
    Labels.QS_ANALYSIS: DatasourcesRelationships.FIELD,
    Labels.QS_VISUALIZATION: DatasourcesRelationships.VISUALIZATION_FIELD,
    Labels.QS_DASHBOARD: DatasourcesRelationships.DASHBOARD_FIELD,
    Labels.QS_DS: DatasourcesRelationships.FIELD,
    Labels.POWERBI_REPORT: DatasourcesRelationships.DASHBOARD_FIELD,
    Labels.POWERBI_VISUALIZATION: DatasourcesRelationships.VISUALIZATION_FIELD,
    Labels.POWERBI_DATAFLOW_TABLE: DatasourcesRelationships.PUBLISHED_FIELD,
    Labels.POWERBI_COPIED_DATAFLOW_TABLE: DatasourcesRelationships.COPY_PUBLISHED_FIELD,
    Labels.POWERBI_SEMANTIC_TABLE: DatasourcesRelationships.EMBEDDED_FIELD,
    Labels.LOOKER_LOOK: DatasourcesRelationships.VISUALIZATION_FIELD,
    Labels.LOOKER_VIEW: DatasourcesRelationships.FIELD,
    Labels.LOOKER_EXPLORE: DatasourcesRelationships.FIELD,
    Labels.LOOKER_VISUAL: DatasourcesRelationships.VISUALIZATION_FIELD,
}


def get_account_bi_connectors(account_id: str):
    connectors_list = ",".join(
        [f'"{conn}"' for conn in list(parent_labels_by_connector.keys())]
    )
    query = f""" 
    MATCH(c:connection{{account_id:$account_id}}) where c.type in [{connectors_list}]
    RETURN collect(distinct c.type) as connections
    """
    connections_types: list[str] = conn.query_read_only(
        query=query,
        parameters={"account_id": account_id},
    )[0]["connections"]
    return connections_types


def get_lineage_base(account_id: str, id: str, label: list[Labels]):
    relationships_filter = ["<connecting"]
    for rel in data_relationships:
        relationships_filter.append(f"""<{rel}""")
    upstream_relationships = "|".join(relationships_filter)
    labelFilter = "|".join(label)
    query = f""" 
    MATCH(n:{labelFilter}{{account_id:$account_id,id:$id}})
    CALL apoc.path.subgraphNodes(n,{{relationshipFilter:'{upstream_relationships}',labelFilter:'/connection'}})
    YIELD node
    WITH node.type as connector, labels(n)[0] as label, n.type as type
    RETURN {{label:label, type:type, connector_type:connector}} as data
    """
    connector_types: list[str] = conn.query_read_only(
        query=query,
        parameters={"account_id": account_id, "id": id},
    )[0]["data"]
    return connector_types


def get_datasources_lineage_of_table(
    account_id: str,
    root_id: str,
    root_label: Labels,
    direction: str,
    columns_ids: list[str] = [],
    level: int = None,
) -> list[dict[str, str]]:
    up_or_down = "<source_of|SQL>"
    source_rel_side = "startNode"
    sql_rel_side = "endNode"
    if direction == "downstream":
        up_or_down = "source_of>|<SQL"
        source_rel_side = "endNode"
        sql_rel_side = "startNode"

    columns_filter = "WHERE c.id in $columns_ids" if columns_ids else ""
    query = f"""
    MATCH(:{root_label}{{account_id:$account_id,id:$root_id}})-[:{children_relationship[root_label]}]->(c:column|field {columns_filter})
    call apoc.path.spanningTree(c,{{
        bfs:false,
        relationshipFilter:'{up_or_down}',
        maxLevel:$level,
        labelFilter:'+alias|+temp_column|+column|+set_op_column' }})
    yield path
    with path, relationships(path) as relationships, nodes(path) as nodes 
    with path, relationships, [n in nodes where n:column or n:temp_column ] as columns
    with path, relationships, columns where size(columns)>1
    CALL (relationships){{
        UNWIND relationships as rel
        WITH rel, properties({sql_rel_side}(rel)) as sql_rel_node, properties({source_rel_side}(rel)) as source_rel_node, properties(rel) as rel_props
        WITH CASE
        WHEN type(rel)='source_of' 
            THEN {{ col_id: source_rel_node.id, sql_id: rel_props.source_sql_id }}
        WHEN type(rel)='SQL' 
            THEN {{ col_id: sql_rel_node.id, sql_id: [rel_props.sql_id] }}
        ELSE {{ col_id: '', sql_id: [] }}
        END AS sql_id_to_node
        RETURN apoc.map.groupByMulti(collect(sql_id_to_node),'col_id') as sqls_map
    }}
    with path, columns, sqls_map where size(columns)>1
    call (columns){{
        with columns
        unwind columns as col
        match(col)<-[:schema]-(t:table|temp_table)<-[:schema]-(schema:schema|temp_schema)<-[:schema]-(db:db)<-[:connecting]-(connector:connection)
        with columns, collect(distinct t{{.type,.id,.name,.deleted,.num_of_queries,breadcrumbs:apoc.text.join([db.name,schema.name], ' \u2022 '),connector_type:connector.type, columns_ids: [col.id]}}) as tables
        return tables
    }}
    with sqls_map,tables where size(tables)>1 and tables[0].id<>tables[-1].id //prevent circular dependencies
    return collect(distinct {{tables:tables,sqls_map:sqls_map}}) as branches"""

    branches: list[list[dict, str, str]] = conn.query_read_only(
        query=query,
        parameters={
            "root_id": root_id,
            "account_id": account_id,
            "columns_ids": columns_ids,
            "level": level if level else 10,
        },
    )[0]["branches"]
    matrix = []
    for branch in branches:
        sqls_map = branch["sqls_map"]
        tables = branch["tables"]
        for level, table in enumerate(tables):
            add_node_to_matrix(
                matrix=matrix,
                direction=direction,
                current_level=level,
                node=table,
                parent_level=None if level == 0 else level - 1,
                parent_id=None if level == 0 else tables[level - 1]["id"],
                sqls_map=sqls_map,
            )
    return matrix


# the order of the links is meaningful! when adding more relationships make sure you add the link by the order of importance
parent_relationships_by_connector = {
    BiConnectors.LOOKER: [
        DatasourcesRelationships.SCHEMA,
        DatasourcesRelationships.FIELD,
        DatasourcesRelationships.VISUALIZATION_FIELD,
    ],
    BiConnectors.TABLEAU: [
        DatasourcesRelationships.SCHEMA,
        DatasourcesRelationships.PUBLISHED_FIELD,
        DatasourcesRelationships.EMBEDDED_FIELD,
        DatasourcesRelationships.SHEET_FIELD,
    ],
    BiConnectors.SISENSE: [
        DatasourcesRelationships.SCHEMA,
        DatasourcesRelationships.FIELD,
        DatasourcesRelationships.WIDGET_FIELD,
    ],
    BiConnectors.QUICKSIGHT: [
        DatasourcesRelationships.SCHEMA,
        DatasourcesRelationships.FIELD,
        DatasourcesRelationships.VISUALIZATION_FIELD,
    ],
    BiConnectors.POWERBI: [
        DatasourcesRelationships.SCHEMA,
        DatasourcesRelationships.PUBLISHED_FIELD,
        DatasourcesRelationships.COPY_PUBLISHED_FIELD,
        DatasourcesRelationships.EMBEDDED_FIELD,
        DatasourcesRelationships.DASHBOARD_FIELD,
        DatasourcesRelationships.VISUALIZATION_FIELD,
    ],
}

parent_labels_by_connector = {
    BiConnectors.LOOKER: [
        Labels.TABLE,
        Labels.LOOKER_EXPLORE,
        Labels.LOOKER_VIEW,
        Labels.LOOKER_LOOK,
        Labels.LOOKER_VISUAL,
    ],
    BiConnectors.TABLEAU: [
        Labels.TABLE,
        Labels.PUBLISHED_DS,
        Labels.EMBEDDED_DS,
        Labels.SHEET,
    ],
    BiConnectors.SISENSE: [
        Labels.TABLE,
        Labels.SISENSE_TABLE,
        Labels.SISENSE_WIDGET,
    ],
    BiConnectors.QUICKSIGHT: [
        Labels.TABLE,
        Labels.QS_DS,
        Labels.QS_VISUALIZATION,
    ],
    BiConnectors.POWERBI: [
        Labels.TABLE,
        Labels.POWERBI_DATAFLOW_TABLE,
        Labels.POWERBI_COPIED_DATAFLOW_TABLE,
        Labels.POWERBI_SEMANTIC_TABLE,
        Labels.POWERBI_REPORT,
        Labels.POWERBI_VISUALIZATION,
    ],
}


def handle_node_type(node: dict) -> dict:
    if "label" not in node:
        return
    ## specifically in PowerBi tables we would like to override the type with the parent type
    if node["label"] in [
        Labels.POWERBI_DATAFLOW_TABLE,
        Labels.POWERBI_COPIED_DATAFLOW_TABLE,
    ]:
        node["type"] = label_to_type(Labels.POWERBI_DATAFLOW)
    if node["label"] == Labels.POWERBI_SEMANTIC_TABLE:
        node["type"] = label_to_type(Labels.POWERBI_SEMANTIC_MODEL)
    return node


def add_node_to_matrix(
    matrix: list[dict[str, dict[str, str]]],
    direction: str,
    current_level: int,
    node: dict,
    sqls_map: dict[str, list[dict[str, str]]],
    parent_level: int = None,
    parent_id: str = None,
):
    if len(matrix) <= current_level:
        matrix.append({})
    node_id = node["id"]
    handle_node_type(node)
    if node_id not in matrix[current_level]:
        matrix[current_level][node_id] = node.copy()
        matrix[current_level][node_id][direction] = []
        matrix[current_level][node_id]["columns_ids"] = []
        matrix[current_level][node_id]["sql_id"] = []
    if "columns_ids" in node:
        matrix[current_level][node_id]["columns_ids"] = list(
            set(matrix[current_level][node_id]["columns_ids"] + node["columns_ids"])
        )
        ## get the sql ids connected from the parent
        for column_id in matrix[current_level][node_id]["columns_ids"]:
            if column_id in sqls_map:
                for col_to_sql in sqls_map[column_id]:
                    matrix[current_level][node_id]["sql_id"].extend(
                        col_to_sql["sql_id"]
                    )
                    matrix[current_level][node_id]["sql_id"] = list(
                        set(matrix[current_level][node_id]["sql_id"])
                    )
    if current_level > 0:
        parent = matrix[parent_level][parent_id]
        parent[direction].append(node_id)


def get_bi_lineage_of_table(
    account_id: str,
    root_id: str,
    root_label: Labels,
    connector: BiConnectors,
    direction: str,
    level: int,
    matrix: list[str] = [],
    columns_ids: list[str] = None,
) -> list[dict[str, str]]:
    relationships = parent_relationships_by_connector[connector]
    if children_relationship[root_label] not in relationships:
        return matrix
    fields_relationships = "<depends_on|source_of>|<SQL"
    source_of_child = "endNode"
    depends_on_child = "startNode"
    if direction == "upstream":
        source_of_child = "startNode"
        depends_on_child = "endNode"
        fields_relationships = "depends_on>|<source_of|SQL>"
    root_relationship = children_relationship[root_label]
    root_rel_ranking = relationships.index(root_relationship)
    is_leaf = root_rel_ranking == len(relationships) - 1 and direction == "downstream"
    if is_leaf:
        return matrix
    # the rank array is the hierarchy of parents that we are looking for
    # F.E if the whole hierarch array is ['schema', 'published_field', 'embedded_field', 'sheet_field']
    # if we will look for the lineage of a published datasource in Tableau
    # the downstream rank would be ['published_field', 'embedded_field', 'sheet_field']
    # and the upstream rank would be: ['published_field', 'schema']
    # when the first rank is always the root's relationships
    relationships_ranking = (
        relationships[: root_rel_ranking + 1]
        if direction == "upstream"
        else relationships[root_rel_ranking:]
    )
    parent_relationships = "|".join(relationships_ranking)
    ranks = relationships_ranking
    if direction == "upstream":
        ranks.reverse()
    if not parent_relationships:
        return matrix
    parent_labels = "|".join(parent_labels_by_connector[connector])
    columns_filter = "WHERE c.id in $columns_ids" if columns_ids else ""
    query = f"""
    MATCH(root:{root_label}{{account_id:$account_id,id:$root_id}})-[:{root_relationship}]->(c:column|field {columns_filter})
    call apoc.path.spanningTree(c,{{
        bfs:false,
        relationshipFilter:'{fields_relationships}',
        maxLevel:$level,
        labelFilter:'+field|+column|+alias|+set_op_column' }})
    yield path
    with path, nodes(path) as nodes, relationships(path) as rels
    CALL (rels){{
        UNWIND rels as rel
        WITH rel, properties({depends_on_child}(rel)) as depends_on_node, properties({source_of_child}(rel)) as source_of_node, properties(rel) as rel_props
        WITH CASE
        WHEN type(rel)='source_of' 
            THEN {{ col_id: source_of_node.id, sql_id: rel_props.source_sql_id }}
        WHEN type(rel)='SQL' 
            THEN {{ col_id: depends_on_node.id, sql_id: [rel_props.sql_id] }}
        WHEN type(rel)='depends_on' AND apoc.map.get(source_of_node,'sql_id',null,false) is not null // rel between bi field and a custom sql node
            THEN {{ col_id: depends_on_node.id, sql_id: [apoc.map.get(source_of_node,'sql_id',null,false)] }}
        ELSE {{ col_id: '', sql_id: [] }}
        END AS sql_id_to_node
        RETURN apoc.map.groupByMulti(collect(sql_id_to_node),'col_id') as sqls_map
    }}
    with path, nodes, sqls_map
    CALL (nodes, path){{
        UNWIND nodes AS node
        // we do this step for 2 reasons: 
        // 1. filter out the columns who's parents are irrelevant (custom sql fields or aliases for example)
        // 2. bi fields could be shared between entities of differnt types , so for example a field could belong to both a sheet and an embedded datasource.
        // the purpose of this step is to get the field X number of parents. 
        // so if we have [field] and the field has 2 parents, this array would turn to -> [field, field] which would later be [embedded datasource, sheet]
        MATCH(node)<-[link:{parent_relationships}]-(parent:{parent_labels})
        WITH path, collect({{ column_id: node.id, parent: parent, link:type(link), parent_id: parent.id }}) as branch, collect(parent.id) as parents_ids
        WITH path, branch, apoc.map.groupByMulti(branch,'parent_id') as parents, parents_ids, apoc.map.groupByMulti(branch,'link') as rels,
        apoc.map.groupByMulti(branch,'column_id') as columns_to_parents
        UNWIND KEYS(parents) as parent_id
        WITH branch, parents_ids,  parents[parent_id][0].parent as parent, parents[parent_id][0].link as link,
        any(col_id in keys(columns_to_parents) where size(columns_to_parents[col_id])>1) as shared_fields,
        // calculated fields could depend on other fields within the same datasource, so we group together all of the columns ids within the same parent
            keys(apoc.map.groupByMulti(parents[parent_id],'column_id')) as columns_ids,
            CASE WHEN parents[parent_id][0].link ='schema' 
                  THEN apoc.coll.indexOf(parents_ids,parent_id)
                  WHEN parents[parent_id][0].link <>'schema' and $direction='downstream' // bi elements should be the downstream of tables
                  THEN apoc.coll.indexOf($ranks, parents[parent_id][0].link)+size(apoc.map.get(rels,'schema',[],false)) // tables count
                  ELSE apoc.coll.indexOf($ranks, parents[parent_id][0].link)
                  END AS depth
        WITH branch, parents_ids, shared_fields, parent{{.type, .id, .name, .num_of_queries, depth:depth, link:link, columns_ids: columns_ids, connector_type: $connector,
        label: labels(parent)[0] }} as parent
        // filter out sheets that use the fields of the root sheet
        WITH branch, parents_ids, shared_fields, parent where not(parent.type='sheet' and parent.depth=0 and parent.id<>$root_id)
        WITH shared_fields, parent ORDER BY parent.depth, apoc.coll.indexOf(parents_ids,parent.id)
        WITH shared_fields, collect(parent) as branch
        WITH branch, shared_fields WHERE SIZE(branch)>1
        RETURN branch, shared_fields
    }}
    return collect({{branch:branch,shared_fields:shared_fields, sqls_map: sqls_map }}) as branches """

    branches = conn.query_read_only(
        query=query,
        parameters={
            "root_id": root_id,
            "account_id": account_id,
            "columns_ids": columns_ids,
            "ranks": ranks,
            "level": level if level else 10,
            "connector": connector,
            "direction": direction,
        },
    )[0]["branches"]

    for branch_details in branches:
        branch: list[dict[str, str]] = branch_details["branch"]
        shared_fields: bool = branch_details["shared_fields"]
        sqls_map: bool = branch_details["sqls_map"]
        types_map: dict[str, list[dict[str, str]]] = {}
        for index, node in enumerate(branch):
            if node["type"] in types_map:
                types_map[node["type"]].append(node)
            else:
                types_map[node["type"]] = [node]
            if index == 0:
                node["level"] = 0
                add_node_to_matrix(
                    matrix=matrix,
                    direction=direction,
                    current_level=0,
                    node=node,
                    sqls_map=sqls_map,
                )
            # visuals use fields of other entities quite often, this means that they can be the downstream of multiple entities
            # even within the same branch
            elif is_visual(node) and shared_fields:
                parent_type = get_visual_parent_type(connector, types_map)
                if not parent_type:
                    continue
                potential_parents = types_map[parent_type]
                if len(potential_parents) == 1:
                    parent = potential_parents[0]
                    add_node_to_matrix(
                        matrix=matrix,
                        direction=direction,
                        current_level=parent["level"] + 1,
                        node=node,
                        parent_level=parent["level"],
                        parent_id=parent["id"],
                        sqls_map=sqls_map,
                    )
                else:
                    node_fields = set(node["columns_ids"])
                    for parent in potential_parents:
                        common_fields = set(parent["columns_ids"]).intersection(
                            node_fields
                        )
                        if len(common_fields) == 0:
                            continue
                        node_to_connect = node.copy()
                        node_to_connect["columns_ids"] = list(common_fields)
                        add_node_to_matrix(
                            matrix=matrix,
                            direction=direction,
                            current_level=parent["level"] + 1,
                            node=node_to_connect,
                            parent_level=parent["level"],
                            parent_id=parent["id"],
                            sqls_map=sqls_map,
                        )
            else:
                node["level"] = index
                add_node_to_matrix(
                    matrix=matrix,
                    direction=direction,
                    current_level=index,
                    node=node,
                    parent_level=index - 1,
                    parent_id=branch[index - 1]["id"],
                    sqls_map=sqls_map,
                )

    return matrix


def get_visual_parent_type(connector: BiConnectors, types_map: dict):
    ancestors_labels = parent_labels_by_connector[connector].copy()
    ancestors_labels = ancestors_labels[: -get_number_of_sheets_by_connector(connector)]
    ancestors_labels.reverse()
    parent_type = next(
        (
            label_to_type(parent_label)
            for parent_label in ancestors_labels
            if label_to_type(parent_label) in types_map
        ),
        None,
    )
    return parent_type


def is_visual(node: dict):  ## sheet, qs_visualization, sisense_widget
    return node["type"] == "sheet"


def get_lineage_data(account_id: str, node_id: str, label: str, connector: str):
    bi_breadcrumbs_root = bi_roots[connector] if connector in bi_roots else ""
    children_rel = (
        children_relationship[label] if label in children_relationship else ""
    )
    query = f"""
    MATCH(node:{label}{{account_id:$account_id, id:$node_id}})
    CALL apoc.case([
            node:table or node:temp_table,
            'MATCH(node)<-[:schema]-(schema:schema)<-[:schema]-(db:db)<-[:connecting]-(connector:connection)
            RETURN apoc.text.join([db.name,schema.name]," \u2022 ") as breadcrumbs, connector.type as connector_type'
            ],
            'WITH node CALL apoc.path.spanningTree(node, 
                {{ relationshipFilter:"<bi_rel", labelFilter:"/{bi_breadcrumbs_root}" }} ) YIELD path
            with reverse(nodes(path))[0..2] as breadcrumb
            RETURN apoc.text.join([n in breadcrumb | n.name]," \u2022 ") as breadcrumbs', 
            {{node:node}}
        )
    YIELD value
    WITH node, value.breadcrumbs as breadcrumbs, coalesce(value.connector_type,$connector) as connector_type
    CALL apoc.case([
        node.type='sheet' and $connector<>'{BiConnectors.POWERBI}',
        'OPTIONAL MATCH(node)<-[:bi_rel]-(dashboard:{type_to_label("dashboard", connector)}|{Labels.QS_ANALYSIS})
        RETURN collect(dashboard{{.id,.name,.type, connector_type: connector}}) as dashboards',
        node.type='sheet' and $connector='{BiConnectors.POWERBI}',
        'OPTIONAL MATCH(node)<-[:bi_rel]-(:{Labels.POWERBI_REPORT}{{account_id:$account_id}})<-[:depends_on]-(dashboard:{Labels.POWERBI_DASHBOARD})
        RETURN collect(dashboard{{.id,.name,.type, connector_type: connector}}) as dashboards'
        ], 
        'RETURN null as dashboards',
        {{ node:node,  connector: $connector, account_id:$account_id }})
    YIELD value
    WITH node, breadcrumbs, connector_type, value.dashboards as dashboards
    {get_tags_query_by_node_alias("node")}
    WITH node, breadcrumbs, connector_type, dashboards, value.tags as root_tags
    CALL apoc.when(
       node.type<>'dashboard',
       'MATCH(node)-[:{children_rel}]->(column:temp_column|column|field)
        WITH column order by column.name
        {get_tags_query_by_node_alias("column")}
        RETURN collect(column{{.id,.type,.name,.deleted,.data_type, tags: value.tags}}) as columns',
       'RETURN [] as columns', {{ node:node, bi_connector:$connector }})
    YIELD value
    WITH node, breadcrumbs, connector_type, dashboards, root_tags, value.columns as columns
    RETURN {{tags:root_tags, breadcrumbs:breadcrumbs, columns:columns, dashboards:dashboards, connector_type:connector_type }} as data
    """
    columns = conn.query_read_only(
        query=query,
        parameters={
            "node_id": node_id,
            "account_id": account_id,
            "connector": connector,
        },
    )[0]["data"]
    return columns
