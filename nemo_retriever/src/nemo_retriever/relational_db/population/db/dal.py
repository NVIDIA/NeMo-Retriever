from datetime import datetime
import logging

import pandas as pd
from vector_store.neo4j_store import get_neo4j_conn

logger = logging.getLogger(__name__)

from graph.dal.utils_dal import get_entity_before_update
from graph.utils import chunks
from graph.model.reserved_words import Labels, SQLType, label_to_type
from graph.dal.schemas_dal import load_schema_from_graph, add_schemas_edge

from graph.model.schema import TEMP_SCHEMA_NAME
conn = get_neo4j_conn()


def db_exists(account_id, db_node):
    db_name = db_node.get_name()
    query = f"""
    MATCH (n:db{{account_id:"{account_id}", name:"{db_name}"}})
    OPTIONAL MATCH (n)-[r]-(v) WHERE NOT v:connection
    RETURN n.id AS id, count(r) AS nbrs
    """
    result_data = conn.query_read_only(
        query=query, parameters={"account_id": account_id}
    )
    if not result_data or len(result_data) == 0:
        return None, None

    nbrs = result_data[0]["nbrs"]

    return result_data[0]["id"], False if nbrs == 0 else True


def update_node_property(account_id, label, node_id, update_properties):
    query = f"""
            match(n:{label}{{account_id:$account_id, id:$node_id}})
            set n += $update_properties    
            """
    conn.query_write(
        query=query,
        parameters={
            "account_id": account_id,
            "node_id": node_id,
            "update_properties": update_properties,
        },
    )


def delete_schema(schema_node_id, account_id, deleted_time=datetime.now()):
    query = """MATCH (n:schema {id: $schema_node_id, account_id: $account_id})-[:schema]->(t:table)-[:schema]->(c:column) 
               SET n.deleted = True
               SET n.deleted_time = $deleted_time
               SET t.deleted = True
               SET t.deleted_time = $deleted_time
               SET c.deleted = True
               SET c.deleted_time = $deleted_time
             """
    conn.query_write(
        query=query,
        parameters={
            "account_id": account_id,
            "schema_node_id": schema_node_id,
            "deleted_time": deleted_time,
        },
    )


def update_disconnected_sqls(account_id):
    query = """
            match (sql:sql{account_id:$account_id, is_sub_select:false})
            call (sql){
            CALL apoc.path.subgraphNodes(sql, {
                relationshipFilter: "SQL>",
                labelFilter: "/table|-column",     
                minLevel: 0})
            YIELD node
            WHERE coalesce(node.deleted,false)=true
            return collect(node.id) as tbls
            }
            with sql, tbls
            SET sql.invalid = size(tbls) > 0
            SET sql.invalid_items = tbls
            SET sql.invalid_time = case when sql.invalid=true and sql.invalid_time is null 
                    then datetime.realtime() else sql.invalid_time end
            """
    conn.query_write(query=query, parameters={"account_id": account_id})
    # delete_sql = [x["sql_id"] for x in result_data]

    query = """
            match (sql:sql{account_id:$account_id, is_sub_select:false})
            call (sql){
            CALL apoc.path.subgraphNodes(sql, {
                relationshipFilter: "SQL>",
                labelFilter: "/column|-table",     
                minLevel: 0})
            YIELD node
            WHERE coalesce(node.deleted,false)=true
            return collect(node.id) as cols
            }
            with sql, cols
            SET sql.invalid = size(cols) > 0
            SET sql.invalid_items = coalesce(sql.invalid_items, []) + cols
            SET sql.invalid_time = case when sql.invalid=true and sql.invalid_time is null 
                    then datetime.realtime() else sql.invalid_time end
            """
    conn.query_write(query=query, parameters={"account_id": account_id})

def delete_unused_temporary_tables_and_columns(account_id):
    query = """
            match(s:temp_schema{account_id:$account_id})-[:schema]->(deleted_table:temp_table{deleted:true})-[:schema]->(c:temp_column)
            detach delete c
            detach delete deleted_table
            with s
            match(s:temp_schema{account_id:$account_id})-[:schema]->(t:temp_table)-[:schema]->(deleted_column{deleted:true})
            detach delete deleted_column
            """
    try:
        conn.query_write(
            query=query,
            parameters={"account_id": account_id},
        )
    except Exception as err:
        logger.error(f"Error in delete_unused_temporary_tables_and_columns: {err}")


def find_and_delete_old_version_source_queries(account_id, source_type):
    query = """ 
                match(c:column|temp_column{account_id:$account_id})<-[r:source_of]-() 
                where exists((:sql{id:r.source_sql_id[0], sql_type:$source_type})-[]-()) 
                with c, apoc.coll.toSet(apoc.coll.flatten(collect(r.source_sql_id))) as sources
                where size(sources)>1
                match(s:sql{account_id:$account_id, sql_type:$source_type}) where s.id in sources
                with c, apoc.coll.sortNodes(collect(s), 'last_query_timestamp')[0] as first_s, sources
                match(s:sql{account_id:$account_id, sql_type:$source_type}) where s.id in sources and s.id<>first_s.id
                with distinct s, s.id as deleted_sql_id
                CALL apoc.path.subgraphNodes(s, { 
                    relationshipFilter:'SQL>', 
                    minLevel: 0}
                ) YIELD node 
                where not(node:table or node:column or node:temp_table or node:temp_column)
                detach delete node
                RETURN collect(distinct deleted_sql_id) as deleted_sql_ids
                """
    try:
        result = conn.query_write(
            query=query,
            parameters={"account_id": account_id, "source_type": source_type},
        )
        if len(result) > 0:
            deleted_sql_ids = result[0]["deleted_sql_ids"]
    except Exception as err:
        raise Exception('Error in "find_and_delete_redundant_views"')

    query = """ 
                CALL apoc.periodic.iterate(
                    "UNWIND $deleted_sql_ids as deleted_sql_id
                    MATCH (c2:column|temp_column{account_id:$account_id})<-[r_to_delete:source_of]-(src_col:column|temp_column{account_id:$account_id})
                    WHERE deleted_sql_id in r_to_delete.source_sql_id
                    RETURN deleted_sql_id, r_to_delete",
                    "CALL apoc.do.when(size(r_to_delete.source_sql_id) > 1,
                        'SET r_to_delete.source_sql_id = [id in r_to_delete.source_sql_id where id <> deleted_sql_id]',
                        'DELETE r_to_delete',
                        {r_to_delete:r_to_delete, deleted_sql_id:deleted_sql_id}
                    )
                    YIELD value
                    RETURN count(*)",
                {batchSize:100, parallel:false, params:{deleted_sql_ids:$deleted_sql_ids, account_id:$account_id}})
                yield batches, total return batches, total
            """
    try:
        conn.query_write(
            query=query,
            parameters={"account_id": account_id, "deleted_sql_ids": deleted_sql_ids},
        )
    except Exception as err:
        raise Exception('Error in "find_and_delete_redundant_views"')


def find_and_connect_unions(account_id):
    query = """
            MATCH(main_union_node:operator{name: "Union", account_id:$account_id})
            WHERE not coalesce(main_union_node.processed, false)
            WITH main_union_node
            MATCH(s:sql{id:main_union_node.sql_id})
            WHERE s.sql_type in $sql_types
            CALL (s){
                // collect all joins nodes in the union's subgraph for them to be "blacklist"
                CALL apoc.path.subgraphNodes(s, {
                    relationshipFilter: "SQL>",
                    labelFilter: ">command",     
                    minLevel: 0})
                YIELD node as joins_node
                WHERE joins_node.name = "Joins"
                MATCH(from_node:command)-[:SQL]->(joins_node)
                RETURN collect(from_node) as blacklist
            }
            CALL apoc.path.subgraphNodes(main_union_node, {
                relationshipFilter: "SQL>",
                labelFilter: "/table",     
                minLevel: 0,
                blacklistNodes: blacklist})
            YIELD node as table_for_union
            WITH main_union_node, collect(table_for_union) as tables_for_union
            UNWIND range(0, size(tables_for_union)-2) as i
            UNWIND range(i+1, size(tables_for_union)-1) as j
            MATCH(t1:table{id:tables_for_union[i].id, account_id:$account_id}), (t2:table{id:tables_for_union[j].id, account_id:$account_id})
            MERGE(t1)-[:union]->(t2)
            """
    conn.query_write(
        query=query,
        parameters={
            "account_id": account_id,
            "sql_types": [SQLType.QUERY, SQLType.MERGE],
        },
    )


def add_schemas_edge_batch(account_id, edges, created):
    """
    If the nodes do not exist in the Neo4j graph, the function adds them.
    Add to the Neo4j graph the given edge.
    :param edge: edge is a tuple of the form (from_node, to_node, edge_properties)
    :return:
    """
    try:
        # in case of match override the existing ID in the graph, in order to correlate with the ID of the parsed Node object
        query = """
            UNWIND $edges as e
            CALL apoc.merge.node.eager([e.from_label], e.from_identProps, e.v_props, {id:e.v_props.id})
            yield node as v1
            set v1.created = case when coalesce(v1.deleted, false) = false then coalesce(v1.created, $created) else $created end
            set v1.deleted = false
            with v1, e
            call apoc.merge.node.eager([e.to_label], e.to_identProps, e.u_props, {id:e.u_props.id})
            yield node as v2
            set v2.created = case when coalesce(v2.deleted, false) = false then coalesce(v2.created, $created) else $created end
            set v2.deleted = false
            MERGE (v1)-[r:schema]->(v2)
            SET r = e.optional_edge_props
            """

        conn.query_write(
            query=query,
            parameters={
                "account_id": account_id,
                "created": created,
                "edges": edges,
            },
        )
    except Exception as err:
        raise Exception('Error in "add_schemas_edge_batch"')


def accumulate_deleted_column_props(account_id, deleted_column, list_for_event_log):
    list_for_event_log.append(
        {
            "id": deleted_column.props_graph["id"],
            "before_update": get_entity_before_update(
                account_id, deleted_column.props_graph["id"], Labels.COLUMN
            ),
            "payload": None,
        }
    )


def accumulate_added_column_props(
    account_id, added_column, list_for_event_log, edges_to_add, new_schema
):
    new_table_node_props = new_schema.get_table_node_props(added_column.table_name)
    new_table_node_match_props = new_schema.get_table_node_match_props(
        added_column.table_name
    )
    edges_to_add.append(
        {
            "from_label": new_table_node_props["label"],
            "from_identProps": new_table_node_match_props,
            "v_props": new_table_node_props,
            "to_label": added_column.props_files["label"],
            "to_identProps": added_column.match_props_files,
            "u_props": added_column.props_files,
            "optional_edge_props": {"schema": added_column.schema},
        }
    )
    list_for_event_log.append(
        {
            "id": added_column.props_files["id"],
            "before_update": None,
            "payload": remove_embedding_property(added_column.props_files),
            "table_name": added_column.table_name,
        }
    )


def accumulate_updated_table(
    account_id, table_in_intersection, items_to_update_in_graph, new_schema
):
    # verify that the node with the correct id is in hand: replace new table id with existing table id
    # new_schema.replace_id(table_in_intersection.props_y["id"], table_in_intersection.props_x["id"])
    new_table_node_props = table_in_intersection.props_files
    new_table_node_props["id"] = table_in_intersection.props_graph["id"]
    items_to_update_in_graph.append(
        {
            "id": table_in_intersection.props_graph["id"],
            "label": Labels.TABLE,
            "props": new_table_node_props,
        }
    )


def accumulate_updated_column(
    account_id, column_in_intersection, items_to_update_in_graph, new_schema
):
    # verify that the node with the correct id is in hand
    # new_schema.replace_id(column_in_intersection.props_y["id"], column_in_intersection.props_x["id"])
    new_column_node_props = column_in_intersection.props_files
    new_column_node_props["id"] = column_in_intersection.props_graph["id"]
    items_to_update_in_graph.append(
        {
            "id": column_in_intersection.props_graph["id"],
            "label": Labels.COLUMN,
            "props": new_column_node_props,
        }
    )


def update_diff_from_existing_schema(new_schema, account_id, latest_timestamp):
    try:
        added_or_modified_tables = {
            "schema": str(new_schema.schema_node.name),
            "tables": set(),
        }

        # load existing schema
        schema_name = new_schema.get_schema_name()
        db_name = new_schema.get_db_name()

        is_temp = new_schema.schema_node.label == Labels.TEMP_SCHEMA
        if is_temp:
            return added_or_modified_tables

        existing_schema = load_schema_from_graph(
            db_name, schema_name, account_id, is_temp=is_temp
        )
        if existing_schema is None:
            return False

        existing_schema_node = existing_schema.get_schema_node()

        existing_table_names = existing_schema.tables_df.table_name.unique()
        new_table_names = new_schema.tables_df.table_name.unique()
        tables_names_to_add = set(new_table_names) - set(existing_table_names)
        logger.info(f"Tables to add in schema {schema_name}: {len(tables_names_to_add)}")
        added_or_modified_tables["tables"].update(tables_names_to_add)

        for table_name in tables_names_to_add:
            edge_params = {"schema": schema_name}
            add_schemas_edge(
                [
                    existing_schema_node,
                    new_schema.get_table_node(table_name),
                    edge_params,
                ],
                account_id,
                latest_timestamp,
            )

            column_names = new_schema.get_table_columns_by_table_name(table_name)
            for column_name in column_names:
                add_schemas_edge(
                    [
                        new_schema.get_table_node(table_name),
                        new_schema.get_column_node(column_name, table_name),
                        edge_params,
                    ],
                    account_id,
                    latest_timestamp,
                )

        tables_names_to_delete = set(existing_table_names) - set(new_table_names)
        logger.info(f"Tables to delete in schema {schema_name}: {len(tables_names_to_delete)}")
        for deleted_table_name in tables_names_to_delete:
            deleted_table_node_props = existing_schema.get_table_node_props(
                deleted_table_name
            )
            delete_table(deleted_table_node_props["id"], account_id)
        # update ids of tables and columns that appear both in the new schema and in the existing schema
        tables_merge = pd.merge(
            existing_schema.tables_df,
            new_schema.tables_df,
            on=["database", "schema", "table_name"],
            how="inner",
            suffixes=("_graph", "_files"),
        )
        list_of_props = ["row_count", "size", "retention_time", "last_altered"]
        if len(list_of_props) > 0:
            table_diffs = []
            for prop in list_of_props:
                # This is not working after upgrading NumPy, as it doesn't allow to compare pd.NA
                # table_diff = tables_merge.loc[
                #     ~pd.isna(tables_merge[f"{prop}_graph"])
                #     & (tables_merge[f"{prop}_graph"] != tables_merge[f"{prop}_files"])
                # ]
                # Converting to string for now (just like columns), we need to think about pd.NA
                table_diff = tables_merge[
                    tables_merge[f"{prop}_graph"].astype(str)
                    != tables_merge[f"{prop}_files"].astype(str)
                ]
                table_diffs.append(table_diff)
            tables_to_update = pd.concat(table_diffs, ignore_index=True, axis=0)
            tables_to_update.drop_duplicates(
                set(tables_to_update.columns)
                - set(
                    [
                        "props_graph",
                        "props_files",
                        "match_props_graph",
                        "match_props_files",
                    ]
                ),
                inplace=True,
            )

            items_to_update_in_graph = []
            tables_to_update.apply(
                lambda x: accumulate_updated_table(
                    account_id, x, items_to_update_in_graph, new_schema
                ),
                axis=1,
            )
            items_to_update_in_graph_chunks = list(
                chunks(items_to_update_in_graph, 1000)
            )
            len_chunks = len(items_to_update_in_graph_chunks)
            for i, chunk in enumerate(items_to_update_in_graph_chunks):
                logger.info(f"Updating tables chunk {i + 1}/{len_chunks}")
                update_properties_in_graph_batch(account_id, chunk)

        # If a table appears in both schemas, identify columns to add and columns to delete.
        columns_merge = pd.merge(
            existing_schema.columns_df,
            new_schema.columns_df,
            on=["database", "schema", "table_name", "column_name"],
            how="left",
            suffixes=("_graph", "_files"),
        )
        deleted_columns = columns_merge.loc[
            columns_merge["column_name_lower_files"].isnull()
        ]
        logger.info(f"Columns to delete in schema {schema_name}: {len(deleted_columns)}")
        entities = []
        deleted_columns.apply(
            lambda x: accumulate_deleted_column_props(account_id, x, entities), axis=1
        )
        delete_columns_batch([x["id"] for x in entities], account_id)
        # filter out columns of deleted tables
        modified_tables = deleted_columns.table_name.unique()
        added_or_modified_tables["tables"].update(modified_tables)

        columns_merge = pd.merge(
            existing_schema.columns_df,
            new_schema.columns_df,
            on=["database", "schema", "table_name", "column_name"],
            how="right",
            suffixes=("_graph", "_files"),
        )
        added_columns = columns_merge.loc[
            columns_merge["column_name_lower_graph"].isnull()
        ]
        logger.info(f"Columns to add in schema {schema_name}: {len(added_columns)}")
        entities = []
        edges_to_merge = []
        added_columns.apply(
            lambda x: accumulate_added_column_props(
                account_id, x, entities, edges_to_merge, new_schema
            ),
            axis=1,
        )
        add_schemas_edge_batch(account_id, edges_to_merge, created=latest_timestamp)
        # filter out columns of added tables
        modified_tables = added_columns.table_name.unique()
        added_or_modified_tables["tables"].update(modified_tables)

        columns_merge = pd.merge(
            existing_schema.columns_df,
            new_schema.columns_df,
            on=["database", "schema", "table_name", "column_name"],
            how="inner",
            suffixes=("_graph", "_files"),
        )

        list_of_props = ["data_type", "default", "is_nullable", "length", "scale"]
        if len(list_of_props) > 0:
            column_diffs = []
            for prop in list_of_props:
                # column_diff = columns_merge.loc[
                #     (columns_merge[f"{prop}_graph"] != columns_merge[f"{prop}_files"])
                #     & ~pd.isna(columns_merge[f"{prop}_graph"])
                # ]
                # https://stackoverflow.com/a/34746437
                column_diff = columns_merge[
                    columns_merge[f"{prop}_graph"].astype(str)
                    != columns_merge[f"{prop}_files"].astype(str)
                ]
                column_diffs.append(column_diff)
            columns_to_update = pd.concat(column_diffs, ignore_index=True, axis=0)
            columns_to_update.drop_duplicates(
                set(columns_to_update.columns)
                - set(
                    [
                        "props_graph",
                        "props_files",
                        "match_props_graph",
                        "match_props_files",
                    ]
                ),
                inplace=True,
            )

            items_to_update_in_graph = []
            columns_to_update.apply(
                lambda x: accumulate_updated_column(
                    account_id, x, items_to_update_in_graph, new_schema
                ),
                axis=1,
            )
            items_to_update_in_graph_chunks = list(
                chunks(items_to_update_in_graph, 1000)
            )
            len_chunks = len(items_to_update_in_graph_chunks)
            for i, chunk in enumerate(items_to_update_in_graph_chunks):
                logger.info(f"Updating columns chunk {i + 1}/{len_chunks}")
                update_properties_in_graph_batch(account_id, chunk)
        delete_table_info_for_embedding_batch(
            account_id, added_or_modified_tables, db_name
        )
        return added_or_modified_tables
    except Exception as err:
        raise Exception(f'Error in "update_diff_from_existing_schema": {err}')


def get_tables_columns(account_id, db_id, schema):
    if db_id is None:
        query = """MATCH(db:db{account_id:$account_id})-[:schema]->(s:schema{name:$schema})-[:schema]->
                    (t:table)-[:schema]->(c:column)
                    WHERE coalesce(s.deleted, false) = false and coalesce(t.deleted, false) = false and
                        coalesce(c.deleted, false) = false 
                    RETURN t.name as table_name, c.name as col_name 
                """
    else:
        query = """MATCH(db:db{id:$db_id,account_id:$account_id})-[:schema]->(s:schema{name:$schema})-[:schema]->
                    (t:table)-[:schema]->(c:column)
                    WHERE coalesce(s.deleted, false) = false and coalesce(t.deleted, false) = false and
                        coalesce(c.deleted, false) = false 
                    RETURN t.name as table_name, c.name as col_name 
                    """
    result = pd.DataFrame(
        conn.query_read_only(
            query=query,
            parameters={"db_id": db_id, "account_id": account_id, "schema": schema},
        )
    )
    result = result.groupby("table_name")["col_name"].apply(list).reset_index()
    result["col_length"] = result["col_name"].apply(lambda x: len(x))
    return result


def remove_embedding_property(props: dict):
    if "embedding" in props:
        del props["embedding"]
    return props


def delete_table(table_id, account_id, deleted_time=datetime.now()):
    query = """ MATCH (n:table {id: $table_id, account_id: $account_id})-[:schema]->(c:column)
                SET n.deleted = True
                SET n.deleted_time = $deleted_time
                SET c.deleted = True
                SET c.deleted_time = $deleted_time
            """
    conn.query_write(
        query=query,
        parameters={
            "account_id": account_id,
            "table_id": table_id,
            "deleted_time": deleted_time,
        },
    )


def update_properties_in_graph_batch(account_id, items):
    query = """
            UNWIND $items as item
            WITH item, item.props.description as new_description, 
            apoc.map.removeKeys(item.props, ["description"]) as item_props_no_description 
            CALL apoc.merge.node.eager([item.label], {id: item.id, account_id: $account_id}, {}, item_props_no_description) 
            YIELD node
            // keep existing description unless it is null
            SET node.description = coalesce(node.description, new_description)  
            """
    conn.query_write(
        query=query,
        parameters={"items": items, "account_id": account_id},
    )


def update_properties_in_graph(account_id, item_id, node_label, new_parameters):
    query = f""" MATCH (item:{node_label}{{account_id: $account_id, id:$table_id}})
                 SET item += $new_parameters
            """
    conn.query_write(
        query=query,
        parameters={
            "table_id": item_id,
            "account_id": account_id,
            "new_parameters": new_parameters,
        },
    )


def delete_table_info_for_embedding_batch(
    account_id, added_or_modified_tables, db_name
):
    account_simple_str = account_id.replace("-", "_")
    tables = [
        {"db": db_name, "schema": added_or_modified_tables["schema"], "name": t}
        for t in added_or_modified_tables["tables"]
    ]
    query = f"""
            UNWIND $tables as table
            MATCH (t:table {{account_id: $account_id, name: table.name, db_name: table.db, schema_name: table.schema}})
            SET t.table_info_for_embedding = null,
            t.table_info_embedding_{account_simple_str} = null
            """
    conn.query_write(
        query=query,
        parameters={
            "account_id": account_id,
            "tables": tables,
        },
    )


def delete_columns_batch(column_ids, account_id, deleted_time=datetime.now()):
    query = """UNWIND $column_ids as column_id
               MATCH (c:column {id: column_id, account_id: $account_id}) 
               SET c.deleted = True 
               SET c.deleted_time = $deleted_time
            """
    conn.query_write(
        query=query,
        parameters={
            "account_id": account_id,
            "column_ids": column_ids,
            "deleted_time": deleted_time,
        },
    )


def delete_column(column_id, account_id, deleted_time=datetime.now()):
    query = """MATCH (c:column {id: $column_id, account_id: $account_id}) 
               SET c.deleted = True 
               SET c.deleted_time = $deleted_time
            """
    conn.query_write(
        query=query,
        parameters={
            "account_id": account_id,
            "column_id": column_id,
            "deleted_time": deleted_time,
        },
    )
