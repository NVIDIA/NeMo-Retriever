from datetime import datetime
import logging

import pandas as pd
from nemo_retriever.relational_db.neo4j_connection import get_neo4j_conn

logger = logging.getLogger(__name__)

from nemo_retriever.relational_db.population.graph.dal.utils_dal import get_entity_before_update
from nemo_retriever.relational_db.population.graph.utils import chunks
from nemo_retriever.relational_db.population.graph.model.reserved_words import Labels, label_to_type
from nemo_retriever.relational_db.population.graph.dal.schemas_dal import load_schema_from_graph, add_schemas_edge

from nemo_retriever.relational_db.population.graph.model.schema import TEMP_SCHEMA_NAME
conn = get_neo4j_conn()


def db_exists(db_node):
    db_name = db_node.get_name()
    query = """
    MATCH (n:Db{name: $db_name})
    OPTIONAL MATCH (n)-[r]-(v) WHERE NOT v:Connection
    RETURN n.id AS id, count(r) AS nbrs
    """
    result_data = conn.query_read_only(
        query=query, parameters={"db_name": db_name}
    )
    if not result_data or len(result_data) == 0:
        return None, None

    nbrs = result_data[0]["nbrs"]

    return result_data[0]["id"], False if nbrs == 0 else True


def update_node_property(label, node_id, update_properties):
    query = f"""
            match(n:{label}{{id:$node_id}})
            set n += $update_properties
            """
    conn.query_write(
        query=query,
        parameters={
            "node_id": node_id,
            "update_properties": update_properties,
        },
    )


def delete_schema(schema_node_id, deleted_time=datetime.now()):
    query = """MATCH (n:Schema {id: $schema_node_id})-[:CONTAINS]->(t:Table)-[:CONTAINS]->(c:Column)
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
            "schema_node_id": schema_node_id,
            "deleted_time": deleted_time,
        },
    )






def add_schemas_edge_batch(edges, created):
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
            MERGE (v1)-[r:CONTAINS]->(v2)
            SET r = e.optional_edge_props
            """

        conn.query_write(
            query=query,
            parameters={
                "created": created,
                "edges": edges,
            },
        )
    except Exception as err:
        raise Exception('Error in "add_schemas_edge_batch"')


def accumulate_deleted_column_props(deleted_column, list_for_event_log):
    list_for_event_log.append(
        {
            "id": deleted_column.props_graph["id"],
            "before_update": get_entity_before_update(
                deleted_column.props_graph["id"], Labels.COLUMN
            ),
            "payload": None,
        }
    )


def accumulate_added_column_props(
    added_column, list_for_event_log, edges_to_add, new_schema
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
    table_in_intersection, items_to_update_in_graph, new_schema
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
    column_in_intersection, items_to_update_in_graph, new_schema
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


def update_diff_from_existing_schema(new_schema, latest_timestamp):
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
            db_name, schema_name, is_temp=is_temp
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
                    latest_timestamp,
                )

        tables_names_to_delete = set(existing_table_names) - set(new_table_names)
        logger.info(f"Tables to delete in schema {schema_name}: {len(tables_names_to_delete)}")
        for deleted_table_name in tables_names_to_delete:
            deleted_table_node_props = existing_schema.get_table_node_props(
                deleted_table_name
            )
            delete_table(deleted_table_node_props["id"])
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
                    x, items_to_update_in_graph, new_schema
                ),
                axis=1,
            )
            items_to_update_in_graph_chunks = list(
                chunks(items_to_update_in_graph, 1000)
            )
            len_chunks = len(items_to_update_in_graph_chunks)
            for i, chunk in enumerate(items_to_update_in_graph_chunks):
                logger.info(f"Updating tables chunk {i + 1}/{len_chunks}")
                update_properties_in_graph_batch(chunk)

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
            lambda x: accumulate_deleted_column_props(x, entities), axis=1
        )
        delete_columns_batch([x["id"] for x in entities])
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
                x, entities, edges_to_merge, new_schema
            ),
            axis=1,
        )
        add_schemas_edge_batch(edges_to_merge, created=latest_timestamp)
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
                    x, items_to_update_in_graph, new_schema
                ),
                axis=1,
            )
            items_to_update_in_graph_chunks = list(
                chunks(items_to_update_in_graph, 1000)
            )
            len_chunks = len(items_to_update_in_graph_chunks)
            for i, chunk in enumerate(items_to_update_in_graph_chunks):
                logger.info(f"Updating columns chunk {i + 1}/{len_chunks}")
                update_properties_in_graph_batch(chunk)

        return added_or_modified_tables
    except Exception as err:
        raise Exception(f'Error in "update_diff_from_existing_schema": {err}')


def get_tables_columns(db_id, schema):
    if db_id is None:
        query = """MATCH(db:Db)-[:CONTAINS]->(s:Schema{name:$schema})-[:CONTAINS]->
                    (t:Table)-[:CONTAINS]->(c:Column)
                    WHERE coalesce(s.deleted, false) = false and coalesce(t.deleted, false) = false and
                        coalesce(c.deleted, false) = false 
                    RETURN t.name as table_name, c.name as col_name 
                """
    else:
        query = """MATCH(db:Db{id:$db_id})-[:CONTAINS]->(s:Schema{name:$schema})-[:CONTAINS]->
                    (t:Table)-[:CONTAINS]->(c:Column)
                    WHERE coalesce(s.deleted, false) = false and coalesce(t.deleted, false) = false and
                        coalesce(c.deleted, false) = false 
                    RETURN t.name as table_name, c.name as col_name 
                    """
    result = pd.DataFrame(
        conn.query_read_only(
            query=query,
            parameters={"db_id": db_id, "schema": schema},
        )
    )
    result = result.groupby("table_name")["col_name"].apply(list).reset_index()
    result["col_length"] = result["col_name"].apply(lambda x: len(x))
    return result




def delete_table(table_id, deleted_time=datetime.now()):
    query = """ MATCH (n:Table {id: $table_id})-[:CONTAINS]->(c:Column)
                SET n.deleted = True
                SET n.deleted_time = $deleted_time
                SET c.deleted = True
                SET c.deleted_time = $deleted_time
            """
    conn.query_write(
        query=query,
        parameters={
            "table_id": table_id,
            "deleted_time": deleted_time,
        },
    )


def update_properties_in_graph_batch(items):
    query = """
            UNWIND $items as item
            WITH item, item.props.description as new_description, 
            apoc.map.removeKeys(item.props, ["description"]) as item_props_no_description 
            CALL apoc.merge.node.eager([item.label], {id: item.id}, {}, item_props_no_description) 
            YIELD node
            // keep existing description unless it is null
            SET node.description = coalesce(node.description, new_description)  
            """
    conn.query_write(
        query=query,
        parameters={"items": items},
    )


def update_properties_in_graph(item_id, node_label, new_parameters):
    query = f""" MATCH (item:{node_label}{{id:$table_id}})
                 SET item += $new_parameters
            """
    conn.query_write(
        query=query,
        parameters={
            "table_id": item_id,
            "new_parameters": new_parameters,
        },
    )





def delete_columns_batch(column_ids, deleted_time=datetime.now()):
    query = """UNWIND $column_ids as column_id
               MATCH (c:Column {id: column_id}) 
               SET c.deleted = True 
               SET c.deleted_time = $deleted_time
            """
    conn.query_write(
        query=query,
        parameters={
            "column_ids": column_ids,
            "deleted_time": deleted_time,
        },
    )


def delete_column(column_id, deleted_time=datetime.now()):
    query = """MATCH (c:Column {id: $column_id}) 
               SET c.deleted = True 
               SET c.deleted_time = $deleted_time
            """
    conn.query_write(
        query=query,
        parameters={
            "column_id": column_id,
            "deleted_time": deleted_time,
        },
    )
