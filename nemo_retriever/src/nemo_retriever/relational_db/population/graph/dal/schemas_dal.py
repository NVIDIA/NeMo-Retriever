import pandas as pd
from infra.Neo4jConnection import get_neo4j_conn
from shared.files.utils import load_tables, load_columns
import logging

from shared.graph.model.node import Node
from shared.graph.model.reserved_words import Labels
from shared.graph.model.schema import Schema

conn = get_neo4j_conn()
logger = logging.getLogger("schemas_dal")


def load_schema_from_graph(
    db_name,
    schema_name,
    account_id,
    db_node=None,
    is_temp=False,
    include_deleted: bool = False,
):
    tables_df = get_schema_tables(db_name, schema_name, account_id, include_deleted)
    columns_df = get_schema_columns(db_name, schema_name, account_id, include_deleted)
    if not tables_df.empty and not columns_df.empty:
        tables_df["is_temp"] = is_temp
        columns_df["is_temp"] = is_temp
    else:
        tables_df = None
        columns_df = None

    if db_node is None:
        db_node = Node(name=db_name, label=Labels.DB, props={"name": db_name})

    schema = Schema(account_id, db_node, tables_df, columns_df)
    schema.create_schema_node(schema_name, account_id, is_temp=is_temp)
    return schema


def get_slim_account_schemas(
    account_id: str, relevant_schemas_ids: list = None
) -> list[dict[str, str]]:
    if relevant_schemas_ids is not None and len(relevant_schemas_ids) > 0:
        query = """ MATCH (db:db{account_id:$account_id})-[:schema]->(schema:schema {account_id:$account_id} WHERE schema.id in $relevant_schemas_ids)
                    -[:schema]->(table:table {account_id:$account_id} WHERE coalesce(table.deleted, FALSE)=FALSE)
                    -[:schema]->(column:column {account_id:$account_id} WHERE coalesce(column.deleted, FALSE)=FALSE AND NOT column.db_name IS NULL)
                    RETURN collect({
                    db_name:column.db_name,
                    id:column.id,
                    name:column.name, 
                    table_name:column.table_name,
                    data_type:column.data_type, 
                    schema_name:column.schema_name, 
                    label:labels(column)[0]
                    }) as data
                    """
    else:
        query = """ MATCH (column:column {account_id:$account_id} WHERE coalesce(column.deleted, FALSE)=FALSE AND NOT column.db_name IS NULL)
                    RETURN collect({
                    db_name:column.db_name,
                    id:column.id,
                    name:column.name, 
                    table_name:column.table_name,
                    data_type:column.data_type, 
                    schema_name:column.schema_name, 
                    label:labels(column)[0]
                    }) as data
                    """
    result = conn.query_read_only(
        query=query,
        parameters={
            "account_id": account_id,
            "relevant_schemas_ids": relevant_schemas_ids,
        },
    )
    if len(result) > 0:
        return result[0]["data"]
    return []


def get_schemas_ids_and_names(
    account_id: str, db_id: str = None, include_deleted: bool = False
):
    db_fiter = "id:$db_id" if db_id else ""
    query = f"""MATCH(db:db{{{db_fiter},account_id:$account_id}})-[:schema]->(s:schema|temp_schema)
                {"" if include_deleted else "WHERE coalesce(s.deleted, false) = false"}
                RETURN s.name as schema_name, s.id as schema_id
            """
    result = pd.DataFrame(
        conn.query_read_only(
            query=query,
            parameters={
                "db_id": db_id,
                "account_id": account_id,
                "include_deleted": include_deleted,
            },
        )
    )
    return result.to_dict(orient="records")


def get_table_id_by_schema_and_table_names(
    account_id,
    schema_name,
    table_name,
):
    query = """match(s:schema{account_id:$account_id,name:$schema_name})-[:schema]-(t:table{name:$table_name}) return t.id as table_id"""
    result = conn.query_read_only(
        query=query,
        parameters={
            "account_id": account_id,
            "schema_name": schema_name,
            "table_name": table_name,
        },
    )
    if len(result) > 0:
        return result[0]["table_id"]
    return None


def get_column_id_by_table(
    account_id, schema_name, table_name, column_name, include_deleted: bool = False
):
    query = f"""
    match(s:schema{{account_id:$account_id,name:$schema_name}})-[:schema]-(t:table{{name:$table_name}})-
    [:schema]-(c:column{{name:$column_name}}) 
    {"" if include_deleted else " where coalesce(c.deleted,false)=false "}
    return c.id as column_id"""
    result = conn.query_read_only(
        query=query,
        parameters={
            "account_id": account_id,
            "schema_name": schema_name,
            "table_name": table_name,
            "column_name": column_name,
        },
    )
    if len(result) > 0:
        return result[0]["column_id"]
    return None


def get_schema_columns(db_name, schema_name, account_id, include_deleted: bool = False):
    query = f"""MATCH (d:db{{account_id:$account_id, name:$db_name}})-[:schema]->
                (s:schema{{account_id:$account_id, name:$schema_name}})-[:schema]->
                (t:table{{account_id:$account_id}})-[:schema]->(c:column{{account_id:$account_id}})
                {"" if include_deleted else " WHERE coalesce(s.deleted, false) = false and coalesce(t.deleted, false) = false and coalesce(c.deleted, false) = false "}
                WITH d.name as database, s.name as schema, t.name as table_name,  
                c.name as column_name, c.id as id, c.data_type as data_type, c.ordinal_position as ordinal_position,
                c.is_nullable as is_nullable, c.default as default, c.length as length, c.description as comment,c.scale as scale
                RETURN collect({{database:database, schema:schema, table_name:table_name, column_name:column_name, id:id, data_type:data_type,
                                ordinal_position: ordinal_position, is_nullable:is_nullable, default:default, length:length, comment:comment, 
                                scale:scale }}) as columns
                """
    res = conn.query_read_only(
        query=query,
        parameters={
            "account_id": account_id,
            "db_name": db_name,
            "schema_name": schema_name,
        },
    )
    return load_columns(res[0]["columns"], False)


def get_schema_tables(db_name, schema_name, account_id, include_deleted: bool = False):
    query = f"""MATCH (d:db{{account_id:$account_id, name:$db_name}})-[:schema]->
                (s:schema{{account_id:$account_id, name:$schema_name}})-[:schema]->
                (t:table{{account_id:$account_id}})
                {"" if include_deleted else " WHERE coalesce(s.deleted, false) = false and coalesce(t.deleted, false) = false "}
                WITH d.name as database, s.name as schema, t.name as table_name, t.id as id, 
                t.table_type as table_type, t.row_count as row_count, t.size as size, 
                t.retention_time as retention_time, tostring(t.created) as created,
                tostring(t.last_altered) as last_altered, t.description as comment
                RETURN collect({{database: database, schema: schema, table_name:table_name, id:id, table_type:table_type, row_count:row_count, 
                size:size, retention_time:retention_time, created: created, last_altered: last_altered, comment:comment }}) as tables
                """
    res = conn.query_read_only(
        query=query,
        parameters={
            "account_id": account_id,
            "db_name": db_name,
            "schema_name": schema_name,
        },
    )
    return load_tables(res[0]["tables"], False)


def get_db_ids_and_names(account_id, connection_id=None):
    if connection_id:
        query = """match (c: connection {id: $conn_id, account_id: $account_id})-[:connecting]->(db:db) 
                   return collect({id: db.id, name:db.name}) as dbs"""
    else:
        query = """match (db:db {account_id: $account_id}) 
                   return collect({id: db.id, name:db.name}) as dbs"""
    return conn.query_read_only(
        query=query, parameters={"account_id": account_id, "conn_id": connection_id}
    )[0]["dbs"]


def add_schemas_edge(edge, account_id, created):
    """
    If the nodes do not exist in the Neo4j graph, the function adds them.
    Add to the Neo4j graph the given edge.
    :param edge: edge is a tuple of the form (from_node, to_node, edge_properties)
    :return:
    """
    try:
        node_from = edge[0]
        node_to = edge[1]

        node_from_label = node_from.get_label()
        node_to_label = node_to.get_label()

        # in case of match override the existing ID in the graph, in order to correlate with the ID of the parsed Node object
        query = """
            CALL apoc.merge.node.eager($from_label, $from_identProps, $v_props, {id:$v_props.id})
            yield node as v1
            set v1.created = case when coalesce(v1.deleted, false) = false then coalesce(v1.created, $created) else $created end
            set v1.deleted = false
            with v1
            call apoc.merge.node.eager($to_label, $to_identProps, $u_props, {id:$u_props.id})
            yield node as v2
            set v2.created = case when coalesce(v2.deleted, false) = false then coalesce(v2.created, $created) else $created end
            set v2.deleted = false
            MERGE (v1)-[r:schema]->(v2)
            SET r = $optional_edge_props
            """

        conn.query_write(
            query=query,
            parameters={
                "account_id": account_id,
                "created": created,
                "from_label": [node_from_label],
                "to_label": [node_to_label],
                "from_identProps": node_from.match_props,
                "to_identProps": node_to.match_props,
                "v_props": node_from.get_properties(),
                "u_props": node_to.get_properties(),
                "optional_edge_props": edge[2],
            },
        )
    except Exception as err:
        logger.exception(err)
        raise Exception(f'Error in "add_schemas_edge" when adding edge: {str(edge)}')


def delete_old_fks(account_id, last_seen):
    query = """ OPTIONAL MATCH (:column{account_id:$account_id})-[old_fk:fk]->(:column{account_id:$account_id})
                WHERE old_fk.last_seen<>$last_seen
                DELETE old_fk               
            """
    conn.query_write(
        query=query,
        parameters={
            "account_id": account_id,
            "last_seen": last_seen,
        },
    )


def add_fks(account_id, fks_df, last_seen):
    # pk_database_name	pk_schema_name	pk_table_name	pk_column_name	fk_database_name	fk_schema_name	fk_table_name	fk_column_name
    query = """UNWIND $fks_dict as fkd
               MATCH (t1:table{account_id:$account_id, name: fkd.pk_table_name, schema_name: fkd.pk_schema_name, db_name: fkd.pk_database_name})-[:schema]->(col1:column{account_id:$account_id, name: fkd.pk_column_name})
               MATCH (t2:table{account_id:$account_id, name: fkd.fk_table_name, schema_name: fkd.fk_schema_name, db_name: fkd.fk_database_name})-[:schema]->(col2:column{account_id:$account_id, name: fkd.fk_column_name})
               MERGE (col1)-[:fk {last_seen: $last_seen}]->(col2)"""
    conn.query_write(
        query=query,
        parameters={
            "fks_dict": fks_df.to_dict(orient="records"),
            "account_id": account_id,
            "last_seen": last_seen,
        },
    )


def reset_pks(account_id):
    query = """MATCH (t:table{account_id:$account_id})
               SET t.pk = NULL"""
    conn.query_write(
        query=query,
        parameters={
            "account_id": account_id,
        },
    )


def add_pks(account_id, pks_df):
    # database_name	schema_name	table_name	column_name
    query = """UNWIND $pks_dict as pkd
               MATCH (t:table{account_id:$account_id, name: pkd.table_name, schema_name: pkd.schema_name, db_name: pkd.database_name})-[:schema]->(col:column{account_id:$account_id, name: pkd.column_name})
               SET t.pk = CASE WHEN t.pk is NULL THEN [col.name] ELSE t.pk + [col.name] END"""
    conn.query_write(
        query=query,
        parameters={
            "pks_dict": pks_df.to_dict(orient="records"),
            "account_id": account_id,
        },
    )


def merge_schema_nodes(nodes, account_id, created):
    # in case of match override the existing ID in the graph, in order to correlate with the ID of the parsed Node object
    merge_nodes_query = """ 
                            UNWIND $nodes as node
                            CALL apoc.merge.node.eager(node.label, node.match_props, node.props, {id:node.props.id})
                            yield node as v1
                            set v1.created = case when coalesce(v1.deleted, false) = false then coalesce(v1.created, $created) else $created end
                            set v1.deleted = false
                            set v1.description = coalesce(v1.description, node.props.description)
                        """
    conn.query_write(
        query=merge_nodes_query,
        parameters={"nodes": nodes, "account_id": account_id, "created": created},
    )


def merge_schema_edges(edges, from_label, to_label, account_id):
    merge_edges_query = f"""
                            UNWIND $edges as edge
                            MATCH (v:{from_label} {{account_id: $account_id, id:edge.vid}})
                            MATCH (u:{to_label} {{account_id: $account_id, id:edge.uid}})
                            CALL apoc.merge.relationship(v, "schema", {{}}, edge.props, u, {{}})
                            YIELD rel RETURN rel
                        """
    conn.query_write(
        query=merge_edges_query, parameters={"edges": edges, "account_id": account_id}
    )
