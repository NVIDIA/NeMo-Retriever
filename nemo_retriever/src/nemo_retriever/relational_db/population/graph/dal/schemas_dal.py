import pandas as pd
from nemo_retriever.relational_db.neo4j_connection import get_neo4j_conn
from nemo_retriever.relational_db.population.graph.utils import load_tables, load_columns
import logging

from nemo_retriever.relational_db.population.graph.model.node import Node
from nemo_retriever.relational_db.population.graph.model.reserved_words import Labels
from nemo_retriever.relational_db.population.graph.model.schema import Schema

conn = get_neo4j_conn()
logger = logging.getLogger("schemas_dal")


def load_schema_from_graph(
    db_name,
    schema_name,
    db_node=None,
    is_temp=False,
    include_deleted: bool = False,
):
    tables_df = get_schema_tables(db_name, schema_name, include_deleted)
    columns_df = get_schema_columns(db_name, schema_name, include_deleted)
    if not tables_df.empty and not columns_df.empty:
        tables_df["is_temp"] = is_temp
        columns_df["is_temp"] = is_temp
    else:
        tables_df = None
        columns_df = None

    if db_node is None:
        db_node = Node(name=db_name, label=Labels.DB, props={"name": db_name})

    schema = Schema(db_node, tables_df, columns_df)
    schema.create_schema_node(schema_name, is_temp=is_temp)
    return schema




def get_schemas_ids_and_names(
    db_id: str = None, include_deleted: bool = False
):
    db_filter = " {id:$db_id}" if db_id else ""
    query = f"""MATCH(db:Db{db_filter})-[:CONTAINS]->(s:Schema|TempSchema)
                {"" if include_deleted else "WHERE coalesce(s.deleted, false) = false"}
                RETURN s.name as schema_name, s.id as schema_id
            """
    result = pd.DataFrame(
        conn.query_read_only(
            query=query,
            parameters={
                "db_id": db_id,
                "include_deleted": include_deleted,
            },
        )
    )
    return result.to_dict(orient="records")


def get_schema_columns(db_name, schema_name, include_deleted: bool = False):
    # Use c_id alias: "id" is reserved in Cypher
    query = f"""MATCH (d:Db{{name:$db_name}})-[:CONTAINS]->
                (s:Schema{{name:$schema_name}})-[:CONTAINS]->
                (t:Table)-[:CONTAINS]->(c:Column)
                {"" if include_deleted else " WHERE coalesce(s.deleted, false) = false and coalesce(t.deleted, false) = false and coalesce(c.deleted, false) = false "}
                WITH d.name as database, s.name as schema, t.name as table_name,
                c.name as column_name, c.id as c_id, c.data_type as data_type, c.ordinal_position as ordinal_position,
                c.is_nullable as is_nullable, c.default as default, c.length as length, c.description as comment, c.scale as scale
                RETURN collect({{database:database, schema:schema, table_name:table_name, column_name:column_name, id:c_id, data_type:data_type,
                                ordinal_position: ordinal_position, is_nullable:is_nullable, default:default, length:length, comment:comment,
                                scale:scale }}) as columns
                """
    res = conn.query_read_only(
        query=query,
        parameters={
            "db_name": db_name,
            "schema_name": schema_name,
        },
    )
    # Neo4j collect() returns a list; load_columns expects a DataFrame
    return load_columns(pd.DataFrame(res[0]["columns"] if res[0]["columns"] else []))


def get_schema_tables(db_name, schema_name, include_deleted: bool = False):
    # Use t_id alias: "id" is reserved in Cypher
    query = f"""MATCH (d:Db{{name:$db_name}})-[:CONTAINS]->
                (s:Schema{{name:$schema_name}})-[:CONTAINS]->
                (t:Table)
                {"" if include_deleted else " WHERE coalesce(s.deleted, false) = false and coalesce(t.deleted, false) = false "}
                WITH d.name as database, s.name as schema, t.name as table_name, t.id as t_id,
                t.table_type as table_type, t.row_count as row_count, t.size as size,
                t.retention_time as retention_time, tostring(t.created) as created,
                tostring(t.last_altered) as last_altered, t.description as comment
                RETURN collect({{database: database, schema: schema, table_name:table_name, id:t_id, table_type:table_type, row_count:row_count,
                size:size, retention_time:retention_time, created: created, last_altered: last_altered, comment:comment }}) as tables
                """
    res = conn.query_read_only(
        query=query,
        parameters={
            "db_name": db_name,
            "schema_name": schema_name,
        },
    )
    # Neo4j collect() returns a list; load_tables expects a DataFrame
    return load_tables(pd.DataFrame(res[0]["tables"] if res[0]["tables"] else []))


def get_db_ids_and_names(connection_id=None):
    if connection_id:
        query = """match (c:Connection {id: $conn_id})-[:CONNECTING]->(db:Db) 
                   return collect({id: db.id, name:db.name}) as dbs"""
    else:
        query = """match (db:Db) 
                   return collect({id: db.id, name:db.name}) as dbs"""
    return conn.query_read_only(
        query=query, parameters={"conn_id": connection_id}
    )[0]["dbs"]


def add_schemas_edge(edge, created):
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
            MERGE (v1)-[r:CONTAINS]->(v2)
            SET r = $optional_edge_props
            """

        conn.query_write(
            query=query,
            parameters={
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


def delete_old_fks(last_seen):
    query = """ OPTIONAL MATCH (:Column)-[old_fk:FOREIGN_KEY]->(:Column)
                WHERE old_fk.last_seen<>$last_seen
                DELETE old_fk
            """
    conn.query_write(
        query=query,
        parameters={"last_seen": last_seen},
    )


def add_fks(fks_df, last_seen):
    # pk_database_name	pk_schema_name	pk_table_name	pk_column_name	fk_database_name	fk_schema_name	fk_table_name	fk_column_name
    query = """UNWIND $fks_dict as fkd
               MATCH (t1:Table{name: fkd.pk_table_name, schema_name: fkd.pk_schema_name, db_name: fkd.pk_database_name})-[:CONTAINS]->(col1:Column{name: fkd.pk_column_name})
               MATCH (t2:Table{name: fkd.fk_table_name, schema_name: fkd.fk_schema_name, db_name: fkd.fk_database_name})-[:CONTAINS]->(col2:Column{name: fkd.fk_column_name})
               MERGE (col1)-[:FOREIGN_KEY {last_seen: $last_seen}]->(col2)"""
    conn.query_write(
        query=query,
        parameters={
            "fks_dict": fks_df.to_dict(orient="records"),
            "last_seen": last_seen,
        },
    )


def reset_pks():
    query = """MATCH (t:Table)
               SET t.pk = NULL"""
    conn.query_write(query=query, parameters={})


def add_pks(pks_df):
    # database_name	schema_name	table_name	column_name
    query = """UNWIND $pks_dict as pkd
               MATCH (t:Table{name: pkd.table_name, schema_name: pkd.schema_name, db_name: pkd.database_name})-[:CONTAINS]->(col:Column{name: pkd.column_name})
               SET t.pk = CASE WHEN t.pk is NULL THEN [col.name] ELSE t.pk + [col.name] END"""
    conn.query_write(
        query=query,
        parameters={"pks_dict": pks_df.to_dict(orient="records")},
    )


def merge_schema_nodes(nodes, created):
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
        parameters={"nodes": nodes, "created": created},
    )


def merge_schema_edges(edges, from_label, to_label):
    merge_edges_query = f"""
                            UNWIND $edges as edge
                            MATCH (v:{from_label} {{id:edge.vid}})
                            MATCH (u:{to_label} {{id:edge.uid}})
                            CALL apoc.merge.relationship(v, "CONTAINS", {{}}, edge.props, u, {{}})
                            YIELD rel RETURN rel
                        """
    conn.query_write(
        query=merge_edges_query, parameters={"edges": edges}
    )
