# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import logging
from nemo_retriever.tabular_data.neo4j import get_neo4j_conn
from nemo_retriever.tabular_data.ingestion.utils import (
    normalize_tables,
    normalize_columns,
)
from nemo_retriever.tabular_data.ingestion.model.neo4j_node import Neo4jNode
from nemo_retriever.tabular_data.ingestion.model.reserved_words import (
    Edges,
    Labels,
    RelTypes,
)
from nemo_retriever.tabular_data.ingestion.model.schema import Schema

logger = logging.getLogger(__name__)

conn = get_neo4j_conn()


def get_db_ids_and_names():
    query = """ match (db:db) 
                return collect({id: db.id, name:db.name}) as dbs"""
    return conn.query_read(
        query=query, parameters={}
    )[0]["dbs"]



def load_schema_from_graph(
    db_name,
    schema_name,
    db_node=None,
):
    tables_df = get_schema_tables(db_name, schema_name)
    columns_df = get_schema_columns(db_name, schema_name)
    if tables_df.empty or columns_df.empty:
        tables_df = None
        columns_df = None

    if db_node is None:
        db_node = Neo4jNode(name=db_name, label=Labels.DB, props={"name": db_name})

    schema = Schema(db_node, tables_df, columns_df)
    schema.create_schema_node(schema_name)
    return schema


def get_schemas_ids_and_names(db_id: str = None):
    db_filter = " {id:$db_id}" if db_id else ""
    query = f"""MATCH(db:{Labels.DB}{db_filter})-[:{Edges.CONTAINS}]->(s:{Labels.SCHEMA})
                RETURN s.name as schema_name, s.id as schema_id
            """
    result = pd.DataFrame(
        get_neo4j_conn().query_read(
            query=query,
            parameters={"db_id": db_id},
        )
    )
    return result.to_dict(orient="records")


def get_schema_columns(db_name, schema_name):
    # Use c_id alias: "id" is reserved in Cypher
    query = f"""MATCH (d:{Labels.DB}{{name:$db_name}})-[:{Edges.CONTAINS}]->
                (s:{Labels.SCHEMA}{{name:$schema_name}})-[:{Edges.CONTAINS}]->
                (t:{Labels.TABLE})-[:{Edges.CONTAINS}]->(c:{Labels.COLUMN})
                WITH d.name as database,
                s.name as schema,
                t.name as table_name,
                c.name as column_name,
                c.id as c_id,
                c.data_type as data_type,
                c.is_nullable as is_nullable
                RETURN collect({{
                    database: database,
                    schema: schema,
                    table_name: table_name,
                    column_name: column_name,
                    id: c_id,
                    data_type: data_type,
                    is_nullable: is_nullable
                }}) as columns
                """
    res = get_neo4j_conn().query_read(
        query=query,
        parameters={
            "db_name": db_name,
            "schema_name": schema_name,
        },
    )
    # Neo4j collect() returns a list; normalize_columns expects a DataFrame
    return normalize_columns(pd.DataFrame(res[0]["columns"] if res[0]["columns"] else []))


def get_schema_tables(db_name, schema_name):
    # Use t_id alias: "id" is reserved in Cypher
    query = f"""MATCH (d:{Labels.DB}{{name:$db_name}})-[:{Edges.CONTAINS}]->
                (s:{Labels.SCHEMA}{{name:$schema_name}})-[:{Edges.CONTAINS}]->
                (t:{Labels.TABLE})
                WITH d.name as database, s.name as schema, t.name as table_name, t.id as t_id,
                tostring(t.created) as created, t.description as description
                RETURN collect({{
                    database: database, schema: schema, table_name: table_name,
                    id:t_id, created: created, description: description
                }}) as tables
                """
    res = get_neo4j_conn().query_read(
        query=query,
        parameters={
            "db_name": db_name,
            "schema_name": schema_name,
        },
    )
    # Neo4j collect() returns a list; normalize_tables expects a DataFrame
    return normalize_tables(pd.DataFrame(res[0]["tables"] if res[0]["tables"] else []))


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

        # in case of match, override the existing ID in the graph
        # to correlate with the ID of the parsed Neo4jNode object
        query = f"""
            CALL apoc.merge.node.eager($from_label, $from_identProps, $v_props, {{id:$v_props.id}})
            yield node as v1
            set v1.created = coalesce(v1.created, $created)
            with v1
            call apoc.merge.node.eager($to_label, $to_identProps, $u_props, {{id:$u_props.id}})
            yield node as v2
            set v2.created = coalesce(v2.created, $created)
            MERGE (v1)-[r:{Edges.CONTAINS}]->(v2)
            SET r = $optional_edge_props
            """

        get_neo4j_conn().query_write(
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
    query = f""" OPTIONAL MATCH (:{Labels.COLUMN})-[old_fk:{Edges.FOREIGN_KEY}]->(:{Labels.COLUMN})
                WHERE old_fk.last_seen<>$last_seen
                DELETE old_fk
            """
    get_neo4j_conn().query_write(
        query=query,
        parameters={"last_seen": last_seen},
    )


def add_fks(fks_df, last_seen):
    # pk_database_name, pk_schema_name, pk_table_name, pk_column_name
    # fk_database_name, fk_schema_name, fk_table_name, fk_column_name
    query = f"""UNWIND $fks_dict as fkd
               MATCH (t1:{Labels.TABLE}{{
                   name: fkd.pk_table_name, schema_name: fkd.pk_schema_name,
                   db_name: fkd.pk_database_name
               }})-[:{Edges.CONTAINS}]->(col1:{Labels.COLUMN}{{name: fkd.pk_column_name}})
               MATCH (t2:{Labels.TABLE}{{
                   name: fkd.fk_table_name, schema_name: fkd.fk_schema_name,
                   db_name: fkd.fk_database_name
               }})-[:{RelTypes.CONTAINS}]->(col2:{Labels.COLUMN}{{name: fkd.fk_column_name}})
               MERGE (col1)-[:{RelTypes.FOREIGN_KEY} {{last_seen: $last_seen}}]->(col2)"""
    get_neo4j_conn().query_write(
        query=query,
        parameters={
            "fks_dict": fks_df.to_dict(orient="records"),
            "last_seen": last_seen,
        },
    )


def reset_pks():
    query = f"""MATCH (t:{Labels.TABLE})
               SET t.pk = NULL"""
    get_neo4j_conn().query_write(query=query, parameters={})


def add_pks(pks_df):
    # database_name, schema_name, table_name, column_name
    query = f"""UNWIND $pks_dict as pkd
               MATCH (t:{Labels.TABLE}{{
                   name: pkd.table_name, schema_name: pkd.schema_name,
                   db_name: pkd.database_name
               }})-[:{Edges.CONTAINS}]->(col:{Labels.COLUMN}{{name: pkd.column_name}})
               SET t.pk = CASE WHEN t.pk is NULL THEN [col.name] ELSE t.pk + [col.name] END"""
    get_neo4j_conn().query_write(
        query=query,
        parameters={"pks_dict": pks_df.to_dict(orient="records")},
    )


def merge_schema_nodes(nodes, created):
    # in case of match, override the existing ID in the graph
    # to correlate with the ID of the parsed Neo4jNode object
    merge_nodes_query = """
                            UNWIND $nodes as node
                            CALL apoc.merge.node.eager(node.label, node.match_props, node.props, {id:node.props.id})
                            yield node as v1
                            set v1.created = coalesce(v1.created, $created)
                            set v1.description = coalesce(v1.description, node.props.description)
                        """
    get_neo4j_conn().query_write(
        query=merge_nodes_query,
        parameters={"nodes": nodes, "created": created},
    )


def merge_schema_edges(edges, from_label, to_label):
    merge_edges_query = f"""
                            UNWIND $edges as edge
                            MATCH (v:{from_label} {{id:edge.vid}})
                            MATCH (u:{to_label} {{id:edge.uid}})
                            CALL apoc.merge.relationship(v, "{Edges.CONTAINS}", {{}}, edge.props, u, {{}})
                            YIELD rel RETURN rel
                        """
    get_neo4j_conn().query_write(query=merge_edges_query, parameters={"edges": edges})
