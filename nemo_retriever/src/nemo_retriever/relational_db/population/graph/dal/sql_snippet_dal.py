import uuid
from typing import Literal
from shared.graph.model.reserved_words import SQLType, Labels
from infra.Neo4jConnection import get_neo4j_conn

conn = get_neo4j_conn()


def create_reaching_edges(account_id: str, attribute_id: str, snippet_id: str = None):
    query = f"""
    MATCH(attribute:attribute{{account_id:$account_id,id:$attribute_id}})-[snippet:attr_of]->(connected_data:alias|sql)
    {"WHERE snippet.sql_snippet_id = $snippet_id" if snippet_id else ""}
    UNWIND snippet.reached_columns as column_id
    CALL (attribute, column_id, snippet){{
        MATCH(column:column{{account_id:$account_id, id:column_id}})
        MERGE(attribute)-[reaching:reaching{{sql_snippet_id:snippet.sql_snippet_id}}]->(column)
        RETURN reaching
    }}
    RETURN collect(properties(reaching)) as reaching_edges
    """
    res = conn.query_write(
        query,
        parameters={
            "attribute_id": attribute_id,
            "account_id": account_id,
            "snippet_id": snippet_id,
        },
    )
    return res


def delete_snippet(
    account_id: str,
    attr_id: str,
    snippet_id: str,
    user_id: str = None,
    delete_reaching_edges: bool = True,
):
    count_query = """
        MATCH(term:term{account_id:$account_id})-[:term_of]->(attribute:attribute{id:$attr_id,account_id:$account_id})
        MATCH(attribute)-[r:attr_of]->(item)
        RETURN count(r) as snippet_count
        """
    count_result = conn.query_read_only(
        query=count_query,
        parameters={
            "account_id": account_id,
            "attr_id": attr_id,
        },
    )

    if len(count_result) == 0 or count_result[0]["snippet_count"] <= 1:
        raise Exception(
            "An attribute must contain at least one snippet. The last snippet cannot be removed."
        )

    update_last_modified = ""
    if user_id is not None:
        update_last_modified = "SET attribute.last_modified = datetime.realtime(), attribute.last_modified_by = $user_id"
    delete_reaching = "|reaching" if delete_reaching_edges else ""
    query = f"""
        MATCH(term:term{{account_id:$account_id}})-[:term_of]->(attribute:attribute{{id:$attr_id,account_id:$account_id}})
        MATCH(attribute)-[r:attr_of{delete_reaching}{{sql_snippet_id:$snippet_id}}]->(item)
        CALL apoc.do.when(
        // if the snippet is a custom sql then delete the custom sql subgraph as well
            item:sql AND item.sql_type = $sql_type AND apoc.node.degree.in(item) = 1,
            'with item ,r
            CALL apoc.path.subgraphNodes(item, {{
                relationshipFilter: ">SQL",
                labelFilter: "-column|-table",
                filterStartNode: true, 
                minLevel: 0}})
            YIELD node 
            DETACH DELETE node ','', {{item:item, r:r}})
        yield value
        DELETE r
        {update_last_modified}
        RETURN attribute.id as id, term.id as term_id
        """
    result = conn.query_write(
        query=query,
        parameters={
            "account_id": account_id,
            "attr_id": attr_id,
            "user_id": user_id,
            "snippet_id": snippet_id,
            "sql_type": SQLType.SEMANTIC,
        },
    )
    if len(result) > 0:
        return result[0]["id"], result[0]["term_id"]
    else:
        raise Exception("no snippet found for id " + snippet_id)


def delete_non_existing_analysis(account_id, nodes_to_connect):
    query = """
            match (sql:sql{account_id:$account_id})<-[r:analysis_of]-(a:analysis)
            where sql.id in $nodes_to_connect and  coalesce(r.non_existing_analysis, false) = true 
            with sql, a
            CALL apoc.path.subgraphNodes(sql, {
                relationshipFilter: ">SQL",
                labelFilter: "-column|-table",
                filterStartNode: true, 
                minLevel: 0})
            YIELD node 
            DETACH DELETE node
            WITH a
            MATCH (sql_ex:sql{account_id:$account_id}) WHERE  sql_ex.id in $nodes_to_connect
            MERGE (a)-[:analysis_of]->(sql_ex)  // change pointer to existing queries, 
            // reconnect other analyses that point to the same synthetic snippet.
            // how can different analyses point to the same snippet? answer: if they are different subsets of the same snippet
            """
    res = conn.query_write(
        query,
        parameters={
            "nodes_to_connect": nodes_to_connect,
            "account_id": account_id,
        },
    )
    return res


def replace_definition_sql(account_id, attr_id, snippet_id, source_clauses):
    query = """
            match (attr:attribute{id:$attr_id, account_id:$account_id})-[r:attr_of {sql_snippet_id:$snippet_id}]->()
            SET r.source_expr = $source_clauses.source_select, 
            r.from_source_ids = $source_clauses.from_source_ids,
            r.from_source = $source_clauses.from_source,
            r.definition_sql_snippet = $source_clauses.definition_sql                
            """
    conn.query_write(
        query,
        parameters={
            "account_id": account_id,
            "attr_id": attr_id,
            "snippet_id": snippet_id,
            "source_clauses": source_clauses,
        },
    )


def save_custom_snippet_in_graph(
    account_id: str,
    attr_id: str,
    snippet_sql: str,
    snippet_data_type: str,
    mode: Literal["single_column", "existing_sqls", "synthetic_sql"],
    nodes_to_connect: list[dict],
    reached_columns: list[str],
    sql_clauses: dict = None,
    definition_sql_clauses: dict = None,
    snippet_id: str = None,
    user_id: str = None,
):
    if snippet_id:
        ## in existing sqls mode, we don't want to delete the reaching edges because they don't change- we just connect the attribute to the existing sqls
        ## instead of the synthetic one (but it's still connected to the same columns)
        ## in any other case we assume that the sql changed and therefore we want to connect to the new columns and delete the existing reaching edges
        delete_reaching = mode != "existing_sqls"
        delete_snippet(
            account_id=account_id,
            attr_id=attr_id,
            snippet_id=snippet_id,
            user_id=user_id,
            delete_reaching_edges=delete_reaching,
        )
    if mode == "single_column":
        query = f"""MATCH(attribute:{Labels.ATTR}{{id:$attr_id, account_id:$account_id}})
                    MATCH(item:{Labels.COLUMN}{{id:$column_id, account_id:$account_id}})
                    WITH attribute, item
                    MERGE(attribute)-[r:attr_of]-(item) 
                    SET r.source_expr = $source_clauses.source_select, 
                    r.from_source_ids = $source_clauses.from_source_ids,
                    r.from_source = $source_clauses.from_source,
                    r.definition_sql_snippet = $source_clauses.definition_sql,
                    r.snippet_select = $sql_clauses.select,
                    r.snippet_from = $sql_clauses.from,
                    r.snippet_where = $sql_clauses.where,
                    r.snippet_select_tc = $sql_clauses.select_tc,
                    r.data_type = $snippet_data_type,
                    r.reached_columns = [$column_id],
                    r.sql_snippet = $snippet_sql
                    {", r.sql_snippet_id = randomUUID()" if not snippet_id else "r.sql_snippet_id = $snippet_id"}
                    RETURN r.sql_snippet_id as snippet_id"""
        response = conn.query_write(
            query,
            parameters={
                "attr_id": attr_id,
                "column_id": nodes_to_connect[0]["node_id"],
                "account_id": account_id,
                "source_clauses": definition_sql_clauses,
                "snippet_data_type": snippet_data_type,
                "snippet_id": snippet_id,
                "snippet_sql": snippet_sql,
                "sql_clauses": sql_clauses,
            },
        )
        snippet_id = response[0]["snippet_id"]
        return snippet_id
    elif mode == "synthetic_sql":
        query = f""" 
                    MATCH(sql:{Labels.SQL}{{id:$sql_id, account_id:$account_id}})
                    MATCH(attribute:{Labels.ATTR}{{id:$attr_id, account_id:$account_id}})
                    WITH attribute, sql
                    MERGE(attribute)-[r:attr_of]->(sql) 
                    SET r.source_expr = $source_clauses.source_select, 
                    r.sql_snippet = $snippet_sql,
                    r.from_source_ids = $source_clauses.from_source_ids,
                    r.from_source = $source_clauses.from_source,
                    r.definition_sql_snippet = $source_clauses.definition_sql,
                    r.snippet_select = $sql_clauses.select,
                    r.snippet_from = $sql_clauses.from,
                    r.snippet_where = $sql_clauses.where,
                    r.snippet_select_tc = $sql_clauses.select_tc,
                    r.reached_columns = $reached_columns,
                    r.data_type = $snippet_data_type
                    {", r.sql_snippet_id = randomUUID()" if not snippet_id else "r.sql_snippet_id = $snippet_id"}
                    RETURN r.sql_snippet_id as snippet_id"""
        response = conn.query_write(
            query,
            parameters={
                "attr_id": attr_id,
                "sql_id": nodes_to_connect[0]["node_id"],
                "account_id": account_id,
                "source_clauses": definition_sql_clauses,
                "sql_clauses": sql_clauses,
                "reached_columns": reached_columns,
                "snippet_data_type": snippet_data_type,
                "snippet_id": snippet_id,
                "snippet_sql": snippet_sql,
            },
        )
        snippet_id = response[0]["snippet_id"]
        create_reaching_edges(
            account_id=account_id, attribute_id=attr_id, snippet_id=snippet_id
        )
        return snippet_id
    elif mode == "existing_sqls":
        query = f"""
                MATCH(attribute:{Labels.ATTR}{{id:$attribute_id, account_id:$account_id}})
                UNWIND $nodes_to_connect as node_to_connect
                MATCH(sql:{Labels.SQL}{{id:node_to_connect.sql_id, account_id:$account_id}})
                WITH attribute, sql
                MERGE(attribute)-[r:attr_of]->(sql) 
                SET r.source_expr = $source_clauses.source_select, 
                r.sql_snippet = $snippet_sql,
                r.from_source_ids = $source_clauses.from_source_ids,
                r.from_source = $source_clauses.from_source,
                r.definition_sql_snippet = $source_clauses.definition_sql,
                r.snippet_select = $sql_clauses.select,
                r.snippet_from = $sql_clauses.from,
                r.snippet_where = $sql_clauses.where,
                r.snippet_select_tc = $sql_clauses.select_tc,
                r.reached_columns = $reached_columns,
                r.data_type = $snippet_data_type,
                r.sql_snippet_id = $sql_snippet_idå
                RETURN r.sql_snippet_id as snippet_id
                """
        response = conn.query_write(
            query,
            parameters={
                "attribute_id": attr_id,
                "account_id": account_id,
                "source_clauses": definition_sql_clauses,
                "sql_clauses": sql_clauses,
                "snippet_data_type": snippet_data_type,
                "sql_snippet_id": snippet_id,
                "reached_columns": reached_columns,
                "sql_type": SQLType.SEMANTIC,
                "nodes_to_connect": nodes_to_connect,
                "snippet_sql": snippet_sql,
            },
        )
        snippet_id = response[0]["snippet_id"]
        return snippet_id


def delete_semantic_sqls_without_parents(account_id: str):
    query = """
            MATCH(sql:sql{account_id:$account_id, sql_type:$sql_type})
            // A semantic sql that has no parents
            WHERE apoc.node.degree.in(sql) = 0
            WITH sql
            CALL apoc.path.subgraphNodes(sql, {
                relationshipFilter:">SQL", 
                labelFilter:"-column|-table|-temp_column|-temp_table", 
                filterStartNode:true, 
            minLevel:0})
            YIELD node
            DETACH DELETE node
            """
    conn.query_write(
        query, parameters={"account_id": account_id, "sql_type": SQLType.SEMANTIC}
    )


def save_analysis_in_graph(
    account_id: str,
    sqls_ids: list[dict],
    analysis_id: str = None,
    name: str = None,
    sql: str = None,
    description: str = None,
    owner_id: str = None,
    recommended: bool = False,
    user_id: str = None,
    reached_columns: list[str] = None,
):
    if analysis_id:
        # connect to queries existing in the graph
        query = """
        MATCH(analysis:analysis{id:$analysis_id, account_id:$account_id})
        SET analysis.reached_columns=$reached_columns
        WITH analysis
        UNWIND $sqls_ids as sql_id
        MATCH(sql:sql{id:sql_id, account_id:$account_id})
        WITH analysis, sql
        MERGE(analysis)-[:analysis_of]->(sql)
        """
        conn.query_write(
            query,
            parameters={
                "analysis_id": analysis_id,
                "account_id": account_id,
                "sqls_ids": sqls_ids,
                "reached_columns": reached_columns,
            },
        )
        # after connecting to queries, delete the synthetic (sql_type="semantic") one
        query = """
                MATCH(analysis:analysis{id:$analysis_id, account_id:$account_id})-[:analysis_of]->(sql:sql{account_id:$account_id})
                CALL apoc.do.when(sql.sql_type=$sql_type,
                    'CALL apoc.path.subgraphNodes(sql, {relationshipFilter:">SQL", labelFilter:"-column|-table|-temp_column|-temp_table", filterStartNode:true, minLevel:0})
                    YIELD node
                    DETACH DELETE node',
                    '',
                    {sql:sql})
                YIELD value
                RETURN True
                """
        conn.query_write(
            query,
            parameters={
                "analysis_id": analysis_id,
                "account_id": account_id,
                "sql_type": SQLType.SEMANTIC,
            },
        )
    else:
        analysis_id = str(uuid.uuid4())
        query = """
        MATCH(sql:sql{account_id:$account_id,id:$sql_id})
        CREATE(analysis:analysis{id:$analysis_id, account_id:$account_id, name:$name, description:$description, owner_id:$owner_id, sql:$sql, recommended:$recommended, created_by:$user_id, type:$type, created_date:datetime.realtime(), reached_columns:$reached_columns })
        MERGE(analysis)-[:analysis_of]->(sql)
        """
        conn.query_write(
            query,
            parameters={
                "analysis_id": analysis_id,
                "account_id": account_id,
                "name": name,
                "description": description,
                "owner_id": owner_id,
                "sql": sql,
                "sql_id": sqls_ids[0],
                "recommended": recommended,
                "user_id": user_id,
                "type": Labels.ANALYSIS,
                "reached_columns": reached_columns,
            },
        )
    return analysis_id


UPSTREAM_SOURCE_CYPHER_STR = """
                OPTIONAL MATCH (c)<-[:source_of]-(src)
                CALL (c, src){
                    CALL apoc.when(src is null, 'return [] as from_sources, [] as upstream_path',
                        // this call is to find all column-source pairs for the replacement of columns with their definition 
                        'CALL (c){
                            CALL apoc.path.expandConfig(c, { 
                                relationshipFilter:"SQL>|<source_of", 
                                labelFilter:">alias|>column|>set_op_column",
                                filterStartNode: true,
                                minLevel: 0}
                            ) YIELD path as p
                            WITH reduce(output = [], r in relationships(p)|output+[case when type(r)="source_of" then 
                            startNode(r) end]) as most_upstream_src, relationships(p) as upstream_path
                            ORDER BY length(p) DESC
                            LIMIT 1
                            RETURN [val in most_upstream_src WHERE val is not null][-1] as most_upstream_src,
                            reduce(output = [], r in upstream_path|output+[case when type(r)="source_of" then 
                            {src: case when startNode(r):alias then startNode(r).aliased_expr 
                            when startNode(r):column then startNode(r).schema_name +"."+startNode(r).table_name+"."+startNode(r).name 
                            else startNode(r).name end,
                            column: endNode(r).schema_name +"."+endNode(r).table_name+"."+endNode(r).name} end]) as upstream_path
                        }
                        // this call is to find the tables for further definition SQL construction (the from clause)
                        CALL apoc.path.subgraphNodes(most_upstream_src, {
                            relationshipFilter: "SQL>",
                            labelFilter: "/column",
                            filterStartNode: true,
                            minLevel: 0})
                        YIELD node 
                        MATCH(node)<-[:schema]-(table:table)
                        RETURN distinct c.schema_name + "." + c.table_name + "." + c.name as sch_tab_col_name, 
                        c.table_name + "." + c.name as tab_col_name, c.name as col_name, c.id as id, 
                        collect(distinct {id: table.id, name: table.schema_name + "." + table.name}) as from_sources,
                        [val in upstream_path WHERE val is not null] as upstream_path', 
                        {c:c})
                    YIELD value
                    RETURN value.from_sources as from_sources, value.upstream_path as upstream_path
                }
            """


def get_upstream_source(account_id, columns_ids):
    query = """UNWIND $columns_ids as c_id
               MATCH (c:column{account_id:$account_id, id:c_id})<-[:schema]-(c_table:table{account_id:$account_id})"""
    query += UPSTREAM_SOURCE_CYPHER_STR
    query += """
             RETURN distinct upstream_path, from_sources
             """
    res = conn.query_write(
        query,
        parameters={
            "account_id": account_id,
            "columns_ids": columns_ids,
        },
    )
    return res


def find_potential_snippet_root(account_id, tables_to_find):
    query = """
            MATCH (sql:sql {account_id:$account_id, is_sub_select:false})-[:SQL*1..2]->(table:table{account_id:$account_id} WHERE table.id in $find_tbls)
            WHERE coalesce(sql.invalid, false) = false and sql.sql_type in $sql_types
            WITH sql, collect(table.id) as tables_ids
            WITH sql WHERE all(t_id IN $find_tbls WHERE t_id IN tables_ids)
            RETURN sql.id as node_id
            """
    roots = conn.query_read_only(
        query,
        parameters={
            "find_tbls": tables_to_find,
            "account_id": account_id,
            "sql_types": [SQLType.QUERY, SQLType.SEMANTIC],
        },
    )
    return roots
