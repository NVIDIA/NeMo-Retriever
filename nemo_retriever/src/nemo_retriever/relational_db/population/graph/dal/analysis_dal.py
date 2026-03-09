from infra.Neo4jConnection import get_neo4j_conn
from notifications.log_dal import produce_signals_bulk
from notifications.types import Events, Actions, NotificationClassification
from shared.graph.services.sql_snippet import parse_analysis
from shared.graph.model.reserved_words import Labels
from shared.graph.dal.usage_dal import get_count_str_by_month
from shared.graph.services.queries_comparison.queries_comparison import (
    find_identical_queries,
)
from shared.graph.dal.tables_dal import load_sqls_to_tables
from shared.graph.parsers.sql.queries_parser import parse_single
from shared.graph.dal.utils_dal import get_analyses_queries

import logging
import datetime
import numpy as np
import pandas as pd
from shared.graph.model.reserved_words import SQLType

conn = get_neo4j_conn()
logger = logging.getLogger("analysis_dal.py")


def is_valid_sql(account_id, schemas, sql: str, dialects: list):
    try:
        parse_single(
            q=sql,
            schemas=schemas,
            dialects=dialects,
            keep_string_vals=False,
            sql_type=SQLType.SEMANTIC,
        )
    except Exception:
        return {"error": "SQL Invalid"}
    try:
        analysis_queries = get_analyses_queries(account_id)
        if len(analysis_queries) > 0:
            (identical_ids_from_graph, _, _, _) = find_identical_queries(
                account_id=account_id,
                main_sql=sql,
                get_parsed_query=lambda sql_query: parse_single(
                    q=sql_query,
                    schemas=schemas,
                    dialects=dialects,
                    is_full_parse=True,
                ),
                is_subgraph=False,
                remove_aliases=False,
                potential_sqls_ids=analysis_queries,
            )
            if len(identical_ids_from_graph) > 0:
                return {"error": "SQL code is already saved as analysis"}
        return {"success": True}
    except Exception as e:
        logger.error(f"Error in is_valid_sql: {str(e)}", exc_info=True)
        return {"error": "Validation failed"}


def filter_new_rec_analysis(account_id, quantile_90):
    query = """ MATCH (an:analysis{account_id:$account_id, recommended: true})- [:analysis_of] -> (s:sql)
                RETURN collect(s.id) as ids
            """
    already_recommended = conn.query_read_only(
        query=query, parameters={"account_id": account_id}
    )[0]["ids"]
    new_rec_analysis = list(set(quantile_90).difference(already_recommended))
    return new_rec_analysis


def recommend_analysis(account_id: str):
    # first step: retrieve 0.1 upper quantile queries by usage
    # second step: return all those who don't have a saves analysis
    preprocess_invalid_analysis(account_id)
    cnt_str = get_count_str_by_month("node")
    query = f"""
            match(node:sql{{account_id:$account_id,is_sub_select:False}}) return collect([node.id, {cnt_str}]) as usages"""
    result = conn.query_read_only(query=query, parameters={"account_id": account_id})[
        0
    ]["usages"]
    # sort the array for quantiles
    positive_result = list(filter(lambda x: x[1] > 0, result))
    if len(positive_result) > 0:
        quantile_90_usages = np.quantile([x[1] for x in positive_result], 0.9)
        quantile_90 = [x for x in positive_result if x[1] >= quantile_90_usages]
        new_recommendations = filter_new_rec_analysis(
            account_id, [x[0] for x in quantile_90]
        )
        query = f"""
                MATCH(node:sql{{account_id:$account_id,is_sub_select:False}}) where node.id in $new_recommendations
                CREATE (an:analysis{{name: "recommended_analysis_" + node.id, account_id: $account_id, recommended: true, id: randomUUID(), sql: node.sql_full_query, label:'analysis'}}) - [:analysis_of] -> (node) 
                RETURN collect({{id:node.id, query_text:node.sql_full_query, usage: {cnt_str}}}) as queries
                """
        result = conn.query_write(
            query=query,
            parameters={
                "account_id": account_id,
                "new_recommendations": new_recommendations,
            },
        )[0]["queries"]

        if len(result) > 0:
            queries_logs = [
                {
                    "id": query["id"],
                    "before_update": None,
                    "payload": {"query": query["query_text"], "usage": query["usage"]},
                }
                for query in result
            ]
            produce_signals_bulk(
                account_id,
                Events.RECOMMENDATION,
                Actions.SAVE,
                Labels.ANALYSIS,
                queries_logs,
                None,
                NotificationClassification.RECOMMENDATION.OPPORTUNITY,
            )


def preprocess_invalid_analysis(account_id):
    query = """MATCH (a:analysis{account_id:$account_id})-[:analysis_of]-(s:sql)
            where coalesce(s.invalid,false)=true and coalesce(a.recommended,false)=true
            detach delete a"""
    conn.query_write(query=query, parameters={"account_id": account_id})


def update_analysis(
    account_id: str,
    id: str,
    user_id: str,
    name: dict,
    description: dict,
    owner_notes: str,
    query: dict,
    owner_id: str = None,
    tags: list = None,
):
    update_query = """ MATCH (n:analysis {id:$analysis_id, account_id:$account_id}) """

    if name:  # check if we update the name, description or query
        if "value" in name:  # check if we change the name or the certification
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
    if query:
        update_query += """
        SET n.certified_query=$certified_query
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
    if owner_notes is not None:
        if owner_notes != "":
            update_query += """SET n.owner_notes=$owner_notes """
        else:
            update_query += """REMOVE n.owner_notes """

    update_query += """
    SET n.last_modified=datetime.realtime()
    SET n.last_modified_by=$user_id

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
    certified_query = query["certified"] if query and "certified" in query else None

    result = conn.query_write(
        query=update_query,
        parameters={
            "account_id": account_id,
            "analysis_id": id,
            "user_id": user_id,
            "tags": tags,
            "name": name_value,
            "certified_name": certified_name,
            "description": description_value,
            "certified_description": certified_description,
            "certified_query": certified_query,
            "owner_notes": owner_notes,
            "owner_id": owner_id,
        },
    )
    return result[0]


def populate_created_analysis(account_id, schemas, dialect, keep_string_values):
    logger.info("Update analysis.")
    logger.info("Update manually created analysis (parse sql snippets).")
    # get from graph_dal all created attributes and their snippets
    created_analysis = get_created_analysis(account_id)
    sqls_tabls_df = load_sqls_to_tables(account_id=account_id)
    # for each attribute
    for analysis in created_analysis:
        parse_analysis(
            account_id=account_id,
            sql=analysis["sql_snippet"],
            schemas=schemas,
            dialects=[dialect],
            analysis_id=analysis["id"],
            sqls_tbls_df=sqls_tabls_df,
        )


def get_created_analysis(account_id):
    query = """MATCH(analysis:analysis{account_id:$account_id} WHERE coalesce(analysis.recommended, FALSE)=FALSE)-[r:analysis_of]->(sql:sql{account_id:$account_id})
    WITH analysis, sql, CASE WHEN sql.sql_type = $sql_type THEN sql.sql_full_query ELSE analysis.sql END as sql_snippet
    RETURN distinct analysis.id as id, analysis.name as name, analysis.owner_id as owner_id, analysis.description as description, 
          analysis.user_id as user_id, sql_snippet as sql_snippet
    """
    result = pd.DataFrame(
        conn.query_read_only(
            query=query,
            parameters={"account_id": account_id, "sql_type": SQLType.SEMANTIC},
        )
    )
    return result.to_dict(orient="records")


def get_analysis_id_by_name(account_id, name):
    query = """MATCH(a:analysis{account_id: $account_id, name: $name})
               RETURN distinct a.id as id
            """

    result = pd.DataFrame(
        conn.query_read_only(
            query=query, parameters={"account_id": account_id, "name": name}
        )
    )
    return result.to_dict(orient="records")


def connect_visual_to_analysis(
    account_id: str,
    visual_id: str,
    analysis_id: str,
    visual_labels: list[Labels],
    parsing_time: datetime,
):
    labels = "|".join(visual_labels)
    query = f"""
            MATCH(visual:{labels}{{account_id: $account_id, id: $visual_id}})
            MATCH(analysis:analysis{{account_id: $account_id, id: $analysis_id}})
            MERGE(visual)-[:analysis_source {{last_read:$last_read}}]->(analysis)
            """
    conn.query_write(
        query=query,
        parameters={
            "account_id": account_id,
            "visual_id": visual_id,
            "analysis_id": analysis_id,
            "last_read": parsing_time,
        },
    )
