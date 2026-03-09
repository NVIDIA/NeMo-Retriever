import json
from shared.graph.parsers.metrics_parser import parse_single_metric
from shared.graph.model.reserved_words import Labels, SQLType
from shared.graph.model.metric import Metric
from shared.graph.model.metric_to_sql import (
    build_sqls_from_metric,
    build_sql_from_bi_fields,
)
from shared.graph.dal.metrics_dal import (
    add_metric_to_graph,
    get_metrics_with_props,
    delete_metric_subgraph,
    get_metric_sqls,
    detach_metric_queries,
    add_usage_to_metric,
    get_queries_for_metric_usage,
    connect_metric_to_bi_fields,
    add_edges_to_queries,
)
from shared.graph.dal.tables_dal import get_tables_queried_today
from shared.graph.services.queries_comparison.queries_comparison import (
    find_identical_queries,
)
from shared.graph.parsers.sql.queries_parser import parse_single

# from shared.graph.parsers.metrics_parser import (
#     find_common_dimensions_and_filters,
# )
from shared.graph.dal.kpi_recommendation_dal import get_kpi_recommendation

# from shared.graph.services.metric_similarity.embedding_structure_similarity import (
#     calculate_embedding_structure_similarity,
# )
import logging
import uuid
from notifications.log_dal import produce_signal
from notifications.types import Events, Actions, NotificationClassification
from pandas import DataFrame

logger = logging.getLogger("metrics.py")


def process_metric(
    account_id,
    schemas,
    name,
    id,
    description,
    formula,
    user_id=None,
    owner_id=None,
    owner_notes=None,
    recommended=False,
    population=False,
    dialects=["postgres", "snowflake", "generic", "ansi"],
    is_existing_metric: bool = True,
    creating_from_bi_fields: list[str] = None,
    snippet_id: str = None,
):
    id = str(id)
    sqls_before_reparse = []

    if is_existing_metric:
        sqls_before_reparse = get_metric_sqls(account_id, id)
        delete_metric_subgraph(account_id, id)

    metric_obj = Metric(
        name,
        id,
        account_id,
        description,
        formula,
        owner_id,
        user_id,
        owner_notes,
        recommended,
        is_bi=bool(creating_from_bi_fields),
    )
    logger.info(f"Parsing metric id:{id} formula:{formula} name:{name}")
    add_metric_obj_to_graph(
        metric_obj,
        account_id,
        creating_from_bi_fields=creating_from_bi_fields,
        population=population,
    )
    logger.info("Building SQLs from metric graph.")
    if creating_from_bi_fields:
        build_sql_from_bi_fields(
            account_id, metric_obj, creating_from_bi_fields, snippet_id
        )
        connect_metric_to_bi_fields(account_id, id, creating_from_bi_fields)
        return
    metrics_sqls = build_sqls_from_metric(
        account_id, metric_obj, from_bi_field=creating_from_bi_fields
    )
    if population and not recommended:
        new_sqls, sqls_unchanged, queried_today = get_metric_sqls_for_usage(
            account_id, metrics_sqls, sqls_before_reparse
        )
        if is_existing_metric:
            existing_sqls = get_key_from_dicts(sqls_unchanged, "sql")
            detach_metric_queries(account_id, str(id), sqls_to_keep=existing_sqls)
        for metric_sql in new_sqls + queried_today:
            try:
                sql = metric_sql["sql"]
                logger.info(f"Compare sql {sql} to queries in graph")
                parse_single(
                    q=sql,
                    schemas=schemas,
                    dialects=dialects,
                    keep_string_vals=False,
                    sql_type=SQLType.SEMANTIC,
                )
                tables_filter = (
                    metric_sql["tables_ids"]
                    if "tables_used_today" not in metric_sql
                    or len(metric_sql["tables_used_today"]) == 0
                    else metric_sql["tables_used_today"]
                )
                queries_to_compare = get_queries_for_metric_usage(
                    account_id, id, sql, tables_filter
                )
                logger.info(f"Checking {metric_obj.name} occurrences in graph.")
                (
                    identical_ids_from_graph,
                    _,
                    subset_ids_from_graph,
                    _,
                ) = find_identical_queries(
                    account_id=account_id,
                    main_sql=sql,
                    get_parsed_query=lambda sql_query: parse_single(
                        q=sql_query,
                        schemas=schemas,
                        dialects=dialects,
                        is_full_parse=True,
                    ),
                    is_subgraph=True,
                    remove_aliases=True,
                    potential_sqls_ids=queries_to_compare,
                )
                if identical_ids_from_graph:
                    add_edges_to_queries(
                        id,
                        identical_ids_from_graph,
                        "identical",
                        account_id,
                        sql,
                    )
                if subset_ids_from_graph:
                    add_edges_to_queries(
                        id, subset_ids_from_graph, "subset", account_id, sql
                    )
            except Exception as e:
                logger.error(
                    f"""Error in retrieving metric queries comparison for metric id:{id} \nname:{name}
                    \nsql:{metric_sql["sql"]},\n error:{str(e)}"""
                )
                continue
        add_usage_to_metric(account_id, id)
    return metric_obj, metrics_sqls


def get_key_from_dicts(dicts: list[dict[str, str]], key: str) -> list[str]:
    return [dict[key] for dict in dicts]


## we check which metric sqls are new and need to be searched in the graph
## we also check for existing sqls- if their tables were queried today, we know to look for
## new queries for those tables too
def get_metric_sqls_for_usage(
    account_id: str,
    metric_sqls: list[dict[str, str]],
    sqls_before_reparse: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    sqls = get_key_from_dicts(metric_sqls, "sql")
    prev_sqls = get_key_from_dicts(sqls_before_reparse, "sql")
    sqls_added = list(set(sqls) - set(prev_sqls))
    sqls_removed = list(set(prev_sqls) - set(sqls))
    new_sqls = [
        metric_sql for metric_sql in metric_sqls if metric_sql["sql"] in sqls_added
    ]
    unchanged_sqls = [
        metric_sql
        for metric_sql in metric_sqls
        if metric_sql not in new_sqls and metric_sql not in sqls_removed
    ]
    queried_today = []
    for metric_sql in unchanged_sqls:
        tables_queried_today = get_tables_queried_today(
            account_id, metric_sql["tables_ids"]
        )
        if len(tables_queried_today) > 0:
            metric_sql["tables_used_today"] = tables_queried_today
            queried_today.append(metric_sql)
    return new_sqls, unchanged_sqls, queried_today


##TODO: make this function great again
# def find_similar_metrics(account_id, query_metric_obj):
# sqls_tbls_df = load_sqls_to_tables(account_id)xw
# edges = query_metric_obj.get_edges().copy()
# tbl_ids, nodes_count = get_table_ids_and_nodes_count(edges)
# identical_id = find_query_in_graph(
#     edges,
#     account_id,
#     tbl_ids,
#     nodes_count,
#     sqls_tbls_df,
#     query_metric_obj,
#     skip_source_of=True,
# )

# if (
#     identical_id is not None and len(identical_id) > 0
# ):  # if metric sql is already in graph - retrieve its id
#     query_metric_obj.id = identical_id
# elif identical_id is None or len(identical_id) == 0:
#     add_query(edges, account_id)

# most_similar = calculate_embedding_structure_similarity(
#     account_id, query_metric_obj
# )
# if (
#     identical_id is None or len(identical_id) == 0
# ):  # metrics sql was added to graph, remove
#     delete_query_by_id(account_id, str(query_metric_obj.id))
# return most_similar


def reparse_metrics_for_usage(account_id, schemas, population):
    logger.info("Update metrics usage.")
    metrics = get_metrics_with_props(account_id, recommended=False)
    for metric in metrics:
        try:
            process_metric(
                account_id,
                schemas,
                name=metric["name"],
                id=metric["id"],
                description=metric["description"],
                formula=metric["formula"],
                owner_id=metric["owner_id"],
                recommended=metric["recommended"],
                population=population,
                is_existing_metric=True,
            )
        except Exception as e:
            logger.error(e)
            continue


def reparse_recommended_metrics_to_delete(account_id, schemas):
    logger.info("Update recommended metrics.")
    all_recommended_metrics = get_metrics_with_props(account_id, recommended=True)
    valid_recommended_metrics = []
    for metric in all_recommended_metrics:
        try:
            process_metric(
                account_id,
                schemas,
                name=metric["name"],
                id=metric["id"],
                description=metric["description"],
                formula=metric["formula"],
                owner_id=metric["owner_id"],
                recommended=metric["recommended"],
                population=True,
                is_existing_metric=True,
            )
            valid_recommended_metrics.append(metric)
        except Exception as e:
            logger.error(e)
            continue

    return valid_recommended_metrics


def check_exist_recommendations(kpi_recommendations: DataFrame, exist_rec_metrics):
    exist_metrics_formula = [x["formula"] for x in exist_rec_metrics]
    kpi_recommendations["exist"] = kpi_recommendations["recommended_kpi"].apply(
        lambda x: True if x.strip() in exist_metrics_formula else False
    )
    new_recommendations = kpi_recommendations.loc[kpi_recommendations["exist"].notna()]
    return new_recommendations


def recommend_kpis(account_id, schemas):
    logger.info(
        "Reparsing recommended metrics to delete invalid (term / attribute changed)."
    )
    valid_recommended_metrics = reparse_recommended_metrics_to_delete(
        account_id, schemas
    )

    logger.info("Recommend kpis. ")
    kpi_recommendations = get_kpi_recommendation(account_id)
    if len(kpi_recommendations) > 0:
        kpi_recommendations = check_exist_recommendations(
            kpi_recommendations, valid_recommended_metrics
        )

    logger.info(f"Adding {len(kpi_recommendations)} recommended kpis to graph.")
    recommended_metric_log = None
    for index, metric in kpi_recommendations.iterrows():
        try:
            id = str(uuid.uuid4())
            process_metric(
                account_id,
                schemas,
                name=f"recommended_kpi_{index}",
                id=id,
                description=json.dumps({"from_sql": metric["main_sql_text"]}),
                formula=metric["recommended_kpi"],
                owner_id=None,
                recommended=True,
                population=True,
                is_existing_metric=False,
            )
            recommended_metric_log = {
                "id": id,
                "payload": {"formula": metric["recommended_kpi"]},
            }
        except Exception as e:
            logger.error(metric.formula_to_kpi)
            logger.error(e)
            continue
        # we only send one event per population
    if recommended_metric_log is not None:
        produce_signal(
            account_id,
            Events.RECOMMENDATION,
            Actions.SAVE,
            None,
            Labels.METRIC,
            recommended_metric_log["id"],
            None,
            recommended_metric_log,
            NotificationClassification.RECOMMENDATION.OPPORTUNITY,
        )


def add_metric_obj_to_graph(
    metric_obj: Metric,
    account_id,
    creating_from_bi_fields: list[str] = None,
    population=False,
):
    parse_single_metric(
        account_id,
        metric_obj,
        from_bi_fields=creating_from_bi_fields,
        population=population,
    )
    logger.info(f"Adding {metric_obj.name} to graph.")
    add_metric_to_graph(account_id, metric_obj)
