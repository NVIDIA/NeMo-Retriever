from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
import uuid
import pandas as pd
from shared.graph.utils import chunks
from shared.graph.dal.tables_dal import load_sqls_to_tables
from shared.graph.model.reserved_words import SQLType
from sqloxide import parse_sql
from tqdm import tqdm

from .queries_parser import parse_single, dispatch_sqls, pre_process
from shared.graph.services.queries_comparison.queries_comparison import (
    find_identical_queries,
)
from shared.graph.dal.schemas_dal import add_schemas_edge
from shared.graph.dal.queries_dal import (
    add_query,
    get_sql_by_full_query,
    update_counters_and_timestamps_for_query_and_affected_data,
)
from shared.graph.parsers.sql.views_parser import parse as parse_views


from shared.graph.model.query import Query, MissingDataError, UnsupportedQueryError

logger = logging.getLogger("queries_parser.py")
keep_string_values = False


def add_temp_schema_to_graph(account_id: str):
    return lambda schema_name, node_from, node_to: add_schemas_edge(
        (node_from, node_to, {"schema": schema_name}),
        account_id,
        created=None,
    )


def filter_by_query_types(
    query_text: str, dialect: str, allowed_sql_types: list[SQLType]
) -> bool:
    try:
        sql_type = next(iter(parse_sql(pre_process(query_text), dialect)[0]))
    except ValueError:
        # sqloxide error
        return False
    except Exception:
        return False
    if sql_type.lower() in allowed_sql_types:
        return True
    return False


def filter_and_sort_queries_df(queries_df: pd.DataFrame, dialect: str):
    drop_dup_columns = (
        ["schema", "query_text"] if "schema" in queries_df else ["query_text"]
    )
    queries_df = queries_df.drop_duplicates(drop_dup_columns)

    allowed_sql_types = [
        SQLType.QUERY,
        SQLType.INSERT,
        SQLType.CREATE_TABLE,
        SQLType.UPDATE,
        SQLType.MERGE,
    ]
    queries_df["allowed"] = queries_df["query_text"].apply(
        lambda x: filter_by_query_types(x, dialect, allowed_sql_types)
    )
    queries_df = queries_df[queries_df["allowed"]]

    if queries_df.empty:
        logger.info(
            "There are no queries that passed the stage of filtering by query type compatability and sqloxide correctness."
        )
        return [], []

    queries_df["is_create_query"] = queries_df["query_text"].apply(
        lambda x: filter_by_query_types(x, dialect, [SQLType.CREATE_TABLE])
    )
    create_tables_queries = queries_df[queries_df["is_create_query"]]
    create_tables_queries.sort_values(by="end_time", inplace=True)

    queries_df = queries_df[~queries_df["is_create_query"]]
    queries_df.sort_values(by="end_time", inplace=True)

    return create_tables_queries, queries_df


def parse_create_tables_queries(
    account_id: str,
    create_tables_queries: pd.DataFrame,
    schemas: dict,
    dialect: str,
    keep_string_vals: bool,
    sqls_tables_from_graph_df: pd.DataFrame,
):
    if schemas is None:
        raise Exception("Schemas are required")
    global keep_string_values
    keep_string_values = keep_string_vals

    failed_queries: list[dict[str, str]] = []
    parsed_queries: dict[str, Query] = dict()
    found_in_graph_or_in_memory = 0

    # The create queries are not ordered according to the creation order, so it is required to loop over and over as
    # long as there is a chance that some creation query depends on another creation query that has not been parsed yet.
    max_no_change = 2  # len(create_temp_tables_queries)
    no_change_counter = 0
    while len(create_tables_queries) > 0:
        if no_change_counter > max_no_change:
            break

        old_len = len(create_tables_queries)

        failed_queries = []
        failed_indexes, found_queries_counter = parse_queries_df(
            account_id=account_id,
            dialect=dialect,
            failed_queries=failed_queries,
            keep_string_values=keep_string_values,
            parsed_queries=parsed_queries,
            queries_df=create_tables_queries,
            schemas=schemas,
            sqls_tables_from_graph_df=sqls_tables_from_graph_df,
        )
        found_in_graph_or_in_memory += found_queries_counter
        # keep for the next round only the failed create queries
        create_tables_queries = create_tables_queries.filter(
            items=failed_indexes, axis=0
        )

        new_len = len(create_tables_queries)
        if old_len == new_len:
            no_change_counter += 1
        else:
            no_change_counter = 0

    return parsed_queries, failed_queries, found_in_graph_or_in_memory


def parse_not_create_queries(
    account_id: str,
    queries_df: pd.DataFrame,
    schemas: dict,
    dialect: str,
    keep_string_vals: bool,
    sqls_tables_from_graph_df: pd.DataFrame,
):
    if schemas is None:
        raise Exception("Schemas are required")
    global keep_string_values
    keep_string_values = keep_string_vals

    failed_queries: list[dict[str, str]] = []
    parsed_queries: dict[str, Query] = dict()
    found_in_graph_or_in_memory = 0

    _, found_queries_counter = parse_queries_df(
        account_id=account_id,
        dialect=dialect,
        failed_queries=failed_queries,
        keep_string_values=keep_string_values,
        parsed_queries=parsed_queries,
        queries_df=queries_df,
        schemas=schemas,
        sqls_tables_from_graph_df=sqls_tables_from_graph_df,
    )
    found_in_graph_or_in_memory += found_queries_counter

    return parsed_queries, failed_queries, found_in_graph_or_in_memory


def parse_queries_df(
    account_id: str,
    dialect: str,
    failed_queries: list[dict[str, str]],
    keep_string_values: bool,
    parsed_queries: dict[str, Query],
    queries_df: pd.DataFrame,
    schemas: dict,
    sqls_tables_from_graph_df: pd.DataFrame,
):
    found_in_graph_or_in_memory = 0
    # unsupported_query_type = 0
    failed_indexes = []
    for index, row in queries_df.iterrows():
        try:
            sql_id = (
                str(row["id"])
                if "id" in row and not (pd.isna(row["id"]) or not row["id"])
                else str(uuid.uuid4())
            )
            q = pre_process(row["query_text"])
            q_utterance = row["utterance"] if "utterance" in row else None
            q_timestamp = row["end_time"]
            q_count = row["count"] if "count" in row else 1
            q_count = int(q_count) if isinstance(q_count, str) else q_count
            default_schema = (
                row["schema"] if "schema" in row and len(row["schema"]) > 0 else None
            )
            q_tag = row["query_tag"] if "query_tag" in row else None
            query_obj = Query(
                schemas=schemas,
                id=sql_id,
                q=q,
                utterance=q_utterance,
                ltimestamp=q_timestamp,
                count=q_count,
                is_subselect=False,
                default_schema=default_schema,
                dialect=dialect,
                query_tag=q_tag,
            )
            parsed_query = parse_sql(sql=q, dialect=dialect)
            is_parsed = dispatch_sqls(
                parsed_query=parsed_query,
                sql_text=q,
                query_obj=query_obj,
                keep_string_values=keep_string_values,
                schemas=schemas,
                is_full_parse=False,
                add_temp_schema_to_graph=add_temp_schema_to_graph(account_id),
            )
            existing_query_id = get_sql_by_full_query(account_id, q)
            if existing_query_id:
                update_counters_and_timestamps_for_query_and_affected_data(
                    account_id=account_id,
                    identical_sql_id=existing_query_id,
                    sql_node=query_obj.sql_node,
                )
                found_in_graph_or_in_memory += 1
                logger.info(f"found in graph by text. index {index}")
                continue
            if is_parsed:
                (
                    identical_ids_from_graph,
                    identical_ids_in_memory,
                    _,
                    _,
                ) = find_identical_queries(
                    account_id=account_id,
                    main_sql=q,
                    get_parsed_query=lambda sql_query: parse_single(
                        q=sql_query,
                        schemas=schemas,
                        dialects=[dialect],
                        default_schema=query_obj.default_schema,
                        is_full_parse=True,
                        add_temp_schema_to_graph=add_temp_schema_to_graph(account_id),
                    ),
                    sqls_tbls_from_graph_df=sqls_tables_from_graph_df,
                    is_subgraph=False,
                    remove_aliases=False,
                    in_memory_queries=parsed_queries,
                )
                if identical_ids_from_graph:
                    identical_sql_id = identical_ids_from_graph[0]
                    update_counters_and_timestamps_for_query_and_affected_data(
                        account_id=account_id,
                        identical_sql_id=identical_sql_id,
                        sql_node=query_obj.sql_node,
                    )
                    found_in_graph_or_in_memory += 1
                elif identical_ids_in_memory:
                    identical_sql_id = identical_ids_in_memory[0]
                    identical_query_obj = parsed_queries[identical_sql_id]
                    identical_query_obj.increase_sql_counter(query_obj)
                    found_in_graph_or_in_memory += 1
                else:
                    query_obj.sql_node.add_property(
                        "nodes_count", query_obj.get_nodes_counter()
                    )
                    parsed_queries.update({query_obj.id: query_obj})
        except MissingDataError as err:
            logger.info("missing data")
            logger.exception(err)
            # logger.info(f"Failed parsing: {row['query_text']}")
            failed_queries.append(row)
            failed_indexes.append(index)
        except UnsupportedQueryError:
            # unsupported_query_type += 1
            logger.info("Unsupported Query Error")
        except RecursionError as r:
            logger.info(r)
        except Exception as err:
            logger.info("Failed parsing query")
            logger.exception(err)
            # logger.info(f"Failed parsing: {row['query_text']}")
            failed_queries.append(row)
            failed_indexes.append(index)
    return failed_indexes, found_in_graph_or_in_memory


def populate_subset_of_queries(
    schemas,
    queries: pd.DataFrame,
    account_id,
    num_workers,
    dialect,
    keep_string_values,
    sqls_tables_from_graph_df,
    is_create_queries: bool,
):
    before_parsing_queries = time.time()
    logger.info(
        f"Starting to parse {len(queries)} {'create ' if is_create_queries else ''}queries."
    )

    failed_queries_total = []
    chunk_size = len(queries) if is_create_queries else 500
    queries_chunks = list(chunks(queries.to_dict(orient="records"), chunk_size))
    for i, chunk in enumerate(queries_chunks):
        logger.info(f"chunk {i + 1} out of {len(queries_chunks)} chunks")
        if is_create_queries:
            parsed_queries, failed_queries, found_in_graph_or_in_memory = (
                parse_create_tables_queries(
                    account_id=account_id,
                    create_tables_queries=pd.DataFrame(chunk),
                    schemas=schemas,
                    dialect=dialect,
                    keep_string_vals=keep_string_values,
                    sqls_tables_from_graph_df=sqls_tables_from_graph_df,
                )
            )
        else:
            parsed_queries, failed_queries, found_in_graph_or_in_memory = (
                parse_not_create_queries(
                    account_id=account_id,
                    queries_df=pd.DataFrame(chunk),
                    schemas=schemas,
                    dialect=dialect,
                    keep_string_vals=keep_string_values,
                    sqls_tables_from_graph_df=sqls_tables_from_graph_df,
                )
            )

        failed_queries_total += failed_queries
        # after_parse_queries = time.time()
        # logger.info(
        #     f"Time took to parse queries:{after_parse_queries - before_parsing_queries}"
        # )
        #
        # logger.info(
        #     f"Found in graph or in memory (batch) {found_in_graph_or_in_memory} queries."
        # )
        # logger.info(
        #     f"Successfully finished parsing {len(parsed_queries)} queries. Starting to insert into the graph."
        # )

        # logger.info(f"Adding queries to graph using {num_workers} workers")

        with tqdm(
            desc="Total Added Queries",
            total=len(parsed_queries),
            mininterval=10,
            maxinterval=10,
        ) as pbar:
            with ThreadPoolExecutor(num_workers) as executor:
                futures = (
                    executor.submit(
                        add_query,
                        q.get_edges(),
                        account_id,
                    )
                    for q in parsed_queries.values()
                )
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)

    logger.info(
        f"Time took to parse and insert queries to the graph:{time.time() - before_parsing_queries}"
    )
    return failed_queries_total


def populate_queries(
    schemas, queries, account_id, num_workers, dialect, keep_string_values
):
    logger.info(
        "Preparation step: Load from graph all existing SQL nodes with their reached tables."
    )
    sqls_tables_from_graph_df = load_sqls_to_tables(account_id)

    logger.info(f"Starting to parse {len(queries)} queries.")
    create_tables_queries, queries_df = filter_and_sort_queries_df(queries, dialect)

    failed_create_queries = []
    failed_queries = []
    if len(create_tables_queries) > 0:
        failed_create_queries = populate_subset_of_queries(
            schemas,
            create_tables_queries,
            account_id,
            num_workers,
            dialect,
            keep_string_values,
            sqls_tables_from_graph_df,
            is_create_queries=True,
        )
    if len(queries_df) > 0:
        # reload the sqls to tables df, because the ids of temp tables have been updated when parsing the
        # create temp table queries
        sqls_tables_from_graph_df = load_sqls_to_tables(account_id)
        failed_queries = populate_subset_of_queries(
            schemas,
            queries_df,
            account_id,
            num_workers,
            dialect,
            keep_string_values,
            sqls_tables_from_graph_df,
            is_create_queries=False,
        )

    logger.info("Finished inserting the queries into the graph.")
    return failed_create_queries + failed_queries


def populate_views(
    schemas: dict,
    views: list,
    account_id: str,
    num_workers: int,
    dialect: str,
    keep_string_values: bool,
):
    logger.info(
        "Preparation step: Load from graph all existing SQL nodes with their reached tables."
    )
    sqls_tables_df = load_sqls_to_tables(account_id, is_view=True)

    before_parsing_views = time.time()
    logger.info(f"Starting to parse {len(views)} views.")
    parsed_views, failed_views = parse_views(
        account_id, views, schemas, dialect, keep_string_values, sqls_tables_df
    )
    views_queries = parsed_views.values()
    after_parse_views = time.time()

    logger.info(f"Time took to parse views:{after_parse_views - before_parsing_views}")
    logger.info(
        f"Successfully finished parsing {len(views_queries)} views. Starting to insert into the graph."
    )
    logger.info(f"Adding views to graph using {num_workers} workers")

    with tqdm(
        desc="Total Added Views",
        total=len(views_queries),
        mininterval=10,
        maxinterval=10,
    ) as pbar:
        with ThreadPoolExecutor(num_workers) as executor:
            futures = (
                executor.submit(add_query, q.get_edges(), account_id)
                for q in views_queries
            )
            for future in as_completed(futures):
                future.result()
                pbar.update(1)

    logger.info(
        f"Time took to insert views to the graph:{time.time() - after_parse_views}"
    )

    logger.info("Finished inserting the views into the graph.")
    return failed_views
