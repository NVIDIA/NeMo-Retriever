# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
import uuid

import pandas as pd
from tqdm import tqdm

from nemo_retriever.tabular_data.ingestion.utils import chunks
from nemo_retriever.tabular_data.ingestion.dal.queries_dal import (
    add_query,
    get_candidate_sql_ids,
    get_sql_by_full_query,
    load_sqls_to_tables,
    update_counters_and_timestamps_for_query_and_affected_data,
)


from nemo_retriever.tabular_data.ingestion.model.query import Query
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Props
from nemo_retriever.tabular_data.ingestion.parsers.sqlglot_extractor import (
    ExtractionResult,
    extract_tables_and_columns,
)
from nemo_retriever.tabular_data.ingestion.parsers.query_comparator import (
    compare_queries,
    normalize_sql,
)

logger = logging.getLogger(__name__)


def parse_query_slim(sql_text: str, query_obj: Query, dialect: str, schemas: dict) -> bool:
    """Parse a SQL query using sqlglot extraction.

    Identifies referenced tables and columns for all SQL statement types without
    building a full AST.  Updates ``query_obj.tables_ids`` and appends SQL→table
    and SQL→column edges to ``query_obj.edges``.  Also stores the sqlglot AST
    node count on the query as a cheap structural fingerprint.

    Returns True when at least one recognised table was found, False otherwise.
    """
    extraction: ExtractionResult = extract_tables_and_columns(
        sql=sql_text,
        dialect=dialect,
        all_schemas=schemas,
    )

    query_obj.ast_node_count = extraction.ast_node_count
    query_obj.join_count = len(extraction.joins)

    if not extraction.tables:
        return False

    resolved_schemas: dict[str, object] = {}

    for table_key, match in extraction.tables.items():
        bare_name = table_key.split(".")[-1]

        schema = schemas.get(match.schema_name) if match.schema_name else None
        if schema is None:
            for s in schemas.values():
                if s.table_exists(bare_name):
                    schema = s
                    break

        if schema is None:
            logger.debug("Table %r not found in any schema – skipping.", bare_name)
            continue

        resolved_schemas[table_key] = schema

        try:
            table_node = schema.get_table_node(bare_name)
        except Exception:
            logger.debug("Failed to get table node for %r – skipping.", bare_name)
            continue

        query_obj.add_table_to_query(table_node, bare_name)
        edge_props = {Props.SQL_ID: str(query_obj.id)}
        query_obj.edges.append((query_obj.sql_node, table_node, edge_props))

        for col_name in match.columns:
            try:
                if schema.is_column_in_table(table_node, col_name):
                    col_node = schema.get_column_node(col_name, bare_name)
                    query_obj.edges.append((query_obj.sql_node, col_node, edge_props))
            except Exception:
                continue

    sql_id = str(query_obj.id)
    for jp in extraction.joins:
        left_schema = resolved_schemas.get(jp.left_table)
        right_schema = resolved_schemas.get(jp.right_table)
        if left_schema is None or right_schema is None:
            continue
        left_bare = jp.left_table.split(".")[-1]
        right_bare = jp.right_table.split(".")[-1]
        try:
            left_table_node = left_schema.get_table_node(left_bare)
            right_table_node = right_schema.get_table_node(right_bare)
            if not left_schema.is_column_in_table(
                left_table_node, jp.left_column
            ) or not right_schema.is_column_in_table(right_table_node, jp.right_column):
                continue
            left_col_node = left_schema.get_column_node(jp.left_column, left_bare)
            right_col_node = right_schema.get_column_node(jp.right_column, right_bare)
            condition = f"{left_bare}.{jp.left_column} {jp.operator} {right_bare}.{jp.right_column}"
            join_edge_props = {
                Props.JOIN: True,
                "join_refs": [f"{sql_id}|{condition}"],
            }
            query_obj.edges.append((left_col_node, right_col_node, join_edge_props))
        except Exception:
            logger.debug(
                "Failed to create JOIN edge: %s.%s → %s.%s",
                left_bare,
                jp.left_column,
                right_bare,
                jp.right_column,
            )
            continue

    return bool(query_obj.get_tables_ids())


def parse_query_single(
    sql: str,
    dialect: str,
    schemas: dict,
) -> Query | None:
    """Parse a single SQL string and return a populated :class:`Query`, or ``None`` if no
    recognised tables were found."""
    from datetime import datetime

    query_obj = Query(
        schemas=schemas,
        id=str(uuid.uuid4()),
        sql_text=sql,
        ltimestamp=datetime.now(),
        count=1,
        dialect=dialect,
    )
    is_parsed = parse_query_slim(sql_text=sql, query_obj=query_obj, dialect=dialect, schemas=schemas)
    if not is_parsed:
        return None
    query_obj.sql_node.add_property("nodes_count", query_obj.get_nodes_counter())
    query_obj.sql_node.add_property("join_count", query_obj.get_join_count())
    return query_obj


def _try_merge_with_graph(
    query_obj: Query,
    sql_text: str,
    dialect: str,
    sqls_tbls_df: pd.DataFrame,
) -> bool:
    """Check whether an equivalent query already exists in the graph.

    1. Exact text match (cheapest).
    2. Structural comparison against candidates pre-filtered by
       table-set, column-set (leaves), and AST node count — mirroring
       the old ``get_sqls_connected_to_tables`` heuristic.

    Returns ``True`` if a match was found and counters were updated.
    """
    exact_id = get_sql_by_full_query(sql_text)
    if exact_id:
        update_counters_and_timestamps_for_query_and_affected_data(
            identical_sql_id=exact_id,
            sql_node=query_obj.sql_node,
        )
        logger.info("Found existing SQL in graph by full text; updated counters.")
        return True

    candidates = get_candidate_sql_ids(
        tbl_ids=query_obj.get_tables_ids(),
        col_ids=query_obj.get_column_ids(),
        nodes_count=query_obj.get_nodes_counter(),
        sqls_tbls_df=sqls_tbls_df,
        join_count=query_obj.get_join_count(),
    )
    for _, cand in candidates.iterrows():
        cand_sql = cand.get("sql_full_query", "")
        if not cand_sql:
            continue
        if compare_queries(sql_text, cand_sql, dialect=dialect, ignore_literals=True):
            update_counters_and_timestamps_for_query_and_affected_data(
                identical_sql_id=cand["sql_id"],
                sql_node=query_obj.sql_node,
            )
            logger.info(
                "Found structurally equivalent SQL %s in graph; updated counters.",
                cand["sql_id"],
            )
            return True
    return False


def _try_merge_in_memory(
    sql_text: str,
    sql_count: int,
    dialect: str,
    query_obj: Query,
    table_index: dict[frozenset, list[str]],
    parsed_queries: dict[str, Query],
    norm_cache: dict[str, str | None],
) -> bool:
    """Check whether a structurally equivalent query is already accumulated in memory.

    Uses a table-set index (``table_index``) so only queries that share the
    exact same set of table IDs are compared, then gates on matching column
    IDs (leaves) and AST node count before falling through to the cached
    normalised-string comparison.

    Returns ``True`` if a duplicate was found and its counter was bumped.
    """
    tbl_key = frozenset(query_obj.get_tables_ids())
    col_set = frozenset(query_obj.get_column_ids())
    nodes_count = query_obj.get_nodes_counter()
    join_count = query_obj.get_join_count()

    new_norm = norm_cache.get(query_obj.id)
    if new_norm is None:
        new_norm = normalize_sql(sql_text, dialect=dialect, ignore_literals=True)
        norm_cache[query_obj.id] = new_norm
    if new_norm is None:
        return False

    for qid in table_index.get(tbl_key, []):
        existing_q = parsed_queries[qid]
        if existing_q.get_nodes_counter() != nodes_count:
            continue
        if existing_q.get_join_count() != join_count:
            continue
        if frozenset(existing_q.get_column_ids()) != col_set:
            continue

        existing_norm = norm_cache.get(qid)
        if existing_norm is None:
            existing_sql = existing_q.sql_node.get_properties().get("sql_full_query", "")
            existing_norm = normalize_sql(existing_sql, dialect=dialect, ignore_literals=True)
            norm_cache[qid] = existing_norm
        if existing_norm is None:
            continue

        if new_norm == existing_norm:
            props = existing_q.sql_node.get_properties()
            props["total_counter"] = props.get("total_counter", 0) + sql_count
            existing_q.sql_node.add_property("total_counter", props["total_counter"])
            # Carry over the incoming query's per-month counters so a
            # different month from the deduplicated query is not lost.
            for key, val in query_obj.sql_node.get_properties().items():
                if key.startswith("count_"):
                    merged = props.get(key, 0) + val
                    existing_q.sql_node.add_property(key, merged)
            logger.info(
                "Merged structurally equivalent query into in-memory query %s.",
                qid,
            )
            return True
    return False


def parse_queries_df(
    dialect: str,
    parsed_queries: dict[str, Query],
    queries_df: pd.DataFrame,
    schemas: dict,
    sqls_tbls_df: pd.DataFrame | None = None,
) -> list[dict[str, str]]:
    """Parse rows from *queries_df* and accumulate unique queries in *parsed_queries*.

    *parsed_queries* is mutated in-place so that each newly parsed query can
    be cross-checked against the queries already parsed in this batch before
    being compared with what is stored in the graph.

    *sqls_tbls_df* is an optional pre-loaded DataFrame from
    :func:`load_sqls_to_tables` (passed in to avoid re-querying the graph
    for every chunk).
    """
    if sqls_tbls_df is None:
        sqls_tbls_df = load_sqls_to_tables()

    table_index: dict[frozenset, list[str]] = defaultdict(list)
    norm_cache: dict[str, str | None] = {}
    for qid, q in parsed_queries.items():
        table_index[frozenset(q.get_tables_ids())].append(qid)

    failed_queries: list[dict[str, str]] = []
    for _, row in queries_df.iterrows():
        try:
            sql_id = str(uuid.uuid4())
            sql_text = row["query_text"]
            sql_timestamp = row["end_time"]
            sql_count = row["count"] if "count" in row else 1
            sql_count = int(sql_count) if isinstance(sql_count, str) else sql_count
            query_obj = Query(
                schemas=schemas,
                id=sql_id,
                sql_text=sql_text,
                ltimestamp=sql_timestamp,
                count=sql_count,
                dialect=dialect,
            )
            is_parsed = parse_query_slim(
                sql_text=sql_text,
                query_obj=query_obj,
                dialect=dialect,
                schemas=schemas,
            )
            if not is_parsed:
                continue

            if _try_merge_with_graph(query_obj, sql_text, dialect, sqls_tbls_df):
                continue

            if _try_merge_in_memory(
                sql_text,
                sql_count,
                dialect,
                query_obj,
                table_index,
                parsed_queries,
                norm_cache,
            ):
                continue

            query_obj.sql_node.add_property("nodes_count", query_obj.get_nodes_counter())
            query_obj.sql_node.add_property("join_count", query_obj.get_join_count())
            parsed_queries[query_obj.id] = query_obj
            tbl_key = frozenset(query_obj.get_tables_ids())
            table_index[tbl_key].append(query_obj.id)
        except Exception as err:
            logger.info("Failed parsing query")
            logger.exception(err)
            failed_queries.append(row)
    return failed_queries


def populate_queries(schemas, queries_df, num_workers, dialect):
    before = time.time()
    logger.info(f"Starting to parse {len(queries_df)} queries.")

    failed_queries: list[dict[str, str]] = []
    if not queries_df.empty:
        queries_chunks = list(chunks(queries_df.to_dict(orient="records"), 500))
        for i, chunk in enumerate(queries_chunks):
            logger.info(f"chunk {i + 1} out of {len(queries_chunks)} chunks")
            sqls_tbls_df = load_sqls_to_tables()
            parsed_queries: dict[str, Query] = {}
            chunk_failed = parse_queries_df(
                dialect=dialect,
                parsed_queries=parsed_queries,
                queries_df=pd.DataFrame(chunk),
                schemas=schemas,
                sqls_tbls_df=sqls_tbls_df,
            )
            failed_queries += chunk_failed

            with tqdm(
                desc="Total Added Queries",
                total=len(parsed_queries),
                mininterval=10,
                maxinterval=10,
            ) as pbar:
                with ThreadPoolExecutor(num_workers) as executor:
                    futures = (executor.submit(add_query, q.get_edges()) for q in parsed_queries.values())
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception:
                            logger.exception("Failed to persist query to graph.")
                        pbar.update(1)

    logger.info(f"Time took to parse and insert queries: {time.time() - before}")
    logger.info("Finished inserting the queries into the graph.")
    return failed_queries
