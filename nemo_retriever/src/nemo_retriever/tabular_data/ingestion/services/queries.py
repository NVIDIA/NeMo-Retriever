from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
import uuid
import pandas as pd
from nemo_retriever.tabular_data.ingestion.utils import chunks
from tqdm import tqdm

from nemo_retriever.tabular_data.ingestion.parsers.sql.queries_parser import pre_process
from nemo_retriever.tabular_data.ingestion.dal.queries_dal import add_query


from nemo_retriever.tabular_data.ingestion.model.query import Query
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Props
from nemo_retriever.tabular_data.ingestion.parsers.sqlglot_extractor import extract_tables_and_columns

logger = logging.getLogger(__name__)


def parse_query_slim(q: str, query_obj: Query, dialect: str, schemas: dict) -> bool:
    """Parse a SQL query in slim mode using sqllineage + sqlglot extraction.

    Identifies referenced tables and columns for all SQL statement types without
    building a full AST.  Populates ``query_obj.tables_ids`` and
    ``query_obj.reached_columns_ids``.

    Returns True when at least one recognised table was found, False otherwise.
    """
    tables_and_columns: dict[str, set[str]] = extract_tables_and_columns(
        sql=q,
        dialect=dialect,
        all_schemas=schemas,
    )

    if not tables_and_columns:
        return False

    column_ids: list[str] = []
    for table_name, columns in tables_and_columns.items():
        if table_name == "<unresolved>":
            continue

        table_node = None
        for schema in schemas.values():
            if schema.table_exists(table_name):
                try:
                    table_node = schema.get_table_node(table_name)
                except Exception:
                    continue
                break

        if table_node is None:
            logger.debug("Table %r not found in any schema – skipping.", table_name)
            continue

        query_obj.add_table_to_query(table_node, table_name)
        edge_props = {Props.SQL_ID: str(query_obj.id)}
        query_obj.edges.append((query_obj.sql_node, table_node, edge_props))

        for col_name in columns:
            for schema in schemas.values():
                try:
                    if schema.is_column_in_table(table_node, col_name):
                        col_node = schema.get_column_node(col_name, table_name)
                        column_ids.append(str(col_node.id))
                        query_obj.edges.append((query_obj.sql_node, col_node, edge_props))
                        break
                except Exception:
                    continue

    query_obj.set_reached_columns_ids(column_ids)
    return bool(query_obj.get_tables_ids())


def parse_queries_df(
    dialect: str,
    failed_queries: list[dict[str, str]],
    parsed_queries: dict[str, Query],
    queries_df: pd.DataFrame,
    schemas: dict,
):
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
                default_schema=default_schema,
                dialect=dialect,
                query_tag=q_tag,
            )
            is_parsed = parse_query_slim(
                q=q,
                query_obj=query_obj,
                dialect=dialect,
                schemas=schemas,
            )
            if is_parsed:
                query_obj.sql_node.add_property(
                    "nodes_count", query_obj.get_nodes_counter()
                )
                parsed_queries.update({query_obj.id: query_obj})
        except Exception as err:
            logger.info("Failed parsing query")
            logger.exception(err)
            failed_queries.append(row)
            failed_indexes.append(index)
    return failed_indexes


def populate_queries(
    schemas, queries_df, num_workers, dialect
):
    before = time.time()
    logger.info(f"Starting to parse {len(queries_df)} queries.")

    failed_queries: list[dict[str, str]] = []
    if not queries_df.empty:
        queries_chunks = list(chunks(queries_df.to_dict(orient="records"), 500))
        for i, chunk in enumerate(queries_chunks):
            logger.info(f"chunk {i + 1} out of {len(queries_chunks)} chunks")
            chunk_failed: list[dict[str, str]] = []
            parsed_queries: dict[str, Query] = {}
            parse_queries_df(
                dialect=dialect,
                failed_queries=chunk_failed,
                parsed_queries=parsed_queries,
                queries_df=pd.DataFrame(chunk),
                schemas=schemas,
            )
            failed_queries += chunk_failed

            with tqdm(
                desc="Total Added Queries",
                total=len(parsed_queries),
                mininterval=10,
                maxinterval=10,
            ) as pbar:
                with ThreadPoolExecutor(num_workers) as executor:
                    futures = (
                        executor.submit(add_query, q.get_edges())
                        for q in parsed_queries.values()
                    )
                    for future in as_completed(futures):
                        future.result()
                        pbar.update(1)

    logger.info(f"Time took to parse and insert queries: {time.time() - before}")
    logger.info("Finished inserting the queries into the graph.")
    return failed_queries
