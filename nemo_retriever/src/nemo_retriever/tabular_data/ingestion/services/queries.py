from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
import uuid
import pandas as pd
from nemo_retriever.tabular_data.ingestion.utils import chunks
from tqdm import tqdm

from nemo_retriever.tabular_data.ingestion.dal.queries_dal import add_query


from nemo_retriever.tabular_data.ingestion.model.query import Query
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Props
from nemo_retriever.tabular_data.ingestion.parsers.sqlglot_extractor import (
    TableMatch,
    extract_tables_and_columns,
)

logger = logging.getLogger(__name__)


def parse_query_slim(q: str, query_obj: Query, dialect: str, schemas: dict) -> bool:
    """Parse a SQL query using sqlglot extraction.

    Identifies referenced tables and columns for all SQL statement types without
    building a full AST.  Populates ``query_obj.tables_ids`` and
    ``query_obj.reached_columns_ids``.

    Returns True when at least one recognised table was found, False otherwise.
    """
    table_matches: dict[str, TableMatch] = extract_tables_and_columns(
        sql=q,
        dialect=dialect,
        all_schemas=schemas,
    )

    if not table_matches:
        return False

    column_ids: list[str] = []
    for table_key, match in table_matches.items():
        # table_key may be "schema.table" or just "table"; bare name is always the last part.
        bare_name = table_key.split(".")[-1]

        # Use the schema identified by the extractor directly; fall back to scanning
        # all schemas when the owning schema could not be determined.
        schema = schemas.get(match.schema_name) if match.schema_name else None
        if schema is None:
            for s in schemas.values():
                if s.table_exists(bare_name):
                    schema = s
                    break

        if schema is None:
            logger.debug("Table %r not found in any schema – skipping.", bare_name)
            continue

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
                    column_ids.append(str(col_node.id))
                    query_obj.edges.append((query_obj.sql_node, col_node, edge_props))
            except Exception:
                continue

    query_obj.set_reached_columns_ids(column_ids)
    return bool(query_obj.get_tables_ids())


def parse_queries_df(
    dialect: str,
    parsed_queries: dict[str, Query],
    queries_df: pd.DataFrame,
    schemas: dict,
) -> list[dict[str, str]]:
    failed_queries: list[dict[str, str]] = []
    for _, row in queries_df.iterrows():
        try:
            sql_id = str(uuid.uuid4())
            q = row["query_text"]
            q_timestamp = row["end_time"]
            q_count = row["count"] if "count" in row else 1
            q_count = int(q_count) if isinstance(q_count, str) else q_count
            q_tag = row["query_tag"] if "query_tag" in row else None
            query_obj = Query(
                schemas=schemas,
                id=sql_id,
                q=q,
                ltimestamp=q_timestamp,
                count=q_count,
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
    return failed_queries


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
            parsed_queries: dict[str, Query] = {}
            chunk_failed = parse_queries_df(
                dialect=dialect,
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
