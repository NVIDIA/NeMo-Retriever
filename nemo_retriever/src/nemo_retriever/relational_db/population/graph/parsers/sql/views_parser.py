import logging
import uuid
from datetime import datetime, timezone
import codecs

import pandas as pd
from sqloxide import parse_sql

from shared.graph.model.query import Query
from shared.graph.model.reserved_words import SQLType
from shared.graph.parsers.sql import sql_view_parser
from shared.graph.dal.queries_dal import (
    get_sql_by_full_query,
    update_counters_and_timestamps_for_query_and_affected_data,
)
from shared.graph.services.queries_comparison.queries_comparison import (
    find_identical_queries,
)
from shared.graph.parsers.sql.queries_parser import parse_single

logger = logging.getLogger("queries_parser.py")
keep_string_values = False


def parse(
    account_id: str,
    views_df: pd.DataFrame,
    schemas: dict,
    dialect: str,
    keep_string_vals: bool,
    sqls_tables_from_graph_df: pd.DataFrame,
):
    if schemas is None:
        raise Exception("A schema is required")
    global keep_string_values
    keep_string_values = keep_string_vals

    parsed_views: dict[str, Query] = dict()
    failed_views: list[dict[str, str]] = []
    # database	schema	name	view_definition
    for index, row in views_df.iterrows():
        try:
            with_no_binding = False
            # in the demo we need the query id from the csv
            sql_id = (
                row["id"] if ("id" in row and not pd.isna(row["id"])) else uuid.uuid4()
            )
            q = codecs.decode(row["view_definition"], "unicode-escape")
            if q.lower().find("with no schema binding".lower()) != -1:
                q = q.lower()
                q = q.replace("with no schema binding", "")
                with_no_binding = True
            # MSSQl
            if q.lower().find("with schemabinding".lower()) != -1:
                q = q.lower()
                q = q.replace("with schemabinding", "")

            # replacing double qoatation with single qoatation causes the sqloxide to
            # identify $path as constant instead of identifier of a column
            q = q.replace('"$path"', "'$path'")

            # Teva: U&'val' is an unsescaped string that sqlparser don't know how to handle.
            q = q.replace("U&'", "'")
            query_obj = Query(
                schemas,
                sql_id,
                q,
                utterance=None,
                ltimestamp=datetime.now(timezone.utc).replace(microsecond=0),
                count=0,
                is_subselect=False,
                default_schema=row["schema"],
                dialect=dialect,
            )
            existing_query_id = get_sql_by_full_query(account_id, q)
            if existing_query_id:
                update_counters_and_timestamps_for_query_and_affected_data(
                    account_id=account_id,
                    identical_sql_id=existing_query_id,
                    sql_node=query_obj.sql_node,
                )
                continue
            parsed_query = parse_sql(sql=q, dialect=dialect)
            query_obj.set_sql_type(SQLType.VIEW)
            is_parsed = sql_view_parser.build_query_obj(
                schema_name=row["schema"],
                table_name=row["name"],
                with_no_binding=with_no_binding,
                query_obj=query_obj,
                parsed_query=parsed_query,
                keep_string_vals=keep_string_vals,
                is_full_parse=False,
            )

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
                        default_schema=row["schema"],
                        is_full_parse=True,
                    ),
                    sqls_tbls_from_graph_df=sqls_tables_from_graph_df,
                    is_subgraph=False,
                    remove_aliases=False,
                    in_memory_queries=parsed_views,
                )
                if identical_ids_from_graph:
                    # There is no need to update the last_query_timestamp for the tables and columns used in the view.
                    # Do nothing.
                    identical_sql_id = identical_ids_from_graph[0]
                    update_counters_and_timestamps_for_query_and_affected_data(
                        account_id=account_id,
                        identical_sql_id=identical_sql_id,
                        sql_node=query_obj.sql_node,
                        update_data_last_query_timestamp=False,
                    )
                elif identical_ids_in_memory:
                    # There is no need to update the counters for views.
                    pass
                else:
                    query_obj.sql_node.add_property(
                        "nodes_count", query_obj.get_nodes_counter()
                    )
                    parsed_views.update({query_obj.id: query_obj})
        except Exception as err:
            logger.exception(err)
            # logger.info(f"Failed parsing: {row['view_definition']}")
            logger.info("Failed parsing query - views parser")
            failed_views.append(row)

    return parsed_views, failed_views
