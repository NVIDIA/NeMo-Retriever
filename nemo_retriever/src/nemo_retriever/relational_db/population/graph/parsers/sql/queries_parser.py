import logging
import uuid
import datetime
import codecs
from sqloxide import parse_sql


from shared.graph.model.node import Node
from shared.graph.model.query import (
    Query,
    NotSelectSqlTypeError,
    NotValidSyntaxError,
    UnsupportedQueryError,
)
from shared.graph.model.reserved_words import SQLType

from shared.graph.parsers.sql import (
    sql_insert_into_parser,
    sql_update_table_parser,
    sql_merge_table_parser,
    sql_create_table_parser,
    sql_create_temp_table_parser,
    sql_select_parser,
    sql_view_parser,
)
from typing import Callable

logger = logging.getLogger("queries_parser.py")
keep_string_values = False


def pre_process(q):
    # fmt: off
    # # q = q.replace("\"", "'").replace("! =", "!=")
    # q = q.replace("  ", " ")
    # q = ' '.join(w for w in q.split())
    q = codecs.decode(q, "unicode-escape")
    q = q.replace('\"', '"')
    q = q.replace('INSERT OVERWRITE INTO', 'INSERT INTO')
    q = q.replace('MERGE INTO IDENTIFIER(', 'MERGE INTO (')

    # Teva: U&'val' is an unsescaped string that sqlparser don't know how to handle.
    q = q.replace("U&'", "'")

    # fmt: on
    return q


def dispatch_sqls(
    parsed_query: list[dict[str, str]],
    sql_text: str,
    query_obj: Query,
    keep_string_values: bool,
    schemas: dict,
    add_temp_schema_to_graph: Callable[[str, Node], None] = None,
    sql_type: SQLType = None,
    allow_only_select: bool = False,
    fks: list[dict[str, str]] = None,
    is_full_parse: bool = False,
):
    if not sql_type:
        sql_type = next(iter(parsed_query[0]))

    if allow_only_select:
        if sql_type.lower() not in [SQLType.QUERY, SQLType.SEMANTIC]:
            raise NotSelectSqlTypeError("Query type is not supported.")

    query_obj.set_sql_type(sql_type)
    if sql_type.lower() in [SQLType.QUERY, SQLType.SEMANTIC]:
        sql_select_parser.build_query_obj(
            query_obj=query_obj,
            parsed_query=parsed_query[0]["Query"],
            keep_string_vals=keep_string_values,
            fks=fks,
            is_full_parse=is_full_parse,
        )
    elif sql_type.lower() == SQLType.INSERT and sql_text.lower().find("select") != -1:
        sql_insert_into_parser.build_query_obj(
            query_obj=query_obj,
            parsed_query=parsed_query,
            keep_string_vals=keep_string_values,
            is_full_parse=is_full_parse,
        )
    elif sql_type.lower() == SQLType.CREATE_TABLE and (
        sql_text.lower().find("select") != -1
        or sql_text.lower().find(" clone ") != -1
        and add_temp_schema_to_graph is not None
    ):
        return sql_create_table_parser.build_query_obj(
            query_obj=query_obj,
            parsed_query=parsed_query,
            keep_string_vals=keep_string_values,
            schemas=schemas,
            is_full_parse=is_full_parse,
            add_temp_schema_to_graph=add_temp_schema_to_graph,
        )
    elif (
        sql_type.lower() == SQLType.CREATE_TABLE
        and add_temp_schema_to_graph is not None
    ):
        is_temporary_without_data = parsed_query[0]["CreateTable"]["temporary"]
        if is_temporary_without_data:
            return sql_create_temp_table_parser.build_query_obj(
                query_obj=query_obj,
                parsed_query=parsed_query,
                keep_string_vals=keep_string_values,
                schemas=schemas,
                is_full_parse=is_full_parse,
                add_temp_schema_to_graph=add_temp_schema_to_graph,
            )
        else:
            raise UnsupportedQueryError("Unsupported query type.")
    elif sql_type.lower() == SQLType.UPDATE and (
        sql_text.lower().find("select") != -1 or sql_text.lower().find("from") != -1
    ):
        sql_update_table_parser.build_query_obj(
            query_obj=query_obj,
            parsed_query=parsed_query,
            keep_string_vals=keep_string_values,
            is_full_parse=is_full_parse,
        )
    elif sql_type.lower() == SQLType.MERGE and (
        sql_text.lower().find("select") != -1 or sql_text.lower().find("using")
    ):
        sql_merge_table_parser.build_query_obj(
            query_obj=query_obj,
            parsed_query=parsed_query,
            keep_string_vals=keep_string_values,
            is_full_parse=is_full_parse,
        )
    elif sql_type.lower() == "createview":
        sql_view_parser.build_query_obj(
            schema_name=query_obj.default_schema,
            table_name=None,
            with_no_binding=False,
            query_obj=query_obj,
            parsed_query=parsed_query,
            keep_string_vals=keep_string_values,
            is_full_parse=is_full_parse,
        )
    elif sql_type.lower() == SQLType.DELETE:
        raise UnsupportedQueryError("Delete query type is not supported")
        # raise Exception(f"Delete query type is not supported. The query: {sql_text}")
    else:
        raise UnsupportedQueryError("Unsupported query type.")
        # raise Exception(f"Unsupported query type. The query: {sql_text}")
    return True


# Schema is used for spider
def parse_single(
    q: str,
    schemas: dict,
    dialects: list[str],
    keep_string_vals: bool = False,
    sql_type: SQLType = None,
    allow_only_select: bool = False,
    default_schema: str = None,
    fks=None,
    is_full_parse=False,
    add_temp_schema_to_graph: Callable[[str, Node], None] = None,
):
    global keep_string_values
    keep_string_values = keep_string_vals
    sql_id = uuid.uuid4()
    for i, dialect in enumerate(dialects):
        try:
            q = pre_process(q)
            parsed_query = parse_sql(sql=q, dialect=dialect)
            # if parsed successfully, break out of the loop
            break
        except Exception:
            is_last_dialect = i == len(dialects) - 1
            if is_last_dialect:
                # if all possible dialects didn't work, then raise exception
                raise NotValidSyntaxError(
                    f"Invalid query syntax or unsupported dialect. Supported dialects: {','.join(dialects)}"
                )
    query_obj = Query(
        schemas=schemas,
        id=sql_id,
        q=q,
        utterance=None,
        ltimestamp=datetime.datetime.now(),
        count=1,
        is_subselect=False,
        default_schema=default_schema,
    )
    dispatch_sqls(
        parsed_query=parsed_query,
        sql_text=q,
        query_obj=query_obj,
        keep_string_values=keep_string_values,
        schemas=schemas,
        sql_type=sql_type,
        allow_only_select=allow_only_select,
        fks=fks,
        is_full_parse=is_full_parse,
        add_temp_schema_to_graph=add_temp_schema_to_graph,
    )
    return query_obj
