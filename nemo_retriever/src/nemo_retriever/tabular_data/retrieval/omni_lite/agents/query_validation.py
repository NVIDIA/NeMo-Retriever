import re
import logging

from nemo_retriever.tabular_data.neo4j.neo4j_connection import get_neo4j_conn

from nemo_retriever.tabular_data.retrieval.omni_lite.utils import (
    Labels,
)

logger = logging.getLogger(__name__)
neo4j_conn = get_neo4j_conn()

base_error_patterns = [
    # Snowflake
    "insufficient privileges",
    "incorrect username or password",
    "user temporarily locked",
    "user is not found",
    "saml response is invalid",
    "failed to connect",
    "connection refused",
    "connection reset",
    "network is unreachable",
    "no trusted certificate found",
    "ssl peer certificate",
    "ssh remote key",
    "failed to find the root",
    "broken pipe",
    "remote host terminated",
    "target server failed",
    "communication error",
    # MSSQL
    "login failed",
    "cannot open database",
    "user does not have permission",
    "the server was not found or was not accessible",
    "error locating server/instance",
    "could not open a connection",
    "timeout expired",
    "cannot generate sspi context",
    "insufficient system memory",
    "filegroup is full",
    "transaction log for database is full",
    "access denied",
    # Databricks
    "permission_denied",
    "insufficient privileges",
    "does not have permission",
    "not authorized",
    "user not authorized",
    "invalid access token",
    "unauthorized",
    "forbidden",
    "authentication failed",
    "connection timed out",
    "connection refused",
    "host not found",
    "no route to host",
    "connection reset",
    "out of memory",
    "memory limit exceeded",
    "no space left on device",
    "insufficient capacity",
    "quota exceeded",
]


def is_infra_or_auth_error(error: Exception | str) -> bool:
    msg = str(error)

    logger.error(f"Error from query execution: {msg}")

    # Explicit exclusion for "Connection <id> not found"
    if re.search(r"^'?Connection [\w-]+ not found'?$", msg, re.IGNORECASE):
        return True

    error_patterns = [re.escape(pattern) for pattern in base_error_patterns]
    combined_pattern = re.compile("|".join(error_patterns), re.IGNORECASE)
    return bool(combined_pattern.search(msg))


def _get_column_breadcrumbs(column_ids: list) -> dict:
    """
    Fetch breadcrumb information (table_name, schema_name, database_name, connection, connection_type) for columns.
    Returns a dict mapping column_id to {table_name, schema_name, database_name, connection, connection_type}.
    """
    if not column_ids:
        return {}

    query = """
        UNWIND $column_ids as col_id
        MATCH (c:column )<-[:schema]-(t:table)<-[:schema]-(s:schema)<-[:schema]-(db:db)<-[:connecting]-(conn:connection)
        RETURN col_id as column_id, t.name as table_name, s.name as schema_name, db.name as database_name, conn.id as connection, conn.type as connection_type
    """
    rows = neo4j_conn.query_read_only(
        query=query,
        parameters={"column_ids": column_ids},
    )

    breadcrumbs = {}
    for row in rows or []:
        column_id = row.get("column_id")
        if column_id:
            breadcrumbs[column_id] = {
                "table_name": row.get("table_name"),
                "schema_name": row.get("schema_name"),
                "database_name": row.get("database_name"),
                "connection": row.get("connection"),
                "connection_type": row.get("connection_type"),
            }
    return breadcrumbs


def query_validation(
    schemas,
    sql: str,
    dialects: list,
    user_participants: list,
    fks=None,
):
    return_dict = {}
    try:
        query = parse_single(
            q=sql,
            schemas=schemas,
            dialects=dialects,
            sql_type=Labels.SEMANTIC,
            allow_only_select=True,
            fks=fks,
        )
        column_ids = query.get_reached_columns_ids()
        return_dict["sql_columns"] = column_ids
        if column_ids:
            pii_columns = get_columns_with_pii_tag(column_ids)
            return_dict["pii_objects"] = pii_columns
            # Extract breadcrumb information for columns
            column_breadcrumbs = _get_column_breadcrumbs(column_ids)
            return_dict["column_breadcrumbs"] = column_breadcrumbs
        if column_ids:
            tables_validation = validate_tables_with_user_participants(
                user_participants, column_ids
            )
            out_of_zone = [
                t["table_name"] for t in tables_validation if not t["in_zone"]
            ]
            if len(out_of_zone) > 0:
                return_dict.update(
                    {
                        "error": f"The following tables are not authorized for use: {', '.join(out_of_zone)}",
                        "another_try": 1,
                    }
                )
                return return_dict

        return_dict["success"] = True
        return return_dict

    except NoFKError as error:
        return_dict.update({"error": str(error), "another_try": 1})
        return return_dict
    except NotSelectSqlTypeError as error:
        return_dict.update({"error": str(error), "another_try": 0})
        return return_dict
    except Exception as error:
        return_dict.update({"error": str(error), "another_try": 1})
        return return_dict
