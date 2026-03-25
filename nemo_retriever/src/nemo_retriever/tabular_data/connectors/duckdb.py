# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DuckDB connector for in-process SQL execution.

Wraps ``duckdb.connect()`` with helpers to register pandas DataFrames or
scan CSV/Parquet/JSON files directly from the filesystem.  No server or Docker
service is required — DuckDB runs fully in-process.

Example
-------
::

    from nemo_retriever.tabular_data.connectors.duckdb import DuckDB

    conn = DuckDB({"database": "./spider2.duckdb"})
    rows = conn.execute("SELECT * FROM Airlines.flights LIMIT 5")
    # rows -> [{"flight_id": 1, ...}]
"""

from __future__ import annotations


import logging
import duckdb
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


class DuckDB:
    """In-process DuckDB connection with convenience helpers.

    Parameters
    ----------
    database:
        Path to a persistent DuckDB database file, or ``None`` / ``":memory:"``
        for an ephemeral in-memory database (default: in-memory).
    read_only:
        Open the database in read-only mode (default: False).
    """

    def __init__(self, connection_string: str) -> None:
        self.conn = duckdb.connect(database=connection_string, read_only=False)
        logger.debug("DuckDB connected (database=%r, read_only=%s).", connection_string, False)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, sql: str, parameters: Optional[list] = None) -> pd.DataFrame:
        """Execute a SQL statement and return a pandas DataFrame.

        Parameters
        ----------
        sql:
            SQL query to execute.
        parameters:
            Optional positional parameters.
        """
        logger.debug("DuckDB executing (→ DataFrame): %s", sql[:200])
        if parameters:
            rel = self.conn.execute(sql, parameters)
        else:
            rel = self.conn.execute(sql)
        return rel.df()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_tables(self) -> pd.DataFrame:
        """Return all tables from information_schema as a DataFrame."""
        return self.execute(
            """
            SELECT
                table_catalog AS "database",
                table_schema  AS "schema",
                table_name    AS "table_name"
            FROM information_schema.tables
            ORDER BY table_catalog, table_schema, table_name
        """
        )

    def get_columns(self) -> pd.DataFrame:
        """Return all columns from information_schema as a DataFrame."""
        return self.execute(
            """
            SELECT
                table_catalog    AS "database",
                table_schema     AS "schema",
                table_name       AS "table_name",
                column_name      AS "column_name",
                data_type        AS "data_type",
                is_nullable      AS "is_nullable"
            FROM information_schema.columns
            ORDER BY table_catalog, table_schema, table_name, ordinal_position
        """
        )

    def get_queries(self) -> pd.DataFrame:
        """DuckDB has no built-in query history — returns an empty DataFrame."""
        return pd.DataFrame(columns=["end_time", "query_text"])

    def get_views(self) -> pd.DataFrame:
        """Return all views from information_schema."""
        return self.execute(
            """
            SELECT
                table_catalog   AS database,
                table_schema    AS schema,
                table_name,
                view_definition
            FROM information_schema.views
            ORDER BY table_catalog, table_schema, table_name
        """
        )

    # Todo: Test as Spider2 has no PKs
    def get_pks(self) -> pd.DataFrame:
        """Return primary key columns from duckdb_constraints() as a DataFrame.

        Columns: database, schema, table_name, column_name, ordinal_position.
        If duckdb_constraints() is unavailable, returns an empty DataFrame with those columns.
        """
        empty = pd.DataFrame(
            columns=[
                "database",
                "schema",
                "table_name",
                "column_name",
                "ordinal_position",
            ]
        )
        try:
            # duckdb_constraints() returns constraint_column_names as list; unnest to one row per column
            df = self.execute(
                """
                SELECT
                    current_database() AS "database",
                    c.schema_name      AS "schema",
                    c.table_name       AS "table_name",
                    unnest(c.constraint_column_names) AS "column_name",
                    unnest(range(1, len(c.constraint_column_names) + 1)) AS "ordinal_position"
                FROM duckdb_constraints() c
                WHERE c.constraint_type = 'PRIMARY KEY'
                ORDER BY c.schema_name, c.table_name, "ordinal_position"
            """
            )
            return df if not df.empty else empty
        except Exception:
            return empty

    # Todo: Test as Spider2 has no FKs
    def get_fks(self) -> pd.DataFrame:
        """Return foreign key columns from duckdb_constraints() as a DataFrame.

        Columns: database, schema, table_name, column_name, and referenced_* if available.
        If duckdb_constraints() is unavailable, returns an empty DataFrame with standard columns.
        """
        empty = pd.DataFrame(
            columns=[
                "database",
                "schema",
                "table_name",
                "column_name",
                "referenced_schema",
                "referenced_table",
                "referenced_column",
            ]
        )
        try:
            df = self.execute(
                """
                SELECT
                    current_database() AS "database",
                    c.schema_name      AS "schema",
                    c.table_name       AS "table_name",
                    unnest(c.constraint_column_names) AS "column_name"
                FROM duckdb_constraints() c
                WHERE c.constraint_type = 'FOREIGN KEY'
                ORDER BY c.schema_name, c.table_name
            """
            )

            return df if not df.empty else empty
        except Exception:
            return empty

    # ------------------------------------------------------------------
    # Context manager / cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the DuckDB connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
