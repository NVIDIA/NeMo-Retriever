# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DuckDB engine wrapper for in-process SQL execution.

Wraps ``duckdb.connect()`` with helpers to register pandas DataFrames or
scan CSV/Parquet/JSON files directly from the filesystem.  No server or Docker
service is required — DuckDB runs fully in-process.

Example
-------
::

    from nemo_retriever.relational_db.db_setup.duckdb_engine import DuckDBEngine

    engine = DuckDBEngine({"database": "./spider2.duckdb"})
    rows = engine.execute("SELECT * FROM Airlines.flights LIMIT 5")
    # rows -> [{"flight_id": 1, ...}]
"""

from __future__ import annotations


import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)


class DuckDBEngine:
    """In-process DuckDB connection with convenience helpers.

    Parameters
    ----------
    database:
        Path to a persistent DuckDB database file, or ``None`` / ``":memory:"``
        for an ephemeral in-memory database (default: in-memory).
    read_only:
        Open the database in read-only mode (default: False).
    """

    def __init__(self, connection: Optional[Dict[str, Any]] = None) -> None:
        try:
            import duckdb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "DuckDB is required. Install it via: pip install 'duckdb>=1.2.0'"
            ) from exc

        self.connection_properties = connection or {}
        db_path = self.connection_properties.get("database", ":memory:")
        read_only = self.connection_properties.get("read_only", False)
        self.conn = duckdb.connect(database=db_path, read_only=read_only)
        logger.debug("DuckDB connected (database=%r, read_only=%s).", db_path, read_only)

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

    def test(self) -> List[Dict[str, Any]]:
        """Return a list of dicts mapping each database to its schemas."""
        return [
            {"db_name": db, "schemas": self.list_db_schemas(db)}
            for db in self.list_databases()
        ]

    @property
    def pull_info(self) -> List[Dict[str, Any]]:
        """List of {db_name, schemas} for merging with tables/columns/views (e.g. in extract_data)."""
        return self.test()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def list_databases(self) -> List[str]:
        """Return all catalog/database names visible in this connection."""
        df = self.execute("SELECT DISTINCT catalog_name FROM information_schema.schemata")
        return df["catalog_name"].tolist()

    def list_schemas(self) -> List[str]:
        """Return all catalog/schema names (alias for list_databases)."""
        return self.list_databases()

    def list_db_schemas(self, db: str) -> List[str]:
        """Return schema names within *db*."""
        df = self.execute(
            f"SELECT DISTINCT schema_name FROM information_schema.schemata "
            f"WHERE catalog_name = '{db}'"
        )
        return df["schema_name"].tolist()

    def schema_tables(self, catalog_or_schema: str) -> List[str]:
        """Return table names in the given catalog (e.g. Spider2 schema name like 'Airlines')."""
        df = self.execute(
            "SELECT DISTINCT table_name FROM information_schema.tables "
            "WHERE table_catalog = ? OR table_schema = ? ORDER BY table_name",
            [catalog_or_schema, catalog_or_schema],
        )
        return df["table_name"].tolist()

    def get_tables(self) -> pd.DataFrame:
        """Return all tables from information_schema as a DataFrame."""
        return self.execute("""
            SELECT
                table_catalog AS "database",
                table_schema  AS "schema",
                table_name    AS "table_name",
                table_type    AS "table_type",
                NULL          AS "created"
            FROM information_schema.tables
            ORDER BY table_catalog, table_schema, table_name
        """)

    def get_columns(self) -> pd.DataFrame:
        """Return all columns from information_schema as a DataFrame."""
        return self.execute("""
            SELECT
                table_catalog    AS "database",
                table_schema     AS "schema",
                table_name       AS "table_name",
                column_name      AS "column_name",
                ordinal_position AS "ordinal_position",
                data_type        AS "data_type",
                is_nullable      AS "is_nullable"
            FROM information_schema.columns
            ORDER BY table_catalog, table_schema, table_name, ordinal_position
        """)

    def get_schemas(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return (tables_df, columns_df) from information_schema."""
        return self.get_tables(), self.get_columns()

    def get_queries(self) -> pd.DataFrame:
        """DuckDB has no built-in query history — returns an empty DataFrame."""
        return pd.DataFrame(columns=["end_time", "query_text"])

    def get_views(self) -> pd.DataFrame:
        """Return all views from information_schema."""
        return self.execute("""
            SELECT
                table_catalog   AS database,
                table_schema    AS schema,
                table_name,
                view_definition
            FROM information_schema.views
            ORDER BY table_catalog, table_schema, table_name
        """)

    def get_pks(self) -> pd.DataFrame:
        """Return primary key columns from duckdb_constraints() as a DataFrame.

        Columns: database, schema, table_name, column_name, ordinal_position.
        If duckdb_constraints() is unavailable, returns an empty DataFrame with those columns.
        """
        empty = pd.DataFrame(
            columns=["database", "schema", "table_name", "column_name", "ordinal_position"]
        )
        try:
            # duckdb_constraints() returns constraint_column_names as list; unnest to one row per column
            df = self.execute("""
                SELECT
                    current_database() AS "database",
                    c.schema_name      AS "schema",
                    c.table_name       AS "table_name",
                    unnest(c.constraint_column_names) AS "column_name",
                    unnest(range(1, len(c.constraint_column_names) + 1)) AS "ordinal_position"
                FROM duckdb_constraints() c
                WHERE c.constraint_type = 'PRIMARY KEY'
                ORDER BY c.schema_name, c.table_name, "ordinal_position"
            """)
            return df if not df.empty else empty
        except Exception:
            return empty

    def get_fks(self) -> pd.DataFrame:
        """Return foreign key columns from duckdb_constraints() as a DataFrame.

        Columns: database, schema, table_name, column_name, and referenced_* if available.
        If duckdb_constraints() is unavailable, returns an empty DataFrame with standard columns.
        """
        empty = pd.DataFrame(
            columns=[
                "database", "schema", "table_name", "column_name",
                "referenced_schema", "referenced_table", "referenced_column",
            ]
        )
        try:
            df = self.execute("""
                SELECT
                    current_database() AS "database",
                    c.schema_name      AS "schema",
                    c.table_name       AS "table_name",
                    unnest(c.constraint_column_names) AS "column_name"
                FROM duckdb_constraints() c
                WHERE c.constraint_type = 'FOREIGN KEY'
                ORDER BY c.schema_name, c.table_name
            """)
            if not df.empty and "referenced_table_name" in self.execute("SELECT * FROM duckdb_constraints() LIMIT 1").columns:
                ref = self.execute("""
                    SELECT schema_name, table_name, constraint_column_names
                    FROM duckdb_constraints()
                    WHERE constraint_type IN ('FOREIGN KEY', 'PRIMARY KEY', 'UNIQUE')
                """)
                # Simplified: return FK side only; referenced_* can be extended if DuckDB exposes it
                for col in ["referenced_schema", "referenced_table", "referenced_column"]:
                    if col not in df.columns:
                        df[col] = None
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
