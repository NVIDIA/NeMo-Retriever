"""PostgreSQL connector implementing the NeMo-Retriever SQLDatabase ABC."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import psycopg2
import psycopg2.extras

from nemo_retriever.tabular_data.sql_database import SQLDatabase


class PostgresDatabase(SQLDatabase):
    """Concrete :class:`SQLDatabase` backed by ``psycopg2``.

    Parameters
    ----------
    connection_string:
        A ``libpq``-style connection URI, e.g.
        ``postgresql://user:pass@host:5432/dbname``.
    """

    def __init__(self, connection_string: str) -> None:
        self._connection_string = connection_string
        self._conn: psycopg2.extensions.connection = psycopg2.connect(connection_string)
        self._database_name: str = self.execute("SELECT current_database()").iloc[0, 0]

    @property
    def dialect(self) -> str:
        return "postgres"

    @property
    def database_name(self) -> str:
        return self._database_name

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, sql: str, parameters: Optional[list] = None) -> pd.DataFrame:
        try:
            with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, parameters)
                if cur.description is None:
                    self._conn.commit()
                    return pd.DataFrame()
                rows = cur.fetchall()
            return pd.DataFrame(rows)
        except psycopg2.Error:
            self._conn.rollback()
            raise

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def get_tables(self) -> pd.DataFrame:
        return self.execute(
            """
            SELECT
                t.table_schema    AS table_schema,
                t.table_name      AS table_name
            FROM information_schema.tables t
            WHERE t.table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY t.table_schema, t.table_name
        """
        )

    def get_columns(self) -> pd.DataFrame:
        return self.execute(
            """
            SELECT
                c.table_schema       AS table_schema,
                c.table_name         AS table_name,
                c.column_name        AS column_name,
                c.data_type          AS data_type,
                c.is_nullable        AS is_nullable,
                c.ordinal_position   AS ordinal_position
            FROM information_schema.columns c
            WHERE c.table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY c.table_schema, c.table_name, c.ordinal_position
        """
        )

    def get_queries(self) -> pd.DataFrame:
        """Return recent queries from ``pg_stat_statements`` if available."""
        try:
            return self.execute(
                """
                SELECT
                    now()      AS end_time,
                    query      AS query_text
                FROM pg_stat_statements
                ORDER BY total_exec_time DESC
                LIMIT 100
            """
            )
        except psycopg2.Error:
            self._conn.rollback()
            return pd.DataFrame(columns=["end_time", "query_text"])

    def get_views(self) -> pd.DataFrame:
        return self.execute(
            """
            SELECT
                v.table_schema     AS table_schema,
                v.table_name       AS table_name,
                v.view_definition  AS view_definition
            FROM information_schema.views v
            WHERE v.table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY v.table_schema, v.table_name
        """
        )

    def get_pks(self) -> pd.DataFrame:
        return self.execute(
            """
            SELECT
                kcu.table_schema         AS table_schema,
                kcu.table_name           AS table_name,
                kcu.column_name          AS column_name,
                kcu.ordinal_position     AS ordinal_position
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema    = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY kcu.table_schema, kcu.table_name, kcu.ordinal_position
        """
        )

    def get_fks(self) -> pd.DataFrame:
        return self.execute(
            """
            SELECT
                kcu.table_schema         AS table_schema,
                kcu.table_name           AS table_name,
                kcu.column_name          AS column_name,
                ccu.table_schema         AS referenced_schema,
                ccu.table_name           AS referenced_table,
                ccu.column_name          AS referenced_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema    = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
              ON tc.constraint_name = ccu.constraint_name
             AND tc.table_schema    = ccu.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY kcu.table_schema, kcu.table_name, kcu.column_name
        """
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()
