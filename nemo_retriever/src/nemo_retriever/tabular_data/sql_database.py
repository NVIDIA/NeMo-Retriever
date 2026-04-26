# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class for SQL database connectors.

Any connector used by the ingestion / text-2-sql agent must implement this
interface.  DuckDB is the bundled reference implementation; other libraries
(Snowflake, BigQuery, PostgreSQL, etc.) can provide their own by subclassing
``SQLDatabase``.

Example
-------
::

    from nemo_retriever.tabular_data.sql_database import SQLDatabase

    class MyConnector(SQLDatabase):
        def __init__(self, connection_string: str) -> None:
            ...

        def execute(self, sql, parameters=None):
            ...

        # implement remaining abstract methods ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class SQLDatabase(ABC):
    """Abstract SQL database connector.

    Subclasses must implement all abstract methods.  The context-manager
    protocol (``__enter__`` / ``__exit__``) is provided by this base class
    and delegates to :meth:`close`.

    Parameters
    ----------
    connection_string:
        A driver-specific connection string or database path.
    dialect:
        SQL dialect used by this connector (e.g. ``"duckdb"``, ``"snowflake"``).
        Used by the text-to-SQL agent to emit dialect-appropriate SQL.
    """

    @abstractmethod
    def __init__(self, connection_string: str) -> None: ...

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @abstractmethod
    def execute(self, sql: str, parameters: Optional[list] = None) -> pd.DataFrame:
        """Execute a SQL statement and return the result as a DataFrame.

        Parameters
        ----------
        sql:
            SQL query to execute.
        parameters:
            Optional positional parameters for parameterised queries.
        """

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    @abstractmethod
    def get_tables(self) -> pd.DataFrame:
        """Return all tables.

        Expected columns: ``table_schema``, ``table_name``.
        """

    @abstractmethod
    def get_columns(self) -> pd.DataFrame:
        """Return all columns.

        Expected columns: ``table_schema``, ``table_name``,
        ``column_name``, ``data_type``, ``is_nullable``.
        """

    @abstractmethod
    def get_queries(self) -> pd.DataFrame:
        """Return recent / historical queries if the backend supports it.

        Expected columns: ``end_time``, ``query_text``.
        Connectors without query history should return an empty DataFrame
        with those two columns.
        """

    @abstractmethod
    def get_views(self) -> pd.DataFrame:
        """Return all views.

        Expected columns: ``table_schema``, ``table_name``,
        ``view_definition``.
        """

    @abstractmethod
    def get_pks(self) -> pd.DataFrame:
        """Return primary key columns.

        Expected columns: ``table_schema``, ``table_name``,
        ``column_name``, ``ordinal_position``.
        """

    @abstractmethod
    def get_fks(self) -> pd.DataFrame:
        """Return foreign key columns.

        Expected columns: ``table_schema``, ``table_name``,
        ``column_name``, ``referenced_schema``, ``referenced_table``,
        ``referenced_column``.
        """

    @property
    @abstractmethod
    def dialect(self) -> str:
        """Return the dialect of the database.

        Returns:
            A sqlglot-compatible dialect name for this engine (e.g. ``"duckdb"``, ``"snowflake"``).
        """

    @property
    @abstractmethod
    def database_name(self) -> str:
        """Return the name of the connected database.

        Returns:
            The database name as reported by the backend (e.g. ``"mydb"``).
        """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by this connector."""

    def __enter__(self) -> "SQLDatabase":
        return self

    def __exit__(self, *args) -> None:
        self.close()
