# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DuckDB connector for in-process SQL execution.

Wraps ``duckdb.connect()`` with helpers to register pandas DataFrames or
scan CSV/Parquet/JSON files directly from the filesystem.  No server or Docker
service is required — DuckDB runs fully in-process.

This is the reference implementation of
:class:`~nemo_retriever.tabular_data.sql_database.SQLDatabase`.

Example
-------
::

    from duckdb_connector import DuckDB  # run from tabular-dev-tools/

    conn = DuckDB("./spider2.duckdb")
    rows = conn.execute("SELECT * FROM Airlines.flights LIMIT 5")
    # rows -> [{"flight_id": 1, ...}]
"""

from __future__ import annotations


import logging
from datetime import datetime
import duckdb
import pandas as pd
from typing import Optional

from nemo_retriever.tabular_data.sql_database import SQLDatabase

logger = logging.getLogger(__name__)


class DuckDB(SQLDatabase):
    """In-process DuckDB connection with convenience helpers.

    Parameters
    ----------
    database:
        Path to a persistent DuckDB database file, or ``None`` / ``":memory:"``
        for an ephemeral in-memory database (default: in-memory).
    read_only:
        Open the database in read-only mode (default: True).  Multiple
        processes can hold a read-only connection simultaneously; set to
        ``False`` only when you need to write to the file.
    """

    def __init__(self, connection_string: str, *, read_only: bool = True) -> None:
        self.conn = duckdb.connect(database=connection_string, read_only=read_only)
        logger.debug("DuckDB connected (database=%r, read_only=%s).", connection_string, read_only)

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

        queries = ["""WITH RecencyScore AS (
    SELECT customer_unique_id,
           MAX(order_purchase_timestamp) AS last_purchase,
           NTILE(5) OVER (ORDER BY MAX(order_purchase_timestamp) DESC) AS recency
    FROM orders
        JOIN customers USING (customer_id)
    WHERE order_status = 'delivered'
    GROUP BY customer_unique_id
),
FrequencyScore AS (
    SELECT customer_unique_id,
           COUNT(order_id) AS total_orders,
           NTILE(5) OVER (ORDER BY COUNT(order_id) DESC) AS frequency
    FROM orders
        JOIN customers USING (customer_id)
    WHERE order_status = 'delivered'
    GROUP BY customer_unique_id
),
MonetaryScore AS (
    SELECT customer_unique_id,
           SUM(price) AS total_spent,
           NTILE(5) OVER (ORDER BY SUM(price) DESC) AS monetary
    FROM orders
        JOIN order_items USING (order_id)
        JOIN customers USING (customer_id)
    WHERE order_status = 'delivered'
    GROUP BY customer_unique_id
),

-- 2. Assign each customer to a group
RFM AS (
    SELECT last_purchase, total_orders, total_spent,
        CASE
            WHEN recency = 1 AND frequency + monetary IN (1, 2, 3, 4) THEN "Champions"
            WHEN recency IN (4, 5) AND frequency + monetary IN (1, 2) THEN "Can't Lose Them"
            WHEN recency IN (4, 5) AND frequency + monetary IN (3, 4, 5, 6) THEN "Hibernating"
            WHEN recency IN (4, 5) AND frequency + monetary IN (7, 8, 9, 10) THEN "Lost"
            WHEN recency IN (2, 3) AND frequency + monetary IN (1, 2, 3, 4) THEN "Loyal Customers"
            WHEN recency = 3 AND frequency + monetary IN (5, 6) THEN "Needs Attention"
            WHEN recency = 1 AND frequency + monetary IN (7, 8) THEN "Recent Users"
            WHEN recency = 1 AND frequency + monetary IN (5, 6) OR
                recency = 2 AND frequency + monetary IN (5, 6, 7, 8) THEN "Potentital Loyalists"
            WHEN recency = 1 AND frequency + monetary IN (9, 10) THEN "Price Sensitive"
            WHEN recency = 2 AND frequency + monetary IN (9, 10) THEN "Promising"
            WHEN recency = 3 AND frequency + monetary IN (7, 8, 9, 10) THEN "About to Sleep"
        END AS RFM_Bucket
    FROM RecencyScore
        JOIN FrequencyScore USING (customer_unique_id)
        JOIN MonetaryScore USING (customer_unique_id)
)

SELECT RFM_Bucket, 
       AVG(total_spent / total_orders) AS avg_sales_per_customer
FROM RFM
GROUP BY RFM_Bucket""", """WITH CustomerData AS (
    SELECT
        customer_unique_id,
        COUNT(DISTINCT orders.order_id) AS order_count,
        SUM(payment_value) AS total_payment,
        JULIANDAY(MIN(order_purchase_timestamp)) AS first_order_day,
        JULIANDAY(MAX(order_purchase_timestamp)) AS last_order_day
    FROM customers
        JOIN orders USING (customer_id)
        JOIN order_payments USING (order_id)
    GROUP BY customer_unique_id
)
SELECT
    customer_unique_id,
    order_count AS PF,
    ROUND(total_payment / order_count, 2) AS AOV,
    CASE
        WHEN (last_order_day - first_order_day) < 7 THEN
            1
        ELSE
            (last_order_day - first_order_day) / 7
        END AS ACL
FROM CustomerData
ORDER BY AOV DESC
LIMIT 3"""]
        

        return pd.DataFrame({"end_time": datetime.today(), "query_text": queries})

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
