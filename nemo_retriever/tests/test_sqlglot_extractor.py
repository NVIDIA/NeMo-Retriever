"""Unit tests for the sqlglot-based SQL extractor.

All tests run without a live Neo4j connection.  Schema information is supplied
via a lightweight ``_MockSchema`` stub that mimics the ``Schema.columns_df``
attribute consumed by ``extract_tables_and_columns``.
"""

import pandas as pd

from nemo_retriever.tabular_data.ingestion.parsers.sqlglot_extractor import (
    JoinPair,
    TableMatch,
    extract_tables_and_columns,
)


# ---------------------------------------------------------------------------
# Schema stub
# ---------------------------------------------------------------------------


class _MockSchema:
    """Minimal stand-in for ``Schema`` that only exposes ``columns_df``."""

    def __init__(self, rows: list[dict]):
        self.columns_df = pd.DataFrame(rows)

    def table_exists(self, table_name: str) -> bool:
        if self.columns_df is None or self.columns_df.empty:
            return False
        return table_name.lower() in self.columns_df["table_name"].str.lower().values


# Shared schema covering all tables used by the test queries below.
_SCHEMA = _MockSchema(
    [
        {"table_name": "orders", "column_name": "order_id"},
        {"table_name": "orders", "column_name": "customer_id"},
        {"table_name": "orders", "column_name": "order_status"},
        {"table_name": "orders", "column_name": "order_purchase_timestamp"},
        {"table_name": "customers", "column_name": "customer_id"},
        {"table_name": "customers", "column_name": "customer_unique_id"},
        {"table_name": "order_items", "column_name": "order_id"},
        {"table_name": "order_items", "column_name": "price"},
        {"table_name": "order_payments", "column_name": "order_id"},
        {"table_name": "order_payments", "column_name": "payment_value"},
    ]
)

_ALL_SCHEMAS = {"ecommerce": _SCHEMA}

# ---------------------------------------------------------------------------
# SQL fixtures  (taken verbatim from duckdb_connector.py)
# ---------------------------------------------------------------------------

# RFM segmentation: 4-CTE query joining orders × customers × order_items.
_SQL_RFM = """
WITH RecencyScore AS (
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
GROUP BY RFM_Bucket
"""

# Customer lifetime value: 1-CTE query joining customers × orders × order_payments.
_SQL_CLV = """
WITH CustomerData AS (
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
LIMIT 3
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rfm_query_tables():
    """All three source tables are detected; no CTE names leak into the result."""
    result = extract_tables_and_columns(_SQL_RFM, dialect="duckdb", all_schemas=_ALL_SCHEMAS)
    assert set(result.tables.keys()) == {"orders", "customers", "order_items"}


def test_rfm_query_schema_name():
    """Each table resolves to the owning schema key."""
    result = extract_tables_and_columns(_SQL_RFM, dialect="duckdb", all_schemas=_ALL_SCHEMAS)
    assert result.tables["orders"].schema_name == "ecommerce"
    assert result.tables["customers"].schema_name == "ecommerce"
    assert result.tables["order_items"].schema_name == "ecommerce"


def test_rfm_query_orders_columns():
    """orders: join-key customer_id, join-key order_id, filter order_status,
    aggregation column order_purchase_timestamp."""
    result = extract_tables_and_columns(_SQL_RFM, dialect="duckdb", all_schemas=_ALL_SCHEMAS)
    assert result.tables["orders"].columns == {"order_id", "customer_id", "order_status", "order_purchase_timestamp"}


def test_rfm_query_customers_columns():
    """customers: join-key customer_id, grouping/select customer_unique_id."""
    result = extract_tables_and_columns(_SQL_RFM, dialect="duckdb", all_schemas=_ALL_SCHEMAS)
    assert result.tables["customers"].columns == {"customer_id", "customer_unique_id"}


def test_rfm_query_order_items_columns():
    """order_items: join-key order_id, aggregation column price."""
    result = extract_tables_and_columns(_SQL_RFM, dialect="duckdb", all_schemas=_ALL_SCHEMAS)
    assert result.tables["order_items"].columns == {"order_id", "price"}


def test_clv_query_tables():
    """All three source tables are detected; CustomerData CTE does not appear."""
    result = extract_tables_and_columns(_SQL_CLV, dialect="duckdb", all_schemas=_ALL_SCHEMAS)
    assert set(result.tables.keys()) == {"customers", "orders", "order_payments"}


def test_clv_query_customers_columns():
    """customers: join-key customer_id, select/group-by customer_unique_id."""
    result = extract_tables_and_columns(_SQL_CLV, dialect="duckdb", all_schemas=_ALL_SCHEMAS)
    assert result.tables["customers"].columns == {"customer_id", "customer_unique_id"}


def test_clv_query_orders_columns():
    """orders: join-key customer_id, explicit order_id reference, timestamp aggregations."""
    result = extract_tables_and_columns(_SQL_CLV, dialect="duckdb", all_schemas=_ALL_SCHEMAS)
    assert result.tables["orders"].columns == {"customer_id", "order_id", "order_purchase_timestamp"}


def test_clv_query_order_payments_columns():
    """order_payments: join-key order_id, aggregation column payment_value."""
    result = extract_tables_and_columns(_SQL_CLV, dialect="duckdb", all_schemas=_ALL_SCHEMAS)
    assert result.tables["order_payments"].columns == {"order_id", "payment_value"}


def test_no_schema_name_when_schemas_empty():
    """Without all_schemas, schema_name is None for every table."""
    result = extract_tables_and_columns(_SQL_RFM, dialect="duckdb", all_schemas={})
    for match in result.tables.values():
        assert match.schema_name is None


def test_no_schema_returns_subset():
    """Without schema assistance qualify() still resolves explicitly-qualified columns."""
    result = extract_tables_and_columns(_SQL_RFM, dialect="duckdb", all_schemas={})
    # join-key columns resolved via qualify's USING→ON expansion must be present
    assert "customer_id" in result.tables.get("orders", TableMatch()).columns
    assert "customer_id" in result.tables.get("customers", TableMatch()).columns
    assert "order_id" in result.tables.get("order_items", TableMatch()).columns


def test_ast_node_count_is_positive():
    """ast_node_count is populated with a positive value for valid SQL."""
    result = extract_tables_and_columns(_SQL_RFM, dialect="duckdb", all_schemas=_ALL_SCHEMAS)
    assert result.ast_node_count > 0


def test_empty_sql_returns_empty():
    result = extract_tables_and_columns("", all_schemas=_ALL_SCHEMAS)
    assert result.tables == {}


def test_invalid_sql_returns_empty():
    result = extract_tables_and_columns("NOT VALID SQL !!!", all_schemas=_ALL_SCHEMAS)
    assert result.tables == {}


# ---------------------------------------------------------------------------
# Join extraction tests
# ---------------------------------------------------------------------------


def _join_set(pairs: list[JoinPair]) -> set[tuple[str, str, str, str]]:
    """Normalise join pairs into a set of tuples for order-independent comparison."""
    result = set()
    for p in pairs:
        key = (p.left_table, p.left_column, p.right_table, p.right_column)
        key_rev = (p.right_table, p.right_column, p.left_table, p.left_column)
        result.add(min(key, key_rev))
    return result


def test_rfm_join_edges():
    """RFM query uses USING joins: orders↔customers on customer_id, orders↔order_items on order_id."""
    result = extract_tables_and_columns(_SQL_RFM, dialect="duckdb", all_schemas=_ALL_SCHEMAS)
    joins = _join_set(result.joins)
    assert ("customers", "customer_id", "orders", "customer_id") in joins
    assert ("order_items", "order_id", "orders", "order_id") in joins


def test_clv_join_edges():
    """CLV query: customers↔orders on customer_id, orders↔order_payments on order_id."""
    result = extract_tables_and_columns(_SQL_CLV, dialect="duckdb", all_schemas=_ALL_SCHEMAS)
    joins = _join_set(result.joins)
    assert ("customers", "customer_id", "orders", "customer_id") in joins
    assert ("order_payments", "order_id", "orders", "order_id") in joins


def test_join_no_duplicates():
    """Same join key used in multiple CTEs should not produce duplicate pairs."""
    result = extract_tables_and_columns(_SQL_RFM, dialect="duckdb", all_schemas=_ALL_SCHEMAS)
    joins = _join_set(result.joins)
    assert len(joins) == len(result.joins)


def test_explicit_on_join():
    """Explicit ON syntax produces join edges."""
    sql = """
    SELECT o.order_id, c.customer_unique_id
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    """
    result = extract_tables_and_columns(sql, dialect="duckdb", all_schemas=_ALL_SCHEMAS)
    joins = _join_set(result.joins)
    assert ("customers", "customer_id", "orders", "customer_id") in joins


def test_multi_table_join_chain():
    """Three-table join chain produces two distinct edges."""
    sql = """
    SELECT o.order_id, c.customer_unique_id, oi.price
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    """
    result = extract_tables_and_columns(sql, dialect="duckdb", all_schemas=_ALL_SCHEMAS)
    joins = _join_set(result.joins)
    assert len(joins) == 2
    assert ("customers", "customer_id", "orders", "customer_id") in joins
    assert ("order_items", "order_id", "orders", "order_id") in joins


def test_no_joins_returns_empty_list():
    """A query with no JOINs returns an empty joins list."""
    sql = "SELECT order_id, customer_id FROM orders"
    result = extract_tables_and_columns(sql, dialect="duckdb", all_schemas=_ALL_SCHEMAS)
    assert result.joins == []


def test_join_without_schema():
    """Join extraction works even without all_schemas (qualify still rewrites USING → ON)."""
    sql = """
    SELECT o.order_id, c.customer_id
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    """
    result = extract_tables_and_columns(sql, dialect="duckdb", all_schemas={})
    joins = _join_set(result.joins)
    assert ("customers", "customer_id", "orders", "customer_id") in joins


def test_using_join_with_schema():
    """USING syntax produces join edges (qualify rewrites USING → ON when schema is available)."""
    sql = """
    SELECT order_id, customer_unique_id
    FROM orders
    JOIN customers USING (customer_id)
    """
    result = extract_tables_and_columns(sql, dialect="duckdb", all_schemas=_ALL_SCHEMAS)
    joins = _join_set(result.joins)
    assert ("customers", "customer_id", "orders", "customer_id") in joins


def test_using_join_without_schema():
    """USING syntax without schema exercises the USING fallback path in _extract_join_pairs."""
    sql = """
    SELECT order_id, customer_id
    FROM orders
    JOIN customers USING (customer_id)
    """
    result = extract_tables_and_columns(sql, dialect="duckdb", all_schemas={})
    joins = _join_set(result.joins)
    assert ("customers", "customer_id", "orders", "customer_id") in joins


def test_using_multi_column():
    """USING with multiple columns produces one edge per column."""
    sql = """
    SELECT a.order_id
    FROM orders a
    JOIN order_items b ON a.order_id = b.order_id
    """
    result = extract_tables_and_columns(sql, dialect="duckdb", all_schemas=_ALL_SCHEMAS)
    joins = _join_set(result.joins)
    assert ("order_items", "order_id", "orders", "order_id") in joins


def test_empty_sql_returns_no_joins():
    result = extract_tables_and_columns("", all_schemas=_ALL_SCHEMAS)
    assert result.joins == []
