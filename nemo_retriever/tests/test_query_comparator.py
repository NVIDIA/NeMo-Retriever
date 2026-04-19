"""Unit tests for the SQL query comparator.

All tests are pure-Python — no database or Neo4j connection required.
"""

from nemo_retriever.tabular_data.ingestion.parsers.query_comparator import (
    compare_queries,
    normalize_sql,
)


# ---------------------------------------------------------------------------
# normalize_sql
# ---------------------------------------------------------------------------


class TestNormalizeSql:
    def test_returns_string_for_valid_sql(self):
        result = normalize_sql("SELECT 1")
        assert isinstance(result, str)

    def test_returns_none_for_unparseable_sql(self):
        assert normalize_sql("NOT VALID SQL !!!") is None

    def test_lowercases_identifiers(self):
        n1 = normalize_sql("SELECT Id FROM Users")
        n2 = normalize_sql("SELECT id FROM users")
        assert n1 == n2

    def test_sorts_select_columns(self):
        n1 = normalize_sql("SELECT a, b, c FROM t")
        n2 = normalize_sql("SELECT c, a, b FROM t")
        assert n1 == n2

    def test_ignore_literals_replaces_strings(self):
        n1 = normalize_sql("SELECT x FROM t WHERE d = '2024-01-01'", ignore_literals=True)
        n2 = normalize_sql("SELECT x FROM t WHERE d = '2025-06-15'", ignore_literals=True)
        assert n1 == n2

    def test_ignore_literals_replaces_numbers(self):
        n1 = normalize_sql("SELECT x FROM t WHERE id > 10", ignore_literals=True)
        n2 = normalize_sql("SELECT x FROM t WHERE id > 999", ignore_literals=True)
        assert n1 == n2

    def test_literals_preserved_when_flag_off(self):
        n1 = normalize_sql("SELECT x FROM t WHERE d = '2024'")
        n2 = normalize_sql("SELECT x FROM t WHERE d = '2025'")
        assert n1 != n2

    def test_ignore_literals_replaces_double_quoted_values(self):
        n1 = normalize_sql('SELECT x FROM t WHERE d = "15/12/2024"', ignore_literals=True)
        n2 = normalize_sql('SELECT x FROM t WHERE d = "01/01/2030"', ignore_literals=True)
        assert n1 == n2

    def test_idempotent(self):
        sql = "SELECT b, a FROM t WHERE x = 1"
        assert normalize_sql(sql) == normalize_sql(sql)


# ---------------------------------------------------------------------------
# compare_queries — identical
# ---------------------------------------------------------------------------


class TestCompareQueriesIdentical:
    def test_same_string(self):
        sql = "SELECT id, name FROM users"
        assert compare_queries(sql, sql) is True

    def test_select_order_insensitive(self):
        assert compare_queries(
            "SELECT id, name FROM users",
            "SELECT name, id FROM users",
        ) is True

    def test_case_insensitive_identifiers(self):
        assert compare_queries(
            "SELECT Id, Name FROM Users",
            "SELECT id, name FROM users",
        ) is True

    def test_select_order_in_cte(self):
        q1 = "WITH cte AS (SELECT a, b FROM t) SELECT * FROM cte"
        q2 = "WITH cte AS (SELECT b, a FROM t) SELECT * FROM cte"
        assert compare_queries(q1, q2) is True

    def test_select_order_in_subquery(self):
        q1 = "SELECT * FROM (SELECT x, y FROM t) sub"
        q2 = "SELECT * FROM (SELECT y, x FROM t) sub"
        assert compare_queries(q1, q2) is True

    def test_ignore_literals_different_filter_values(self):
        assert compare_queries(
            "SELECT id FROM orders WHERE created > '2024-01-01'",
            "SELECT id FROM orders WHERE created > '2025-06-15'",
            ignore_literals=True,
        ) is True

    def test_ignore_literals_different_numeric_values(self):
        assert compare_queries(
            "SELECT id FROM orders WHERE amount > 100",
            "SELECT id FROM orders WHERE amount > 500",
            ignore_literals=True,
        ) is True

    def test_combined_select_order_and_ignore_literals(self):
        assert compare_queries(
            "SELECT order_total_price, order_id FROM orders WHERE order_date = '15/12/2024'",
            "SELECT order_id, order_total_price FROM orders WHERE order_date = '15/12/2025'",
            ignore_literals=True,
        ) is True


# ---------------------------------------------------------------------------
# compare_queries — different
# ---------------------------------------------------------------------------


class TestCompareQueriesDifferent:
    def test_different_tables(self):
        assert compare_queries(
            "SELECT id FROM users",
            "SELECT id FROM orders",
        ) is False

    def test_different_columns(self):
        assert compare_queries(
            "SELECT id FROM users",
            "SELECT name FROM users",
        ) is False

    def test_extra_where_clause(self):
        assert compare_queries(
            "SELECT id FROM users",
            "SELECT id FROM users WHERE active = 1",
        ) is False

    def test_different_join(self):
        assert compare_queries(
            "SELECT a.id FROM a JOIN b ON a.id = b.id",
            "SELECT a.id FROM a JOIN c ON a.id = c.id",
        ) is False

    def test_different_literals_without_flag(self):
        assert compare_queries(
            "SELECT id FROM t WHERE x = '2024'",
            "SELECT id FROM t WHERE x = '2025'",
            ignore_literals=False,
        ) is False

    def test_different_aggregation(self):
        assert compare_queries(
            "SELECT COUNT(id) FROM users",
            "SELECT SUM(id) FROM users",
        ) is False


# ---------------------------------------------------------------------------
# compare_queries — edge cases
# ---------------------------------------------------------------------------


class TestCompareQueriesEdgeCases:
    def test_unparseable_sql_returns_false(self):
        assert compare_queries("NOT SQL", "SELECT 1") is False

    def test_both_unparseable_returns_false(self):
        assert compare_queries("GARBAGE", "ALSO GARBAGE") is False

    def test_single_column_select_order_irrelevant(self):
        assert compare_queries(
            "SELECT id FROM t",
            "SELECT id FROM t",
        ) is True

    def test_dialect_parameter(self):
        assert compare_queries(
            "SELECT a, b FROM t",
            "SELECT b, a FROM t",
            dialect="duckdb",
        ) is True

    def test_complex_cte_with_reordered_selects(self):
        q1 = """
        WITH sales AS (
            SELECT customer_id, SUM(amount) AS total, COUNT(*) AS cnt
            FROM orders
            GROUP BY customer_id
        )
        SELECT total, cnt, customer_id FROM sales WHERE total > 100
        """
        q2 = """
        WITH sales AS (
            SELECT customer_id, COUNT(*) AS cnt, SUM(amount) AS total
            FROM orders
            GROUP BY customer_id
        )
        SELECT customer_id, cnt, total FROM sales WHERE total > 100
        """
        assert compare_queries(q1, q2) is True
