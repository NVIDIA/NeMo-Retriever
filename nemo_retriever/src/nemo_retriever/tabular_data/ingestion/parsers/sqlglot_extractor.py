"""Extract tables and columns referenced in a SQL query.

Strategy
--------
sqlglot parses the full query into an AST to enumerate all CTEs and inline
subqueries.  For each scope (CTE body or inline subquery) that references at
least one **real** (non-CTE) table, sqllineage is run on that scope in
isolation so it can accurately attribute columns to their source tables.

Scopes that only reference other CTEs are skipped — their columns are already
resolved by the base-CTE scopes that first introduced the real tables.

Any column that survives all scopes without a real-table parent is looked up
in ``all_schemas`` (Neo4j metadata):
  - unique match within the query's source tables → attributed to that table
  - ambiguous / not found → remains ``"<unresolved>"``

When ``all_schemas`` is ``None`` (default), ``get_account_schemas()`` is
called automatically.  Pass ``{}`` to skip schema-assisted resolution.

Usage (standalone):
    python sqlglot_extractor.py [--dialect <dialect>]
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqllineage.runner import LineageRunner

_SYNTHETIC_TARGET = "__target__"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bare_column_name(raw_name: str) -> str:
    """Extract the innermost column identifier from a SQL expression.

    Strips function wrappers, DISTINCT modifiers, and type casts so that
    e.g. ``MAX(order_purchase_timestamp)`` → ``order_purchase_timestamp``
    and ``COUNT(DISTINCT order_id)`` → ``order_id``.
    """
    try:
        parsed = sqlglot.parse_one(raw_name)
        # Walk the expression tree and return the first leaf Column identifier.
        for col in parsed.find_all(exp.Column):
            return col.name.lower()
        # Bare identifier with no table qualifier (e.g. just "price")
        if isinstance(parsed, exp.Identifier):
            return parsed.name.lower()
    except Exception:
        pass
    # Fallback: recursively strip the outermost function wrapper with regex.
    import re
    m = re.match(r"^\w+\s*\((.+)\)\s*$", raw_name.strip(), re.DOTALL)
    if m:
        inner = re.sub(r"^(?:distinct|all)\s+", "", m.group(1).strip(), flags=re.IGNORECASE)
        first_arg = inner.split(",")[0].strip()
        return _bare_column_name(first_arg) if "(" in first_arg else first_arg.lower()
    return raw_name.lower().strip()


def _build_column_to_tables(all_schemas: dict) -> dict[str, list[str]]:
    """``{column_name: [table_name, ...]}`` reverse-index from Neo4j schemas."""
    col_to_tables: dict[str, list[str]] = {}
    for schema in all_schemas.values():
        df = schema.columns_df
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            col = str(row["column_name"]).lower()
            tbl = str(row["table_name"]).lower()
            col_to_tables.setdefault(col, [])
            if tbl not in col_to_tables[col]:
                col_to_tables[col].append(tbl)
    return col_to_tables


def _real_tables_in_select(select_node: exp.Expression, cte_names: set[str]) -> set[str]:
    """Return the set of real (non-CTE) table names referenced inside *select_node*."""
    return {
        t.name.lower()
        for t in select_node.find_all(exp.Table)
        if t.name.lower() not in cte_names
    }


def _using_columns(scope_node: exp.Expression) -> set[str]:
    """Return column names that appear in USING clauses within *scope_node*.

    These are shared join keys that genuinely belong to all joined tables,
    so attributing them to multiple tables is correct.
    """
    cols: set[str] = set()
    for join in scope_node.find_all(exp.Join):
        for using_col in join.args.get("using") or []:
            cols.add(using_col.name.lower())
    return cols


def _run_lineage_on_scope(
    scope_sql: str,
    scope_node: exp.Expression,
    sqllineage_dialect: str,
    scope_real_tables: set[str],
    result: dict[str, set[str]],
    col_to_tables: dict[str, list[str]] | None,
) -> None:
    """Run sqllineage on a single scope and merge column→table into *result*.

    After sqllineage, unresolved columns are looked up in *col_to_tables*
    restricted to *scope_real_tables*:
      - unique match  → attributed to that table
      - multiple matches AND column is a USING join key → attributed to all
      - multiple matches but NOT a join key → left as ``"<unresolved>"``
      - no match      → left as ``"<unresolved>"``
    """
    scope_unresolved: set[str] = set()
    wrapped = f"INSERT INTO {_SYNTHETIC_TARGET} {scope_sql}"
    try:
        runner = LineageRunner(wrapped, dialect=sqllineage_dialect)
        for path in runner.get_column_lineage():
            src_col = path[0]
            col_name = _bare_column_name(src_col.raw_name)
            parent = src_col.parent
            if parent is not None:
                tbl = parent.raw_name.lower()
                if tbl != _SYNTHETIC_TARGET.lower():
                    result.setdefault(tbl, set()).add(col_name)
                    continue
            scope_unresolved.add(col_name)
    except Exception:
        pass

    # Schema-assisted resolution scoped to this scope's real tables only.
    if col_to_tables and scope_unresolved:
        join_keys = _using_columns(scope_node)
        for col_name in scope_unresolved:
            candidates = col_to_tables.get(col_name, [])
            in_scope = [t for t in candidates if t in scope_real_tables]
            if len(in_scope) == 1:
                result.setdefault(in_scope[0], set()).add(col_name)
            elif len(in_scope) > 1 and col_name in join_keys:
                # Verified USING join key — belongs to all joined tables.
                for tbl in in_scope:
                    result.setdefault(tbl, set()).add(col_name)
            else:
                # Ambiguous: column exists in multiple tables but is not a
                # confirmed join key — leave unresolved rather than guess.
                result.setdefault("<unresolved>", set()).add(col_name)
    else:
        result.setdefault("<unresolved>", set()).update(scope_unresolved)


# Mapping from sqlglot dialect names to sqllineage/sqlfluff dialect names.
_DIALECT_MAP = {
    "spark": "sparksql",
    "spark2": "sparksql",
    "tsql": "tsql",
}


def _to_sqllineage_dialect(dialect: str) -> str:
    return _DIALECT_MAP.get(dialect.lower(), dialect.lower())


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def extract_tables_and_columns(
    sql: str,
    dialect: str = "sqlite",
    all_schemas: dict | None = None,
) -> dict[str, set[str]]:
    """Return ``{table_name: {column, ...}}`` for all real source tables in *sql*.

    Parameters
    ----------
    sql:
        Raw SQL string.
    dialect:
        sqlglot / sqllineage dialect (e.g. ``"sqlite"``, ``"duckdb"``,
        ``"snowflake"``).  Defaults to ``"sqlite"``.
    all_schemas:
        Optional ``{schema_name: Schema}`` dict from Neo4j.  When ``None``
        (default), ``get_account_schemas()`` is called automatically.
        Pass ``{}`` to skip schema-assisted resolution.

    Returns
    -------
    dict
        ``{table_name: set_of_column_names}``.
    """
    statement = sqlglot.parse_one(sql, dialect=dialect)
    sqllineage_dialect = _to_sqllineage_dialect(dialect)

    # ── Collect all CTE names (virtual – not real tables) ────────────────────
    cte_names: set[str] = {cte.alias.lower() for cte in statement.find_all(exp.CTE)}

    # ── All real tables referenced anywhere in the query ────────────────────
    source_table_names: set[str] = {
        t.name.lower()
        for t in statement.find_all(exp.Table)
        if t.name.lower() not in cte_names
    }

    result: dict[str, set[str]] = {t: set() for t in source_table_names}
    result["<unresolved>"] = set()

    # ── Fetch schema metadata once (used per-scope below) ───────────────────
    if all_schemas is None:
        from nemo_retriever.tabular_data.ingestion.services.schema import get_account_schemas
        all_schemas = get_account_schemas()

    col_to_tables = _build_column_to_tables(all_schemas) if all_schemas else None

    # ── Run sqllineage scope by scope ────────────────────────────────────────
    #
    # Collect all scopes: CTE bodies + inline subqueries.
    # Only process those that reference at least one real table directly.
    scopes: list[exp.Expression] = []

    for cte in statement.find_all(exp.CTE):
        scopes.append(cte.this)

    for subquery in statement.find_all(exp.Subquery):
        scopes.append(subquery.this)

    if isinstance(statement, exp.Select):
        scopes.append(statement)
    elif hasattr(statement, "this") and isinstance(statement.this, exp.Select):
        scopes.append(statement.this)

    for scope in scopes:
        scope_real_tables = _real_tables_in_select(scope, cte_names)
        if not scope_real_tables:
            continue  # scope only references other CTEs — skip
        _run_lineage_on_scope(
            scope_sql=scope.sql(dialect=dialect),
            scope_node=scope,
            sqllineage_dialect=sqllineage_dialect,
            scope_real_tables=scope_real_tables,
            result=result,
            col_to_tables=col_to_tables,
        )

    if not result.get("<unresolved>"):
        result.pop("<unresolved>", None)

    return result


# ---------------------------------------------------------------------------
# Example SQL
# ---------------------------------------------------------------------------

_EXAMPLE_SQL = """
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
            WHEN recency = 1 AND frequency + monetary IN (1, 2, 3, 4) THEN 'Champions'
            WHEN recency IN (4, 5) AND frequency + monetary IN (1, 2) THEN 'Can''t Lose Them'
            WHEN recency IN (4, 5) AND frequency + monetary IN (3, 4, 5, 6) THEN 'Hibernating'
            WHEN recency IN (4, 5) AND frequency + monetary IN (7, 8, 9, 10) THEN 'Lost'
            WHEN recency IN (2, 3) AND frequency + monetary IN (1, 2, 3, 4) THEN 'Loyal Customers'
            WHEN recency = 3 AND frequency + monetary IN (5, 6) THEN 'Needs Attention'
            WHEN recency = 1 AND frequency + monetary IN (7, 8) THEN 'Recent Users'
            WHEN recency = 1 AND frequency + monetary IN (5, 6) OR
                recency = 2 AND frequency + monetary IN (5, 6, 7, 8) THEN 'Potential Loyalists'
            WHEN recency = 1 AND frequency + monetary IN (9, 10) THEN 'Price Sensitive'
            WHEN recency = 2 AND frequency + monetary IN (9, 10) THEN 'Promising'
            WHEN recency = 3 AND frequency + monetary IN (7, 8, 9, 10) THEN 'About to Sleep'
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract tables and columns from SQL.")
    parser.add_argument("--dialect", default="sqlite", help="SQL dialect (default: sqlite)")
    args = parser.parse_args()

    # Pass all_schemas={} to skip Neo4j when running standalone.
    tables_and_columns = extract_tables_and_columns(_EXAMPLE_SQL, dialect=args.dialect)

    print(f"Tables and columns extracted (dialect={args.dialect!r}):")
    print("=" * 50)
    for table, columns in sorted(tables_and_columns.items()):
        print(f"\n  {table}")
        for col in sorted(columns):
            print(f"    - {col}")
