"""Extract tables and columns referenced in a SQL query.

Strategy
--------
sqlglot parses the full query into an AST, then the ``qualify`` optimizer
pass annotates every ``Column`` node with its resolved source table by
propagating schema information through CTEs and subquery scopes.

A pre-pass builds an alias→table map so that table aliases
(``SELECT o.id FROM orders AS o``) are transparently resolved when reading
``col.table`` from the qualified AST.

Columns that remain unresolved after qualification are looked up in
``all_schemas`` (Neo4j metadata) and attributed when the match is
unambiguous within the query's source tables.

Pass ``all_schemas={}`` (the default) to skip schema-assisted resolution
and rely solely on ``qualify()``.

Usage (standalone):
    python sqlglot_extractor.py [--dialect <dialect>]
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _qualified_table_name(table: exp.Table) -> str:
    """Return the fully-qualified table name as ``schema.table`` when a schema is
    present in the AST, or just ``table`` otherwise."""
    parts = [p for p in [table.db, table.name] if p]
    return ".".join(parts).lower()


def _alias_to_table_map(statement: exp.Expression, cte_names: set[str]) -> dict[str, str]:
    """Return ``{alias_or_bare_name: qualified_table_name}`` for every real table.

    Qualified names are ``schema.table`` when the SQL references a schema prefix,
    or plain ``table`` otherwise.  Both the alias *and* the bare table name are
    registered so that ``col.table`` from the qualified AST always resolves,
    regardless of whether an alias was used.
    """
    mapping: dict[str, str] = {}
    for table in statement.find_all(exp.Table):
        if table.name.lower() in cte_names:
            continue
        qualified = _qualified_table_name(table)
        alias = table.alias.lower() if table.alias else table.name.lower()
        mapping[alias] = qualified
        mapping[table.name.lower()] = qualified  # fallback for unaliased bare references
    return mapping


def _build_schema_dict(all_schemas: dict) -> dict[str, dict[str, str]]:
    """Build a flat ``{table: {col: "TEXT"}}`` dict for sqlglot's qualify pass.

    qualify() resolves bare column references using this flat mapping.  A flat
    dict works for both unqualified SQL (``FROM orders``) and schema-qualified
    SQL (``FROM schema_a.orders AS a``): in the latter case qualify() resolves
    columns through the alias, so the schema prefix is not needed here.
    Multi-schema disambiguation is handled separately via ``alias_map`` and
    ``source_table_names`` (which use fully-qualified keys).
    """
    schema: dict[str, dict[str, str]] = {}
    for s in all_schemas.values():
        df = s.columns_df
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            tbl = str(row["table_name"]).lower()
            col = str(row["column_name"]).lower()
            schema.setdefault(tbl, {})[col] = "TEXT"
    return schema


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def extract_tables_and_columns(
    sql: str,
    dialect: str = "sqlite",
    all_schemas: dict = {},
) -> dict[str, set[str]]:
    """Return ``{table_name: {column, ...}}`` for all real source tables in *sql*.

    Parameters
    ----------
    sql:
        Raw SQL string.
    dialect:
        sqlglot dialect (e.g. ``"sqlite"``, ``"duckdb"``, ``"snowflake"``).
        Defaults to ``"sqlite"``.
    all_schemas:
        ``{schema_name: Schema}`` dict from Neo4j.  Pass ``{}`` (default) to
        skip schema-assisted resolution and rely solely on ``qualify()``.

    Returns
    -------
    dict
        ``{table_name: set_of_column_names}``.
        Keys are qualified (``schema.table``) when the SQL uses a schema prefix,
        or bare (``table``) otherwise.
    """
    try:
        statement = sqlglot.parse_one(sql, dialect=dialect)
    except Exception:
        return {}

    # CTE names are virtual — not real source tables.
    cte_names: set[str] = {cte.alias.lower() for cte in statement.find_all(exp.CTE)}

    # Real (non-CTE) table names referenced anywhere in the query.
    # Uses qualified names (``schema.table``) when the SQL includes a schema prefix,
    # so that same-named tables from different schemas are kept distinct.
    source_table_names: set[str] = {
        _qualified_table_name(t)
        for t in statement.find_all(exp.Table)
        if t.name.lower() not in cte_names
    }

    if not source_table_names:
        return {}

    # alias → real table name (e.g. "o" → "orders")
    alias_map = _alias_to_table_map(statement, cte_names)

    result: dict[str, set[str]] = {t: set() for t in source_table_names}
    unresolved: set[str] = set()

    # qualify() mutates the AST in-place, rewriting USING into ON — extract
    # join keys now, before qualify destroys them.
    # Maps each USING column to the set of real tables joined on that column,
    # derived from the ordered FROM + JOIN chain of each SELECT scope.
    join_keys: dict[str, set[str]] = {}
    for select in statement.find_all(exp.Select):
        from_clause = select.args.get("from_")  # sqlglot uses "from_" — "from" is a Python keyword
        joins = select.args.get("joins") or []
        if not from_clause or not joins:
            continue
        # Build the ordered list of real tables for this SELECT's join chain.
        chain: list[str] = []
        ft = from_clause.this
        if isinstance(ft, exp.Table):
            n = alias_map.get((ft.alias or ft.name).lower())
            if n:
                chain.append(n)
        for join in joins:
            right = join.this
            right_name = None
            if isinstance(right, exp.Table):
                right_name = alias_map.get((right.alias or right.name).lower())

            # Collect join-key column names from USING or equivalent ON conditions.
            # Some dialects (or sqlglot itself) convert USING to ON during parsing,
            # so we check both: USING args first, then ON equalities where both sides
            # share the same column name (t1.col = t2.col ≡ USING (col)).
            key_cols: list[str] = [c.name.lower() for c in (join.args.get("using") or [])]
            if not key_cols:
                on_expr = join.args.get("on")
                if on_expr:
                    for eq in on_expr.find_all(exp.EQ):
                        lc, rc = eq.left, eq.right
                        if (
                            isinstance(lc, exp.Column)
                            and isinstance(rc, exp.Column)
                            and lc.name.lower() == rc.name.lower()
                        ):
                            key_cols.append(lc.name.lower())

            for col_name in key_cols:
                participants = {t for t in chain if t in source_table_names}
                if right_name and right_name in source_table_names:
                    participants.add(right_name)
                if participants:
                    join_keys.setdefault(col_name, set()).update(participants)

            if right_name:
                chain.append(right_name)

    # qualify() annotates every Column node with its resolved source table.
    schema_dict = _build_schema_dict(all_schemas) if all_schemas else {}
    try:
        qualified = qualify(
            statement,
            dialect=dialect,
            schema=schema_dict,
            qualify_columns=True,
            validate_qualify_columns=False,
            expand_stars=bool(schema_dict),
        )
    except Exception:
        qualified = statement  # fall back to unqualified AST if optimizer fails

    # Walk every Column node; after qualify, col.table names the resolved table/alias.
    for col in qualified.find_all(exp.Column):
        col_name = col.name.lower() if col.name else None
        if not col_name:
            continue
        table_ref = col.table.lower() if col.table else None
        real_table = alias_map.get(table_ref) if table_ref else None
        if real_table and real_table in source_table_names:
            result[real_table].add(col_name)
        elif not table_ref or table_ref in cte_names:
            # Bare / CTE-referencing column — candidate for schema-assisted lookup.
            unresolved.add(col_name)
        # else: alias or subquery reference that doesn't map to a real table — skip.

    # Schema-assisted resolution for columns not attributed by qualify.
    if unresolved and all_schemas:
        col_to_tables: dict[str, list[str]] = {}
        for schema_name, s in all_schemas.items():
            df = s.columns_df
            if df is None or df.empty:
                continue
            skey = schema_name.lower()
            for _, row in df.iterrows():
                col_n = str(row["column_name"]).lower()
                tbl_n = str(row["table_name"]).lower()
                # Try the qualified name first (schema.table); fall back to bare
                # table name for SQL that doesn't prefix tables with a schema.
                qualified = f"{skey}.{tbl_n}"
                matched = qualified if qualified in source_table_names else (
                    tbl_n if tbl_n in source_table_names else None
                )
                if matched and matched not in col_to_tables.get(col_n, []):
                    col_to_tables.setdefault(col_n, []).append(matched)

        for col_name in unresolved:
            candidates = col_to_tables.get(col_name, [])
            if len(candidates) == 1:
                result[candidates[0]].add(col_name)
            elif len(candidates) > 1 and col_name in join_keys:
                # Cross-validate: only attribute to tables that are both in the
                # schema candidates AND in the actual USING join for this column.
                matched = [t for t in candidates if t in join_keys[col_name]]
                for tbl in (matched or candidates):
                    result[tbl].add(col_name)
            # else: ambiguous — omit rather than guess.

    # Drop real-table entries that ended up with no columns attributed.
    return {k: v for k, v in result.items() if v}


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

    import pandas as pd

    parser = argparse.ArgumentParser(description="Extract tables and columns from SQL.")
    parser.add_argument("--dialect", default="sqlite", help="SQL dialect (default: sqlite)")
    args = parser.parse_args()

    from nemo_retriever.tabular_data.ingestion.services.schema import get_account_schemas
    all_schemas = get_account_schemas()

    tables_and_columns = extract_tables_and_columns(
        _EXAMPLE_SQL, dialect=args.dialect, all_schemas=all_schemas
    )

    print(f"Tables and columns extracted (dialect={args.dialect!r}):")
    print("=" * 50)
    for table, columns in sorted(tables_and_columns.items()):
        print(f"\n  {table}")
        for col in sorted(columns):
            print(f"    - {col}")
