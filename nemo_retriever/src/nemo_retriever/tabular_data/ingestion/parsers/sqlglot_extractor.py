# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify


@dataclass
class TableMatch:
    """Extraction result for a single source table.

    Attributes
    ----------
    columns:
        Set of column names referenced in the SQL for this table.
    schema_name:
        The ``all_schemas`` key (Neo4j schema name) that owns this table,
        or ``None`` when the owning schema could not be determined.
    """

    columns: set[str] = field(default_factory=set)
    schema_name: str | None = None


@dataclass
class JoinPair:
    """A single join condition between two resolved columns.

    Both table references are resolved through the alias map to their
    real (non-CTE, non-alias) qualified table names.
    """

    left_table: str
    left_column: str
    right_table: str
    right_column: str
    operator: str = "="


@dataclass
class ExtractionResult:
    """Container for the full output of :func:`extract_tables_and_columns`.

    Attributes
    ----------
    tables:
        ``{table_key: TableMatch}`` mapping.
    ast_node_count:
        Total number of nodes in the sqlglot AST.  Cheap structural
        fingerprint used to pre-filter duplicate candidates.
    joins:
        Equi-join column pairs extracted from explicit ``JOIN … ON``
        and ``JOIN … USING`` clauses, with aliases fully resolved.
    """

    tables: dict[str, TableMatch] = field(default_factory=dict)
    ast_node_count: int = 0
    joins: list[JoinPair] = field(default_factory=list)


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


_BINARY_CMP = (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.NEQ)

_OP_SYMBOL: dict[type, str] = {
    exp.EQ: "=",
    exp.GT: ">",
    exp.GTE: ">=",
    exp.LT: "<",
    exp.LTE: "<=",
    exp.NEQ: "!=",
}


def _unwrap_column(node: exp.Expression) -> exp.Column | None:
    """Return the single ``Column`` inside *node*, unwrapping functions/casts.

    When *node* contains zero or more than one ``Column`` the reference is
    ambiguous (e.g. ``col_a + col_b``), so ``None`` is returned.
    """
    if isinstance(node, exp.Column):
        return node
    cols = list(node.find_all(exp.Column))
    return cols[0] if len(cols) == 1 else None


def _build_derived_col_map(
    qualified: exp.Expression,
    alias_map: dict[str, str],
    source_table_names: set[str],
) -> dict[str, dict[str, tuple[str, str]]]:
    """Map derived-table aliases to their output columns' real source.

    For ``(SELECT order_items.order_id, ... FROM order_items) AS d``, produces:
    ``{"d": {"order_id": ("order_items", "order_id"), ...}}``

    Only maps columns that trace unambiguously to a single real source table.
    """
    derived: dict[str, dict[str, tuple[str, str]]] = {}
    for subq in qualified.find_all(exp.Subquery):
        alias = subq.alias.lower() if subq.alias else None
        if not alias:
            continue
        inner_select = subq.find(exp.Select)
        if not inner_select:
            continue
        col_map: dict[str, tuple[str, str]] = {}
        for expr in inner_select.expressions:
            output_name = expr.alias.lower() if isinstance(expr, exp.Alias) else None
            source_expr = expr.this if isinstance(expr, exp.Alias) else expr
            col = _unwrap_column(source_expr)
            if col is None:
                continue
            if output_name is None:
                output_name = col.name.lower() if col.name else None
            if not output_name:
                continue
            tbl_ref = col.table.lower() if col.table else None
            real_table = alias_map.get(tbl_ref) if tbl_ref else None
            if real_table and real_table in source_table_names:
                col_map[output_name] = (real_table, col.name.lower())
        if col_map:
            derived[alias] = col_map
    return derived


def _extract_join_pairs(
    qualified: exp.Expression,
    alias_map: dict[str, str],
    source_table_names: set[str],
) -> list[JoinPair]:
    """Extract join column pairs from the (post-qualify) AST.

    After ``qualify()``, ``USING`` clauses are rewritten into ``ON``, so
    walking ``JOIN … ON`` covers both syntaxes.  Falls back to raw
    ``USING`` args when ``ON`` is absent (``qualify()`` failed or the
    dialect preserved ``USING``).

    Handled condition types:

    * Binary comparisons (``=``, ``>``, ``>=``, ``<``, ``<=``, ``!=``)
    * ``BETWEEN … AND …``
    * Function-wrapped / cast-wrapped columns (e.g. ``LOWER(t.col)``)
    * Columns referencing derived tables (traced back to source real table)
    """
    derived_map = _build_derived_col_map(qualified, alias_map, source_table_names)
    pairs: list[JoinPair] = []
    seen: set[tuple[str, str, str, str]] = set()

    def _resolve_col(col: exp.Column) -> tuple[str | None, str]:
        """Resolve a column to (real_table, col_name), using derived_map as fallback."""
        col_name = col.name.lower()
        tbl_ref = col.table.lower() if col.table else None
        real_table = alias_map.get(tbl_ref) if tbl_ref else None
        if real_table and real_table in source_table_names:
            return real_table, col_name
        if tbl_ref and tbl_ref in derived_map:
            entry = derived_map[tbl_ref].get(col_name)
            if entry:
                return entry
        return None, col_name

    def _try_add(lc: exp.Column, rc: exp.Column, op: str = "=") -> None:
        """Resolve aliases and append a canonically-ordered, deduplicated ``JoinPair``."""
        lt, l_col = _resolve_col(lc)
        rt, r_col = _resolve_col(rc)
        if not lt or not rt:
            return
        if lt == rt and l_col == r_col:
            return
        if (lt, l_col) > (rt, r_col):
            lt, l_col, rt, r_col = rt, r_col, lt, l_col
        key = (lt, l_col, rt, r_col)
        if key not in seen:
            seen.add(key)
            pairs.append(
                JoinPair(
                    left_table=lt,
                    left_column=l_col,
                    right_table=rt,
                    right_column=r_col,
                    operator=op,
                )
            )

    for select_node in qualified.find_all(exp.Select):
        from_clause = select_node.args.get("from_")
        join_nodes = select_node.args.get("joins") or []
        if not join_nodes:
            continue

        chain: list[str] = []
        if from_clause:
            ft = from_clause.this
            if isinstance(ft, exp.Table):
                resolved = alias_map.get((ft.alias or ft.name).lower())
                if resolved:
                    chain.append(resolved)

        for join_node in join_nodes:
            right_ast = join_node.this
            right_resolved = None
            if isinstance(right_ast, exp.Table):
                right_resolved = alias_map.get((right_ast.alias or right_ast.name).lower())

            on_expr = join_node.args.get("on")
            if on_expr:
                for cmp in on_expr.find_all(*_BINARY_CMP):
                    lc = _unwrap_column(cmp.left)
                    rc = _unwrap_column(cmp.right)
                    if lc and rc:
                        _try_add(lc, rc, _OP_SYMBOL.get(type(cmp), "="))

                for between in on_expr.find_all(exp.Between):
                    main_c = _unwrap_column(between.this)
                    low_arg = between.args.get("low")
                    high_arg = between.args.get("high")
                    low_c = _unwrap_column(low_arg) if low_arg else None
                    high_c = _unwrap_column(high_arg) if high_arg else None
                    if main_c and low_c:
                        _try_add(main_c, low_c, "BETWEEN")
                    if main_c and high_c:
                        _try_add(main_c, high_c, "BETWEEN")
            else:
                using_cols = join_node.args.get("using") or []
                if using_cols and chain and right_resolved:
                    left_table = chain[-1]
                    for col_ident in using_cols:
                        col_name = col_ident.name.lower()
                        if left_table in source_table_names and right_resolved in source_table_names:
                            lt, lc, rt, rc = left_table, col_name, right_resolved, col_name
                            if (lt, lc) > (rt, rc):
                                lt, lc, rt, rc = rt, rc, lt, lc
                            key = (lt, lc, rt, rc)
                            if key not in seen:
                                seen.add(key)
                                pairs.append(
                                    JoinPair(
                                        left_table=lt,
                                        left_column=lc,
                                        right_table=rt,
                                        right_column=rc,
                                    )
                                )

            if right_resolved:
                chain.append(right_resolved)

    return pairs


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def extract_tables_and_columns(
    sql: str,
    dialect: str = "sqlite",
    all_schemas: dict = {},
) -> ExtractionResult:
    """Return an :class:`ExtractionResult` for all real source tables in *sql*.

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
    ExtractionResult
        ``.tables`` — ``{table_key: TableMatch}`` where *table_key* is
        ``"schema.table"`` when the SQL uses a schema prefix, or bare
        ``"table"`` otherwise.  Each ``TableMatch.schema_name`` carries
        the *all_schemas* key that owns the table (``None`` when
        undetermined).

        ``.ast_node_count`` — total number of sqlglot AST nodes; used as
        a cheap structural fingerprint to pre-filter duplicate candidates.
    """
    try:
        statement = sqlglot.parse_one(sql, dialect=dialect)
    except Exception:
        return ExtractionResult()

    ast_node_count = sum(1 for _ in statement.walk())

    # CTE names are virtual — not real source tables.
    cte_names: set[str] = {cte.alias.lower() for cte in statement.find_all(exp.CTE)}

    # Real (non-CTE) table names referenced anywhere in the query.
    # Uses qualified names (``schema.table``) when the SQL includes a schema prefix,
    # so that same-named tables from different schemas are kept distinct.
    source_table_names: set[str] = {
        _qualified_table_name(t) for t in statement.find_all(exp.Table) if t.name.lower() not in cte_names
    }

    if not source_table_names:
        return ExtractionResult(ast_node_count=ast_node_count)

    # alias → real table name (e.g. "o" → "orders")
    alias_map = _alias_to_table_map(statement, cte_names)

    # Pre-build table_key → schema_key so every TableMatch knows its owner.
    #
    # Two-pass strategy to avoid scanning every table in every schema:
    #   1. Schema-qualified keys ("schema_a.orders") — the schema name is
    #      already embedded; resolve directly against all_schemas keys.
    #   2. Bare keys ("orders") — only look up the tables we actually need,
    #      stopping as soon as all are resolved.
    table_to_schema: dict[str, str] = {}
    bare_tables: set[str] = set()
    schema_keys_lower = {k.lower() for k in all_schemas}

    for tbl_key in source_table_names:
        if "." in tbl_key:
            schema_part, _ = tbl_key.split(".", 1)
            if schema_part in schema_keys_lower:
                table_to_schema[tbl_key] = schema_part
        else:
            bare_tables.add(tbl_key)

    for schema_key, s in all_schemas.items():
        if not bare_tables:
            break
        skey = schema_key.lower()
        for tbl in list(bare_tables):
            if s.table_exists(tbl):
                table_to_schema[tbl] = skey
                bare_tables.discard(tbl)

    result: dict[str, TableMatch] = {t: TableMatch(schema_name=table_to_schema.get(t)) for t in source_table_names}
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
            result[real_table].columns.add(col_name)
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
                qualified_name = f"{skey}.{tbl_n}"
                matched = (
                    qualified_name
                    if qualified_name in source_table_names
                    else (tbl_n if tbl_n in source_table_names else None)
                )
                if matched and matched not in col_to_tables.get(col_n, []):
                    col_to_tables.setdefault(col_n, []).append(matched)

        for col_name in unresolved:
            candidates = col_to_tables.get(col_name, [])
            if len(candidates) == 1:
                result[candidates[0]].columns.add(col_name)
            elif len(candidates) > 1 and col_name in join_keys:
                # Cross-validate: only attribute to tables that are both in the
                # schema candidates AND in the actual USING join for this column.
                matched = [t for t in candidates if t in join_keys[col_name]]
                for tbl in matched or candidates:
                    result[tbl].columns.add(col_name)
            # else: ambiguous — omit rather than guess.

    join_pairs = _extract_join_pairs(qualified, alias_map, source_table_names)

    # Drop real-table entries that ended up with no columns attributed.
    return ExtractionResult(
        tables={k: v for k, v in result.items() if v.columns},
        ast_node_count=ast_node_count,
        joins=join_pairs,
    )
