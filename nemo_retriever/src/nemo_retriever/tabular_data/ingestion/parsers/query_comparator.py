# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare two SQL queries for structural equivalence.

The comparison is **insensitive to SELECT-column order** in the main SELECT,
CTEs, and nested subqueries.  An optional ``ignore_literals`` flag neutralises
all constant values (strings, numbers, booleans) so that two queries differing
only in filter values are considered equal.

Example
-------
::

    from query_comparator import compare_queries

    same = compare_queries(
        "SELECT id, name FROM users WHERE created > '2024-01-01'",
        "SELECT name, id FROM users WHERE created > '2025-06-15'",
        ignore_literals=True,
    )
    # same -> True
"""

from __future__ import annotations

import logging

import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)


def _normalize(tree: exp.Expression, dialect: str | None, ignore_literals: bool) -> str:
    """Return a canonical SQL string for *tree* suitable for equality comparison.

    1. Optionally replace every literal value with a ``?`` placeholder.
    2. Sort the SELECT expressions of every ``SELECT`` clause alphabetically
       so that column order does not affect the comparison.
    3. Lower-case the resulting SQL for case-insensitive identifier matching.
    """
    tree = tree.copy()

    if ignore_literals:
        for literal in tree.find_all(exp.Literal):
            literal.replace(exp.Placeholder())
        # Double-quoted values (e.g. "15/12/2024") are parsed as quoted
        # identifiers rather than string literals.  Neutralise unqualified
        # quoted-identifier Column nodes that almost certainly represent
        # constant values rather than real column references.
        for col in list(tree.find_all(exp.Column)):
            ident = col.this
            if (
                isinstance(ident, exp.Identifier)
                and ident.args.get("quoted")
                and not col.table
            ):
                col.replace(exp.Placeholder())

    for select_node in tree.find_all(exp.Select):
        exprs = list(select_node.expressions)
        if len(exprs) > 1:
            exprs.sort(key=lambda e: e.sql(dialect=dialect).lower())
            select_node.set("expressions", exprs)

    return tree.sql(dialect=dialect).lower()


def normalize_sql(
    sql: str,
    dialect: str | None = None,
    ignore_literals: bool = False,
) -> str | None:
    """Parse *sql* and return its canonical normalised form.

    Returns ``None`` when the SQL cannot be parsed.  The returned string can
    be cached and compared with ``==`` to detect structural equivalence
    without re-parsing.
    """
    try:
        tree = sqlglot.parse_one(sql, dialect=dialect)
    except sqlglot.errors.ParseError:
        return None
    return _normalize(tree, dialect, ignore_literals)


def compare_queries(
    sql1: str,
    sql2: str,
    dialect: str | None = None,
    ignore_literals: bool = False,
) -> bool:
    """Return ``True`` when *sql1* and *sql2* are structurally equivalent.

    Parameters
    ----------
    sql1, sql2:
        SQL query strings to compare.
    dialect:
        Optional sqlglot dialect (e.g. ``"duckdb"``, ``"snowflake"``).
    ignore_literals:
        When ``True``, all constant values (strings, numbers) are replaced
        with placeholders before comparison.  This makes
        ``WHERE x = '2024'`` equal to ``WHERE x = '2025'``.
    """
    n1 = normalize_sql(sql1, dialect=dialect, ignore_literals=ignore_literals)
    n2 = normalize_sql(sql2, dialect=dialect, ignore_literals=ignore_literals)
    if n1 is None or n2 is None:
        return False
    return n1 == n2
