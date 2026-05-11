---
name: sql-rules
description: Hard SQL syntax and shape rules that every SELECT must obey — fully-qualified identifiers, JOIN-vs-WHERE separation, single-quote literals, SELECT-only enforcement, banned dialect features.
---

# SQL Rules

Every SQL the agent emits — whether produced by `plan_query`, `generate_sql`, or
`fix_sql` — MUST satisfy every rule below. A violation is a hard validation
failure.

## Identifiers

- **Tables**: `SCHEMA.TABLE AS alias` — never a bare table name.
- **Columns**: `alias.column` — never a bare column name and never qualified
  by table name (`table.column` is also wrong; only `alias.column`).
- Every alias used anywhere in `SELECT` / `WHERE` / `GROUP BY` / `HAVING` /
  `ORDER BY` MUST be defined in `FROM` / `JOIN`.
- Use ONLY tables that appear in the retrieval context's `relevant_tables`
  (or, in Phase 2, the plan's `tables_to_use`). Never reference unlisted
  schema objects.

## JOIN vs WHERE

- When more than one table is involved, link them with an explicit
  `JOIN ... ON ...` clause. The FK predicate goes in the `JOIN` `ON`, NOT
  in `WHERE`.
- `WHERE` is reserved for row-level filters.
- Use ONLY foreign-key relationships listed in `relevant_fks` for `JOIN`
  conditions. Never invent a JOIN that's not in the FK list.
- Prefer `INNER JOIN` unless the question requires preserving unmatched
  rows (`LEFT` / `RIGHT JOIN`).

## No correlated subqueries for joinable conditions

When a related table is used to compute a `COUNT` / `SUM` / `EXISTS` /
comparison against a row of the main table, express it as
`JOIN ... ON ...` plus `GROUP BY` and `HAVING` — NOT as a
`(SELECT ... WHERE outer.col = inner.col) <op> ...` correlated subquery.
`EXISTS` / `IN (SELECT ...)` are also discouraged when an explicit JOIN
expresses the same intent.

## Literals & casing

- String literals MUST use single quotes. Double quotes are for identifiers
  only (and even then, prefer no quoting).
- Never alter the capitalisation of user-supplied values.

## SELECT-only

- No `INSERT` / `UPDATE` / `DELETE` / `DROP` / `ALTER` / `CREATE`.
- No SQL comments. No markdown fences in the output.

## Banned constructs

- `::` casts.
- `FILTER (WHERE ...)`.
- `QUALIFY`.
- `DISTINCT ON`.
- `GROUP BY ALL`.
- PostgreSQL-only syntax.

## Time windows

- "last week / month / year" = most recent **completed** calendar period.
- No rolling windows (e.g. `DATEADD(day,-7,...)` is not "last week").
- Use dialect-appropriate date functions.

## ORDER BY

- May reference only aggregated fields (by alias) or columns present in
  `SELECT` / `GROUP BY`.

## Pre-output checklist

Before emitting SQL, verify:

1. Every table is written as `SCHEMA.TABLE AS alias`.
2. Every column reference uses `alias.column`.
3. Every alias used in any clause is defined in `FROM` / `JOIN`.
4. Every JOIN predicate is in `ON`, never in `WHERE`.
5. No banned construct appears anywhere.
6. The query is `SELECT`-only.
