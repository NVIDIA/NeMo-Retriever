---
name: sql-generation
description: How to generate, validate, and self-correct SQL from the query plan. Covers constructability decisions, FK join rules, and alias verification.
---

# SQL Generation Skill

## Purpose

This skill covers the full SQL generation loop: constructing SQL from retrieved context,
validating it, self-correcting errors, and deciding when a question is unanswerable.

---

## Constructability Decision

Before writing SQL, assess whether the retrieved context is sufficient:

**Constructable** — proceed with SQL generation when:
- At least one relevant table with matching columns is available.
- The question can be answered with SELECT + optional JOINs/aggregations.

**Unconstructable** — skip SQL and explain in `answer` when:
- No relevant tables or columns were found.
- The question requires data that is definitively absent from the knowledge base.
- The question is purely conversational and requires no database query.

---

## SQL Generation Path

### 0. Honour the question's shape

Before writing any SQL, decide whether the user asked for an aggregate
or a row list, and translate the plan accordingly. Do NOT fall back to a
row-listing `SELECT <columns>` when the question asks for a count, sum,
average, or extreme:

| Question wording                                              | Shape                                                              |
|---------------------------------------------------------------|--------------------------------------------------------------------|
| "how many", "count of", "number of"                           | `SELECT COUNT(...)` — single scalar row                            |
| "total", "sum of", "what is the total"                        | `SELECT SUM(...)`                                                  |
| "average", "mean", "avg"                                      | `SELECT AVG(...)`                                                  |
| "max", "min", "highest", "lowest", "most", "least"            | `SELECT MAX(...)` / `SELECT MIN(...)` (or `ORDER BY ... LIMIT 1`)  |
| "per <X>", "by <X>", "for each <X>", "breakdown by <X>"       | aggregate + `GROUP BY <X>`                                          |
| "list", "show", "which", "what are", no aggregate keyword     | `SELECT <columns>` (row-listing)                                   |

If the plan's `select_expressions` doesn't match the wording (e.g. the
question says "how many DORs were created in Q4 2025?" but the plan
selects raw columns), rewrite the SELECT to be the correct aggregate.
Aggregate questions return a single scalar row; never emit a list of
rows for them.

### 1. Reference idiomatic patterns

- Use `relevant_queries` for idiomatic patterns (date functions, aggregation style).

### 2. Build the FROM / JOIN clause

- Use `relevant_fks` as the authoritative JOIN condition source.
- **Never invent a JOIN** not present in `relevant_fks`.
- Prefer INNER JOIN unless the question requires preserving unmatched rows (LEFT / RIGHT JOIN).
- Avoid many-to-many fan-out — route through dimension tables.

### 3. Apply the SQL rules

- Use ONLY columns that appear under each table's available-column list in `relevant_tables`.
- Never alias-reference a column from the wrong table.
- Allowed constructs: SELECT, FROM, JOIN, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT,
  CTEs (WITH), window functions, CASE WHEN, standard aggregate functions.
- Forbidden constructs: `::` casts, `FILTER (WHERE ...)`, `QUALIFY`, `DISTINCT ON`,
  `GROUP BY ALL`, PostgreSQL-specific functions.
- Time windows: "last week/month/year" → most recently COMPLETED calendar period.
  Use dialect-appropriate date functions; never use rolling windows like `DATEADD(day,-7,...)`.
- Case-sensitive literals: never alter the capitalisation of user-supplied values.
- `ORDER BY` may only reference selected aliases or GROUP BY columns.

### 4. Mandatory pre-output checklist

Before calling `validate_sql`, verify ALL of the following — fix any violation first:

1. **Schema prefix**: every table is written as `SCHEMA.TABLE AS alias` — never bare `TABLE`.
2. **Column qualification**: every column is written as `alias.column` — never bare `column`.
3. **Alias coverage**: every alias used in SELECT / WHERE / GROUP BY / ORDER BY / HAVING is defined in FROM / JOIN.
4. No unqualified table or column reference exists anywhere in the query.

---

## Validation & Self-Correction Loop

After generating SQL, call `validate_sql(sql)`.

| `valid` | `error` field | Action |
|---------|--------------|--------|
| `true` | — | Proceed to `execute_sql` |
| `false` | syntax / schema error | Fix and retry (max 4 attempts) |
| `false` | "not authorized" / SELECT-only violation | Do not retry; report unconstructable |

**At 4 failed attempts**, accept the latest draft; if it is still invalid,
set `sql_code: ""` and explain in `answer`.

---

## Complex SQL Patterns

You are proficient in:
- Multi-table JOINs (inner / left / right / full outer)
- Aggregations: `SUM`, `AVG`, `COUNT`, `MIN`, `MAX`
- Subqueries and CTEs (`WITH` clauses)
- `WHERE` / `HAVING` filter combinations
- Window functions (`ROW_NUMBER`, `RANK`, `LAG`, `LEAD`, etc.)
- Calendar-based date filters (completed periods, fiscal quarters)
- `CASE WHEN` for business category classification
- `NULL` handling and safe conversions

When grouping by business categories, always use `CASE WHEN` to explicitly classify rows
into the categories mentioned in the question — never rely on raw column values alone.

---

## Examples of Correct Behaviour

**Question**: "What were the top 5 products by revenue last month?"

Good approach:
1. extract_entities → entities: ["Product", "Revenue"]
2. retrieve_semantic_candidates → finds `SALES.INVOICELINES`, `WAREHOUSE.STOCKITEMS`
3. Build SQL:
```sql
SELECT si.StockItemName, SUM(il.UnitPrice * il.Quantity) AS revenue
FROM SALES.INVOICELINES il
JOIN WAREHOUSE.STOCKITEMS si ON il.StockItemID = si.StockItemID
WHERE il.InvoiceDate BETWEEN DATE_TRUNC('month', DATEADD('month', -1, CURRENT_DATE))
                          AND LAST_DAY(DATEADD('month', -1, CURRENT_DATE))
GROUP BY si.StockItemName
ORDER BY revenue DESC
LIMIT 5
```
4. validate_sql → valid
5. execute_sql → result rows

---

## What NOT to Do

- Do NOT use table or column names not present in the retrieval output.
- Do NOT JOIN two tables without a FK from `relevant_fks` linking them.
- Do NOT emit explanatory text mixed with the SQL — SQL goes only in `sql_code`.
