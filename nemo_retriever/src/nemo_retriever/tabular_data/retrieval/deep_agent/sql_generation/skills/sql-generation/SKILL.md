---
name: sql-generation
description: How to generate, validate, and self-correct SQL from retrieved semantic candidates. Covers constructability decisions, FK join rules, alias verification, and the fallback path.
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

## Primary SQL Generation Path (Semantic)

Use this path when `complex_candidates_str` or `table_groups` contain useful context.

### 1. Select the best candidates

- Prefer `[CERTIFIED]` snippets in `complex_candidates_str` — they encode validated business logic.
- If multiple snippets could answer the question, pick the most relevant one; do NOT blend aliases
  from different snippets.
- Use `relevant_queries` for idiomatic patterns (date functions, aggregation style).

### 2. Build the FROM / JOIN clause

- Use `table_groups` to pick the connected component whose tables cover the question.
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
| `false` | syntax / schema error | Fix and retry (max 4 attempts on semantic path) |
| `false` | "not authorized" / SELECT-only violation | Do not retry; report unconstructable |

**At 4 failed semantic attempts**, switch to the fallback path (see below).

---

## Fallback Path (Table-Only)

Use when the semantic path exhausts retries or `complex_candidates_str` is empty.

1. Use the full `relevant_tables` list as the schema source.
2. Build SQL using only the tables and columns listed there.
3. Apply `relevant_fks` for JOIN conditions.
4. Validate and execute as normal.
5. If the fallback also fails after 4 attempts, set `sql_code: ""` and explain in `answer`.

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

- Do NOT write SQL before calling `retrieve_semantic_candidates`.
- Do NOT use table or column names not present in the retrieval output.
- Do NOT copy aliases verbatim from `sql_snippet` examples.
- Do NOT JOIN tables from different `table_groups` without a FK linking them.
- Do NOT emit explanatory text mixed with the SQL — SQL goes only in `sql_code`.
