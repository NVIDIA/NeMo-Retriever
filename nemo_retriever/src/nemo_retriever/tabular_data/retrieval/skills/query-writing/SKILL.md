---
name: query-writing
description: For writing and executing SQL queries - from simple single-table queries to complex multi-table JOINs and aggregations
---

# Query Writing Skill

## When to Use This Skill

Use this skill when you need to answer a question by writing and executing a SQL query.

The built-in `write_todos` tool can be used to plan your work, but:
- Todo lists and their statuses are **internal**.
- Do **not** show todo items or statuses to the user.
- Do **not** stop after planning; always continue until you have executed SQL
  and produced the final structured answer (via the `answer-formatting` skill).

## Workflow for Simple Queries

For straightforward questions about a single table:

1. **Identify the table** - Which table has the data?
2. **Get the schema** - Use `sql_db_schema` to see columns
3. **Write the query** - SELECT relevant columns with WHERE/LIMIT/ORDER BY
4. **Execute** - Run with `sql_db_query`
5. **Format answer (MANDATORY)**  
   - Call the `answer-formatting` skill with:
     - the exact SQL query you executed
     - the raw result returned by `sql_db_query` (number, row list, etc.)
   - Use **only** the structured object returned by `answer-formatting` as your final answer.

## Workflow for Complex Queries

For questions requiring multiple tables:

### 1. Plan Your Approach
**Use `write_todos` to break down the task:**
- Identify all tables needed
- Map relationships (foreign keys)
- Plan JOIN structure
- Determine aggregations

### 2. Examine Schemas
Use `sql_db_schema` for EACH table to find join columns and needed fields.

### 3. Construct Query
- SELECT - Columns and aggregates
- FROM/JOIN - Connect tables on FK = PK
- WHERE - Filters before aggregation
- GROUP BY - All non-aggregate columns
- ORDER BY - Sort meaningfully
- LIMIT - Default 5 rows

### 4. Validate and Execute
Check all JOINs have conditions, GROUP BY is correct, then run query.

### 5. Final Answer (MANDATORY)
- After executing the query, you MUST finish by returning a **single JSON object** with:
  - `sql_code`: the exact SQL you executed
  - `answer`: 1–3 sentences explaining what the result means for the question
  - `result`: the raw DB result (number, list of rows, list of objects, etc.)
- Do **not** end with planning or status updates (e.g. “I am currently working on…”).
- Do **not** add any extra text or markdown around the JSON object.

## Example: Revenue by Country
```sql
SELECT
    c.Country,
    ROUND(SUM(i.Total), 2) as TotalRevenue
FROM Invoice i
INNER JOIN Customer c ON i.CustomerId = c.CustomerId
GROUP BY c.Country
ORDER BY TotalRevenue DESC
LIMIT 5;
```

## Quality Guidelines

- Query only relevant columns (not SELECT *)
- Always apply LIMIT (5 default)
- Use table aliases for clarity
- For complex queries: use write_todos to plan
- After running SQL to answer a question, **always** finish by using the
  `answer-formatting` skill. Do not end with intermediate thoughts or planning
  messages; the final output should be the structured `{sql_code, answer, result}`.
- Never use DML statements (INSERT, UPDATE, DELETE, DROP)
