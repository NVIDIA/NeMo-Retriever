---
name: query-planning
description: How to turn a user question plus a RetrievalContext (entities, relevant tables, FKs) into a structured query plan that the SQL author can translate verbatim.
---

# Query Planning

The planner reads the question and the Phase 1 `RetrievalContext` and
produces a structured plan. A good plan lets the SQL author write SQL
without re-deriving anything.

## Inputs available to the planner

- `question` — the user question.
- `entity_coverage` — entities resolved by Phase 1 (`metric`, `dimension`,
  `time_filter`, `value`) plus `resolved_as` (`column`, `custom_analysis`,
  `expression`, `value`, `time_filter`, `unresolved`) and an optional
  `sql_expression` for entities that needed synthesis.
- `relevant_tables` — list of allowed tables with their columns.
- `relevant_fks` — the only FK pairs you may use for JOINs.
- `coverage_complete` — whether all metric / dimension entities resolved.

## Plan structure

Produce a plan with the following fields (the `plan_query` Pydantic schema):

- `tables_to_use`: list of `SCHEMA.TABLE AS alias` strings the SQL will use.
- `join_conditions`: list of `alias.col = alias.col` strings, drawn from
  `relevant_fks`.
- `select_expressions`: list of `alias.column` or aggregate expressions.
  Embed `sql_expression` from `entity_coverage` verbatim where applicable.
- `where_conditions`: list of row-level filters (NEVER join predicates).
- `group_by`, `having_conditions`, `order_by`: lists of `alias.column` or
  aggregate references.
- `use_cte`: bool — true when a multi-step decomposition (CTE) makes the
  query clearer.
- `notes`: free-text notes (e.g. which entity was unresolved).

## Decision rules

1. **Match the question's shape first.** Read the user question and pick
   the SQL shape it is actually asking for *before* writing any
   `select_expressions`:

   | Question wording                                              | Shape                                                              |
   |---------------------------------------------------------------|--------------------------------------------------------------------|
   | "how many", "count of", "number of"                           | `SELECT COUNT(...)` — single scalar row                            |
   | "total", "sum of", "what is the total"                        | `SELECT SUM(...)`                                                  |
   | "average", "mean", "avg"                                      | `SELECT AVG(...)`                                                  |
   | "max", "min", "highest", "lowest", "most", "least"            | `SELECT MAX(...)` / `SELECT MIN(...)` (or `ORDER BY ... LIMIT 1`)  |
   | "per <X>", "by <X>", "for each <X>", "breakdown by <X>"       | aggregate + `GROUP BY <X>`                                          |
   | "list", "show", "which", "what are", no aggregate keyword     | `SELECT <columns>` (row-listing)                                   |

   The aggregate keyword wins. Never return a row list when the user
   asked "how many / total / average / …": the answer in that case is a
   single aggregate value, not a column dump.

2. **Pick the smallest table set** that covers all required entities. Avoid
   pulling in tables whose columns just happened to match.
3. **Use only listed FKs** for JOINs. If two tables you need lack a FK,
   you can't JOIN them — flag this in `notes`.
4. **For every `entity_type=expression`**: embed the entity's
   `sql_expression` directly in `select_expressions` (or `having_conditions`
   when it's an aggregate filter).
5. **If `coverage_complete=false`**: still produce the best plan possible
   and note the unresolved entity in `notes`.
6. **GROUP BY discipline**: when aggregating, every non-aggregated
   `select_expression` must appear in `group_by`. ORDER BY may use the same
   aliases or aggregate aliases. For pure scalar aggregates ("how many /
   total / average") with no "per <X>" / "by <X>" phrasing, leave
   `group_by` empty.
7. **JOIN vs WHERE**: never put a join predicate into `where_conditions`.
   Aggregate filters go in `having_conditions`, not `where_conditions`.

## Output discipline

- Every identifier in every list is fully qualified (see `sql-rules` skill).
- Lists are JSON-serialisable strings — no SQL fragments that span multiple
  lines or contain SQL comments.
- The plan must be executable by `generate_sql` without re-consulting the
  schema; if the planner needed the schema to make a decision, that
  decision must already be reflected in the plan.
