---
name: entity-decomposition
description: Atomic decomposition of a user question into typed database-retrieval entities (metric / dimension / time_filter / value). Hard noun-only rule and clause-splitting heuristics.
---

# Entity Decomposition

Decompose every user question into the smallest possible set of typed
entities. Each entity must map to ONE database artifact: one table, one
column, one measurable, one time period, or one literal value. Never
combine multiple concepts into a single entity.

## Entity types

| Type | Description |
|---|---|
| `metric` | A measurable value or aggregate to compute (revenue, count, average, …) |
| `dimension` | A schema concept that maps to a table or column (student, product, customer, …) |
| `time_filter` | A time period or date expression (last month, Q3 2024, yesterday, …) |
| `value` | A specific named literal that becomes a `WHERE` filter (Seattle, Enterprise, John, …) |

`dimension` vs `value`: "city" / "student" / "product" are dimensions
(general concepts). "Seattle" / "John" / "Enterprise" are values
(specific instances → `WHERE col = 'X'`).

## How to split

- Each output column the user asks for → its own entity.
- Each subject / object noun → its own entity.
- Each predicate / filter / threshold → its own entity, paired with the
  field it modifies (keep the threshold / value with its qualifier).
- Each time period or proper-noun literal → its own entity.

## Hard "concrete noun" rule

Every entity term MUST contain a concrete noun (the subject / field /
concept being referred to). A term made up only of comparators, numbers,
qualifiers, quantifiers, or prepositions — with no noun — is invalid and
MUST be merged into the entity for the noun it modifies, by including
that noun in the term itself.

Before emitting an entity, check: does this term name a thing that can be
looked up in a database? If not, fix it.

## Style

- Each entity term is plain natural-language text — spell out comparators
  in words rather than using mathematical symbols.
- Normally up to 3 words per term.
- A single entity describing the whole question is almost always wrong.
